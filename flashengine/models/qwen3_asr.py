"""Qwen3-ASR-1.7B（`model_type=qwen3_asr`）目标模型：音频塔 + 多模态投影 + Qwen3 文本骨干。

音频塔是 Whisper 风格 encoder（conv 下采样 → 加性正弦位置 → 24 层非因果窗口注意力 →
ln_post），投影到文本 hidden 后按 audio_pad token 的位置 scatter 进文本输入嵌入。

数值口径（唯一权威）= transformers 5.15 的
`transformers/models/qwen3_asr/modeling_qwen3_asr.py`：

- conv 栈按 `n_window * 2 = 100` 帧一块切分（`[B,128,T] → [B*nchunk,1,128,100]`），
  三次 `Conv2d(3, stride=2, padding=1)` + gelu，每块输出 `[480,16,13]`；
- `conv_out` 把 `[C=480, F=16]` 按 **通道在外、频带在内** 展平成 7680 维；
- 加性正弦位置表（13 x 1024，`log(10000)/(channels//2-1)` 口径）；
- 有效帧（`input_features_mask` 为前缀掩码）压平成跨块连续序列，注意力按
  `window_aftercnn = max_len_after_cnn * (n_window_infer // (n_window*2))` = 104 切窗，
  **窗口内全连接、非因果**；
- 每层 pre-LN（**LayerNorm 带 weight+bias**，eps 取 torch 默认 1e-5）→ 窗口注意力 →
  残差 → pre-LN → fc1 → gelu → fc2 → 残差；**fp16 时每层输出 clamp(±(finfo.max-1000))**；
- `ln_post` → `proj1(1024→1024)+bias` → gelu → `proj2(1024→2048)+bias`。

实现约束（见 docs/QWEN3_ASR_DESIGN.md 3.2）：音频部分只用 `torch.nn` / `F.gelu` /
显式 `matmul + softmax`，不调用 flash-attn / ixformer / SDPA / 任何后端专属算子，
因此 CUDA / CoreX / Ascend310P 三后端共用同一份代码。文本骨干直接复用
`flashengine/models/qwen3.py` 的 `Qwen3Model`（层、RoPE、RMSNorm 全部沿用）。

注意：设计文档 1 节已核对——checkpoint `text_config` 里的 mrope 字段被 HF 忽略，
本仓 `qwen3.py` 走普通 RoPE，**不要**补 mrope。
"""

from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from flashengine.layers.embed_head import ParallelLMHead
from flashengine.models.qwen3 import (
    PACKED_MODULES_MAPPING,
    Qwen3Model,
)
from flashengine.utils.loader import default_weight_loader

# 特殊 token（checkpoint 实测；Qwen3-ASR 全系一致）。
AUDIO_PAD_ID = 151676
AUDIO_START_ID = 151669
AUDIO_END_ID = 151670


class SinusoidsPositionEmbedding(nn.Module):
    """加性正弦位置表 `[length, channels]`（Whisper 式，非学习参数）。

    逐行对齐 HF `SinusoidsPositionEmbedding.compute_default_singular_positional_embedding`：
    `log_timescale_increment = np.log(max_timescale) / (channels//2 - 1)`，
    `cat([sin, cos], dim=1)`。表是 non-persistent buffer（checkpoint 里没有它）。
    """

    def __init__(self, length: int, channels: int, max_timescale: int = 10000) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding 需要偶数 channels")
        self.length = length
        self.channels = channels
        self.max_timescale = max_timescale
        self.register_buffer(
            "positional_embedding", self._build(length, channels, max_timescale),
            persistent=False,
        )

    @staticmethod
    def _build(length: int, channels: int, max_timescale: int) -> torch.Tensor:
        # 与 HF 完全同序：numpy 标量 log → float32 指数/广播（保留逐位一致性）。
        log_timescale_increment = np.log(max_timescale) / (channels // 2 - 1)
        inv_timescales = torch.exp(
            -log_timescale_increment * torch.arange(channels // 2).float()
        )
        scaled_time = torch.arange(length)[:, np.newaxis] * inv_timescales[np.newaxis, :]
        return torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)

    def forward(self, seqlen: int) -> torch.Tensor:
        return self.positional_embedding[:seqlen, :]


def _window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """单个窗口内的非因果全连接注意力（q/k/v: `[1, heads, len, head_dim]`）。

    **刻意不用 `F.scaled_dot_product_attention`**：310P（torch_npu 2.10）与 CoreX
    （ixformer）上 SDPA 的支持面不一致，容易只在板卡上炸；音频塔规模很小
    （30s ≈ 390 token / 4 窗口 / 24 层 / d_model=1024），显式 matmul 代价可忽略，
    且三后端逐位语义一致。窗口内没有 padding（有效帧事先已压平成连续序列），
    因此不需要任何 mask——`softmax` 的 fp32 累加口径与 HF eager 路径相同。
    """
    attn = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = torch.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    return torch.matmul(attn, v)


class Qwen3ASRAudioAttention(nn.Module):
    """音频塔多头注意力：**每个窗口内**全连接、非因果（非因果 = 不做 causal mask）。

    HF 的非 flash 路径把压平序列按 `cu_seqlens` 切窗后逐窗调用注意力接口；本类同构：
    一次性投影 → reshape `[1, H, N, D]` → 按窗口 `torch.split` → 逐窗显式注意力 →
    拼回。窗口内无 padding（有效帧已压平），故不需要 mask。
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.num_heads = config.encoder_attention_heads
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.embed_dim}) 必须能被子数 encoder_attention_heads"
                f" ({self.num_heads}) 整除"
            )
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        seq_length = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(seq_length, self.num_heads, self.head_dim)
        # [N, H, D] → [1, H, N, D]
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        # 窗口边界（压平序列上的累计长度）；token 数很少（30s ≈ 390 token / 4 窗口），
        # 逐窗调用即可，显存友好且与 HF 非 flash 路径逐窗口数值一致。
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        attn_outputs = [
            _window_attention(qq, kk, vv, self.scaling)
            for qq, kk, vv in zip(
                torch.split(q, lengths, dim=2),
                torch.split(k, lengths, dim=2),
                torch.split(v, lengths, dim=2),
            )
        ]
        # 逐窗拼接：各窗长度不同，必须在序列维（dim=2）cat；HF 的 eager 路径先
        # transpose(1,2) 成 [1,N,H,D] 再 cat dim=1，等价。
        attn_output = torch.cat(attn_outputs, dim=2)  # [1, H, N, D]
        # 与 HF 同序：[1,H,N,D] → [1,N,H,D] → [N, H*D]（先转置再展平，否则列序错位）
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.out_proj(attn_output)


class Qwen3ASRAudioEncoderLayer(nn.Module):
    """音频塔单层：pre-LN 注意力 + 残差，pre-LN FFN + 残差（fp16 输出 clamp）。"""

    def __init__(self, config) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Qwen3ASRAudioAttention(config)
        # 注意：checkpoint 里这两个 LayerNorm **带 bias**，不能用 RMSNorm 替代。
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(self, hidden_states: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cu_seqlens)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states

        # 310P 强制 fp16：HF 同款 clamp，避免 fp16 溢出成 inf。
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value)
        return hidden_states


def _post_cnn_length(lengths: torch.Tensor) -> torch.Tensor:
    """三次 (k=3,s=2,p=1) 卷积后的长度；零长度输入保持 0（与 HF 同式）。"""
    for _ in range(3):
        lengths = torch.where(
            lengths > 0, (lengths - 1) // 2 + 1, torch.zeros_like(lengths))
    return lengths


def feat_extract_output_lengths(input_lengths: torch.Tensor, n_window: int = 50) -> torch.Tensor:
    """HF `_get_feat_extract_output_lengths`：音频帧数 → 音频 token 数。

    公式：`rem = frames % (2*n_window)`；`f = (rem-1)//2+1`；
    `tokens = ((f-1)//2+1 -1)//2+1 + (frames//(2*n_window)) * time_steps`，
    其中 `time_steps` = 100 帧经三次 stride-2 conv 后的长度（=13）。
    """
    chunk_len = n_window * 2
    extra = _post_cnn_length(torch.tensor([chunk_len]))[0].item()
    input_lengths_leave = input_lengths % chunk_len
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    return ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // chunk_len) * extra


class Qwen3ASRAudioEncoder(nn.Module):
    """Whisper 风格音频塔：conv 下采样 + 正弦位置 + N 层窗口注意力 + ln_post。"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        embed_dim = config.d_model
        self.embed_dim = embed_dim
        self.num_mel_bins = config.num_mel_bins
        self.n_window = config.n_window
        self.n_window_infer = config.n_window_infer
        chunk_len = self.n_window * 2
        # 每块 conv 输出的频带数：128 → 64 → 32 → 16（三次 (k3,s2,p1)）。
        freq_bins = chunk_len  # 占位，下面按 num_mel_bins 计算，避免歧义
        freq_bins = self.num_mel_bins
        for _ in range(3):
            freq_bins = (freq_bins + 1) // 2
        self.num_freq_bins = freq_bins
        self.time_steps = _post_cnn_length(torch.tensor([chunk_len]))[0].item()
        self.downsample_hidden_size = config.downsample_hidden_size

        self.conv2d1 = nn.Conv2d(1, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d2 = nn.Conv2d(
            config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv2d3 = nn.Conv2d(
            config.downsample_hidden_size, config.downsample_hidden_size, 3, 2, padding=1)
        self.conv_out = nn.Linear(
            config.downsample_hidden_size * freq_bins, embed_dim, bias=False)

        self.positional_embedding = SinusoidsPositionEmbedding(
            config.max_position_embeddings, embed_dim)
        self.layers = nn.ModuleList(
            [Qwen3ASRAudioEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.ln_post = nn.LayerNorm(embed_dim)

    # ---- 窗口切分（对齐 HF get_audio_cu_seqlens）----
    def _cu_seqlens(
        self,
        chunk_lengths: torch.Tensor,
        feature_lens: torch.Tensor,
    ) -> torch.Tensor:
        """按样本把压平序列切成 ≤window_aftercnn 的窗口，返回 int32 累计边界。"""
        aftercnn_lens = feat_extract_output_lengths(feature_lens, self.n_window)
        feature_lens_after_cnn = feat_extract_output_lengths(
            chunk_lengths, self.n_window)
        max_len_after_cnn = feature_lens_after_cnn.max().item()
        window_aftercnn = max_len_after_cnn * (
            self.n_window_infer // (self.n_window * 2))
        if window_aftercnn <= 0:
            raise ValueError(
                f"非法注意力窗口 window_aftercnn={window_aftercnn}"
                f"（n_window_infer={self.n_window_infer}, n_window={self.n_window}）")

        cu_chunk_lens = [0]
        for cnn_len in aftercnn_lens.tolist():
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            remainder = cnn_len % window_aftercnn
            if remainder != 0:
                cu_chunk_lens += [remainder]
        # 常量张量在 CPU 建好再搬设备（310P 旧驱动对 NPU 侧构造更敏感）。
        return torch.tensor(cu_chunk_lens, dtype=torch.int64).to(
            feature_lens.device).cumsum(-1, dtype=torch.int32)

    def forward(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor,
    ) -> torch.Tensor:
        """`input_features`: `[B,128,T_padded]`（T 为 chunk_len 整数倍，右侧补零）；
        `input_features_mask`: `[B,T_padded]`，1=真实帧、0=padding。返回压平的
        `[N_audio_tokens, d_model]`（跨样本连续，与 HF last_hidden_state 同序）。"""
        if input_features.dim() != 3:
            raise ValueError(
                f"input_features 期望 [B, mel, T]，收到 {tuple(input_features.shape)}")
        batch_size, num_mel_bins, padded_feature_length = input_features.shape
        chunk_len = self.n_window * 2
        if padded_feature_length % chunk_len != 0:
            raise ValueError(
                f"mel 帧数必须是 n_window*2={chunk_len} 的整数倍，"
                f"收到 {padded_feature_length}")
        num_chunks = padded_feature_length // chunk_len

        # 计算 dtype 跟随卷积权重（引擎统一 dtype；parity 脚本显式给 fp32/bf16）。
        input_features = input_features.to(self.conv2d1.weight.dtype)
        input_features_mask = input_features_mask.to(torch.long)

        feature_lens = input_features_mask.sum(-1)
        chunk_lengths = (
            input_features_mask
            .view(batch_size, num_chunks, chunk_len)
            .sum(dim=-1)
            .reshape(-1)
        )
        cu_seqlens = self._cu_seqlens(chunk_lengths, feature_lens)

        # ---- conv 栈：按 chunk 切分，输出 [B*nchunk, C=480, F=16, Ts=13] ----
        chunked = (
            input_features
            .view(batch_size, num_mel_bins, num_chunks, chunk_len)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * num_chunks, 1, num_mel_bins, chunk_len)
        )
        conv_out = F.gelu(self.conv2d1(chunked))
        conv_out = F.gelu(self.conv2d2(conv_out))
        conv_out = F.gelu(self.conv2d3(conv_out))
        total_chunks, conv_channels, freq_bins, time_steps = conv_out.size()
        # 通道在外、频带在内 → [chunk, 13, 480*16]
        conv_out = self.conv_out(
            conv_out.permute(0, 3, 1, 2).contiguous()
            .view(total_chunks, time_steps, conv_channels * freq_bins))
        conv_out = conv_out + self.positional_embedding(time_steps).to(conv_out.dtype)

        # ---- 有效帧压平：每块只保留前 chunk_post_cnn_len 行 ----
        chunk_post_cnn_lens = _post_cnn_length(chunk_lengths)
        # 与 HF 的 valid_mask.flatten().nonzero() 等价，但在 host 侧构造索引：
        # 避免 NPU 上 nonzero 的动态形状（padding 掩码恒为前缀）。
        counts = chunk_post_cnn_lens.detach().to("cpu").tolist()
        valid_indices = [
            chunk * time_steps + row
            for chunk, count in enumerate(counts)
            for row in range(int(count))
        ]
        if not valid_indices:
            raise ValueError(
                "音频塔没有有效帧（input_features_mask 全 0 或 valid_frames=0）")
        index = torch.tensor(valid_indices, dtype=torch.long)
        hidden_states = torch.index_select(
            conv_out.reshape(-1, conv_out.shape[-1]), 0, index.to(conv_out.device))

        for layer in self.layers:
            hidden_states = layer(hidden_states, cu_seqlens)
        return self.ln_post(hidden_states)


class Qwen3ASRMultiModalProjector(nn.Module):
    """音频 → 文本 hidden 投影：`linear_1 + gelu + linear_2`（checkpoint 的 proj1/proj2）。"""

    def __init__(self, audio_config) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(audio_config.d_model, audio_config.d_model)
        self.linear_2 = nn.Linear(audio_config.d_model, audio_config.output_dim)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.linear_2(F.gelu(self.linear_1(audio_features)))


def _thinker_section(hf_config, key: str):
    """取原始 Qwen 检查点 `config.json` 里 `thinker_config[key]` 的 dict（若有）。

    原始 checkpoint 的 `config.json` 用 `thinker_config` 包住 audio/text 两段；
    `AutoConfig.from_pretrained` 不认识这个包装，会用 `Qwen3ASRConfig` 的默认值
    （audio 段默认 `output_dim=3584`，与本 checkpoint 的 2048 不符），故 audio 段
    优先读 `thinker_config.audio_config` 的真值。
    """
    thinker = getattr(hf_config, "thinker_config", None)
    if isinstance(thinker, dict):
        section = thinker.get(key)
        if isinstance(section, dict):
            return section
    return None


def resolve_audio_config(hf_config):
    """解析音频塔 config：优先原始 `thinker_config.audio_config`，其次 hf_config.audio_config。"""
    from transformers.models.qwen3_asr.configuration_qwen3_asr import (
        Qwen3ASREncoderConfig,
    )

    section = _thinker_section(hf_config, "audio_config")
    if section is not None:
        allowed = set(Qwen3ASREncoderConfig().to_dict().keys())
        kwargs = {k: v for k, v in section.items() if k in allowed}
        # 原始 checkpoint 的 model_type 是 "qwen3_asr_audio_encoder"（CONFIG_MAPPING
        # 里没有），构造 Qwen3ASREncoderConfig 时按类默认值处理即可。
        kwargs.pop("model_type", None)
        return Qwen3ASREncoderConfig(**kwargs)
    audio_config = getattr(hf_config, "audio_config", None)
    if audio_config is None or not hasattr(audio_config, "d_model"):
        raise NotImplementedError(
            "无法从 hf_config 解析 Qwen3-ASR 的 audio_config：既没有 "
            "thinker_config.audio_config，也没有 audio_config 属性"
        )
    return audio_config


def resolve_text_config(hf_config):
    """文本骨干 config：多模态壳取 `text_config`，纯文本 config 原样返回。"""
    return getattr(hf_config, "text_config", hf_config)


class Qwen3ASRForConditionalGeneration(nn.Module):
    """Qwen3-ASR 目标模型：音频塔 + 投影 + Qwen3 文本骨干（tied lm_head）。

    引擎接口（docs/QWEN3_ASR_DESIGN.md 3.2）：
    - `encode_audio(mel, valid_frames) -> [N, 2048]`：一次算好音频嵌入（引擎缓存在
      Sequence 上，只在 prefill 前调一次）；
    - `forward(input_ids, positions, audio_embeds, audio_slots)`：先 `embed_tokens`
      再把音频嵌入写到**扁平 batch** 的 `audio_slots` 行，最后过文本层；
    - `compute_logits(hidden)` / `load_weights(weights)`（处理 `thinker.*` 前缀与
      `proj1/proj2 → multi_modal_projector.linear_1/linear_2` 映射）。

    纯文本路径（`audio_embeds=None`）与 `Qwen3ForCausalLM` 完全同构——本类不改
    `qwen3.py`，只在 ASR 子类内联同构的 embed→layers→norm 前向，保证既有 Qwen3/
    EAGLE3/DFlash 路径零影响。
    """

    packed_modules_mapping = PACKED_MODULES_MAPPING

    def __init__(self, hf_config) -> None:
        super().__init__()
        self.config = hf_config
        self.text_config = resolve_text_config(hf_config)
        self.audio_config = resolve_audio_config(hf_config)
        self.audio_token_id = int(
            getattr(hf_config, "audio_token_id", AUDIO_PAD_ID))

        # 音频塔 + 投影（纯 torch）
        self.audio_tower = Qwen3ASRAudioEncoder(self.audio_config)
        self.multi_modal_projector = Qwen3ASRMultiModalProjector(self.audio_config)

        # 文本骨干沿用 qwen3.py（层 / RoPE / RMSNorm / qk_norm 全复用）
        self.model = Qwen3Model(self.text_config)
        self.lm_head = ParallelLMHead(
            self.text_config.vocab_size, self.text_config.hidden_size)
        if getattr(self.text_config, "tie_word_embeddings", True):
            # 与 Qwen3ForCausalLM 同款绑定：lm_head 与 embed_tokens 共享存储。
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    # ---------------- 音频 ----------------
    def encode_audio(
        self,
        mel: torch.Tensor,
        valid_frames: int | torch.Tensor,
    ) -> torch.Tensor:
        """音频塔 + 投影：`mel [128,T]`（或 `[B,128,T]`）→ 计算 dtype 的 `[N,2048]`。

        `valid_frames`：真实 mel 帧数（T 右侧的补零不计）。多 batch 时可给
        int 或逐样本序列。返回行数 = 音频 token 数（见 `feat_extract_output_lengths`），
        跨样本/跨块连续排列，与输入 `audio_pad` 的出现顺序一致。
        """
        chunk_len = self.audio_tower.n_window * 2
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            valid = [int(valid_frames)]
        elif mel.dim() == 3:
            if isinstance(valid_frames, torch.Tensor):
                valid = [int(v) for v in valid_frames.tolist()]
            elif isinstance(valid_frames, (list, tuple)):
                valid = [int(v) for v in valid_frames]
            else:
                valid = [int(valid_frames)] * mel.shape[0]
        else:
            raise ValueError(f"mel 期望 [128,T] 或 [B,128,T]，收到 {tuple(mel.shape)}")

        num_mel_bins, total_frames = mel.shape[1], mel.shape[2]
        # 右侧补零到 chunk 整数倍（processor 已补时此处是 no-op）
        pad = (-total_frames) % chunk_len
        if pad:
            mel = F.pad(mel, (0, pad))
        padded_frames = mel.shape[2]
        device = mel.device
        positions = torch.arange(padded_frames, device=device)
        mask = torch.stack([
            (positions < max(0, min(int(v), padded_frames))).to(torch.long)
            for v in valid
        ])

        with torch.no_grad():
            hidden_states = self.audio_tower(mel, mask)
            audio_embeds = self.multi_modal_projector(hidden_states)

        # token 数自检：行数必须等于公式值（否则是 mel 补零/valid_frames 口径不一致）。
        frames_tensor = torch.tensor(valid, dtype=torch.long)
        expected = int(
            feat_extract_output_lengths(
                frames_tensor, self.audio_tower.n_window).sum().item())
        if audio_embeds.shape[0] != expected:
            raise ValueError(
                "音频 token 数与公式不一致：塔输出 "
                f"{audio_embeds.shape[0]} 行 vs 期望 {expected}（valid_frames={valid}）")
        return audio_embeds

    # ---------------- 文本 ----------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        audio_embeds: torch.Tensor | None = None,
        audio_slots: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model.embed_tokens(input_ids)
        if audio_embeds is not None:
            if audio_slots is None:
                raise ValueError("audio_embeds 非空时必须给 audio_slots（扁平 batch 行号）")
            hidden_states = hidden_states.index_copy(
                0,
                audio_slots.to(torch.long).to(hidden_states.device),
                audio_embeds.to(hidden_states.dtype).to(hidden_states.device),
            )
        # 与 qwen3.py `Qwen3Model.forward` 的 embed→layers→norm 逐行同构
        #（ASR 不开投机，无需 aux 隐层捕获）。
        residual = None
        for layer in self.model.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.model.norm(hidden_states, residual)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    # ---------------- 权重装载 ----------------
    def _copy_param(self, target: str, tensor: torch.Tensor) -> None:
        param = self.get_parameter(target)
        if param.shape != tensor.shape:
            raise ValueError(
                f"权重形状不匹配 {target}: checkpoint {tuple(tensor.shape)} vs "
                f"模型 {tuple(param.shape)}")
        param.data.copy_(tensor)

    def load_weights(self, weights) -> None:
        """vllm 式流式装载。`weights`: Iterable[(name, tensor)]。

        映射（对齐原始 Qwen checkpoint）：
        - `thinker.` 前缀去掉；
        - `thinker.audio_tower.proj1/proj2` → `multi_modal_projector.linear_1/2`；
        - 其余 `audio_tower.*` 同名装载（conv/layers/ln_post）；
        - `thinker.model.*` → 文本骨干（走 packed 映射：q/k/v → qkv_proj，
          gate/up → gate_up_proj）；
        - `thinker.lm_head.weight`：tied 时跳过（与 embed_tokens 共享存储，
          由 embed_tokens 装载即可；不跳过会用同一份权重二次覆盖）。
        """
        tied = bool(getattr(self.text_config, "tie_word_embeddings", True))
        loaded: set[str] = set()
        ignored: list[str] = []
        for name, tensor in weights:
            if name.startswith("thinker."):
                name = name[len("thinker."):]

            if name.startswith("audio_tower.proj1."):
                target = "multi_modal_projector.linear_1." + name.rsplit(".", 1)[-1]
            elif name.startswith("audio_tower.proj2."):
                target = "multi_modal_projector.linear_2." + name.rsplit(".", 1)[-1]
            elif name.startswith("audio_tower."):
                target = name
            elif name == "lm_head.weight":
                if tied:
                    # tied 时 lm_head 与 embed_tokens 共享存储：embed_tokens 的权重
                    # 已覆盖同一块参数，跳过；若也装载会二次覆盖（数值相同但浪费）。
                    ignored.append(name)
                    continue
                target = name
            elif name.startswith("model."):
                for key, (module_name, shard_id) in PACKED_MODULES_MAPPING.items():
                    if key in name:
                        target = name.replace(key, module_name)
                        param = self.get_parameter(target)
                        param.weight_loader(param, tensor, shard_id)
                        loaded.add(target)
                        break
                else:
                    param = self.get_parameter(name)
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader)
                    weight_loader(param, tensor)
                    loaded.add(name)
                continue
            else:
                raise NotImplementedError(f"Qwen3-ASR 未识别的权重键 {name!r}")

            self._copy_param(target, tensor)
            loaded.add(target)

        self._num_loaded_weights = len(loaded)
        self._ignored_weight_keys = tuple(ignored)
