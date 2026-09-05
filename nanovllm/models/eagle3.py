"""EAGLE3 draft 模型（官方 SafeAILab/EAGLE cnets.py 的 chain 推理等价实现）。

权重结构（Qwen3 系列 EAGLE-3 checkpoint）：
  - fc:              Linear(3*hidden -> hidden)，输入为 target 3 个 aux 隐层拼接
  - midlayer:        单层解码层；qkv 输入为 [input_layernorm(embed), hidden_norm(x)] 拼接（2*hidden）
  - norm / lm_head:  draft 头（draft_vocab_size，如 32000）
  - d2t / t2d:       draft 词表 -> target 词表的偏移映射（real_id = draft_idx + d2t[draft_idx]）
  - embed_tokens:    与 target 共享，checkpoint 中不含
"""

import os
from glob import glob

import torch
import torch.nn.functional as F
from torch import nn

from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import (
    ReplicatedLinear,
    QKVParallelLinear,
    RowParallelLinear,
    MergedColumnParallelLinear,
)
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.utils.loader import default_weight_loader


class Eagle3MidLayer(nn.Module):
    """对应官方 cnets.LlamaDecoderLayeremb（midlayer）。"""

    def __init__(self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
        self.self_attn = Qwen3AttentionModule(config, head_dim, hidden_size)
        self.mlp = Qwen3MLPModule(config)
        # hidden_norm 作用于 fc 输出（或上一步 draft 隐层），input_layernorm 作用于 embedding。
        self.hidden_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(self, positions: torch.Tensor, x: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        # 对齐 vLLM llama_eagle3 LlamaDecoderLayer：embeds 过 input_layernorm，
        # seed 过 hidden_norm；注意力输入 = cat([normed embeds, normed seed])。
        residual = x
        h = self.hidden_norm(x)
        e = self.input_layernorm(embeds)
        o = self.self_attn(positions, torch.cat([e, h], dim=-1))
        h = residual + o
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h


class Qwen3MLPModule(nn.Module):
    """gate/up/down MLP（权重名 midlayer.mlp.*，经 packed_modules_mapping 重映射）。"""

    def __init__(self, config) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        assert config.hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class Qwen3AttentionModule(nn.Module):
    """midlayer 的注意力：qkv 输入为 2*hidden 拼接向量，无 q/k norm（llama 风格）。"""

    def __init__(self, config, head_dim: int, hidden_size: int) -> None:
        super().__init__()
        total_num_heads = config.num_attention_heads
        total_num_kv_heads = config.num_key_value_heads
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads
        self.head_dim = head_dim
        self.q_size = total_num_heads * head_dim
        self.kv_size = total_num_kv_heads * head_dim
        self.scaling = head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            2 * hidden_size,
            head_dim,
            total_num_heads,
            total_num_kv_heads,
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.q_size,
            hidden_size,
            bias=False,
        )
        self.rotary_emb = get_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=config.max_position_embeddings,
            base=getattr(config, "rope_theta", 1000000),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q = qkv[..., :self.q_size].view(-1, self.num_heads, self.head_dim)
        k = qkv[..., self.q_size:self.q_size + self.kv_size].view(-1, self.num_kv_heads, self.head_dim)
        v = qkv[..., self.q_size + self.kv_size:].view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        return self.o_proj(o.flatten(1, -1))


class Eagle3DraftModel(nn.Module):

    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config, target_embed_tokens: nn.Module) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        # embedding 与 target 共享（权重由 target 的加载流程填充，checkpoint 中无此权重）。
        self.embed_tokens = target_embed_tokens
        self.fc = ReplicatedLinear(3 * hidden_size, hidden_size, bias=False)
        self.midlayer = Eagle3MidLayer(config)
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ReplicatedLinear(hidden_size, config.draft_vocab_size, bias=False)
        # real_id = draft_idx + d2t[draft_idx]（官方实现存的偏移量，见 cnets.topK_genrate）。
        self.register_buffer("d2t", torch.zeros(config.draft_vocab_size, dtype=torch.int64))
        self.register_buffer("t2d", torch.zeros(config.vocab_size, dtype=torch.bool))

    def forward_decode(
        self,
        positions: torch.Tensor,
        seed: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单 token 起草步。seed 为上一步 midlayer 输出（首步为 fc(aux) 结果）。

        返回 (draft_hidden, draft_logits)：draft_hidden 直接作为下一步 seed。
        """
        embeds = self.embed_tokens(token_ids)
        out = self.midlayer(positions, seed, embeds)
        logits = F.linear(self.norm(out), self.lm_head.weight)
        return out, logits

    def forward_prefill(
        self,
        positions: torch.Tensor,
        seed: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """prefill 路径（varlen，context 已由调用方设置）；输出仅用于回传形状，可丢弃。"""
        embeds = self.embed_tokens(token_ids)
        return self.midlayer(positions, seed, embeds)

    def project_aux(self, aux: torch.Tensor) -> torch.Tensor:
        """fc(3H->H) 投影：3 个 aux 隐层拼接映射回 draft 隐层空间。"""
        return F.linear(aux, self.fc.weight)

    def map_to_target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids + self.d2t[draft_ids]


def load_eagle3_draft(model: Eagle3DraftModel, path: str) -> None:
    bin_path = os.path.join(path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu", weights_only=True)

        def items():
            yield from state.items()
    else:
        from safetensors import safe_open
        files = glob(os.path.join(path, "*.safetensors"))
        assert files, f"EAGLE3 draft 目录缺少权重文件: {path}"

        def items():
            for file in files:
                with safe_open(file, "pt", "cpu") as f:
                    for name in f.keys():
                        yield name, f.get_tensor(name)

    for name, tensor in items():
        if name in ("d2t", "t2d"):
            buf = getattr(model, name)
            buf.data.copy_(tensor.to(buf.device))
            continue
        target_name = name
        shard_id = None
        for k, (v, shard) in model.packed_modules_mapping.items():
            if k in name:
                target_name = name.replace(k, v)
                shard_id = shard
                break
        param = model.get_parameter(target_name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        if shard_id is not None:
            weight_loader(param, tensor, shard_id)
        else:
            weight_loader(param, tensor)
