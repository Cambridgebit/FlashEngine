# Qwen3-ASR-1.7B 适配设计（内部契约文档）

状态：**实现中**（2026-11-04）。本文是多方并行开发的接口契约，先于代码存在；
完成后摘要进 `docs/QWEN3_ASR.md`（用户向）与 `docs/SUPPORT_MATRIX.md`。

## 0. 目标与范围

把本地 checkpoint `D:\Materials\AVIC\models\Qwen3-1.7B-ASR`（原始 Qwen 命名，
`model_type=qwen3_asr`，`thinker_config` 包装）接进 FlashEngine：

- 音频输入（16 kHz 单声道 wav）→ 一次离线转写；
- 目标后端：`cuda`（本机 4070 WSL 验证）、`ascend310p`（ssh 板卡）、`corex`；
- **不实现**：流式/麦克风、强制对齐（token classification head）、投机解码、TP>1、W8A8。

## 1. 权威参考实现（唯一口径）

`transformers 5.15.0` 自带完整实现，已拷贝到
`C:\Users\yjq040828\AppData\Local\Temp\qwen3_asr_ref\`：

| 文件 | 关键内容 |
|---|---|
| `modeling_qwen3_asr.py` | `Qwen3ASRAudioEncoder` / `Qwen3ASRAudioAttention` / `SinusoidsPositionEmbedding` / `Qwen3ASRMultiModalProjector` / `Qwen3ASRModel.forward` |
| `feature_extraction_qwen3_asr.py` | log-mel（Whisper 风格，`mel_filter_bank(slaney/slaney, 0–8000Hz)`） |
| `processing_qwen3_asr.py` | 音频 token 数公式、prompt 拼接、输出解析 |
| `configuration_qwen3_asr.py` | 两个 config 类 |

模型自带 `README.md`（`D:\Materials\AVIC\models\Qwen3-1.7B-ASR\README.md`）给出官方
prompt/解码示例；样例音频（WSL 内可直接 `curl` 下载成功）：

- `https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav`
- `https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav`

**关键结论（已核对源码）**：HF 的 `Qwen3ASRModel.forward` 把 `position_ids` 直接透传给
`AutoModel.from_config(text_config)`（= `Qwen3Model`），而 `qwen3` 建模文件里**没有任何
mrope 代码**，`position_ids=None` → 标准 `arange`。即 text_config 里那段
`rope_scaling={"mrope_section":[24,20,20],"mrope_interleaved":true}` 被 HF **忽略**，
等价于普通 RoPE（三路位置相同时 interleaved-mrope 与普通 RoPE 数值等价）。
→ FlashEngine 侧沿用 `qwen3.py` 的普通 RoPE，**不要**实现 mrope。

## 2. 张量口径（以 checkpoint 实测为准）

- 特殊 token：`audio_start=151669`、`audio_end=151670`、`audio_pad=151676`；eos=`[151643,151645]`，pad=151643。
- 音频编码器输入：log-mel `[128, T]`（T = 帧数，右侧补零到 `2*n_window=100` 的整数倍）。
- conv 栈：`conv2d1(1→480,k3,s2,p1) → gelu → conv2d2(480→480) → gelu → conv2d3(480→480) → gelu`，
  输入按 100 帧一块切分：`[B,128,T] → [B,128,nchunk,100] → permute [B,nchunk,128,100] → [B*nchunk,1,128,100]`。
  每块输出 `[1,480,16,13]`。
- `conv_out = Linear(480*16=7680 → 1024, bias=False)`，输入是
  `conv_out_conv.permute(0,3,1,2).reshape(nchunk, 13, 480*16)`（**通道在外、频带在内**）。
- 加性正弦位置表：`SinusoidsPositionEmbedding(max_position_embeddings=13, 1024)`，
  `log_timescale_increment = log(10000)/(channels//2 - 1)`，`cat([sin, cos], dim=1)`，只取前 `13` 行。
- 有效帧筛选：`chunk_post_cnn_len` = 每块 `input_features_mask` 求和后连做三次 `(n-1)//2+1`；
  `valid_mask = arange(13) < len`，`index_select` 把有效行**压平成一个序列**（跨块连续）。
- 24 层 audio encoder layer：pre-LN（**LayerNorm 带 weight+bias**）→ 非因果窗口注意力 → 残差；
  pre-LN → `fc1(1024→4096)` → gelu → `fc2(4096→1024)` → 残差；
  `dtype==fp16` 时每层输出做 `clamp(±(finfo.max-1000))`。
- 注意力窗口：`window_aftercnn = max_len_after_cnn * (n_window_infer // (n_window*2)) = 13*8 = 104`，
  每个样本的压平序列按 104 切窗（末窗为余数），**窗口内全连接、非因果**；`head_dim=1024/16=64`。
- `ln_post`（LayerNorm）→ `proj1(1024→1024)+bias` → gelu → `proj2(1024→2048)+bias` = 音频嵌入。
- 合并：`inputs_embeds[input_ids == 151676] = audio_embeds`（`masked_scatter`，按位置顺序）。
- 音频 token 数：`chunk_len=100`；`rem = frames % 100`；`f=(rem-1)//2+1`；
  `tokens = ((f-1)//2+1 -1)//2+1 + (frames//100)*13`（30 s → 3000 帧 → 30 块 ×13 = **390**）。
- **权重键名**：`thinker.audio_tower.{conv2d1,conv2d2,conv2d3,conv_out,layers.N.*,ln_post,proj1,proj2}`、
  `thinker.model.{embed_tokens,layers.N.*,norm}`、`thinker.lm_head.weight`。
  其中 `proj1/proj2` 对应 HF 的 `multi_modal_projector.linear_1/linear_2`。

## 3. 模块划分与接口契约

### 3.1 `flashengine/asr_processor.py`（owner: asr-ref）

纯 CPU、无 torch 依赖的目标（numpy 即可，可选用 torch 实现 STFT）：
音频 → (mel, input_ids, audio_span)。

```python
AUDIO_PAD_ID = 151676; AUDIO_START_ID = 151669; AUDIO_END_ID = 151670
DEFAULT_SYSTEM_PROMPT = ""          # 官方默认空 system

class AsrProcessor:
    @classmethod
    def from_pretrained(cls, model_path: str) -> "AsrProcessor"
        # 读 tokenizer_config.json（chat_template）+ preprocessor_config.json
    def log_mel(self, waveform: np.ndarray) -> tuple[np.ndarray, int]
        # waveform: float32 mono 16k；返回 (mel[128, T_padded] 到 100 的倍数, valid_frames)
    def num_audio_tokens(self, valid_frames: int) -> int
    def build(self, audio: str | np.ndarray, language: str | None = None,
              system_prompt: str | None = None) -> dict:
        # {"input_ids": list[int], "audio_span": (s, e), "num_audio_tokens": int,
        #  "mel": np.ndarray[128, T], "valid_frames": int, "language": str|None}
    def parse(self, text: str) -> dict   # {"language": str|None, "transcription": str}
```

约束：
- `audio_span` = `input_ids` 中连续 `AUDIO_PAD_ID` 的区间 `[s, e)`，长度必须等于 `num_audio_tokens`；
- prompt 结构（官方 `apply_transcription_request`）：`system(可选)` + `user: <|audio_start|><|audio_pad|>×N<|audio_end|>`
  + `assistant:` 前缀（`language <NAME><asr_text>` 强制语言；默认空 → 模型自己输出 `language <XX><asr_text>`）；
- 直接复用 tokenizer 的 chat_template（`tokenizer_config.json` / `chat_template.json`），不要手写 ChatML；
- wav 读取用标准库 `wave` + numpy；非 16 kHz 需重采样（线性/`np.interp` 足够，或复用 `librosa` 若已装）。

### 3.2 `flashengine/models/qwen3_asr.py`（owner: asr-model）

```python
class Qwen3ASRForConditionalGeneration(nn.Module):
    def __init__(self, hf_config)            # hf_config: transformers Qwen3ASRConfig
    def encode_audio(self, mel: torch.Tensor, valid_frames: int) -> torch.Tensor
        # mel [128,T]（CPU 或 device，计算 dtype bf16/fp16）→ [N_audio_tokens, 2048]
    def forward(self, input_ids, positions, audio_embeds=None, audio_slots=None) -> hidden
        # audio_embeds: [N,2048]（本步要注入的全部行）; audio_slots: [N] int64（在扁平 batch 里的行号）
    def compute_logits(self, hidden)  -> logits
    def load_weights(self, weights)    # 处理 thinker.* 前缀与 proj1/proj2 映射
```

实现约束：
- 文本骨干**复用** `flashengine/models/qwen3.py`（`Qwen3Model` / `Qwen3DecoderLayer` / `Qwen3Attention` /
  `Qwen3MLP` / `RMSNorm`），音频部分纯 `torch.nn`（`nn.Conv2d` / `nn.LayerNorm` / `nn.Linear`），
  **禁止**任何后端专属算子（三后端通吃）。注意力用 `F.scaled_dot_product_attention`
  按窗口做（padding + bool mask，`is_causal=False`），不允许调用 flash-attn / ixformer。
- 中文注释、无新增第三方依赖。
- 需要给 `Qwen3Model.forward` 增加 `inputs_embeds` 入口（或 ASR 子类自带 embed 后调用 layers），
  改动必须**不改变纯文本路径的现有行为**（`qwen3.py` 现有测试/E 锚点不能漂）。
- `flashengine/models/registry.py` 注册 adapter：`name="qwen3_asr"`，
  `model_types=("qwen3_asr",)`，`architectures=("Qwen3ASRForConditionalGeneration",)`，
  `model_config_selector=_text_model_config`（文本 config 供引擎用），
  `supported_spec_methods=frozenset({"none"})`，`aux_layer_resolver=_layer_output_aux`（不会被调用）。
- `flashengine/support_matrix.py`：`TARGETS += ("qwen3_asr",)`，`_TARGET_SPEC_METHODS["qwen3_asr"]={"none"}`，
  `evaluate()` 对它拒绝 `quantization != "none"`（与 qwen3_5 同款门）。

### 3.3 引擎集成（owner: lead，**已实现**）

- `Sequence`：`audio`（processor.build() 产物中的 `mel`/`valid_frames`/`audio_span`/`num_audio_tokens`）、
  `audio_embeds`（device 缓存，抢占/分块 prefill 只编码一次）、`skip_prefix_cache`（带音频即 True）。
  `__setstate__` 里同步初始化（TP>1 不传音频，ASR 已门控 TP=1）。
- `EngineRequests.add_request(prompt, sampling_params, audio=None)`：`validate_audio()` 校验
  `audio_span` 落在 prompt 内、`num_audio_tokens == end-start`、mel 为二维；带音频的 seq `skip_prefix_cache=True`。
  `request_snapshot` 的 stop 判定改用 `config.eos_ids`。
- `LLMEngine.generate(prompts, sampling_params, use_tqdm=True, audio=None)`；audio 支持单个 dict（广播）
  或与 prompts 等长的列表。
- `ModelRunner.prepare_prefill`：对 `audio` 且 `audio_embeds is None` 的 seq 调
  `model.encode_audio(mel, valid_frames)`（mel 转到 device/计算 dtype），本步 query 区间与
  `audio_span` 求交后压平成 `(audio_embeds[N,hidden], audio_slots[N])`，暂存 `self._prefill_audio`，
  由 `run(seqs, is_prefill=True)` 取走传给 `run_model`；`_model_forward` 无音频时保持原调用签名。
  decode 路径不变（无音频）。
- **前缀缓存**：`BlockManager.can_allocate` / `hash_blocks` 对 `seq.skip_prefix_cache` 直接走
  "无缓存"分支（不查表、不登记），纯文本请求不受影响。
- **eos 集合**：`Config.eos_ids`（由 `collect_eos_token_ids` 合并 config.json / generation_config.json /
  tokenizer），`LLMEngine` 在 tokenizer 加载后刷新；`Scheduler.eos_ids` + `_maybe_finish` 用 `in` 判定，
  `config.eos` 保留为单值兼容口。
- **门控**：`Config.__post_init__` 对 `target_adapter.audio_input`（或 name == "qwen3_asr"）强制 TP=1。

### 3.4 脚本 / 服务 / 文档（owner: asr-ops）

- `scripts/asr_benchmark.py`：读 wav 列表 → 逐条转写，打印文本、CER、TTFT/TPS/RTF，支持
  `--backend` / `--eager` / `--language` / `--reference-jsonl`（对比参考转写）。
- `scripts/openai_server.py`：`POST /v1/audio/transcriptions`（multipart/form-data，标准库解析）
  → `{"text": ...}`，`model`/`language` 参数透传。
- `docs/QWEN3_ASR.md` 用户向文档；`docs/SUPPORT_MATRIX.md` + `AGENTS.md` 增量段落。

## 4. 验证口径与结果

1. **音频塔位等**（验收期临时脚本，已随精简移除）：同一 mel 输入，本仓音频塔输出与
   `transformers.Qwen3ASREncoder`（原始 checkpoint 权重，键名去 `thinker.audio_tower.` 前缀）对比。
   实测用例 `full_3000f→390`、`partial_420of500→55`、`single_chunk_100f→13`、`batch2_400/250→[52,33]`、
   `synthetic_200f→26`；fp32 CPU / fp32 CUDA / bf16 CUDA 三档 × 5 用例 × 3 级
   （audio_tower / projector / projector_isolated）**`max_abs_err=0.000e+00`（逐位一致）**。
   全量装载 `keys=623`（`lm_head.weight` 为 tied，跳过）、params 2349.2M。
2. **端到端转写**（CUDA 4070 8GB，`enforce_eager=True, max_model_len=1024, gpu_mem_util=0.85`）：
   - prompt 构造：`AsrProcessor.build()` 的 `input_ids` 与 HF 手工模板（`apply_chat_template` +
     `proc(text=, audio=)`）**逐位一致**：asr_en `214/214`、asr_zh `70/70`；asr_zh 的 mel **逐位相同**。
   - 引擎输出（`LLM.generate(... temperature=0, max_tokens=64)`）：
     - asr_zh（auto）→ `甚至出现交易几乎停滞的情况。`（解析 `language=Chinese`）
     - asr_en（强制 English）→ `Uh huh. Oh yeah, yeah. He wasn't even that big when I started
       listening to him. But and his solo music didn't do overly well. ...`
   - 分块 prefill（`max_num_batched_tokens=64`，prompt 被切成多块）/ 同批多音频 / 音频与纯文本
     混批三种路径均与单块结果一致（见 §4.5）。
3. **三后端**：用户已确认 310P 与 CoreX 机器当前都不可用，本轮**只交付代码路径**。
   - `cuda`：本机 4070 端到端实测通过（上述数字）。
   - `ascend310p` / `corex`：音频塔为纯 torch 算子（显式 matmul+softmax，**不用**
     `F.scaled_dot_product_attention`、不依赖 flash-attn / ixformer / triton），权重加载与
     `index_copy` 走通用路径，因此代码在两条后端上应当可跑，但**未在任何真机验证**；
     `docs/SUPPORT_MATRIX.md` 与 `docs/QWEN3_ASR.md` 必须保持"未验证"标注。
4. **纯文本回归**：Qwen3-1.7B 文本模型 smoke 通过（`'我叫李明，是某大学的计算机专业学生…'`，
   `eos_ids=(151645,151643)`），ASR 改动对纯文本路径零影响（`_model_forward` 无音频时按原签名调用）。
5. **分块/批量回归**（`C:\Users\yjq040828\AppData\Local\Temp\fe_asr_chunk.py`，单进程一次启动，
   `max_num_batched_tokens=64` 远小于 prompt 长度，强制 `prepare_prefill` 按块切分）——**全部通过**：
   | 用例 | 内容 | 结果 |
   |---|---|---|
   | T1 | 单条 asr_zh（prompt 70 tok → 切 2 块） | `甚至出现交易几乎停滞的情况。`（与不切块一致） |
   | T2 | 单条 asr_en（prompt 211 tok → 切 4 块） | 与不切块逐字一致 |
   | T3 | 同批两条音频（70 + 211 tok，各自分块） | 两条都正确，互不串扰 |
   | T4 | 音频 + 纯文本混批 | 音频转写正确；文本请求 `我是一个AI助手，能够回答各种问题…` 正常 |
   | T5 | 同一音频重复请求（`seq.audio_embeds` 缓存路径） | 与 T1 一致 |

   结论：`ModelRunner.prepare_prefill` 里「本步 query 区间 ∩ `audio_span` → 压平成
   `(audio_embeds, audio_slots)`」的扁平行号在**分块 prefill / 多序列同批 / 音频与文本混批**
   三种情形下都正确（`audio_slots` 是拼接后 query 行的全局行号，`flat_offset` 按 `seqlen_q` 累加）。

## 5. 交付清单

| 文件 | 内容 |
|---|---|
| `flashengine/asr_processor.py` | wav→mel、chat template 渲染、`<\|audio_pad\|>` 展开、输出解析、抗混叠重采样 |
| `flashengine/models/qwen3_asr.py` | 纯 torch 音频塔 + 多模态 forward + 原始 `thinker.*` 权重装载 |
| `flashengine/models/registry.py` | `qwen3_asr` adapter（`audio_input=True`） |
| `flashengine/support_matrix.py` | `qwen3_asr`：三后端 × spec=none × quant=none |
| `flashengine/{config,llm}.py`、`flashengine/engine/*` | dtype/eos 解析、Sequence 音频字段、多模态 prefill |
| `scripts/asr_benchmark.py` | 转写基准（TTFT/RTF/TPS/CER） |
| `scripts/openai_server.py` | `POST /v1/audio/transcriptions` |
| `docs/QWEN3_ASR.md`、`docs/SUPPORT_MATRIX.md`、`AGENTS.md` | 用户文档与支持矩阵 |

环境：一切 python 走 WSL `~/vllm-env/bin/python3`（Python 3.12 / torch 2.11+cu130 / transformers 5.15）；
模型目录 Windows `D:\Materials\AVIC\models\Qwen3-1.7B-ASR`
= WSL `/mnt/d/Materials/AVIC/models/Qwen3-1.7B-ASR`。GPU 为 4070 Laptop 8GB（会被桌面占用 ~1GB）。
