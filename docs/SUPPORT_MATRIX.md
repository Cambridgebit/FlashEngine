# 支持矩阵（Support Matrix）

本文是 FlashEngine 当前**声明支持**的组合清单：目标模型 × 后端 × 量化 × 投机方法。
单一事实来源是 `flashengine/support_matrix.py`；`Config` 在启动时调用
`require_supported(...)`，不支持的组合会**干净拒绝**（`NotImplementedError`，不静默降级）。
本文件与 `flashengine/support_matrix.py` 保持一致（后者为代码单一事实来源，启动即校验）。

> 「声明支持」≠「已实机验收」。运行时能力（310P 缺 SplitFuse、图捕获失败、草稿 aux 层
> 不匹配等）可能进一步收紧并优雅降级，见各 backend 的 `normalize_config`/`validate_config`
> 与图捕获回退逻辑。

## 1. 总览

| 目标模型 | 普通推理 | 投机推理 | 量化推理 |
|---|---|---|---|
| **Qwen3 dense** | ✅ cuda / corex / ascend310p | ✅ EAGLE3 / DFlash / DSpark（三后端） | ✅ W8A8，仅 ascend310p |
| **Qwen3 MoE**（Qwen3-30B-A3B） | ✅ cuda / corex / ascend310p | ✅ DFlash（三后端） | ✅ W8A8，仅 ascend310p |
| **Qwen3.5 dense / MoE 文本** | ✅ cuda / corex / ascend310p | ✅ DFlash（三后端） | ❌（当前 fp16/bf16） |
| **Qwen3-ASR**（qwen3_asr，语音转写） | ✅ cuda（已实测）/ corex / ascend310p（后两者未实测；TP=1，单条音频） | ❌（仅 none） | ❌（仅 fp16/bf16） |

> Qwen3 MoE 在 cuda/corex 上走纯 torch 专家前向（`BackendBase.fused_moe`：index_select +
> bmm，形状静态、可被 CUDA Graph 捕获）；310P 覆盖为 npu grouped matmul（含 W8A8_DYNAMIC）。
>
> **dtype**：cuda/corex 的 flash-attn 仅支持 fp16/bf16，fp32 检查点由
> `BackendBase.normalize_config` 自动降为 bf16（打印告警，加载期完成转换）；310P 用
> `default_dtype=fp16` 强制 fp16。
>
> **INT8（Quark W8A8）**：AMD Quark 导出的 int8 检查点（config.json 的
> `quantization_config.quant_method == quark`）在 **cuda/corex** 上于**加载期**按
> per-channel `weight_scale`/`weight_zero_point` 反量化成计算 dtype（`utils/loader.py`），
> 运行时与普通 bf16 模型完全一致——**是 int8 检查点适配，不是 int8 激活/GEMM 计算**
> （激活不量化）。310P 的 W8A8 走其自有算子，不认 Quark 格式。

## 2. 完整组合矩阵（target × backend × quantization × spec_method）

`spec_method = none` 表示普通推理。`quantization = none` 表示 fp16/bf16。

| target | backend | quantization | 支持投机方法 |
|---|---|---|---|
| qwen3 | cuda | none | none / eagle3 / dflash / dspark |
| qwen3 | corex | none | none / eagle3 / dflash / dspark |
| qwen3 | ascend310p | none | none / eagle3 / dflash / dspark |
| qwen3 | ascend310p | w8a8 | none / eagle3 / dflash / dspark |
| qwen3 | cuda / corex | w8a8 | ❌（W8A8 仅 310P） |
| qwen3 | cuda / corex | quark_w8a8 | none / eagle3 / dflash / dspark |
| qwen3 | ascend310p | quark_w8a8 | ❌（Quark 仅 cuda/corex） |
| qwen3_moe | cuda / corex | none | none / dflash |
| qwen3_moe | ascend310p | none | none / dflash |
| qwen3_moe | ascend310p | w8a8 | none / dflash |
| qwen3_moe | cuda / corex | w8a8 | ❌（W8A8 仅 310P） |
| qwen3_moe | cuda / corex | quark_w8a8 | none / dflash |
| qwen3_moe | ascend310p | quark_w8a8 | ❌（Quark 仅 cuda/corex） |
| qwen3_moe | cuda / corex / ascend310p | 任意 | ❌ eagle3 / dspark（无 MoE 草稿） |
| qwen3_5 / qwen3_5_moe | cuda / corex / ascend310p | none | none / dflash |
| qwen3_5 / qwen3_5_moe | cuda / corex / ascend310p | w8a8 / quark_w8a8 | ❌ |
| qwen3_5 / qwen3_5_moe | cuda / corex / ascend310p | 任意 | ❌ eagle3 / dspark |
| qwen3_asr | cuda / corex / ascend310p | none | none |
| qwen3_asr | 任意 | w8a8 / quark_w8a8 | ❌（ASR 未接入量化） |
| qwen3_asr | 任意 | 任意 | ❌ eagle3 / dflash / dspark（无音频投机草稿） |

Qwen3-ASR（`qwen3_asr`）是**音频输入**目标模型：prompt 由
`flashengine/asr_processor.py` 在 CPU 上渲染（mel + `<|audio_pad|>` span），
引擎对该请求做一次音频塔前向得到音频嵌入，再按普通文本 decode。
约束：TP=1（`audio_input` 在 `config.py` 强制）、单请求单条音频、音频请求
`skip_prefix_cache`（音频嵌入不能跨请求复用）、仅 fp16/bf16。一次请求的音频
建议 ≤30s（模型窗口），超长请分块。命令行入口与 OpenAI 转写端点见
[`docs/QWEN3_ASR.md`](QWEN3_ASR.md)。

Qwen3.5 当前 TP=1、仅文本，支持完整块 KV + conv/SSM 快照的 prefix cache，
以及保留 conv/SSM 历史的分块 prefill；DFlash 另恢复草稿 KV。
CUDA 的 FlashAttention 支持普通与 DFlash 整轮图；
FlashInfer 当前 DFlash 仍走 eager。Qwen3.5 dense 的 CUDA 与 310P 4B 已实机验收；
MoE 本轮使用 64.6MB 公开 tiny 检查点在 CUDA/FlashAttention 验收普通/DFlash 图和前缀缓存，
CoreX/310P MoE 仍待板卡验收。图捕获失败按既有机制回退。
新版 DFlash 的逐层滑窗/因果语义已适配；310P 的窗口草稿走稠密掩码，
旧式全局非因果草稿仍用分页 SplitFuse。tiny 草稿未训练，功能通过不代表已验证加速比。
详细记录见 [Qwen3.5](QWEN35.md) 与 [Qwen3.5 MoE](QWEN35_MOE.md)。

## 3. 规则（`support_matrix.evaluate`）

1. **后端**：`qwen3` 与 `qwen3_moe` 三后端全放行；MoE 专家前向 cuda/corex 走
   `BackendBase` 纯 torch 实现，ascend310p 走 npu grouped matmul。
2. **量化**：
   - `w8a8`（Ascend msmodelslim）：仅 `ascend310p`，由模型目录的
     `quant_model_description.json` 自动识别（注意力 qkv/o = `W8A8` 静态，
     MoE 专家 = `W8A8_DYNAMIC`，gate/embed/lm_head 保持 fp16）；
   - `quark_w8a8`（AMD Quark）：仅 `cuda`/`corex`，由 config.json 的
     `quantization_config.quant_method == quark` 自动识别；加载期反量化为计算
     dtype（`utils/loader.py:iter_safetensors_dequant`），gate/embed/lm_head/归一化层
     不量化。
3. **投机方法**：取目标 adapter 的 `supported_spec_methods`：
   - `qwen3` = {eagle3, dflash, dspark}；
   - `qwen3_moe` = {dflash}（EAGLE3/DSpark 无 MoE 草稿；MoE 与 dense 草稿的 aux
     口径相同——都抽残差流，故 DFlash 草稿可直接用于 MoE 目标，`target_layer_ids + 1`）。
4. **量化 × 投机**：二者在实现上相互独立（量化改目标前向，投机加草稿与 verify），
   故组合放行；但该组合未单独实机验证（见第 4 节）。
5. **音频目标（`qwen3_asr`）**：`supported_spec_methods` = {none}、量化仅 none；
   `audio_input=True` 的 adapter 使 `Config` 强制 TP=1。音频请求由调用方带
   `audio={mel, valid_frames, audio_span, num_audio_tokens}` 提交，引擎校验
   span/token 数后对该序列关闭前缀缓存；chat/文本请求不受影响。

## 4. 待实机验收（cuda MoE 已冒烟；corex / 310P 无硬件未运行）

| 组合 | 验收点 |
|---|---|
| qwen3 / qwen3_moe 普通（cuda） | 已实机冒烟：`tiny_qwen3moe_hf` 跑通 benchmark，decode 图 100% 捕获；与 HF bf16 贪心 2/3 prompt 逐 token 全等，1 条为 bf16 平票翻转（HF fp32 top-1 = 引擎选择） |
| qwen3 / qwen3_moe 普通（corex） | 与 cuda 同源的纯 torch 专家前向；待 CoreX 实机位等 |
| qwen3 / qwen3_moe 普通（310P） | 算子链已静态对齐 vllm-ascend 310P（接线与数值经无硬件纯 torch 桩回归验证）；与 transformers 逐 token 位等（含 decoder_sparse_step>1 的混合 dense/稀疏层结构）仍待实机 |
| qwen3 dense + EAGLE3/DFlash/DSpark | 与现有基线一致（E 锚点） |
| qwen3 / qwen3_moe W8A8 | 与 fp16 输出对齐（或已知 ULP 平票豁免）；量化尺度键名/shape 与真实 checkpoint 对齐 |
| qwen3_moe 普通 | MoE decode 图捕获率（失败按桶降级 eager） |
| qwen3_moe + DFlash | spec-vs-base 位等 + **E 显著 > 1.5**（E 是 aux 正确性信号） |
| qwen3_moe quark_w8a8（cuda） | 已实机：`amd/tiny-qwen3-moe-w8a8-int8`（8 专家/top-2/2 层）跑通 benchmark，decode 图 100% 捕获；与 HF 用同批反量化权重贪心生成 **32/32 逐 token 全等** |
| qwen3_moe quark_w8a8（corex） | 与 cuda 同源的加载期反量化；待 CoreX 实机 |
| 量化 + 投机 | 组合冒烟（未单独验证） |
| qwen3_asr 普通（cuda） | ✅ 已实测（2026-11-04，4070 8GB）：CPU processor `build()` + 音频塔 vs HF fp32 位等（max_abs_err=0.0）+ 端到端转写 asr_zh/asr_en 文本正确、**CER 0.0%/0.0%**（字符级口径）、TTFT 均值 78.8–90.8ms、RTF 聚合 0.114–0.133、TPS 聚合 22.6–26.4（eager/mml1024/ms1/bf16）；非 eager 稳态 zh RTF 0.0588 / en 0.0522。细节见 [`docs/QWEN3_ASR.md`](QWEN3_ASR.md) §4.1 |
| qwen3_asr 普通（corex） | ❌ 未在真机验证（用户已确认本轮 CoreX 不可用，只加代码路径：音频塔纯 torch、无特定 attention 依赖、无新依赖；等机器可用时按 `--backend corex` 跑同一条 `scripts.asr_benchmark`） |
| qwen3_asr 普通（310P） | ❌ 未在真机验证（用户已确认本轮 310P 不可用，只加代码路径；310P 硬约束照旧适用：fp16、block{64,128}、TP=1） |
| qwen3_asr server `/v1/audio/transcriptions` | ✅ 已实测（CUDA）：multipart 单测 + curl json/text/语言自动检测/401 鉴权/非 ASR 模型 400；`flashengine_transcription_requests_total 3`、`errors 0`、`audio_seconds_total 23.459125`；chat 路径不受影响 |

## 5. 使用示例

```bash
# Qwen3 dense 普通
python3 -m scripts.benchmark /models/Qwen3-8B --backend cuda

# Qwen3 dense + DFlash 投机
python3 -m scripts.benchmark /models/Qwen3-8B --spec-model /models/Qwen3-8B-DFlash --backend cuda

# Qwen3 MoE 普通（cuda / corex，纯 torch 专家前向）
python3 -m scripts.benchmark /models/Qwen3-30B-A3B --backend cuda \
  --max-model-len 2048 --max-num-seqs 4
python3 -m scripts.benchmark /models/Qwen3-30B-A3B --backend corex \
  --gpu-mem-util 0.85 --max-model-len 2048 --max-num-seqs 4

# Qwen3 MoE 普通（310P）
python3 -m scripts.benchmark /models/Qwen3-30B-A3B --backend ascend310p \
  --gpu-mem-util 0.85 --max-model-len 2048 --max-num-seqs 4

# Qwen3 MoE W8A8（Ascend msmodelslim：目录含 quant_model_description.json 即自动识别）
python3 -m scripts.benchmark /models/Qwen3-30B-A3B-W8A8 --backend ascend310p ...

# INT8 Quark W8A8 检查点（cuda/corex：config.json quant_method=quark 即自动识别，加载期反量化）
python3 -m scripts.benchmark /models/tiny-qwen3-moe-w8a8-int8 --backend cuda \
  --max-model-len 2048 --max-num-seqs 4


# Qwen3-ASR 语音转写（TP=1、单条音频、仅 none 投机/量化）
python3 -m scripts.asr_benchmark /models/Qwen3-1.7B-ASR --backend cuda \\
  --wav-dir bench_data/asr --reference-jsonl bench_data/asr/reference.jsonl

# Qwen3 MoE + DFlash（310P）
python3 -m scripts.benchmark /models/Qwen3-30B-A3B --backend ascend310p \
  --spec-model /models/Qwen3-30B-A3B-DFlash --no-think
```

组合不支持时启动即报错，例如：

```
NotImplementedError: Unsupported combination: target=qwen3_moe, backend=cuda,
quantization=w8a8, spec_method=none. W8A8 量化仅支持后端 ['ascend310p']（当前 cuda）
```
