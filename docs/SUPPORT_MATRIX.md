# 支持矩阵（声明支持）

> **状态**（2026-11-24）：本文件只描述**声明支持**——哪些 `(目标, 后端, 量化, 投机方法)`
> 组合会被引擎接受。**实测效果与证据强度在 [能力状态](STATUS.md)**，两者不可互相替代。

单一事实来源是 [`flashengine/support_matrix.py`](../flashengine/support_matrix.py)：
`Config` 在解析完目标适配器/后端/量化/投机适配器后调用 `require_supported(...)`，
不支持组合**启动即拒**（`NotImplementedError`，不静默降级）。
本文件第 3 节由该模块的 `format_matrix()` 生成，`python -m scripts.check_docs` 校验一致性。

> ⚠️ 声明支持 ≠ 能跑出收益。最典型的反例是 `qwen3 + ascend310p + w8a8 + dflash`：
> 组合放行、也能跑，但实测接受长度 E≈1（草稿提案几乎全被拒），比不开投机更慢。
> 这类「放行但别用」的组合集中在 [STATUS §3 与 §9](STATUS.md)。

## 1. 总览

| 目标模型 | 普通推理 | 投机推理 | 量化推理 |
|---|---|---|---|
| **Qwen3 dense** | ✅ cuda / corex / ascend310p | ✅ EAGLE3 / DFlash / DSpark（三后端） | ✅ W8A8，仅 ascend310p；✅ Quark W8A8，仅 cuda/corex |
| **Qwen3 MoE**（Qwen3-30B-A3B） | ✅ cuda / corex / ascend310p | ✅ DFlash（三后端） | ✅ W8A8，仅 ascend310p；✅ Quark W8A8，仅 cuda/corex |
| **Qwen3.5 dense / MoE 文本** | ✅ cuda / corex / ascend310p | ✅ DFlash（三后端） | ❌（当前 fp16/bf16） |
| **Qwen3-ASR**（`qwen3_asr`） | ✅ cuda / corex / ascend310p（TP=1、单条音频） | ❌（仅 none） | ❌（仅 fp16/bf16） |

三后端的实际验证进度差别很大：**只有 CUDA 与 310P 覆盖了大部分组合，CoreX 只验证过
Qwen3 dense + EAGLE3/普通推理与 ASR 冒烟**。逐格证据见 [STATUS](STATUS.md)。

补充说明：

- **Qwen3 MoE** 在 cuda/corex 走纯 torch 专家前向（`BackendBase.fused_moe`：`index_select` +
  `bmm`，形状静态、可被图捕获）；310P 走 npu grouped matmul（含 `W8A8_DYNAMIC`）。
- **dtype**：cuda/corex 的 flash-attn 只支持 fp16/bf16，fp32 检查点由
  `BackendBase.normalize_config` 自动降为 bf16（打印告警，加载期转换）；310P 用
  `default_dtype=fp16` 强制 fp16。
- **INT8（Quark W8A8）**：AMD Quark 导出的 int8 检查点（`config.json` 的
  `quantization_config.quant_method == quark`）在 cuda/corex 上于**加载期**按 per-channel
  `weight_scale`/`weight_zero_point` 反量化成计算 dtype（`utils/loader.py`），运行时与普通
  bf16 模型完全一致——**是 int8 检查点适配，不是 int8 激活/GEMM 计算**（激活不量化）。
  310P 的 W8A8 走自有算子，不认 Quark 格式。
- **Qwen3-ASR** 是**音频输入**目标：prompt 由 `flashengine/asr_processor.py` 在 CPU 侧渲染
  （mel + `<|audio_pad|>` span），引擎对该请求做一次音频塔前向得到音频嵌入，再按普通文本
  decode。约束：TP=1（`audio_input` 使 `Config` 强制）、单请求单条音频、音频请求
  `skip_prefix_cache`（音频嵌入不可跨请求复用）、仅 fp16/bf16。建议单段 ≤30s。
  入口见 [ASR](QWEN3_ASR.md)。
- **Qwen3.5** 当前 TP=1、仅文本；支持完整块 KV + conv/SSM 快照的 prefix cache 与保留
  conv/SSM 历史的分块 prefill，DFlash 另恢复草稿 KV；新版 DFlash 的逐层滑窗/因果语义已适配。
  详见 [Qwen3.5](QWEN35.md)、[Qwen3.5 MoE](QWEN35_MOE.md)、[前缀缓存](QWEN35_PREFIX_CACHE.md)。

## 2. 判定规则（`support_matrix.evaluate`）

1. **后端**：`qwen3` 全三后端；`qwen3_moe`/`qwen3_5_moe` 还要求后端声明 `supports_moe`。
2. **量化**：
   - `w8a8`（Ascend msmodelslim）：仅 `ascend310p`，由模型目录的
     `quant_model_description.json` 自动识别（注意力 qkv/o 静态 W8A8，MoE 专家
     `W8A8_DYNAMIC`，gate/embed/lm_head 保持 fp16）；
   - `quark_w8a8`（AMD Quark）：仅 `cuda`/`corex`，由 `config.json` 的
     `quantization_config.quant_method == quark` 自动识别；
   - `qwen3_5`/`qwen3_5_moe`/`qwen3_asr` 的量化一律拒绝（尚未接入）。
3. **投机方法**：取目标适配器的 `supported_spec_methods`，由同一张表派生，杜绝两处口径漂移：
   - `qwen3` = {eagle3, dflash, dspark}；
   - `qwen3_moe` = {dflash}（EAGLE3/DSpark 无 MoE 草稿；MoE 与 dense 的 aux 口径相同，
     都抽残差流，故 dense DFlash 草稿可直接用于 MoE 目标，`target_layer_ids + 1`）；
   - `qwen3_5` / `qwen3_5_moe` = {dflash}；`qwen3_asr` = {none}。
4. **量化 × 投机**：二者实现上独立（量化改目标前向，投机加草稿与 verify），故**放行**；
   但 310P 上 W8A8 + DFlash 实测 E≈1，属「放行但无效」，见 [STATUS §3](STATUS.md)。
5. **EAGLE3 × 310P**：能力位放行（`backends/ascend.py` 的 `supports_spec/spec_graph` 为 True），
   但**没有任何 310P EAGLE3 的 E/TPS/正确性实测**，属「放行但未验证」；不要把
   `supported=yes` 读成「310P 上 EAGLE3 可用且有效」。

## 3. 全枚举矩阵（由 `support_matrix.format_matrix()` 生成）

共 5 目标 × 3 后端 × 3 量化 × 4 投机方法 = 180 行。改代码后运行
`python -m scripts.check_docs --fix` 重新生成，不要手改本区块。

<!-- BEGIN:GEN_MATRIX -->
| target | backend | quantization | spec_method | supported |
|---|---|---|---|---|
| qwen3 | cuda | none | none | yes |
| qwen3 | cuda | none | eagle3 | yes |
| qwen3 | cuda | none | dflash | yes |
| qwen3 | cuda | none | dspark | yes |
| qwen3 | cuda | w8a8 | none | no |
| qwen3 | cuda | w8a8 | eagle3 | no |
| qwen3 | cuda | w8a8 | dflash | no |
| qwen3 | cuda | w8a8 | dspark | no |
| qwen3 | cuda | quark_w8a8 | none | yes |
| qwen3 | cuda | quark_w8a8 | eagle3 | yes |
| qwen3 | cuda | quark_w8a8 | dflash | yes |
| qwen3 | cuda | quark_w8a8 | dspark | yes |
| qwen3 | corex | none | none | yes |
| qwen3 | corex | none | eagle3 | yes |
| qwen3 | corex | none | dflash | yes |
| qwen3 | corex | none | dspark | yes |
| qwen3 | corex | w8a8 | none | no |
| qwen3 | corex | w8a8 | eagle3 | no |
| qwen3 | corex | w8a8 | dflash | no |
| qwen3 | corex | w8a8 | dspark | no |
| qwen3 | corex | quark_w8a8 | none | yes |
| qwen3 | corex | quark_w8a8 | eagle3 | yes |
| qwen3 | corex | quark_w8a8 | dflash | yes |
| qwen3 | corex | quark_w8a8 | dspark | yes |
| qwen3 | ascend310p | none | none | yes |
| qwen3 | ascend310p | none | eagle3 | yes |
| qwen3 | ascend310p | none | dflash | yes |
| qwen3 | ascend310p | none | dspark | yes |
| qwen3 | ascend310p | w8a8 | none | yes |
| qwen3 | ascend310p | w8a8 | eagle3 | yes |
| qwen3 | ascend310p | w8a8 | dflash | yes |
| qwen3 | ascend310p | w8a8 | dspark | yes |
| qwen3 | ascend310p | quark_w8a8 | none | no |
| qwen3 | ascend310p | quark_w8a8 | eagle3 | no |
| qwen3 | ascend310p | quark_w8a8 | dflash | no |
| qwen3 | ascend310p | quark_w8a8 | dspark | no |
| qwen3_moe | cuda | none | none | yes |
| qwen3_moe | cuda | none | eagle3 | no |
| qwen3_moe | cuda | none | dflash | yes |
| qwen3_moe | cuda | none | dspark | no |
| qwen3_moe | cuda | w8a8 | none | no |
| qwen3_moe | cuda | w8a8 | eagle3 | no |
| qwen3_moe | cuda | w8a8 | dflash | no |
| qwen3_moe | cuda | w8a8 | dspark | no |
| qwen3_moe | cuda | quark_w8a8 | none | yes |
| qwen3_moe | cuda | quark_w8a8 | eagle3 | no |
| qwen3_moe | cuda | quark_w8a8 | dflash | yes |
| qwen3_moe | cuda | quark_w8a8 | dspark | no |
| qwen3_moe | corex | none | none | yes |
| qwen3_moe | corex | none | eagle3 | no |
| qwen3_moe | corex | none | dflash | yes |
| qwen3_moe | corex | none | dspark | no |
| qwen3_moe | corex | w8a8 | none | no |
| qwen3_moe | corex | w8a8 | eagle3 | no |
| qwen3_moe | corex | w8a8 | dflash | no |
| qwen3_moe | corex | w8a8 | dspark | no |
| qwen3_moe | corex | quark_w8a8 | none | yes |
| qwen3_moe | corex | quark_w8a8 | eagle3 | no |
| qwen3_moe | corex | quark_w8a8 | dflash | yes |
| qwen3_moe | corex | quark_w8a8 | dspark | no |
| qwen3_moe | ascend310p | none | none | yes |
| qwen3_moe | ascend310p | none | eagle3 | no |
| qwen3_moe | ascend310p | none | dflash | yes |
| qwen3_moe | ascend310p | none | dspark | no |
| qwen3_moe | ascend310p | w8a8 | none | yes |
| qwen3_moe | ascend310p | w8a8 | eagle3 | no |
| qwen3_moe | ascend310p | w8a8 | dflash | yes |
| qwen3_moe | ascend310p | w8a8 | dspark | no |
| qwen3_moe | ascend310p | quark_w8a8 | none | no |
| qwen3_moe | ascend310p | quark_w8a8 | eagle3 | no |
| qwen3_moe | ascend310p | quark_w8a8 | dflash | no |
| qwen3_moe | ascend310p | quark_w8a8 | dspark | no |
| qwen3_5 | cuda | none | none | yes |
| qwen3_5 | cuda | none | eagle3 | no |
| qwen3_5 | cuda | none | dflash | yes |
| qwen3_5 | cuda | none | dspark | no |
| qwen3_5 | cuda | w8a8 | none | no |
| qwen3_5 | cuda | w8a8 | eagle3 | no |
| qwen3_5 | cuda | w8a8 | dflash | no |
| qwen3_5 | cuda | w8a8 | dspark | no |
| qwen3_5 | cuda | quark_w8a8 | none | no |
| qwen3_5 | cuda | quark_w8a8 | eagle3 | no |
| qwen3_5 | cuda | quark_w8a8 | dflash | no |
| qwen3_5 | cuda | quark_w8a8 | dspark | no |
| qwen3_5 | corex | none | none | yes |
| qwen3_5 | corex | none | eagle3 | no |
| qwen3_5 | corex | none | dflash | yes |
| qwen3_5 | corex | none | dspark | no |
| qwen3_5 | corex | w8a8 | none | no |
| qwen3_5 | corex | w8a8 | eagle3 | no |
| qwen3_5 | corex | w8a8 | dflash | no |
| qwen3_5 | corex | w8a8 | dspark | no |
| qwen3_5 | corex | quark_w8a8 | none | no |
| qwen3_5 | corex | quark_w8a8 | eagle3 | no |
| qwen3_5 | corex | quark_w8a8 | dflash | no |
| qwen3_5 | corex | quark_w8a8 | dspark | no |
| qwen3_5 | ascend310p | none | none | yes |
| qwen3_5 | ascend310p | none | eagle3 | no |
| qwen3_5 | ascend310p | none | dflash | yes |
| qwen3_5 | ascend310p | none | dspark | no |
| qwen3_5 | ascend310p | w8a8 | none | no |
| qwen3_5 | ascend310p | w8a8 | eagle3 | no |
| qwen3_5 | ascend310p | w8a8 | dflash | no |
| qwen3_5 | ascend310p | w8a8 | dspark | no |
| qwen3_5 | ascend310p | quark_w8a8 | none | no |
| qwen3_5 | ascend310p | quark_w8a8 | eagle3 | no |
| qwen3_5 | ascend310p | quark_w8a8 | dflash | no |
| qwen3_5 | ascend310p | quark_w8a8 | dspark | no |
| qwen3_5_moe | cuda | none | none | yes |
| qwen3_5_moe | cuda | none | eagle3 | no |
| qwen3_5_moe | cuda | none | dflash | yes |
| qwen3_5_moe | cuda | none | dspark | no |
| qwen3_5_moe | cuda | w8a8 | none | no |
| qwen3_5_moe | cuda | w8a8 | eagle3 | no |
| qwen3_5_moe | cuda | w8a8 | dflash | no |
| qwen3_5_moe | cuda | w8a8 | dspark | no |
| qwen3_5_moe | cuda | quark_w8a8 | none | no |
| qwen3_5_moe | cuda | quark_w8a8 | eagle3 | no |
| qwen3_5_moe | cuda | quark_w8a8 | dflash | no |
| qwen3_5_moe | cuda | quark_w8a8 | dspark | no |
| qwen3_5_moe | corex | none | none | yes |
| qwen3_5_moe | corex | none | eagle3 | no |
| qwen3_5_moe | corex | none | dflash | yes |
| qwen3_5_moe | corex | none | dspark | no |
| qwen3_5_moe | corex | w8a8 | none | no |
| qwen3_5_moe | corex | w8a8 | eagle3 | no |
| qwen3_5_moe | corex | w8a8 | dflash | no |
| qwen3_5_moe | corex | w8a8 | dspark | no |
| qwen3_5_moe | corex | quark_w8a8 | none | no |
| qwen3_5_moe | corex | quark_w8a8 | eagle3 | no |
| qwen3_5_moe | corex | quark_w8a8 | dflash | no |
| qwen3_5_moe | corex | quark_w8a8 | dspark | no |
| qwen3_5_moe | ascend310p | none | none | yes |
| qwen3_5_moe | ascend310p | none | eagle3 | no |
| qwen3_5_moe | ascend310p | none | dflash | yes |
| qwen3_5_moe | ascend310p | none | dspark | no |
| qwen3_5_moe | ascend310p | w8a8 | none | no |
| qwen3_5_moe | ascend310p | w8a8 | eagle3 | no |
| qwen3_5_moe | ascend310p | w8a8 | dflash | no |
| qwen3_5_moe | ascend310p | w8a8 | dspark | no |
| qwen3_5_moe | ascend310p | quark_w8a8 | none | no |
| qwen3_5_moe | ascend310p | quark_w8a8 | eagle3 | no |
| qwen3_5_moe | ascend310p | quark_w8a8 | dflash | no |
| qwen3_5_moe | ascend310p | quark_w8a8 | dspark | no |
| qwen3_asr | cuda | none | none | yes |
| qwen3_asr | cuda | none | eagle3 | no |
| qwen3_asr | cuda | none | dflash | no |
| qwen3_asr | cuda | none | dspark | no |
| qwen3_asr | cuda | w8a8 | none | no |
| qwen3_asr | cuda | w8a8 | eagle3 | no |
| qwen3_asr | cuda | w8a8 | dflash | no |
| qwen3_asr | cuda | w8a8 | dspark | no |
| qwen3_asr | cuda | quark_w8a8 | none | no |
| qwen3_asr | cuda | quark_w8a8 | eagle3 | no |
| qwen3_asr | cuda | quark_w8a8 | dflash | no |
| qwen3_asr | cuda | quark_w8a8 | dspark | no |
| qwen3_asr | corex | none | none | yes |
| qwen3_asr | corex | none | eagle3 | no |
| qwen3_asr | corex | none | dflash | no |
| qwen3_asr | corex | none | dspark | no |
| qwen3_asr | corex | w8a8 | none | no |
| qwen3_asr | corex | w8a8 | eagle3 | no |
| qwen3_asr | corex | w8a8 | dflash | no |
| qwen3_asr | corex | w8a8 | dspark | no |
| qwen3_asr | corex | quark_w8a8 | none | no |
| qwen3_asr | corex | quark_w8a8 | eagle3 | no |
| qwen3_asr | corex | quark_w8a8 | dflash | no |
| qwen3_asr | corex | quark_w8a8 | dspark | no |
| qwen3_asr | ascend310p | none | none | yes |
| qwen3_asr | ascend310p | none | eagle3 | no |
| qwen3_asr | ascend310p | none | dflash | no |
| qwen3_asr | ascend310p | none | dspark | no |
| qwen3_asr | ascend310p | w8a8 | none | no |
| qwen3_asr | ascend310p | w8a8 | eagle3 | no |
| qwen3_asr | ascend310p | w8a8 | dflash | no |
| qwen3_asr | ascend310p | w8a8 | dspark | no |
| qwen3_asr | ascend310p | quark_w8a8 | none | no |
| qwen3_asr | ascend310p | quark_w8a8 | eagle3 | no |
| qwen3_asr | ascend310p | quark_w8a8 | dflash | no |
| qwen3_asr | ascend310p | quark_w8a8 | dspark | no |
<!-- END:GEN_MATRIX -->

## 4. 运行时还会进一步收紧

声明通过后，运行时能力仍可能收紧或降级，**这不改变声明**，但会改变可用性：

| 收紧点 | 后果 | 依据 |
|---|---|---|
| 310P 缺 SplitFuse 算子 | 自动关闭前缀缓存并告警 | `backend.prefix_cache_disable_reason()` |
| 310P 缺 v3/SplitFuse 算子 | 回落 legacy flash attention | `FLASHENGINE_310P_FLASH_OP` |
| 图捕获失败 | 按桶回退 eager（不整体放弃） | `NPUGraphDecodeRunner` |
| spec 整轮图捕获失败 | 该轮回落 eager spec | `capture_spec_cudagraph` 异常分支 |
| FlashInfer | `supports_spec_graph=False`，投机恒 eager | [FLASHINFER](FLASHINFER.md) |
| 草稿 aux 层与目标不匹配 | 配置期直接报错 | `config.py` 的 aux 数量/宽度校验 |
| 显存预算不足 | 启动即报错（显式 KV 块数也受校验） | `engine/memory_budget.py` |

## 5. 使用示例

```bash
# Qwen3 dense 普通 / + DFlash 投机
python3 -m scripts.benchmark /models/Qwen3-8B --backend cuda
python3 -m scripts.benchmark /models/Qwen3-8B --spec-model /models/Qwen3-8B-DFlash --backend cuda

# Qwen3 MoE 普通（cuda / corex 纯 torch 专家前向；310P grouped matmul）
python3 -m scripts.benchmark /models/Qwen3-30B-A3B --backend cuda   --max-model-len 2048 --max-num-seqs 4
python3 -m scripts.benchmark /models/Qwen3-30B-A3B --backend corex  --gpu-mem-util 0.85 --max-model-len 2048
python3 -m scripts.benchmark /models/Qwen3-30B-A3B --backend ascend310p --gpu-mem-util 0.85 --max-model-len 2048

# 310P W8A8（目录含 quant_model_description.json 即自动识别）；注意别和投机一起开
python3 -m scripts.benchmark /models/Qwen3-30B-A3B-W8A8 --backend ascend310p --max-num-seqs 1

# Quark int8 检查点（cuda/corex：加载期反量化）
python3 -m scripts.benchmark /models/tiny-qwen3-moe-w8a8-int8 --backend cuda --max-model-len 2048

# Qwen3-ASR 语音转写（TP=1、单条音频、无投机/量化）
python3 -m scripts.asr_benchmark /models/Qwen3-1.7B-ASR --backend cuda \
  --wav-dir bench_data/asr --reference-jsonl bench_data/asr/reference.jsonl

# Qwen3.5 + DFlash（310P 生产配置：k=7 是实测甜点，不要用默认 k=15）
python3 -m scripts.benchmark /models/Qwen3.5-9B --backend ascend310p \
  --spec-model /models/Qwen3.5-9B-DFlash --spec-num-tokens 7 --no-think
```

组合不支持时启动即报错：

```
NotImplementedError: Unsupported combination: target=qwen3_moe, backend=cuda,
quantization=w8a8, spec_method=none. W8A8 量化仅支持后端 ['ascend310p']（当前 cuda）
```

## 6. 相关

- 实测状态与「不要宣称」清单：[STATUS.md](STATUS.md)
- 文档地图与维护规则：[README.md](README.md)
- 代码单一事实来源：[`flashengine/support_matrix.py`](../flashengine/support_matrix.py)
