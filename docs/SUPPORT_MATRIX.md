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

# Qwen3 MoE + DFlash（310P）
python3 -m scripts.benchmark /models/Qwen3-30B-A3B --backend ascend310p \
  --spec-model /models/Qwen3-30B-A3B-DFlash --no-think
```

组合不支持时启动即报错，例如：

```
NotImplementedError: Unsupported combination: target=qwen3_moe, backend=cuda,
quantization=w8a8, spec_method=none. W8A8 量化仅支持后端 ['ascend310p']（当前 cuda）
```
