# Ascend 310P Graph、EAGLE3 与 DFlash 适配说明

本文说明 nano-vLLM 针对 Atlas 300V Pro / Ascend 310P 所做的推理加速适配，重点描述普通
decode、EAGLE3 和 DFlash 的算子路径、KV-cache 布局、NPUGraph 捕获方法，以及它们与
vLLM-Ascend 310P 路径的对应关系。

本文所说的“对齐 vLLM-Ascend”指以下语义和性能原则保持一致：

- 使用 310P 原生的 `FRACTAL_NZ` 权重与 KV-cache；
- prefill、decode、spec verify 使用对应的 NPU attention 算子；
- 用静态地址、静态 shape 和 batch bucket 捕获 ACL/NPUGraph；
- 把动态 block table、context length、slot mapping 移到图外准备；
- 避免图中的 AICPU 整数算子、动态分配和设备到主机同步；
- 图捕获不兼容时安全降级到 eager，保证功能优先。

nano-vLLM 没有复制 vLLM 的编译器和调度框架。vLLM-Ascend 的
`compilation-config mode=3 + PIECEWISE` 由 vLLM compilation backend 负责切图；本项目则用
更轻量的手工静态 buffer、batch bucket 和 `torch.npu.NPUGraph` 实现相同的执行目标。

---

## 1. 总体执行结构

### 1.1 普通解码

```text
Host scheduler
    │
    ├─ 填充 pinned CPU metadata
    │    input_ids / positions / slots / context_lens / block_tables
    │
    ├─ 非阻塞 copy_ 到每个 bucket 的固定 NPU tensor
    │
    └─ NPUGraph.replay()
          ├─ Qwen3 forward
          ├─ _npu_reshape_and_cache
          ├─ _npu_paged_attention
          ├─ FRACTAL_NZ GEMM / fused norm / fused SwiGLU / fused RoPE
          ├─ LM head
          └─ greedy 时图内 argmax
```

### 1.2 EAGLE3 整轮图

```text
scatter metadata
    → draft chain
    → target verify [last_token, d1, ..., dk]
    → longest-prefix acceptance
    → virtual-window refill
    → pack committed tokens + next draft token
    → one D2H readback
```

### 1.3 DFlash / DSpark 整轮图

```text
scatter metadata
    → append accepted target aux into private draft paged KV
    → one non-causal DFlash block forward
    → target verify [anchor, d1, ..., dk]
    → longest-prefix acceptance
    → pack committed tokens
    → one D2H readback
```

---

## 2. 与 vLLM-Ascend 的路径对应

| 维度 | vLLM-Ascend 310P 的方向 | nano-vLLM 310P 实现 |
|---|---|---|
| 权重布局 | 310P GEMM 使用 NZ 内部格式 | 加载后把 Linear 权重转为 `FRACTAL_NZ`；LM head 保留 ND embedding 权重并增加独立 NZ 投影视图 |
| KV-cache | 分页 NZ KV | 每层 `(blocks, hidden/16, block_size, 16)` 的 `FRACTAL_NZ` K/V |
| KV 写入 | ReshapeAndCache | `torch_npu._npu_reshape_and_cache(..., slot_indices=)` |
| Prefill | Flash Attention 系列 | legacy `_npu_flash_attention`，支持时自动选择 v3；token 数按 16 对齐 |
| Decode | Paged Attention | `_npu_paged_attention`，静态 output buffer |
| Chunk/spec attention | SplitFuse 或展开后的 PA | 有 SplitFuse 时走 SplitFuse；verify 使用 host 预展开的 per-query PA metadata |
| Graph | ACLGraph/NPUGraph + shape bucket | 独立 `NPUGraphDecodeRunner`；spec 使用每 bucket 一张整轮图 |
| 动态 metadata | 图外更新，图内固定地址 | pinned CPU blob → 固定 NPU tensor → graph replay |
| Padding | 合法 dummy block/slot | 永久保留最后一个 KV block，padding 请求只访问该块 |
| Sampling | greedy 可图内执行 | 310P greedy 使用图内 `argmax`；非 greedy 自动回 eager/native sampler |
| 失败处理 | 不兼容 shape/op 回退 | decode 按 bucket 回退；spec 整轮捕获失败后统一 eager fallback |

两者最重要的共同点不是 Python 类名，而是：**图中的 tensor 地址和 shape 固定，实际请求的
动态状态只通过固定 buffer 内容更新。**

---

## 3. 310P 基础算子与存储适配

### 3.1 强制 fp16

310P attention/ATB 路径统一使用 `torch.float16`。当 checkpoint 声明为 bf16 或 fp32 时，启动
信息会提示权重在加载阶段转换：

```text
Backend ascend310p requires dtype=torch.float16; model dtype=torch.bfloat16.
Weights will be converted while loading.
```

### 3.2 KV-cache 约束

- `kvcache_block_size` 只允许 64 或 128；其他值自动改为 128；
- 要求 `block_size * head_dim <= 128 * 128`；
- 每层 K/V 都由 `empty_with_format(..., FRACTAL_NZ)` 分配；
- 最后一个物理 block 永久作为 graph dummy block，不交给 BlockManager；
- target、EAGLE3 draft、分页 DFlash draft 使用相同的 block id 和 slot 编号语义，但拥有
  各自独立的 K/V tensor。

### 3.3 权重 NZ 化

普通 `LinearBase` 权重在加载后直接转换为 NZ。LM head 可能与 embedding 共享权重，因此不能把
原 Parameter 原地换成只适合 GEMM 的格式：

- `weight` 保留 ND，供 `F.embedding` 使用；
- `weight_nz` 供 LM-head `F.linear` 使用；
- DFlash 额外的全层 fused K/V projection 也转换为 NZ；
- DFlash 共享 target LM-head 时使用独立的 `lm_head_weight_nz` 普通 tensor 属性，不能覆盖已注册
  的 `lm_head_weight` Parameter。

### 3.4 融合算子

310P 后端优先使用：

- `npu_rms_norm`；
- `npu_add_rms_norm`；
- `npu_swiglu`；
- `npu_apply_rotary_pos_emb`，不可用时尝试 `_npu_rotary_embedding`；
- Linear/LM-head 的 FRACTAL_NZ GEMM。

---

## 4. 普通 decode 的 NPUGraph 对齐

实现文件：`nanovllm/backends/ascend310p/graph.py`。

### 4.1 Batch bucket

捕获尺寸为：

```text
1, 2, 4, 8, 16, 32, ...，以及 max_num_seqs 尾桶
```

运行时选择第一个不小于真实 batch size 的桶。padding 行使用 dummy block，因此不会写脏真实
请求的 KV。

### 4.2 固定输入地址

每个 bucket 永久持有：

- `input_ids: int64`；
- `positions: int64`；
- `slot_mapping: int32`；
- `context_lens: int32`；
- `block_tables: int32`；
- output logits 和 greedy token buffer。

Host 侧为同样的 metadata 分配 pinned CPU tensor，并缓存 NumPy view。每一步只修改已有内存，
然后用非阻塞 `copy_` 更新固定 NPU tensor，不再反复创建 Python nested list 和临时 tensor。

所有 pinned tensor 都显式指定 `device="cpu"`。这是为了解决 ModelRunner 已把默认 device 设置为
NPU 后，`pin_memory=True` 被错误应用到 NPU tensor，触发：

```text
RuntimeError: Only dense CPU tensors can be pinned
```

### 4.3 图内采样

310P 默认 `graph_sample=False`，含义是随机采样不进入图，而不是完全禁用图：

- `temperature=0`：LM head 与 `argmax` 均在图内；
- 非 greedy：图返回 logits，使用 native sampler；
- spec 非 greedy：整轮 spec 图不会把语义静默改成 greedy，而是自动回退 eager spec round。

### 4.4 图捕获失败

decode 图按 bucket 独立捕获。某一个 bucket 失败只禁用该 bucket，不阻止引擎启动；运行时该 shape
走 eager。benchmark 会打印：

```text
NPUGraph runtime = captures_ok=..., captures_failed=..., replays=...,
                   fallbacks=..., bucket_hits={...}
```

---

## 5. EAGLE3 的 310P 适配

### 5.1 保留 vLLM EAGLE3 的 shifted-stream 语义

EAGLE3 没有改成 310P 专属算法，仍保持与 vLLM padded 分支一致的语义：

- rope 位置 `p` 的 draft 行使用 `tok@p+1` embedding；
- 条件来自 target 的 look-ahead aux；
- 输出预测 `tok@p+2`；
- verify 一次处理 `[last_token, d1, ..., dk]`；
- 接受最长匹配前缀，再提交一个 target bonus token；
- refill 重建下一轮需要的虚构 draft 窗口。

### 5.2 Draft KV 分页化

EAGLE3 draft attention 拥有独立的 NZ paged KV，但与 target 共用逻辑 block table 和 slot mapping。
因此：

- draft 链每一步通过 `_npu_reshape_and_cache` 写入自己的 KV；
- decode attention 使用 `_npu_paged_attention`；
- 被拒 draft 的物理槽无需清零，后续写入会覆盖，且 context length 不会读到无效残留。

### 5.3 Host 预展开 verify metadata

310P 上不能在图内临时执行以下动态整数操作：

- 根据 batch 行号 `index_select` block table；
- `context_len + arange`；
- 动态构造 per-query table；
- 依赖设备值的 Python `.item()`。

这些操作容易落入 AICPU，并在 NPUGraph replay 时出现 507011 或 `OPS feature not supported`。

因此 `_fill_spec_meta` 在 host 侧直接生成：

- 每个 verify query 的 `position`；
- 每个 query 的物理 `slot`；
- `context_lens[i, j] = sequence_length + j`；
- 按 query 行复制的 `block_tables_expanded`；
- draft chain 每一步的 position、context length 和 slot。

图内只做固定地址的 `copy_` 和模型算子。

### 5.4 整轮单图

EAGLE3 每个 batch bucket 捕获一张完整图：

```text
_sg_scatter
  → _sg_draft_chain
  → _sg_verify
  → _sg_acceptance
  → _sg_refill
```

这样消除了每轮多次 Python/kernel launch。第一轮因为还没有 carry，走同构 eager 路径建立 carry；
后续轮进入 graph replay。

### 5.5 Graph-safe acceptance

原始 longest-prefix acceptance 中的动态 `arange/full/cat` 和整数 Add 对 310P 图不友好。当前实现：

- `arange`、clamp index、`-1` token buffer 在捕获前静态分配；
- 匹配前缀长度使用 `float32 cumprod + sum`；
- `count = accepted + 1` 使用 float Add 后转回 int64；
- committed tokens、count 和下一轮 draft token 打包在一个 `commit_buf`；
- 每轮只做一次 D2H 读回。

### 5.6 Padding 修复

当真实 batch 小于 bucket 时，必须按序列轴复位：

```python
v_slots[n:, :]
```

不能写成 `v_slots[:, n:]`。后者会清掉真实序列后半部分的 verify slot，造成 KV 未写入、NaN、
零 token 流或全异常字符。padding 请求统一指向末尾 dummy block。

---

## 6. DFlash 的 310P 适配

CoreX/CUDA 原有 DFlash 使用按序列分配的稠密上下文 KV 和通用 SDPA。为了避免在 310P 上产生
大规模 gather、concat、GQA expand 和 SDPA 分解，310P 增加了专用分页路径。

### 6.1 DFlash 算法语义

DFlash draft 通常有 5 层，block size 通常为 16：

- 1 行 anchor；
- 15 行 mask query；
- block 内非因果，所有 query 行互相可见；
- draft 无独立 embedding、LM head 和词表，复用 target；
- target aux 经 `fc → hidden_norm → 每层 K/V projection` 生成 draft 上下文 KV；
- target 一次 verify 后接受最长匹配前缀并提交 bonus token。

### 6.2 私有分页 NZ Draft KV

310P 为 DFlash 每个 draft layer 分配一对独立 NZ K/V：

```text
(num_blocks, draft_kv_hidden/16, block_size, 16)
```

它与 target 使用相同的 block table 和物理 slot 编号，因此无需额外的分页管理器：

- prefill aux 投影结果直接写入 prompt 对应的物理 slot；
- 上轮已提交行的 aux 在下一轮覆盖对应上下文 slot；
- 当前 block 的 query K/V 写入预留的未来 slot；
- 被拒 query 残留会被下一轮覆盖，且有效 context length 不会读取错误位置。

DFlash draft KV 字节数已经加入每个 KV block 的显存预算，避免只按 target KV 计算后分配过量 block。

### 6.3 非因果 SplitFuse

每层 DFlash attention 使用一次 `_npu_paged_attention_splitfuse` 或 v2：

- `query length = k + 1`；
- `context_lens = historical_context + query_block`；
- mask 全零，表示历史上下文和整个 query block 全部可见；
- v2 使用固定 compressed-mask ABI；
- legacy 使用转换为 NZ 的零 mask。

这代替了稠密路径中的：

```text
gather dense context
  → concatenate context K/V and query K/V
  → repeat_interleave for GQA
  → generic scaled_dot_product_attention
```

可通过以下变量做 A/B：

```bash
# 默认：存在 SplitFuse 时自动启用分页路径
NANOVLLM_310P_DFLASH_PAGED=auto

# 强制旧的稠密 SDPA 路径
NANOVLLM_310P_DFLASH_PAGED=0

# 要求分页路径；缺少 SplitFuse 时启动直接报错
NANOVLLM_310P_DFLASH_PAGED=1
```

### 6.4 DFlash 整轮图

每个 bucket 捕获：

```text
_sg_scatter + _df_scatter
  → _df_append_context_graph
  → _df_draft_block_graph
  → _df_verify_graph
  → _sg_acceptance
```

DFlash 专用 metadata 包括：

- `df_ctx_pos`：本轮 aux 追加位置；
- `df_ctx_phys_slots`：对应的物理 slot；
- `df_total_lens`：SplitFuse 读取的总 KV 长度；
- `df_ctx_aux`：固定大小的 GPU aux staging buffer。

block query 的 position 直接复用 host 已展开的 `v_positions`，不在图内执行 int64
`arange + Add`。

### 6.5 Dummy request

DFlash query block 有 16 行，因此 padding 请求不能只把 16 行全部映射到同一个 slot。当前做法为：

- block table 指向最后一个保留 block；
- query slots 指向该 block 的 `0..rows-1`；
- positions 为 `0..rows-1`；
- target verify context lengths 为 `1..rows`；
- DFlash SplitFuse total length 为 `rows`。

这既满足算子输入约束，也不会污染真实请求。

### 6.6 Prefix cache（2026-09 更新）

310P 分页 DFlash/DSpark 的 draft K/V 与 target K/V 使用相同的物理 block id 与 slot 编号，
并在首次 prefill 时写入已投影的上下文状态。因此 prefix hit 复用某个 target block 时，
对应的 draft block 同时保留，不需要重新计算 target aux。运行时仍需 SplitFuse；缺失时
prefix cache 关闭。块查询会覆盖 anchor 所在的 prompt 末块，故该末块始终不注册为
可复用哈希；下一次请求最多多做一个 block 的 prefill，换取草稿上下文正确性。

这条规则只适用于分页 draft KV。CUDA/CoreX 的稠密 draft KV 通过有界 CPU block
快照恢复。310P 路径尚需真实板卡上的前缀命中/图回放/输出一致性验收。

### 6.7 初始化顺序兼容

`warmup_model()` 用于测量模型峰值，执行时间早于正式 KV-cache 分配。此时 dummy sequence 没有
block table，draft NZ cache 也尚未创建。310P paged DFlash 在这一轮只执行 target prefill/aux，
跳过物理 draft KV 写入；正式 KV 分配后的 graph warmup 和 LLMEngine runtime warmup 会执行完整
DFlash 路径。

该门控仅对 `dflash_paged=True` 生效；CoreX/CUDA 的稠密 DFlash warmup 行为保持不变。

## 7. Graph metadata 与同步策略

### 7.1 MetaBlob

spec 路径把动态 metadata 合并为两个连续 pinned CPU 平面：

- int64 平面；
- float32 平面。

每个平面各有一个固定地址的 NPU twin。每轮只进行两次 H2D，然后图内 scatter 到最终静态
buffer。这样减少了多个小 tensor 的分配和 copy launch。

### 7.2 数据类型

310P 约束如下：

| metadata | dtype |
|---|---|
| token id / position | int64 |
| slot mapping | int32 |
| context length | int32 |
| block table | int32 |
| cu_seqlens | int32 |
| temperature / top-p | float32 |

MetaBlob 统一以 int64 存 host 整数数据，scatter 时通过固定目标 buffer 的 `copy_` 完成类型转换。
DFlash 对 SplitFuse 要求更严格，因此 `df_total_lens` 和最终 block table 使用预分配、连续的
int32 graph buffer，不在 attention 调用内部临时 `.to()`。

### 7.3 同步

- metadata H2D 与 graph replay 在同一当前 stream 排队；
- replay 前不做每 token host synchronize；
- 最终 `.tolist()`/commit readback 是该轮必要的完成点；
- capture 后立即 smoke replay 并 synchronize，用于发现“捕获成功但首次 replay 失败”的
  AICPU/算子兼容问题。

---

## 8. 降级与能力边界

| 场景 | 行为 |
|---|---|
| decode 某个 bucket 捕获失败 | 仅该 bucket eager fallback |
| EAGLE3/DFlash 整轮图捕获失败 | 释放图池，全部 spec round 使用同构 eager 路径 |
| spec 请求非 greedy，310P graph sampler 关闭 | 自动 eager spec，不改变采样语义 |
| DFlash 无 SplitFuse | 默认使用 dense-KV/SDPA fallback |
| `NANOVLLM_SPEC_GRAPH=0` | 不尝试 spec graph |
| `--eager` | decode/spec 均不捕获图 |

当前边界：

- 310P speculative decoding 只支持 TP=1；
- 310P 强制 fp16；
- 当前 target model 为 dense Qwen3；
- nano-vLLM 没有 vLLM 的异步调度重叠，因此即使算子与图路径对齐，吞吐仍可能存在 host
  scheduling 差距；
- 不同 CANN/torch-npu 版本对 SplitFuse v2、compressed mask 和 graph-task update 的支持不同，
  实际能力以启动信息为准。

---

## 9. 启动信息判读

### 9.1 普通 decode

```text
Graph mode: decode_graph
Attention backend: torch-npu (flash=legacy|v3, splitfuse=legacy|v2|off)
Graph backend: NPUGraph/ACLGraph
KV-cache layout: FRACTAL_NZ
```

benchmark 结束时应看到 `replays > 0`、`fallbacks = 0`。

### 9.2 EAGLE3

```text
Speculative method: eagle3
Speculative tokens: 3
Graph mode: spec_graph
```

benchmark 结束时重点检查：

```text
Spec graph runtime = captures_ok>0, captures_failed=0,
                     replays>0, eager_rounds≈cold-start rounds
```

### 9.3 DFlash

```text
Speculative method: dflash
Speculative tokens: 15
DFlash attention path: paged-KV/SplitFuse-legacy|v2
Graph mode: spec_graph
Prefix cache: enabled when SplitFuse is available; final prompt block excluded from prefix hashing (requires hardware validation)
```

如果显示 `dense-KV/SDPA-fallback`，说明 SplitFuse 不可用或分页路径被环境变量关闭。

## 10. 验证命令

### 10.1 普通 decode graph

```bash
python3 -m scripts.benchmark /host_root/models/Qwen3-8B/ \
    --questions bench_data/question.jsonl \
    --max-model-len 2048 --max-num-seqs 1 \
    --no-think --max-tokens 64
```

### 10.2 EAGLE3 graph

```bash
python3 -m scripts.benchmark /host_root/models/Qwen3-8B/ \
    --questions bench_data/question.jsonl \
    --max-model-len 2048 --max-num-seqs 1 \
    --no-think --max-tokens 64 \
    --spec-model /host_root/models/eagle3-8B/
```

### 10.3 DFlash paged graph

```bash
python3 -m scripts.benchmark /host_root/models/Qwen3-8B/ \
    --questions bench_data/question.jsonl \
    --max-model-len 2048 --max-num-seqs 1 \
    --no-think --max-tokens 64 \
    --spec-model /host_root/models/Qwen3-8B-Dflash/
```

### 10.4 建议的 A/B

```bash
# Graph vs eager
python3 -m scripts.benchmark ...
python3 -m scripts.benchmark ... --eager

# DFlash paged SplitFuse vs dense SDPA
NANOVLLM_310P_DFLASH_PAGED=1 python3 -m scripts.benchmark ... --spec-model DFLASH
NANOVLLM_310P_DFLASH_PAGED=0 python3 -m scripts.benchmark ... --spec-model DFLASH

# Spec graph vs eager spec
NANOVLLM_SPEC_GRAPH=1 python3 -m scripts.benchmark ... --spec-model DRAFT
NANOVLLM_SPEC_GRAPH=0 python3 -m scripts.benchmark ... --spec-model DRAFT
```

A/B 时必须保持模型、prompt 顺序、`--no-think`、temperature、max tokens、max model length 和
max sequences 完全一致，同时报告：

- Output token throughput；
- steady-state decode throughput；
- TTFT；
- graph replay/fallback 统计；
- EAGLE3/DFlash acceptance length；
- token acceptance rate 和逐位置接受率。

---

## 11. 实机问题与修复记录

| 问题 | 根因 | 修复 |
|---|---|---|
| `Only dense CPU tensors can be pinned` | 默认 device 已设为 NPU，未显式声明 CPU 的 pinned tensor 被错误创建 | 所有 graph host staging tensor 显式 `device="cpu"` |
| DFlash NZ LM-head 赋值 TypeError | `lm_head_weight` 已注册为 Parameter，不能替换成普通 NPU Tensor | 保留 Parameter，增加独立 `lm_head_weight_nz` 属性 |
| DFlash warmup `_slot_of` 越界 | 模型峰值 warmup 早于 KV/BlockManager 分配，dummy sequence 没有 block table | 仅 paged DFlash 在预算 warmup 跳过 draft KV 写入，正式 warmup 仍完整执行 |
| NPUGraph replay 507011/AICPU 风险 | 图内动态 int Add、arange、index、table expansion | metadata 在 host 预展开，acceptance 使用静态 buffer 和 float 运算 |
| padding 写脏真实 KV | padding slot 使用 0 或切错 `v_slots` 轴 | 保留末尾 dummy block，按 `v_slots[n:, :]` 复位 |
| DFlash 图捕获直接失败 | `_df_verify_graph` 使用未定义的 `use_cuda_graph` | 图路径明确传入 graph sampling 语义 |

---

## 12. 关键源码索引

- `nanovllm/backends/ascend.py`：310P 算子、NZ 权重/KV、SplitFuse、图能力声明；
- `nanovllm/backends/ascend310p/graph.py`：普通 decode NPUGraph bucket runner；
- `nanovllm/engine/model_runner.py`：spec metadata、EAGLE3/DFlash 整轮图、捕获与降级；
- `nanovllm/models/eagle3.py`：EAGLE3 draft 模型与 shifted-stream 前向；
- `nanovllm/models/dflash.py`：DFlash context projection、分页/稠密 attention 分支；
- `nanovllm/backends/ascend310p/aclgraph_attn.py`：paged-attention graph-task update 能力探测；
- `docs/EAGLE3.md`：EAGLE3 算法说明；
- `docs/DFlash.md`：DFlash 算法与多后端说明。
