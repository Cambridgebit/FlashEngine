# nano-vLLM CoreX CUDA Graph 适配指导

## 一、目标

当前已经完成：

```text
Qwen3-1.7B
CoreX 4.4.0
IXFormer
TP=1
eager
端到端推理 PASS
```

下一阶段增加 CUDA Graph。

实现策略必须对齐天数 CoreX vLLM：

```text
Prefill            → Eager
Decode(query_len=1) → Full CUDA Graph
```

即：

```text
FULL_DECODE_ONLY
```

不要实现：

```text
torch.compile graph
piecewise graph
prefill graph
mixed prefill/decode graph
```

CoreX CUDA Graph 仍使用：

```python
torch.cuda.CUDAGraph
torch.cuda.graph
graph.replay
graph.pool
```

不增加 IXFormer Graph runtime。

---

# 二、为什么这样做

天数官方 CoreX 4.4.0 vLLM 的运行配置：

```bash
export VLLM_ENFORCE_CUDA_GRAPH=1

--compilation-config \
'{
    "cudagraph_mode":"FULL_DECODE_ONLY",
    "level":0
}'
```

说明 CoreX 的策略是：

```text
vLLM CUDA Graph Manager
        │
        ▼
torch.cuda.CUDAGraph
        │
        ▼
CoreX CUDA-compatible runtime
        │
        ▼
IXFormer Attention
```

而不是：

```text
vLLM
 ↓
IXFormerGraph
```

因此 nano-vLLM 也应该继续使用自己的：

```python
capture_cudagraph()
run_model()
```

只对 CoreX Attention ABI 做适配。

---

# 三、参考 vLLM 0.11.2 的关键设计

vLLM 0.11.2 的 `CUDAGraphWrapper` 做：

```text
第一次：
torch.cuda.CUDAGraph()
torch.cuda.graph(...)
capture

以后：
graph.replay()
```

但 vLLM 明确说明：

```text
CUDAGraphWrapper 本身不负责 static buffer。

persistent input buffer
和 runtime → static buffer copy

由 ModelRunner 管理。
```

因此 nano-vLLM 当前：

```text
graph_vars
input_ids
positions
slot_mapping
context_lens
block_tables
```

这个设计本身就是正确方向。

不要删除。

---

# 四、修改文件

主要修改：

```text
nanovllm/utils/context.py

nanovllm/backends/base.py
nanovllm/backends/corex.py

nanovllm/engine/model_runner.py
```

新增：

```text
scripts/check_corex_cudagraph.py

tests/corex/test_cudagraph.py
```

不要修改：

```text
scheduler.py
block_manager.py
sequence.py
qwen3.py
layernorm.py
activation.py
rotary_embedding.py
```

---

# 五、Context 增加 Graph metadata

`nanovllm/utils/context.py`

增加：

```python
use_cuda_graph: bool = False
```

已有：

```python
max_context_len
```

继续保留。

最终至少具备：

```python
@dataclass
class Context:
    is_prefill: bool

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None

    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    max_context_len: int | None = None

    use_cuda_graph: bool = False
```

---

# 六、CoreX Attention 对齐 IXFormer Graph API

修改：

```text
nanovllm/backends/corex.py
```

当前 decode 已经通过 correctness test。

只增加：

```python
use_cuda_graph=context.use_cuda_graph
```

最终：

```python
def decode_attention(
    self,
    q,
    k_cache,
    v_cache,
    context,
    scale,
):
    return flash_attn_with_kvcache(
        q.unsqueeze(1),
        k_cache,
        v_cache,

        cache_seqlens=context.context_lens,
        block_table=context.block_tables,

        softmax_scale=scale,
        causal=True,

        max_context_len=context.max_context_len,

        use_cuda_graph=context.use_cuda_graph,
    )
```

不要修改其他 Attention 参数。

普通 eager：

```text
use_cuda_graph=False
```

Graph capture：

```text
use_cuda_graph=True
```

---

# 七、增加 Graph slot_mapping dtype API

这是 CoreX 特别需要注意的一点。

当前已经验证：

```text
IXFormer reshape_and_cache_flash

要求：

slot_mapping = torch.long
```

普通 eager 路径里 CoreXBackend 可以：

```python
slot_mapping.to(torch.long)
```

但 Graph 模式尽量不要让这个转换发生在 capture 内。

因此在 backend 增加：

```python
def graph_slot_mapping_dtype(self):
    return torch.int32
```

CUDA 默认：

```python
def graph_slot_mapping_dtype(self):
    return torch.int32
```

CoreX：

```python
def graph_slot_mapping_dtype(self):
    return torch.long
```

然后 CUDA Graph 的 static buffer：

```python
slot_mapping = torch.full(
    (max_bs,),
    -1,
    dtype=backend.graph_slot_mapping_dtype(),
    device="cuda",
)
```

这样：

```text
Runtime slot_mapping int32
        │
        │ copy_
        ▼
Graph static slot_mapping int64
        │
        ▼
IXFormer
```

dtype 转换发生在 `graph.replay()` 之前。

Graph 本身直接读取 long buffer。

这个方案比 capture 内执行：

```python
slot_mapping.to(torch.long)
```

更稳定，也更接近 vendor kernel 对 static input 的要求。

---

# 八、Graph 只 Capture Decode

保持 `run_model()`：

```python
if is_prefill:
    return eager
```

完整判断：

```python
if (
    is_prefill
    or self.enforce_eager
    or input_ids.size(0) > 512
):
    return self.model.compute_logits(
        self.model(
            input_ids,
            positions,
        )
    )
```

所以：

```text
Prefill → 永远 Eager

Decode → Graph
```

这就是 nano-vLLM 版本的：

```text
FULL_DECODE_ONLY
```

---

# 九、Graph batch bucket

继续使用 nano-vLLM 原方案：

```python
self.graph_bs = (
    [1, 2, 4, 8]
    + list(
        range(
            16,
            max_bs + 1,
            16,
        )
    )
)
```

例如：

```text
真实 BS=1 → graph 1

真实 BS=3 → graph 4

真实 BS=7 → graph 8

真实 BS=20 → graph 32
```

不要为每一个 batch size 创建 Graph。

---

# 十、创建 Static Buffers

`capture_cudagraph()`：

```python
input_ids = torch.zeros(
    max_bs,
    dtype=torch.int64,
    device="cuda",
)

positions = torch.zeros(
    max_bs,
    dtype=torch.int64,
    device="cuda",
)

slot_mapping = torch.full(
    (max_bs,),
    -1,
    dtype=self.backend.graph_slot_mapping_dtype(),
    device="cuda",
)

context_lens = torch.zeros(
    max_bs,
    dtype=torch.int32,
    device="cuda",
)
```

block table：

```python
max_num_blocks = (
    config.max_model_len
    + self.block_size
    - 1
) // self.block_size + 1
```

然后：

```python
block_tables = torch.full(
    (
        max_bs,
        max_num_blocks,
    ),
    -1,
    dtype=torch.int32,
    device="cuda",
)
```

这里必须：

```text
+ 1
```

因为 nano-vLLM 自己已有：

```text
runtime block_tables width
>
captured block_tables width
```

的公开 bug。

---

# 十一、Graph Context 必须静态

capture 时：

```python
set_context(
    False,

    slot_mapping=slot_mapping[:bs],

    context_lens=context_lens[:bs],

    block_tables=block_tables[:bs],

    max_context_len=config.max_model_len,

    use_cuda_graph=True,
)
```

注意：

```text
max_context_len
```

不要用：

```python
context_lens.max().item()
```

Graph capture 时必须固定。

CoreX 推荐：

```python
max_context_len=config.max_model_len
```

例如：

```text
2048
4096
8192
```

---

# 十二、为什么 CoreX 需要 max_context_len

IXFormer：

```python
flash_attn_with_kvcache(...)
```

明确要求：

```python
max_context_len is not None
```

Graph 下这个值同时参与 kernel/workspace/scheduling 选择。

因此 capture 期间必须使用静态上界：

```python
config.max_model_len
```

而不能每次 decode 改变 Python integer。

---

# 十三、Capture 流程

实现：

```python
@torch.inference_mode()
def capture_cudagraph(self):
    config = self.config
    hf_config = config.hf_config

    max_bs = min(
        config.max_num_seqs,
        512,
    )

    max_num_blocks = (
        config.max_model_len
        + self.block_size
        - 1
    ) // self.block_size + 1

    input_ids = torch.zeros(
        max_bs,
        dtype=torch.int64,
        device="cuda",
    )

    positions = torch.zeros(
        max_bs,
        dtype=torch.int64,
        device="cuda",
    )

    slot_mapping = torch.full(
        (max_bs,),
        -1,
        dtype=self.backend.graph_slot_mapping_dtype(),
        device="cuda",
    )

    context_lens = torch.zeros(
        max_bs,
        dtype=torch.int32,
        device="cuda",
    )

    block_tables = torch.full(
        (
            max_bs,
            max_num_blocks,
        ),
        -1,
        dtype=torch.int32,
        device="cuda",
    )

    outputs = torch.zeros(
        (
            max_bs,
            hf_config.hidden_size,
        ),
        dtype=hf_config.dtype,
        device="cuda",
    )

    self.graph_bs = (
        [1, 2, 4, 8]
        + list(
            range(
                16,
                max_bs + 1,
                16,
            )
        )
    )

    self.graphs = {}

    self.graph_pool = None

    for bs in reversed(self.graph_bs):

        graph = torch.cuda.CUDAGraph()

        set_context(
            False,

            slot_mapping=slot_mapping[:bs],

            context_lens=context_lens[:bs],

            block_tables=block_tables[:bs],

            max_context_len=config.max_model_len,

            use_cuda_graph=True,
        )

        #
        # Warmup
        #

        outputs[:bs] = self.model(
            input_ids[:bs],
            positions[:bs],
        )

        torch.cuda.synchronize()

        #
        # Capture
        #

        with torch.cuda.graph(
            graph,
            pool=self.graph_pool,
        ):
            outputs[:bs] = self.model(
                input_ids[:bs],
                positions[:bs],
            )

        if self.graph_pool is None:
            self.graph_pool = graph.pool()

        self.graphs[bs] = graph

        torch.cuda.synchronize()

        reset_context()

    self.graph_vars = {
        "input_ids": input_ids,
        "positions": positions,

        "slot_mapping": slot_mapping,

        "context_lens": context_lens,

        "block_tables": block_tables,

        "outputs": outputs,
    }
```

---

# 十四、为什么必须 Warmup

不要：

```text
第一次执行
直接 capture
```

应该：

```text
Warmup
 ↓
synchronize
 ↓
capture
```

原因包括 IXFormer 可能在首次执行发生：

```text
kernel lazy initialization
workspace allocation
runtime initialization
```

这些应该在正式 capture 前完成。

---

# 十五、Replay 时绝对不要重新分配 Buffer

错误：

```python
graph_vars["input_ids"] = input_ids
```

正确：

```python
graph_vars[
    "input_ids"
][:bs].copy_(input_ids)
```

CUDA Graph 捕获的是：

```text
Tensor address
```

不是 Python variable。

---

# 十六、run_model Replay 实现

推荐：

```python
@torch.inference_mode()
def run_model(
    self,
    input_ids,
    positions,
    is_prefill,
):

    if (
        is_prefill
        or self.enforce_eager
        or input_ids.size(0) > 512
    ):
        return self.model.compute_logits(
            self.model(
                input_ids,
                positions,
            )
        )

    bs = input_ids.size(0)

    context = get_context()

    graph_bs = next(
        (
            x
            for x in self.graph_bs
            if x >= bs
        ),
        None,
    )

    if graph_bs is None:
        return self.model.compute_logits(
            self.model(
                input_ids,
                positions,
            )
        )

    gv = self.graph_vars

    #
    # Safety
    #

    cols = context.block_tables.size(1)

    if cols > gv["block_tables"].size(1):
        return self.model.compute_logits(
            self.model(
                input_ids,
                positions,
            )
        )

    #
    # Copy dynamic input
    #

    gv["input_ids"][:bs].copy_(
        input_ids
    )

    gv["positions"][:bs].copy_(
        positions
    )

    #
    # Clear old metadata
    #

    gv[
        "slot_mapping"
    ].fill_(-1)

    gv[
        "context_lens"
    ].zero_()

    gv[
        "block_tables"
    ].fill_(-1)

    #
    # Runtime → static buffer
    #

    gv[
        "slot_mapping"
    ][:bs].copy_(
        context.slot_mapping
    )

    gv[
        "context_lens"
    ][:bs].copy_(
        context.context_lens
    )

    gv[
        "block_tables"
    ][:bs, :cols].copy_(
        context.block_tables
    )

    #
    # Replay
    #

    self.graphs[
        graph_bs
    ].replay()

    return self.model.compute_logits(
        gv[
            "outputs"
        ][:bs]
    )
```

---

# 十七、必须清空 block_tables

每次 replay：

```python
block_tables.fill_(-1)
```

然后：

```python
block_tables[
    :bs,
    :cols
].copy_(...)
```

不能省。

nano-vLLM 已存在公开 correctness bug：

```text
上一请求：

[3, 7, 12]

下一请求：

[5]
```

如果不清空，就可能留下：

```text
[5, 7, 12]
```

这是错误的物理 KV block。

---

# 十八、Padding 必须正确

例如：

```text
真实 batch = 3

选择 graph_bs = 4
```

那么 static buffers：

```text
index 0 真实
index 1 真实
index 2 真实
index 3 padding
```

padding 行必须保持：

```text
slot_mapping = -1

context_lens = 0

block_tables = -1
```

所以 replay 前：

```python
fill_ / zero_
```

是必须的。

---

# 十九、KV Cache 不复制

已经验证正确的 CoreX HND KV：

```text
[num_blocks,
 num_kv_heads,
 block_size,
 head_dim]
```

继续保持。

CUDA Graph 使用：

```text
同一个 KV Cache 地址
```

整个模型生命周期：

```text
allocate KV
     ↓
capture graph
     ↓
decode
     ↓
decode
     ↓
decode
```

不能：

```text
capture 后 realloc KV cache
```

---

# 二十、不要引入 torch.compile

天数官方：

```json
{
    "cudagraph_mode":
        "FULL_DECODE_ONLY",

    "level": 0
}
```

这里非常关键。

说明 CoreX 的 Full Graph 路线不要求：

```python
torch.compile(model)
```

所以 nano-vLLM 不需要为了 CUDA Graph 新增 compile。

继续：

```text
PyTorch eager model
      +
torch.cuda.CUDAGraph
```

即可。

---

# 二十一、第一项测试：CoreX CUDA Graph Runtime

新增：

```text
scripts/check_corex_cudagraph.py
```

测试：

```python
import torch


x = torch.zeros(
    1024,
    device="cuda",
)

y = torch.zeros_like(x)

graph = torch.cuda.CUDAGraph()


# Warmup

y.copy_(x * 2)

torch.cuda.synchronize()


# Capture

with torch.cuda.graph(graph):
    y.copy_(x * 2)


for i in range(20):

    src = torch.randn_like(x)

    x.copy_(src)

    graph.replay()

    torch.testing.assert_close(
        y,
        src * 2,
    )


print(
    "CoreX CUDA Graph PASS"
)
```

必须先 PASS。

---

# 二十二、第二项测试：CoreX Attention Graph

增加：

```text
tests/corex/test_cudagraph.py
```

覆盖：

```text
Decode
GQA
Paged KV
block_table
context_lens
slot_mapping
```

过程：

```text
Eager output

vs

Graph output
```

必须：

```python
torch.testing.assert_close(
    graph_output,
    eager_output,
    atol=1e-2,
    rtol=1e-2,
)
```

---

# 二十三、最后做 Qwen3 E2E

当前已有 eager baseline。

用：

```python
SamplingParams(
    temperature=0.0,
    max_tokens=64,
)
```

注意必须：

```text
temperature=0
```

避免采样随机性。

先：

```python
LLM(
    model,
    enforce_eager=True,
)
```

记录 token。

然后：

```python
LLM(
    model,
    enforce_eager=False,
)
```

记录 token。

要求：

```text
token IDs 完全一致
```

不是只比较看起来相似的文本。

---

# 二十四、测试顺序

严格按照：

```text
1
torch.cuda.CUDAGraph
simple probe

↓

2
IXFormer Decode Attention
CUDA Graph

↓

3
Qwen3 BS=1

↓

4
BS=2

↓

5
BS=3
验证 graph bucket 4

↓

6
BS=7
验证 graph bucket 8

↓

7
context 128

↓

8
context 512

↓

9
context 1024

↓

10
Prefix Cache

↓

11
256/512 token 长 Decode

↓

12
连续 100 请求
```

只有这些全部稳定后再做：

```text
TP2 + CUDA Graph
```

---

# 二十五、不要做的修改

本阶段禁止：

```text
修改 Scheduler

修改 BlockManager

修改 KV layout

替换 RMSNorm

替换 RoPE

替换 Silu

切 ixformer.distributed

修改 TP

增加 torch.compile

自己实现 Graph runtime
```

---

# 二十六、验收标准

CoreX CUDA Graph 第一阶段完成必须同时满足：

```text
Qwen3-1.7B eager        PASS

Qwen3-1.7B CUDA Graph   PASS

Eager / Graph tokens    一致

BS1/2/3/7               PASS

Paged KV                PASS

Prefix Cache            PASS

长 Decode               PASS

连续请求无 KV 污染       PASS
```

性能不是这一阶段的验收标准。

正确性全部完成后，再 benchmark：

```text
Eager Decode tok/s

vs

CUDA Graph Decode tok/s
```

---

# 二十七、核心原则

本次实现应保持：

                  nano-vLLM
                      │
                ModelRunner
                      │
             FULL_DECODE_ONLY
                      │
              torch.cuda.Graph
                      │
               CoreX Runtime
                      │
                  IXFormer


而不是：

                  nano-vLLM
                      │
                CoreXGraph
                      │
             自研 Graph runtime


天数 CoreX 的价值就在于：

```text
继续保持 CUDA execution model，
只替换真正需要 vendor optimization 的 kernel。
```

因此 nano-vLLM 也应该采用同样的设计。