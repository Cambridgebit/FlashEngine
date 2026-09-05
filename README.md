<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM（EAGLE3 fork）

从零实现的轻量级 vLLM 推理引擎。在上游（prefix caching / Tensor Parallel / torch.compile / chunked prefill）之上，本 fork 新增：

- **EAGLE3 投机解码**：draft 链式起草 + target 单次验证，**单图整轮 CUDA Graph**（每轮仅 2 次 H2D + 1 次 D2H）
- **两级 CUDA Graph**：decode 单步图（含端到端图内采样）+ spec 整轮图，失败自动降级
- **多硬件后端**：NVIDIA CUDA（flash-attn）/ 天数智芯 CoreX（ixformer），能力声明式切换，Ascend 预留接口位
- **OpenAI 兼容 server**：并发 continuous batching + 逐 token SSE 流式（纯 stdlib，零新依赖）

> 开发约定、测试命令与"勿回退"清单见 [AGENTS.md](AGENTS.md)。

## 快速开始

```bash
git clone <本仓库> && cd nano-vllm && pip install -e .
pip install tqdm flash-attn --no-build-isolation   # tqdm 未在 pyproject 声明
```

模型需要本地目录（不能用 HF hub id），如 Qwen3 系列：

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3-1.7B",
    spec_model="/path/to/Qwen3-1.7B-EAGLE3",   # EAGLE3 draft 权重目录（官方 AngelSlim 等）
    spec_num_tokens=3,                          # 每轮起草深度 k
)
outputs = llm.generate(["你好，介绍一下你自己。"], SamplingParams(temperature=0, max_tokens=512))
print(outputs[0]["text"])
```

OpenAI 兼容服务与多硬件切换：

```bash
python3 scripts/openai_server.py /path/to/Qwen3-1.7B [--port 8000] [--eager]
# 多硬件：LLM(..., backend="corex") 或环境变量 NANOVLLM_BACKEND=corex（默认 auto 探测）
```

## 架构分层

引擎共六层，上两层纯 Python 调度（零 GPU 代码），下三层纯 GPU 算子与厂商 kernel（零调度逻辑），中间执行层 `ModelRunner` 是唯一两边都碰的"总指挥"：

```
┌─────────────────────────────────────────────────────────────────┐
│ API 层    LLM (llm.py)                                          │
│           generate(prompts) → add_request → step 循环            │
├─────────────────────────────────────────────────────────────────┤
│ 引擎层    LLMEngine (engine/llm_engine.py)                      │
│           step() 普通主循环 / step_spec() 投机主循环              │
│    ┌────────────────────┬───────────────────────────┐           │
│    │ Scheduler           │ BlockManager              │           │
│    │ (engine/scheduler)  │ (engine/block_manager)    │           │
│    │ 连续批调度/抢占/     │ paged KV 分配 +            │           │
│    │ spec 容量预分配/提交 │ prefix cache 块级 hash     │           │
│    └────────────────────┴───────────────────────────┘           │
├─────────────────────────────────────────────────────────────────┤
│ 执行层    ModelRunner (engine/model_runner.py)                  │
│           prepare_*（host 元数据 → pinned blob）                  │
│           → MetaBlob 2 次 H2D → CUDA Graph | eager → 读回         │
├─────────────────────────────────────────────────────────────────┤
│ 模型层    Qwen3ForCausalLM (models/qwen3.py)   ← target          │
│           Eagle3DraftModel (models/eagle3.py)  ← draft（共享词表）│
├─────────────────────────────────────────────────────────────────┤
│ 算子层    layers/：Attention · RMSNorm · Linear(TP 切分) ·       │
│           VocabParallelEmbedding · RoPE · SiluAndMul · Sampler   │
├─────────────────────────────────────────────────────────────────┤
│ 硬件层    backends/：BackendBase 接口                            │
│           ├─ CUDABackend   → flash-attn kernels + triton store   │
│           └─ CoreXBackend  → ixformer 专用算子（vendor .so）      │
└─────────────────────────────────────────────────────────────────┘
```

各层之间只通过三种途径耦合，没有横向越级调用：

1. **方法调用**：上层 `getattr` 下层（`ModelRunner.call` 同时兼容 TP 广播）；
2. **全局 context**（`utils/context.py`）：`set_context()` 在每次前向前写入当前批的
   slot_mapping / block_tables / cu_seqlens，`Attention.forward` 内部 `get_context()` 读取——
   元数据不进模型签名，图捕获时换成静态 buffer 视图即可无缝切换；
3. **backend 单例**（`backends/__init__.py`）：`Attention` 持有 `get_backend()` 返回的实例，
   所有厂商 kernel 调用集中在 `store_kvcache / prefill_attention / decode_attention` 三个方法。

## 一次 generate 的完整调用流

```
LLM.generate(prompts, params)
 │  tokenize → Scheduler.add(seq)×N
 └─ while 未全部完成: LLMEngine.step()
      │
      ├─ Scheduler.schedule() → (seqs, is_prefill)
      │    # prefill：chunked 组批（首个 seq 允许切分）+ prefix cache 命中裁剪
      │    # decode ：连续批滚动，can_append 不足时抢占
      │
      ├─【prefill 分支】ModelRunner.run / run_prefill_spec
      │    prepare_prefill()  组 ids/positions/slots/cu_seqlens → H2D
      │    model(ids, pos)    target 前向（spec 模式同时捕获 3 个 aux 隐层）
      │    Sampler → Scheduler.postprocess()（hash_blocks + eos 判定）
      │    （spec 模式：draft 同 chunk prefill，末行 aux 存入 spec_hidden）
      │
      ├─【decode，spec 关】ModelRunner.run
      │    prepare_decode()   → MetaBlob 填充 → 2 次 H2D → 图内 scatter
      │    run_model()        → decode 单步 CUDA Graph replay（图内采样）
      │    Scheduler.postprocess()
      │
      └─【decode，spec 开】一轮投机 = 多 token 提交
           Scheduler.prepare_spec_round()   # 为每个 running seq 预分配 len+k 容量
           ModelRunner.spec_round()         # 见下方时序图
           Scheduler.commit_spec()          # 逐 token 追加 + eos/max_tokens 截断
```

Tensor Parallel（TP>1）：rank>0 子进程跑 `ModelRunner.loop()`，主进程把每个方法调用经
`SharedMemory` 广播（`write_shm/read_shm`），线性层的部分结果经 `backend.all_reduce` 汇聚；
spec 模式依赖 rank0 的 draft 状态，仅支持 TP=1（config 断言）。

## EAGLE3 投机解码：一轮发生了什么

以 k=3（spec_num_tokens=3）为例，l = 序列当前已提交长度：

```
 一轮 spec_round
 ────────────────────────────────────────────────────────────────────
 ① 采样行引导   draft.forward_decode(rope l-2, seed=fc(aux), last_token)
                → draft1（本轮首个提案）+ 隐层 carry（图轮经上一轮 refill 续流）
 ② 链步 1..k-1  forward_decode(rope l-2+i, 读窗 l+i-1)
                → d2, d3（TTT 流式续写，写自己的流槽）
 ③ verify       target([last_token, d1..dk]) 单次前向，行 j 预测位置 l+j
 ④ 图内接受     match(d_j == t_j) 取最长前缀 m → 提交 [d1..dm, bonus]
 ⑤ refill       虚构窗口重放（复用 verify 的位置/槽位/aux，embed 右移一位，
                行 m 覆盖为 bonus）→ 产出下一轮 draft1 + 采样行隐层 carry
 ────────────────────────────────────────────────────────────────────
 成本：k+1 次 draft/target 前向换 m+1 个 token；每轮 2 次 H2D + 1 次 D2H
```

三个关键口径（详见 AGENTS.md"勿回退"清单）：

- **shifted draft 流**：rope p 的行 embed `tok@p+1`、条件 `fc(aux@p)`（look-ahead）、
  预测 `tok@p+2`——draft prefill、链步、refill 全部同口径；
- **采样行 = refill 行 m**（对齐 vLLM `index_to_sample = 段尾 - 被拒数`），其隐层经
  `spec_carry[seq_id]` 交给下一轮链步 1 当种子；
- **冷启动**：prefill 后首轮无 carry，走与图轮同构的 eager 体补齐，之后进图。

cuda graph 捕获失败 / 桶缺失时自动回落到同构 eager 体，功能不受影响。

## 两级 CUDA Graph 与 MetaBlob

```
decode 单步图                          spec 单图整轮
┌───────────────────────────┐   ┌─────────────────────────────────────┐
│ host 填 pinned blob        │   │ host 填 pinned blob（_sblob）        │
│   ids/pos/slots/ctx/bt/topk│   │   链步/verify 位置槽位、采样参数、bt  │
│ ↓ 2 次 H2D                 │   │ ↓ 2 次 H2D                          │
│ 图内 scatter 分发到静态     │   │ 图内 scatter → draft 链 → verify     │
│ buffer（int64→int32 转型   │   │ → 图内链式接受 → refill（含下一轮     │
│ 在 copy 内完成）           │   │   draft1 采样）                      │
│ ↓ replay                   │   │ ↓ replay                            │
│ 图内 lm_head+Sampler       │   │ commit_buf 读回（每轮 1 次 D2H）     │
│ → token_ids                │   │ [d1..dm, bonus, cnt] 打包最后一列    │
└───────────────────────────┘   └─────────────────────────────────────┘
```

- 桶按 batch 大小分档捕获（[1,2,4,8]+16 步进，自动补 max_bs 尾桶）；
- **降级链**：spec 图捕获失败 → 同构 eager spec 轮；decode 图捕获失败 → 全 eager 前向；
  环境逃生口 `NANOVLLM_SPEC_GRAPH=0` / `NANOVLLM_GRAPH_SAMPLE=0`。任何失败只降速，不崩溃；
- `MetaBlob`（声明式 pinned 元数据）：段声明 → 自动偏移/缓存视图/设备孪生，热路径零临时分配。

## 多硬件适配

```
                    ┌─ backend="cuda"  ──→ CUDABackend   → flash-attn + triton store
backend 选择 ──────┤
(参数/env/自动探测)  └─ backend="corex" ──→ CoreXBackend   → ixformer（vendor .so）
```

厂商差异全部收在 `BackendBase` 的六个方法 + 两个能力属性里：

| 抽象点 | CUDA | CoreX |
|---|---|---|
| KV 写入 `store_kvcache` | triton kernel（slot=-1 跳过） | ixformer `reshape_and_cache_flash` |
| attention | `flash_attn_varlen/with_kvcache` | ixformer 同名接口（HND 布局） |
| KV 布局 / 图 slot dtype | 连续布局 / int32 | HND 布局 / int64 |
| TP 通信 `dist_name` | nccl | nccl（厂商兼容层） |
| spec 图捕获 `supports_spec_graph` | True（已实证） | True（尝试 + 失败自动降级，实机验证见下） |

CoreX 实机验证：`bash scripts/verify_corex.sh <target> <draft> [questions.jsonl]`
五步清单（decode 图回归 → graph_sample 对照 → spec eager E → spec 图捕获尝试 → TPS sweep）。
Ascend 等后续硬件：新增一个 Backend 子类 + 设备代理层（预留接口与方案见 AGENTS.md「CoreX（Iluvatar）移植状态」节）。

## 性能（RTX 4070 Laptop 8GB / WSL2，Qwen3-1.7B + EAGLE3）

| 场景 | vLLM 0.26 | 本引擎 |
|---|---|---|
| spec_bench coding，k=3，串行（接受长度 E） | 2.25 | 2.095 |
| 同场景吞吐 | 129.8 tok/s | ~112-114 tok/s（重构后；重构前 106-112，见 AGENTS.md） |
| decode 吞吐（bs=1，CUDA Graph） | — | ~69 tok/s（benchmark graph 默认负载实测） |

消融（4 并发 + 1100 token 共享前缀 + 96 token 生成）：
基线 126 tok/s → +CUDA Graph **155**（+23%）→ +投机 **179**（+42%，spec 必须有图）→
全开 **187**（+49%）；prefix cache 使后续请求 TTFT **-46%**。

剩余与 vLLM 的 ~14% 差距：~7% 为每轮 host 耗时（vLLM 有异步调度重叠），~6% 为两边
生成文本数值漂移导致的接受率统计差（结构已完全对齐，见 AGENTS.md）。

## 目录结构

```
nanovllm/
├── llm.py                 # LLM = LLMEngine 门面（generate API）
├── config.py              # 引擎配置（slotted dataclass，kwargs 按字段过滤）
├── sampling_params.py     # temperature / top_p / top_k / max_tokens
├── engine/
│   ├── llm_engine.py      # step / step_spec 主循环 + generate
│   ├── scheduler.py       # 连续批调度、spec 容量预分配、多 token 提交
│   ├── block_manager.py   # paged KV + prefix cache（block 级 xxhash）
│   ├── sequence.py        # 序列状态机
│   └── model_runner.py    # 执行层总指挥（prepare → 图/eager → 读回）
├── models/
│   ├── qwen3.py           # target 模型（aux 隐层捕获，forward 返回 (hidden, aux)）
│   └── eagle3.py          # EAGLE3 draft（官方 cnets 链式推理等价实现）
├── layers/                # Attention / RMSNorm / Linear(TP) / Embedding / RoPE / MLP / Sampler
├── backends/              # BackendBase 接口 ← CUDABackend / CoreXBackend
└── utils/                 # context 全局上下文 + 权重加载
scripts/
├── benchmark.py           # graph / prefix 吞吐基准
├── check_prefix_cache.py  # prefix cache 数值回归门（退出码非零 = FAIL）
├── check_eagle3.py        # EAGLE3 验收：E / 逐位接受率 / 直方图 / TPS（--save-out 可比对）
├── verify_corex.sh        # CoreX 实机五步验证清单
└── openai_server.py       # OpenAI 兼容 server
```

## 环境变量

| 变量 | 作用 |
|---|---|
| `NANOVLLM_BACKEND` | 强制后端：`cuda` / `corex`（默认 auto 探测 `torch.corex` 标记） |
| `NANOVLLM_SPEC_GRAPH=0` | 关闭 spec 单图整轮捕获（强制 eager spec 轮） |
| `NANOVLLM_GRAPH_SAMPLE=0` | 关闭 decode 图内端到端采样（采样回图外） |

上游基础特性（prefix caching / TP / compile / chunked prefill）的使用方式与原版一致，
参见原版 README 中的基准：Qwen3-0.6B 256 序列，vLLM 1361.84 tok/s vs nano-vLLM 1434.13 tok/s。
