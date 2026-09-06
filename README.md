# Nano-vLLM（EAGLE3 fork）

从零实现的轻量级推理引擎（上游 nano-vllm ≈1200 行 + EAGLE3 ≈1200 行）。在上游 prefix caching / Tensor Parallel / torch.compile / chunked prefill 之上新增：

- **EAGLE3 投机解码**：draft 链式起草 + target 单次验证 + **spec 单图整轮**（每轮 2 次 H2D + 1 次 D2H）
- **两级 CUDA Graph**：decode 单步图（图内采样）+ spec 整轮图；捕获失败自动降级 eager，只降速不崩溃
- **多硬件 backend**：NVIDIA CUDA（已实证）/ 天数智芯 CoreX（代码就绪待实机）/ 华为 Ascend 310P（代码就绪待实机）
- **OpenAI 兼容 server**：continuous batching + SSE 流式，纯 stdlib 零新依赖，支持投机

> 开发约定、实测数据明细与「勿回退」清单见 [AGENTS.md](AGENTS.md)。

## 快速开始

```bash
pip install -e . && pip install tqdm flash-attn --no-build-isolation   # tqdm 未声明进 pyproject
```

模型需本地目录（HF hub id 不可用）。离线推理：

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-1.7B",
          spec_model="/path/to/Qwen3-1.7B-EAGLE3",   # EAGLE3 draft（官方 AngelSlim 等）
          spec_num_tokens=3)                         # 每轮起草深度 k
print(llm.generate(["你好"], SamplingParams(temperature=0, max_tokens=512))[0]["text"])
```

OpenAI 兼容 server 与基准（必须从仓库根目录以 `-m` 方式运行，直跑会 ModuleNotFoundError）：

```bash
python3 -m scripts.openai_server /path/to/Qwen3-1.7B --port 8000 --max-model-len 2048
#   投机解码：--spec-model /path/to/Qwen3-1.7B-EAGLE3 --spec-num-tokens 3
#   默认 max_num_seqs=8（8GB 卡实测宽桶拖慢单步回放；需高并发显式传 --max-num-seqs）
python3 -m scripts.benchmark graph /path/to/Qwen3-1.7B --batch 1 2 4 8               # 吞吐基准
python3 -m scripts.benchmark jsonl /path/to/Qwen3-1.7B --spec-model /path/to/Qwen3-1.7B-EAGLE3   # 真实问题集 TPS/TTFT
python3 -m scripts.check_eagle3 /path/to/Qwen3-1.7B --draft /path/to/Qwen3-1.7B-EAGLE3   # 验收
```

## 目录结构

```
nanovllm/
├── llm.py                  # LLM 门面（generate API）
├── config.py               # 引擎配置（slotted dataclass，kwargs 按字段过滤）
├── sampling_params.py      # temperature / top_p / top_k / max_tokens
├── engine/
│   ├── llm_engine.py       # step / step_spec 主循环
│   ├── scheduler.py        # 连续批调度、spec 容量预分配、多 token 提交
│   ├── block_manager.py    # paged KV 分配 + prefix cache（block 级 xxhash）
│   ├── sequence.py         # 序列状态机
│   └── model_runner.py     # 执行层总指挥（prepare → 图/eager → 读回）
├── models/
│   ├── qwen3.py            # target 模型（aux 隐层捕获）
│   └── eagle3.py           # EAGLE3 draft（官方 cnets 链式推理等价实现）
├── layers/                 # attention / activation / embed_head / layernorm / linear / rope / sampler
├── backends/               # base 接口 ← cuda / corex / ascend310p
└── utils/                  # context 全局上下文 + 权重加载
scripts/
├── benchmark.py            # graph / prefix / jsonl 吞吐与 TTFT 基准
├── check_eagle3.py         # EAGLE3 验收（E / 逐位接受率 / 直方图 / TPS，--save-out 可回归比对）
├── check_prefix_cache.py   # prefix cache 数值回归门（退出码非零 = FAIL）
├── openai_server.py        # OpenAI 兼容 server
├── verify_corex.sh         # CoreX 实机五步验证清单
└── verify_ascend310p.sh    # Ascend 310P 实机验证清单（环境预检+载荷算子探测+四步）
```

## 架构

六层单向分层：上两层纯 Python 调度（零 GPU 代码），下三层纯算子与厂商 kernel（零调度逻辑），`ModelRunner` 是唯一跨界的「总指挥」。

```
API 层    LLM（generate → add_request → step 循环）
引擎层    LLMEngine（step / step_spec 主循环）
          ├ Scheduler     连续批调度 / 抢占 / spec 容量预分配与提交
          └ BlockManager  paged KV 分配 + prefix cache 块级 hash
执行层    ModelRunner：prepare_*（host 元数据 → pinned MetaBlob，2 次 H2D）
          → CUDA Graph | eager → 读回（每轮 1 次 D2H）
模型层    Qwen3ForCausalLM（target）/ Eagle3DraftModel（draft，共享词表）
算子层    Attention / RMSNorm / Linear(TP) / Embedding / RoPE / MLP / Sampler
硬件层    backends/：CUDABackend（flash-attn + triton）/ CoreXBackend（ixformer）/ Ascend310PBackend（torch_npu）
```

层间仅三种耦合，无横向越级调用：

1. **方法调用**（`ModelRunner.call` 兼容 TP 广播）；
2. **全局 context**（`utils/context.py`：前向前写入 slot_mapping / block_tables / cu_seqlens，
   `Attention.forward` 内读取——元数据不进模型签名，图捕获时换静态 buffer 视图即可无缝切换）；
3. **backend 单例**（厂商 kernel 全部集中在 `store_kvcache / prefill_attention / decode_attention` 三方法）。

**EAGLE3 一轮（k=3，l=已提交长度）**：采样行引导（draft1 + 隐层 carry）→ 链步 1..k-1 →
verify（target 单次前向，行 j 预测位置 l+j）→ 图内链式接受（最长匹配前缀 m，提交 m+1 token）→
refill（虚构窗口重放，产出下一轮 draft1）。成本：k+1 次前向换 m+1 个 token；每轮 2 H2D + 1 D2H。
关键口径（shifted draft 流 / 采样行=refill 行 m / 冷启动同构 eager）见 AGENTS.md「勿回退」清单。

**两级 CUDA Graph**：decode 单步图（图内 lm_head + 采样）与 spec 整轮图按 batch 桶捕获
（[1,2,4,8]+16 步进，自动补 max_bs 尾桶）。降级链：spec 图失败 → 同构 eager spec 轮；
decode 图失败 → 全 eager 前向——任何失败只降速，不崩溃。

**TP>1**：rank>0 子进程跑 `ModelRunner.loop()`，方法调用经 SharedMemory 广播，部分结果
`backend.all_reduce` 汇聚；spec 依赖 rank0 draft 状态，仅支持 TP=1。

## 性能（RTX 4070 Laptop 8GB / WSL2，Qwen3-1.7B + EAGLE3，单会话实测）

**离线**（spec_bench coding 前 10 条，串行 bs=1，temp0，k=3）：

| 配置 | TPS（含 prefill，保守） | 接受长度 E |
|---|---|---|
| ours-decode | 66.2 | — |
| **ours-spec(k=3)** | **108.7** | 2.095 |
| nanovllm-initial（上游原版） | 68.2 | — |
| vLLM 0.26 decode | 69.5 | — |
| vLLM 0.26 eagle3(k=3) | 125.6 | — |

**在线 OpenAI server**（stream_bench：3 条中文散文提示词，temp0.7，串行）：

| 服务 | TPS 中位 | 平均 TTFT |
|---|---|---|
| vLLM 0.26 | 66.7 | 34ms |
| vLLM 0.26 eagle3(k=3) | 64.4 | 34ms |
| nanovllm-initial（变体 server） | 70.1 | 53ms |
| ours（spec off） | 61.7 | 54ms |
| ours（spec on, k=3） | 59.7 | 70ms |

- **decode 与上游原版持平**：背靠背配对 A/B 68.1 vs 70.2 = **0.97**（离线口径 66.2/68.2 = 0.97 一致）。
- **spec 结构性收益**：ours 108.7/66.2 = **1.64×**，vLLM 125.6/69.5 = **1.81×**；ours-spec =
  原始版最优形态（纯 decode）的 **1.59×**（原始版无投机能力）。
- **spec 吃负载**：低接受负载（中文散文 E≈1.2）两引擎都负收益（ours 61.7→59.7，vLLM 66.7→64.4
  互证）——低 E 负载勿开 `--spec-model`；高接受负载（coding/temp0）+60% 以上。
- **与 vLLM spec 残差 ~16%**：decode 基础差 ~5% + 每轮 host 轮时 ~7%（vLLM 异步调度重叠，本引擎同步）
  + 接受长度数值漂移 ~6%（同权重不同实现的固有发散，不可收敛）。
- 消融（4 并发 + 1100 token 共享前缀 + 96 token 生成）：基线 126 → +图 155（+23%）→ +spec 179
  （+42%，spec 必须有图）→ 全开 187（+49%）；prefix cache 使后续请求 TTFT **-46%**。

## 多硬件适配

切换：`LLM(..., backend="cuda|corex|ascend310p")` / `NANOVLLM_BACKEND=<同名>` / 自动探测（`torch.corex` 标记 → corex；`torch.npu` 存在且可用 → ascend310p；否则 cuda）。
厂商差异全部收在 `BackendBase` 的方法与能力声明里，业务代码零感知：

| 抽象点 | CUDA | CoreX | Ascend 310P |
|---|---|---|---|
| KV 布局 | 连续 [2,L,B,S,H,D] | HND [2,L,B,H,S,D] | 同 CUDA |
| 图 slot dtype | int32 | int64 | int32 |
| KV 写入 / attention | triton + flash-attn | ixformer（vendor .so） | torch_npu（`_npu_reshape_and_cache` + TND prefill / BNSD decode） |
| spec 图捕获 | 已实证 | **已实证**（失败自动降级兜底） | ❌ 引擎层拒绝（paged-TND 待后续） |
| TP `dist_name` | nccl | nccl（厂商兼容层） | hccl |

- **CoreX 状态**：**实机验证通过**（2026-09，32GB 板卡 / corex-4.4.0 / transformers 4.57.6）——
  `bash scripts/verify_corex.sh <target> <draft>` 五步全过，spec 单图整轮一次捕获成功：
  decode 图收益 **9.3×**（eager 7.8 → 图 72.9 tok/s @bs=1；launch 税为 CoreX 主要瓶颈，图是刚需），
  spec 图 E=2.050 / **103.7 tok/s**（≈4070 的 108.7 同量级），spec eager 11.3 无收益（与 CUDA「spec 必须有图」互证）。
  jsonl 三配置（coding 前 10 条，串行 temp0）：eager 8.1 → graph 76.5 → **spec+graph 104.1**（+36%，vs eager 12.9×）。
  实机修复：图捕获 `context_lens` 独立 int32（ixformer 硬检查）、空板卡 KV 贴地需 `--gpu-mem-util 0.85`、
  旧版 transformers `apply_chat_template` 返回 list 兼容（7 处）。
  判读：`benchmark graph` 的 correct 列仅参考（temp0 下图内/图外 lm_head 的 ULP 发散；主判据 = 同配置位等，全 PASS）。
- **Ascend 310P 状态**：代码就绪（torch_npu 懒加载+承重算子守卫、设备代理 `torch.npu.*`、NPUGraph 图工厂、NPU 融合算子），**待实机验证**——`bash scripts/verify_ascend310p.sh <target> <draft>`。已知限制：prefix 命中 prefill 的 k/v=整池 4D 切片，TND 算子不消费（与参考实现同款，需 paged-TND 后续适配）——spec 全路径因此被引擎层拒绝（`supports_spec=False`）。

## 环境变量

| 变量 | 作用 |
|---|---|
| `NANOVLLM_BACKEND` | 强制后端：cuda / corex / ascend310p（默认 auto 探测） |
| `NANOVLLM_SPEC_GRAPH=0` | 关闭 spec 整轮图（强制 eager spec 轮） |
| `NANOVLLM_GRAPH_SAMPLE=0` | 关闭 decode 图内采样（采样回图外） |
