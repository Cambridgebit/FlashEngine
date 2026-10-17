# Python 显存管理与 Triton 调用接口

本轮只为 Python FlashEngine 提供显存规划、请求生命周期和外部 Triton Python Backend 适配入口。
不实现 Triton Server、CV 模型后端、跨模型调度或 ensemble。外部 Triton 可以分别加载本适配器及其他模型。

## 显存预算

```python
from flashengine import LLM

engine = LLM(
    "/models/Qwen3-0.6B",
    backend="cuda",
    device_ids=(0,),                 # 进程可见设备索引，每个 TP rank 一项
    memory_budget_bytes=4 * 2**30,   # 当前引擎的规划预算，0 表示只按比例规划
    device_memory_reserve_bytes=2**30,
    memory_workspace_bytes=256 * 2**20,
    graph_memory_reserve_bytes=256 * 2**20,
    max_num_seqs=8,
    max_pending_requests=1024,
)
```

权重、recurrent 状态、稠密草稿上下文和投机静态缓冲先分配并计量，再规划目标与分页草稿 KV。
工作区取配置值与预热临时峰值的较大者；非 eager 另留 Graph 余量。
KV 可用字节取以下限制的最小值，再减工作区与 Graph 预留：

- 整卡 `total × gpu_memory_utilization − 当前整卡占用`。
- 当前空闲显存减 `device_memory_reserve_bytes`。
- 非零 `memory_budget_bytes` 减当前引擎常驻分配量。

显式 `num_kvcache_blocks` 同样必须通过预算检查；310P 的末块 dummy 单独计入块数。
初始化各阶段检查分配峰值；无预算或实际容量的请求返回明确错误。
Graph 捕获保留原有按桶回退机制，不自动关闭投机或截短请求。
未知或超限请求不会被静默改写；批量提交中途失败会取消已提交的本批请求。

预算是启动容量规划，**不是跨进程显存硬隔离**。第三方 native 分配、后续其他模型分配和超出预热负载的工作区仍需留足余量；设备执行失败时适配器停止接收新请求并释放引擎，需由外部管理系统重新加载。
增大 `device_memory_reserve_bytes` 用于给 CV 等模型留空间，FlashEngine 不负责测量或调度其他模型。
Prefix/KV 的空闲逻辑块仍驻留池内，不会在每次请求完成后缩池。

`runtime_snapshot()["memory"]` 返回预算、阶段 resident/peak、当前 allocated/peak 及实际设备。
数字来自该进程 allocator 统计和整卡空闲量，不把它们称为整卡进程硬配额。
`scripts.benchmark`、`scripts.openai_server` 共用 `--device`、`--memory-budget-mib`、
`--device-reserve-mib`、`--memory-workspace-mib`、`--graph-reserve-mib`。

## 引擎生命周期

```python
from flashengine import SamplingParams

seq = engine.add_request("Explain paged KV.", SamplingParams(temperature=0, max_tokens=32))
engine.step()
increment = engine.request_snapshot(seq, offset=0)
engine.cancel_request(seq)  # 也接受 seq.seq_id；重复取消返回 False
engine.close()             # 幂等
```

这些底层方法须由同一驱动线程串行调用。`EngineService` 为外部服务提供有界入流、
持续合批、期限和取消检查，并在该线程完成初始化、step、取消、关闭。
完成请求不保存在引擎的永久结果表内，句柄由调用方持有。
取消同时归还 KV、partial prefill、recurrent/投机状态。

TP=1 使用实例独立的 FileStore，无固定端口竞争；TP>1 自动选端口并使用唯一共享内存名称，
也可通过 `hccl_port` 固定部署端口。每个 Python 模型实例应独占一个进程的默认分布式进程组。
设备映射为 `device_ids[rank]`。初始化失败恢复 PyTorch 默认设备和 dtype 并清理半成品；
仅销毁本引擎拥有的进程组。NPU 模型卸载清理后端张量缓存，保留进程级运行时以允许再次加载。
RoPE 使用包含设备与 dtype 的弱引用共享缓存，模型卸载后不再被全局缓存持有。

## Triton Python Backend 契约

外部模型仓的 `1/model.py` 只需：

```python
from flashengine.triton_backend import TritonPythonModel
```

可复制 `examples/triton/flashengine/` 作为外部模型配置模板，按部署环境修改本地模型目录和预算。
FlashEngine 及后端依赖需已安装在外部 Python Backend 的运行环境中；
`triton_python_backend_utils` 由外部 Triton 提供，本包普通导入不要求它。
它与 CUDA kernel 编译器 `triton` 是不同组件。

| 方法 | 行为 |
|---|---|
| `initialize(args)` | 解析模型配置、设备、预算；在驱动线程创建 LLM |
| `execute(requests)` | 将输入复制成独立 str/list，提交到引擎，转换响应；每请求独立报告参数错误 |
| `finalize()` | 停止接收，取消在途/排队请求，等待驱动退出并关闭引擎；允许重复调用 |

要求 `max_batch_size: 0`，跨请求批处理交给引擎，避免额外一层张量 batch 语义。
`model_path` 参数必须为本地模型目录；`engine_config` 为 JSON 对象，字段使用 Config 原名，
未知字段显式拒绝。超时配置为字符串参数 `request_timeout`（默认 300 秒）、
`startup_timeout`（600 秒）、`shutdown_timeout`（30 秒），必须为正有限数。
期限与取消只在调度点生效，无法中断正在运行的设备 kernel。
关闭等待超时时抛错，不跨线程强制释放仍在执行的资源。

输入契约：

| 输入 | 类型、形状 | 含义 |
|---|---|---|
| `PROMPT` | STRING `[1]`，可选 | UTF-8 原始 prompt；不自动套 chat template |
| `INPUT_IDS` | INT64 `[N]`，可选 | 已分词的非空 token IDs；与 PROMPT 恰选其一 |
| `STREAM` | BOOL `[1]`，可选 | decoupled 默认 True；普通模式仅允许 False |

采样参数放在 Triton 请求 `parameters`：`temperature`（默认 0）、`top_p`、`top_k`、
`max_tokens`（默认 64）、`ignore_eos`。本轮不加入 OpenAI `messages` 转换。
输入长度加输出上限必须满足上下文与实际单请求 KV 容量；不自动截断。

每个响应包含 `TOKEN_IDS`（INT64 `[N]`）、`TEXT`（STRING `[1]`）、
`FINISH_REASON`（STRING `[1]`）、`PROMPT_TOKENS`/`COMPLETION_TOKENS`（INT64 `[1]`）。
流式 TOKEN_IDS 是新增 token；TEXT 是**累计文本**，避免 BPE 未完成字符造成拼接损坏。
中间响应的 FINISH_REASON 为空，最终为 `stop` 或 `length`。
取消、期限、资源不足和执行失败通过 TritonError 返回。

普通模式按输入顺序返回等长响应列表；decoupled 模式返回 None，使用 sender 发送多次响应，
成功、取消、关闭及参数错误均发送一次且仅一次 FINAL（传输本身失败不重试）。
异步路径不保留原 request/输入数组视图。
`model_transaction_policy { decoupled: true }` 可同时服务 STREAM=True/False，
但客户端必须使用支持 decoupled 的双向 gRPC；HTTP Infer 和普通单响应 gRPC 不支持该模式。
取消 API 要求外部 Triton Python Backend 至少支持 23.10 的取消接口。

GPU 实例使用 `model_instance_device_id` 映射 CUDA 兼容的进程可见设备，限 TP=1。
CoreX/310P 推荐由厂商运行环境中的 CPU 实例托管 Python stub，显式配置 backend 和 device_ids；
CPU 实例不表示引擎在 CPU 推理。NPU ID 不能取 NVIDIA GPU instance ID。
多卡 TP 也使用 CPU 实例和完整 device_ids。Python stub ABI、厂商 SDK 和 Triton 构建需分别验收。

## 验证与边界

```bash
python3 -B -m unittest discover -s tests -v
python3 -B -m scripts.check_python_memory /models/Qwen3-0.6B \
  --backend cuda --cycles 2 --rounds 3 --memory-budget-mib 4096 \
  --device-reserve-mib 512 --output bench_results/python-memory/dense.json
```

同一入口支持三后端及现有目标/草稿注册组合，不扩大支持矩阵。用 `--eager`、
`--no-prefix-cache`、`--spec-model`、`--spec-num-tokens` 分别验证执行路径。
用 `--backend corex/ascend310p` 在对应真实环境运行，不能在 CUDA 上代验。
合成草稿只验证状态与内存行为，不证明训练质量或加速收益。

测试检查真实 Prefix 命中、连续批处理负载、partial prefill 取消、后续推理、
稳定轮次显存、权重/KV 弱引用销毁和反复加载卸载。
首轮允许有界运行库缓存（默认 32 MiB），后续循环检查增量不持续增长（默认 16 MiB），
并独立检查模型张量全部销毁，不能靠容许量掩盖模型缓存泄漏。
PyTorch 2.11 会按 cuBLAS handle/stream 保留工作区；不在引擎卸载时清除进程内其他模型可能使用的全局运行库资源。

官方接口依据：[Python Backend](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/python_backend/README.html)、
[Decoupled 模式](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/decoupled_models.html)、
[PyTorch 2.11 cuBLAS 工作区](https://docs.pytorch.org/docs/2.11/notes/cuda.html#cublas-workspaces)。

本轮实际结果见 [测试记录](../bench_results/python-memory/REPORT.md)：48 项主机测试、2 项真实 CUDA 接口测试，以及 9 个 CUDA 配置的 18 次加载/卸载循环通过。Triton 替身契约测试不代表真实服务器验收。
CoreX、310P、未提供的模型组合及 TP 多卡结果必须单独保留未验收状态。
