# FlashEngine V2.1

## Introduction

本项目基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，用于研究和学习大模型推理引擎。代码覆盖请求调度、模型执行、KV cache 管理和硬件后端，提供 Python API 与 OpenAI 兼容 HTTP 服务。

当前目标模型为 Qwen3、Qwen3 MoE 与 Qwen3.5 dense/MoE 文本模型，后端为 NVIDIA CUDA、天数智芯 CoreX、华为 Ascend 310P。Qwen3 支持 EAGLE3、DFlash、DSpark；Qwen3 MoE 与 Qwen3.5 系列支持 DFlash。Qwen3 的 W8A8 支持范围不变，Qwen3.5 当前为 fp16/bf16、TP=1，不加载视觉塔和 MTP。完整组合见 [支持矩阵](docs/SUPPORT_MATRIX.md)，Qwen3.5 的实现与验证边界见 [适配说明](docs/QWEN35.md)。CoreX/310P 的 Qwen3.5 路径尚无实机验收。

## Quickstart

Qwen3.5 DFlash 的 CUDA 性能热点已融合，保守复验约 135 tok/s（修复前约 72），
本机已测 k=15 最快。配置、数值边界与复现口径见 [DFlash 性能修复](docs/DFLASH_PERFORMANCE.md)。

需要 Python 3.10–3.12、对应加速卡运行环境及本地模型目录。以下命令在 Linux 或 WSL 的仓库根目录运行，模型参数不能直接使用 HuggingFace hub ID。

先安装与硬件匹配的 PyTorch，再安装项目依赖（核心依赖含 `tqdm`、`numpy`、`safetensors`，已声明于 `pyproject.toml`）：

```bash
python3 -m pip install -e .
# 或：python3 -m pip install -r requirements.txt
```

CUDA 用 `.[cuda]` 安装 `triton` 与 `flash-attn`；CoreX 需要厂商环境和 `ixformer`（`.[corex]`），Ascend 310P 需要匹配的 CANN 与 `torch-npu`（`.[ascend]`）。Ascend 310P 的完整实测版本清单见 [`requirements-ascend.txt`](requirements-ascend.txt)（**不含 vLLM / vLLM-Ascend**——引擎在 310P 上不依赖它们）。例如：

```bash
python3 -m pip install -e ".[cuda]"
# 等价于手动安装：
#   python3 -m pip install triton
#   python3 -m pip install flash-attn --no-build-isolation
```

### 启动服务

```bash
python3 -m scripts.openai_server /path/to/Qwen3-1.7B \
    --host 127.0.0.1 --port 8000 --max-model-len 2048 \
    --max-num-seqs 4 --gpu-mem-util 0.85 --no-think
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3-1.7B","messages":[{"role":"user","content":"你好"}],"max_tokens":128,"stream":true}'
```

开启 EAGLE3，在启动命令后加 `--spec-model /path/to/Qwen3-1.7B-EAGLE3 --spec-num-tokens 3`。DFlash 或 DSpark 同样通过 `--spec-model` 指定与目标模型匹配的草稿目录，草稿长度按 checkpoint 自动派生。用 `--backend cuda|corex|ascend310p` 指定后端，默认自动探测。显存不足时降低上下文长度、并发数，或加 `--eager` 关闭 Graph。

启动后打开 [http://127.0.0.1:8000/](http://127.0.0.1:8000/) 使用内置对话与性能面板。

### 运行基准

```bash
python3 -m scripts.benchmark /path/to/Qwen3-1.7B \
    --questions bench_data/question.jsonl --num-prompts 10 \
    --max-model-len 2048 --max-num-seqs 4 --gpu-mem-util 0.85 --no-think
```

`--spec-model` 开启投机基准，`--no-prefix-cache` 关闭前缀缓存。脚本以 `python3 -m scripts.<名称>` 运行。HTTP 并发压测使用 `scripts.bench_concurrent`，与进程内串行基准分别记录，不混用测量口径。

## 代码目录与项目架构

```text
flashengine/
├── llm.py / config.py / sampling_params.py  # API、配置、采样参数
├── engine/
│   ├── llm_engine.py                       # 引擎主循环与生命周期
│   ├── scheduler.py / sequence.py          # 调度与请求状态
│   ├── block_manager.py                   # KV 分页与前缀缓存
│   ├── model_runner.py                    # 模型执行、Graph、投机轮
│   └── observer.py                        # 调度与提交事件
├── models/                                # Qwen3、草稿模型、目标注册表
├── speculative/                           # 投机方法注册表与运行策略
├── layers/                                # Attention、Linear、Norm、RoPE、采样
├── backends/                              # CUDA、CoreX、Ascend 算子适配
│   └── ascend310p/                         # NPUGraph 与图内注意力
└── utils/                                 # 执行上下文与权重加载
scripts/                                   # HTTP 服务、Web 控制台、基准脚本
bench_data/                                # 英文、中文、共享前缀数据集
docs/                                      # 算法、后端与压测说明
```

```text
Python API / HTTP 服务
        ↓
LLMEngine → Scheduler + BlockManager
        ↓
ModelRunner + 投机运行策略
        ↓
目标模型 / 草稿模型 → 通用层 → 硬件后端算子
```

引擎将请求加入队列，调度器选择 prefill 或 decode 批次并分配 KV 块，执行器调用模型，最后提交输出 token 并释放已完成请求的状态。普通解码每步为各请求提交一个 token；投机解码每轮可提交多个 token。

目标模型与投机方法分别通过注册表解析；后端提供设备、数据类型、KV 布局、注意力算子与 Graph 能力。`runtime_snapshot()` 和观察接口为服务及基准提供统一统计。实现约定见 [引擎架构说明](docs/ENGINE_RUNTIME.md)。

## 推理加速与服务功能

### Paged KV cache 与 Prefix cache

KV cache 按固定大小的块分配，请求通过 block table 查找物理块，减少为每个请求预留最大上下文造成的浪费。Prefix cache 对完整输入块及其前缀计算哈希，相同前缀可复用已计算的 KV，仅执行剩余输入，主要降低重复长前缀的 prefill 开销和首 token 延迟。

CUDA 默认每块 256 token，共享前缀至少覆盖一个完整块才会命中。Qwen3 + DFlash/DSpark 还需恢复草稿上下文：CUDA/CoreX 使用有界 CPU 快照；310P 使用与目标共索引的分页草稿 KV，并排除可能被草稿查询覆盖的末块。310P 的前缀复用依赖 SplitFuse 算子，缺失时自动关闭。用 `--num-repeats 2` 与 `--no-prefix-cache` 可做缓存对照。

Qwen3.5 普通推理与 DFlash 默认支持前缀缓存，命中时还恢复完整块边界的 GDN conv/SSM 快照。主机快照默认预算 256 MiB，可用 `--recurrent-prefix-cache-mb` 调整；0 或 `--no-prefix-cache` 关闭。快照被淘汰时自动重算。实现与验证见 [Qwen3.5 前缀缓存](docs/QWEN35_PREFIX_CACHE.md)。

### Continuous batching 与 Chunked prefill

连续批处理在每个调度步接纳新请求、移除已完成请求。分块预填充将长 prompt 拆成多个 token 预算内的计算块，避免一次长 prefill 持续占用执行器。调度器在有工作时交替选择 prefill 与 decode；每次模型执行仍为同类批次。

`--max-num-batched-tokens` 控制每步 prefill 预算，`--max-num-partial-prefills` 控制未完成 prefill 的并发数，`--disable-chunked-prefill` 关闭分块。KV 不足时可抢占请求，释放状态并重新 prefill。

### Graph

Graph 预先捕获固定形状的算子调用，随后回放，减少 Python 调度和设备 kernel 启动开销。普通解码捕获一步；投机解码捕获包含起草、验证与接受处理的一整轮。不同 batch 大小使用预先捕获的桶，不足桶大小时填充。

CUDA/CoreX 使用其兼容的图接口，310P 使用 NPUGraph/ACLGraph。捕获失败会回退到 eager，`--eager` 可直接关闭图。Graph 主要加速 decode，也会占用额外设备内存。

### Speculative decoding

草稿模型先提出多个 token，目标模型一次前向验证，接受连续有效的提案并补充一个目标 token，以减少逐 token 的目标前向次数。

| 方法 | 起草方式 |
|---|---|
| EAGLE3 | 利用目标中间隐层，以链式前向逐个提出 token |
| DFlash | 用非因果块草稿前向并行预测多个位置 |
| DSpark | 复用块草稿骨干，由 vanilla Markov head 逐位置生成提案 |

三种方法当前都要求 TP=1。DSpark 使用固定草稿长度，尚未接入基于 confidence 的动态调度。平均接受长度 `E` 表示每轮提交的 token 数（含 bonus）；低接受率时，草稿开销可能超过收益。310P 投机图仅支持 greedy，随机采样请求回退到 eager spec。

算法说明：[EAGLE3](docs/EAGLE3.md)、[DFlash / DSpark](docs/DFlash.md)。

### 后端算子与 Ascend 310P 的 NZ 格式

CUDA 可设置 `FLASHENGINE_CUDA_ATTENTION=flashinfer` 使用与 vLLM 相同的 FlashInfer
公开算子库，默认仍为 `flash-attn`。普通 decode 支持 Graph、前缀缓存和分块 prefill；
FlashInfer 整轮投机图当前关闭，投机走 eager。安装、WSL 命令与回归说明见
[FlashInfer 适配](docs/FLASHINFER.md)。

CUDA 使用 FlashAttention；CoreX 使用 IXFormer。310P 使用厂商 FlashAttention、PagedAttention 与 SplitFuse，并调用融合 RMSNorm、残差加归一化、SwiGLU 和 RoPE 算子，减少中间张量与独立 kernel 调用。

310P 路径使用 fp16，将线性层及 LM head 的矩阵权重转换为 FRACTAL_NZ，并按 NZ 格式分配分页 KV。NZ 按硬件矩阵计算所需的分块布局存储数据，供对应算子直接读取；布局转换后数据精度仍为 fp16。310P 的 KV block 仅支持 64 或 128，且 `block_size × head_dim ≤ 16384`，当前仅 TP=1。布局由后端处理，模型文件无需手工转换。

详细算子与图路径见 [310P 算子说明](docs/ASCEND310P_OPS.md) 和 [310P 图规格](docs/ASCEND310P_GRAPH_SPEC.md)。

### OpenAI 兼容服务

`scripts.openai_server` 使用单个后台线程驱动引擎，HTTP 线程负责入队与发送输出，支持非流式响应和 SSE 流式响应。

| 功能 | 说明 |
|---|---|
| API | `/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/health`；每次请求仅支持 `n=1` |
| 生成控制 | `temperature`、`top_p`、`top_k`、`max_tokens`、文本 `stop`；默认 greedy；`--no-think` 关闭 thinking |
| 上下文管理 | 淘汰最旧历史、保留 system 与最近消息，为输出预留空间，输出上限按剩余上下文夹紧 |
| 统计 | 请求日志记录 TTFT、TPOT、TPS、排队、缓存与执行路径；`/metrics` 提供 Prometheus 指标 |
| 访问控制与预热 | `--api-key` 启用 Bearer 校验；启动预热减少首次真实请求的冷启动开销 |
| 自动续写 | 未传 `max_tokens` 时启用自动续写；必要时滚动保留 token 尾部重新 prefill，直到 EOS、停止或中止条件 |
| Web 控制台 | 同端口 `/` 提供多会话对话与性能面板；`/v1/stats` 提供统计；`--no-ui` 同时关闭这两项 |

兼容范围为常用文本生成接口。API key 不保护控制台静态页面与 `/metrics`，对外访问范围由部署环境控制。

## 性能参考

下表保留仓库已有的历史结果，未填数字的项尚无可引用的测量；当前工作区没有重新测量，不能据此比较两个版本的性能差异。来源及详细条件见 [历史测量记录](AGENTS.md)。

已填项的目标为 Qwen3-1.7B，使用 spec_bench coding 前 10 条、串行 batch=1、greedy、thinking-off；EAGLE3 为匹配的 AngelSlim 草稿，`k=3`。TPS 为含 prefill 的输出吞吐，CUDA 与 CoreX 属不同历史运行。TTFT 单位为 ms，TPS 单位为 token/s。

| 硬件 / 后端 | 执行方式 | TPS | TTFT | E |
|---|---|---:|---:|---:|
| RTX 4070 Laptop 8GB / CUDA | Eager decode | | | |
| RTX 4070 Laptop 8GB / CUDA | Graph decode | 66.2 | | |
| RTX 4070 Laptop 8GB / CUDA | Graph + EAGLE3 | 108.7 | | 2.095 |
| RTX 4070 Laptop 8GB / CUDA | Graph + DFlash | | | |
| RTX 4070 Laptop 8GB / CUDA | Graph + DSpark | | | |
| 天数智芯 32GB / CoreX | Eager decode | 8.1 | | |
| 天数智芯 32GB / CoreX | Graph decode | 76.5 | | |
| 天数智芯 32GB / CoreX | Graph + EAGLE3 | 104.1 | | |
| 天数智芯 32GB / CoreX | Graph + DFlash | | | |
| 天数智芯 32GB / CoreX | Graph + DSpark | | | |
| Ascend 310P | Eager decode | | | |
| Ascend 310P | Graph decode | | | |
| Ascend 310P | Graph + EAGLE3 | | | |
| Ascend 310P | Graph + DFlash | | | |
| Ascend 310P | Graph + DSpark | | | |

串行基准、进程内 batch 基准与 HTTP 并发测量分别记录，不混用吞吐和延迟。补测时应记录模型/草稿、数据集、输出长度、batch、缓存、thinking 和实际 Graph 路径，并保持预热与温度条件一致。

## 代码规模

按 2026-10-13 当前工作区统计物理行数，含注释和空行。按模块互斥分组；Graph、投机与缓存共享执行代码，不将同一段重复计为各功能的独立成本。

| 功能模块 | 统计范围 | Python 行数 |
|---|---|---:|
| 接口、配置与公共工具 | 包根目录、`llm_engine`、`observer`、目标注册表、`utils/` | 1062 |
| 调度、KV 分页与前缀缓存 | `scheduler`、`block_manager`、`sequence` | 542 |
| 模型执行、Graph 与投机轮编排 | `model_runner` | 1864 |
| 目标模型与通用算子 | `qwen3`、`layers/` | 1123 |
| 草稿模型与投机策略 | `eagle3`、`dflash`、`dspark`、`speculative/` | 1003 |
| 多后端适配 | `backends/`，含 310P 图执行器 | 2345 |
| HTTP 服务 | `scripts/openai_server.py` | 1318 |
| 基准与校验工具 | `scripts/` 中其余 Python 文件 | 1190 |
| **运行代码合计** | **`flashengine/` + `scripts/`** | **10447** |

其中引擎 7939 行、脚本 2508 行。服务实际加载的 `scripts/webui.html` 为 3661 行 HTML，备用页面 `webui-aero.html` 为 5555 行，均不计入 Python 合计。行数用于说明维护规模，不作为性能指标。
