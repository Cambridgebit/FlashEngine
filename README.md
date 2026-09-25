# Nano-vLLM

基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的增强版：一个从零实现的轻量级大模型推理引擎。上游约 1200 行代码（本项目分叉点实测 1450 行）；当前 `nanovllm/` 与 `scripts/` 合计约 9617 行 Python（另有内置控制台的 `scripts/webui.html` 1321 行），演化过程见[代码演化路线](#代码演化路线)。

在上游已有的前缀缓存和张量并行基础上，本项目新增：

- **EAGLE3 投机解码**：小草稿模型先猜 `k` 个 token，主模型一次前向批量验证，一轮生成多个 token。
- **Qwen3 + DFlash**：复用 Qwen3 主模型接入 DFlash 块草稿路径；CUDA/CoreX 使用稠密 draft KV，Ascend 310P 使用分页 NZ draft KV + 非因果 SplitFuse，并支持整轮图。
- **Qwen3 + DSpark**：读取 DeepSeek Qwen3 DSpark 格式（vanilla Markov head）；块骨干并行、Markov head 逐位置起草、target 批量验证。当前为固定 `k`，confidence head 权重加载但尚未参与自适应验证长度选择。
- **CUDA Graph / NPUGraph 加速**：普通解码一步录一张图，EAGLE3、DFlash 与 DSpark 投机解码一整轮录一张图；捕获失败自动退回 eager。
- **多硬件后端**：NVIDIA CUDA、天数智芯 CoreX、华为 Ascend 310P。
- **连续批处理与分块预填充**：prefill/decode 公平轮转，长 prompt 可跨多个调度步，并支持运行时持续接纳新请求。
- **OpenAI 兼容服务**：流式输出、前缀缓存统计与实时 TPS 日志，服务端仅使用 Python 标准库。
- **内置 Web 控制台**：服务启动后访问 `http://<host>:<port>/` 即得对话界面（多会话、流式、Markdown）与实时性能面板（TPS / TTFT / KV 占用 / prefix 命中 / 投机 E / 图 replay 曲线 + 最近请求明细），前端是自包含单文件、零外部依赖。

> 开发约定、硬件限制与实测数据见 [AGENTS.md](AGENTS.md)。本文档以当前代码与当前分支为准。

专题文档：[EAGLE3 算法与实现](docs/EAGLE3.md)、[DFlash / DSpark 块草稿投机](docs/DFlash.md)、
[Ascend 310P 图与投机路径规格](docs/ASCEND310P_GRAPH_SPEC.md)、[Ascend 310P 算子清单](docs/ASCEND310P_OPS.md)。

## 代码演化路线

| 阶段 | 内容 | 锚点 | 引擎侧 | scripts | 合计 |
|---|---|---|---:|---:|---:|
| ① 基线 | 上游 nano-vllm | `bb823b3` | 1385 | — | **1450** |
| ② 第一批 | 多后端抽层 + **CoreX** 后端适配 + **EAGLE3** 投机推理 | `88b5d77` V1.4 | 2926 | 951 | **3877** |
| ③ 第二版 | + **310P** 后端适配与 310P EAGLE3 | `fc26e87` V1.7 | 3438 | 1565 | **5003** |
| ④ 第三批 | + 块草稿投机的实验分支 | `64f752a` V2.1.0 | 5744 | 1338 | **7082** |
| ⑤ 当前 main | + target/spec 双注册表、**DSpark**、连续批处理与分块预填充、310P 图路径模块化、内置 Web 控制台与并发压测脚本（Qwen3.5 hybrid 移出 main，见分支 `qwen35-dflash`） | working tree | **7425** | **2192** | **9617** |

非引擎侧 `scripts/` 当前 2192 行：`openai_server.py` 1374、`bench_concurrent.py` 411（多并发/连续批处理 HTTP 压测，不需 GPU）、`benchmark.py` 407（前端控制台 `webui.html` 另计 1321 行 HTML）。

## 快速开始

模型必须是本地目录，不支持直接传 HuggingFace hub id。所有脚本都要从仓库根目录以模块方式运行，使用 `python scripts/X.py` 会遇到包路径问题。

安装基础依赖：

```bash
pip install -e .
pip install tqdm
```

CUDA 环境还需要：

```bash
pip install flash-attn --no-build-isolation
```

CoreX 需要目标机器提供 `ixformer`；Ascend 310P 需要专用的 CANN 与 `torch-npu` 环境（实测组合为 `torch-npu==2.10.0.post4` / CANN 9.1）。

### OpenAI 兼容服务

Qwen3 + EAGLE3：

```bash
python3 -m scripts.openai_server /path/to/Qwen3-1.7B \
    --spec-model /path/to/Qwen3-1.7B-EAGLE3 \
    --spec-num-tokens 3 \
    --max-model-len 2048 --max-num-seqs 8 \
    --port 8000 --no-think
```

普通 Qwen3 服务：

```bash
python3 -m scripts.openai_server /path/to/Qwen3-1.7B \
    --max-model-len 2048 --max-num-seqs 8 --port 8000
```

常用行为：

- 默认 `max-num-seqs` 为 8；小显存卡可显式降低并发。
- `--max-num-batched-tokens` 控制单次 prefill token 预算；预算小于 prompt 时自动分块。`--disable-chunked-prefill` 可关闭该行为，`--max-num-partial-prefills` 限制同时处于未完成 prefill 的请求数（默认 1，其余请求按 FIFO 等待）。
- 连续批处理由后台引擎线程统一 `step()` 驱动：prefill 与 decode 在同一步里公平轮转，运行中可持续接纳新请求，KV 不足时按 LIFO 抢占并重跑 prefill。
- 请求没有显式传 `temperature` 时默认使用 greedy（`temperature=0`），与离线 benchmark
  保持同一执行路径；客户端显式传入随机采样参数时仍严格保留其语义。Ascend 310P 的
  speculative Graph 当前只支持 greedy，随机采样请求会在日志中显示
  `path=eager_spec(sampling)` 并给出一次告警。
- 默认 thinking 交给 chat template（Qwen3 通常为开启）；`--no-think` 显式关闭。单请求可用 `chat_template_kwargs.enable_thinking` 覆盖。
- prompt 超过上下文上限时保留 system 与最近对话，从最旧历史开始淘汰，并为输出预留 `--reserve-output`（默认 512）token；`/v1/completions` 超长时保留尾部（最新内容）。只剩一条消息仍放不下时，裁的是这条消息**文本的 token 尾部**并重新套模板（不切渲染后的 id 流，否则 ChatML 结构被破坏会让模型退化成重复 token），因此**对话可以无限续下去**，只是更早的轮次会滑出窗口。不变式：`_chat_prompt_ids_fit` 要么返回不超过预算的 prompt，要么返回 400 并提示调大窗口——不会静默放行超长 prompt。引擎侧另有上下文硬停：任何序列（含投机一轮的 `k+1` 个 token）都不会越过 `max_model_len` 继续写 KV，到顶即以 `finish_reason=length` 结束，并在 `[DONE ]`/`[WARN ]` 里给出 `ctx` 与 `preempt` 供判断是否需要调大窗口或降低并发。
- 端点为 `/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/health`、`/metrics`、`/v1/stats`（均有去掉 `/v1` 前缀的别名）；`--host` 默认 `0.0.0.0`，传 `--api-key` 后按 `Authorization: Bearer` 校验。`n` 仅支持 1，`stop` 做文本级截断并中止生成，`max_tokens` 未传表示生成到上下文顶。
- **Web 控制台**挂在同端口的 `/`（另接受 `/index.html`、`/ui`），页面本身不鉴权、其中所有 `/v1/*` 调用仍受 `--api-key` 约束（首次 401 时界面会提示填一次，保存在浏览器本地）。`--no-ui` 同时关闭页面与 `/v1/stats`，`/metrics` 不受影响。性能曲线由前端每秒轮询 `/v1/stats` 并对累计计数器做差得到——服务端不额外起采样线程，避免与 `[INFO ]` 的窗口排他读取冲突。
- 日志使用固定标签：`[HTTP ]`、`[REQ  ]`、`[DONE ]`、`[INFO ]`，异常为 `[WARN ]` / `[ERROR]`。
  `[DONE ]` 包含 TTFT、TPOT、TPS、queue、E2E、cache 与实际执行路径；`[INFO ]` 默认每
  10 秒打印 vLLM 风格的 prompt/generation throughput、running/waiting、KV cache usage、
  prefix hit rate、Graph replay rate 和 speculative E，可用 `--stats-interval 0` 关闭。
  控制台每秒轮询的 `GET /v1/stats` 不逐条打印，只累计到 `[INFO ]` 行末尾的 `ui_polls=N`。

请求示例：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3-1.7B","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

### 离线 benchmark

benchmark 以 `question.jsonl` 串行测量吞吐与首字延迟，默认对齐 `vllm bench serve` 的 spec_bench 口径：seed shuffle、chat template、首条 prompt 预热和单墙钟吞吐。仓库内已附三份数据集（`bench_data/`，schema 相同）：`question.jsonl`（spec_bench 英文）、`Zh_question.jsonl`（中文 coding）、`agent_question.jsonl`（Agent 工作流共享前缀）。注意 `--questions` 的默认值是 `../question.jsonl`（历史工作区布局），使用仓库内数据集时显式传路径。

```bash
# 普通解码
python3 -m scripts.benchmark /path/to/Qwen3-1.7B \
    --questions bench_data/question.jsonl \
    --max-model-len 2048 --max-num-seqs 8 --no-think

# EAGLE3 投机解码
python3 -m scripts.benchmark /path/to/Qwen3-1.7B \
    --questions bench_data/question.jsonl \
    --spec-model /path/to/Qwen3-1.7B-EAGLE3 --spec-num-tokens 3 \
    --max-model-len 2048 --max-num-seqs 8 --no-think

# DFlash 块草稿（k 缺省自动取 draft block_size-1，CLI 的 3/4 视为未指定）
python3 -m scripts.benchmark /path/to/Qwen3-8B \
    --spec-model /path/to/Qwen3-8B-DFlash \
    --max-model-len 2048 --max-num-seqs 1 --no-think --max-tokens 64

# DSpark（DeepSeek vanilla Markov head 格式，固定 k）
python3 -m scripts.benchmark /path/to/Qwen3-8B \
    --spec-model /path/to/dspark_qwen3_8b_block7 \
    --max-model-len 2048 --max-num-seqs 1 --no-think --max-tokens 64
```

常用对照：

```bash
# 关闭 CUDA Graph，适合排查图捕获问题
python3 -m scripts.benchmark /path/to/model --eager

# 前缀缓存冷/热 A/B；重复请求的第 2 次起应命中共享前缀
python3 -m scripts.benchmark /path/to/model --num-repeats 2
python3 -m scripts.benchmark /path/to/model --num-repeats 2 --no-prefix-cache

# Agent 工作流共享前缀数据集；-1 保持文件顺序，便于组内相邻复用
python3 -m scripts.benchmark /path/to/model \
    --questions bench_data/agent_question.jsonl \
    --category agent --seed -1 --num-prompts 24 --no-think
```

`--spec-model` 时 benchmark 会额外汇总平均接受长度 `E`、token 接受率和逐位置接受率。历史性能锚点使用 `--no-think`；默认 thinking-on 与 vLLM 对齐，但不能直接和 thinking-off 的 `E` 比较。

结果块分三段：`Benchmark Result`（吞吐、Mean/Median/P99 的 TTFT / TPOT / E2EL）、`Engine Details`（输入输出长度分布、Request-wall 吞吐、host 开销、稳态 decode 吞吐）、以及按配置出现的 `Decode/Spec graph runtime`（捕获/回放/回退/命中桶）、`Prefix cache`（命中率与块计数）、`Draft prefix snapshots`（块草稿快照的保存/恢复/淘汰与主机内存）、`Acceptance length (E)`。

多并发 / 连续批处理压测（打已跑起来的 server，纯 stdlib、不需要 GPU）：

```bash
python3 -m scripts.openai_server /path/to/Qwen3-1.7B --port 8000 --max-num-seqs 8
python3 -m scripts.bench_concurrent http://127.0.0.1:8000 \
    --concurrency 1,4,8 --num-prompts 16 --max-tokens 128 --no-think

# 并发下的前缀缓存：--shared-prefix 给每条请求前置约 N token 的固定 system 文本
python3 -m scripts.bench_concurrent http://127.0.0.1:8000 \
    --concurrency 8 --num-prompts 24 --shared-prefix 1200 --no-think
```

口径与串行 benchmark **不可同表比较**：同批请求用 Barrier 同步发车（不排队头消耗），聚合吞吐 = Σ输出 token / 该档墙钟；判断有没有真正进批看 `peak running` 是否贴住 server 的 `max_num_seqs`，并配合 `preempt` 与 `waiting` 峰值。服务端读数全部来自 `/v1/stats` 差值，不会刷 server 的 `[HTTP ]` 日志。

### Python API

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/path/to/Qwen3-1.7B",
    spec_model="/path/to/Qwen3-1.7B-EAGLE3",  # 可选
    spec_num_tokens=3,
    max_model_len=2048,
    max_num_seqs=8,
)
outputs = llm.generate(
    ["你好"],
    SamplingParams(temperature=0, max_tokens=128),
)
print(outputs[0]["text"])
llm.close()  # terminal cleanup; exit() remains as a backward-compatible alias
```

## 支持矩阵与硬件限制

| 目标模型 / 草稿模型 | 后端 | 主要限制 |
|---|---|---|
| Qwen3 / 无 | CUDA、CoreX、Ascend 310P | Ascend 仅 fp16、TP=1；其余限制见下文 |
| Qwen3 + EAGLE3 | CUDA、CoreX、Ascend 310P | spec 当前要求 TP=1；图捕获失败自动回退 eager spec |
| Qwen3 + DFlash | CUDA、CoreX、Ascend 310P | TP=1；CUDA/CoreX 用有界 CPU 草稿-KV 快照恢复命中块；310P 用共索引分页 NZ 草稿 KV + SplitFuse，末块不缓存 |
| Qwen3 + DSpark | CUDA、CoreX、Ascend 310P（待实机验收） | DeepSeek 官方 vanilla/full-vocab 格式；固定 `k`，无动态 confidence 调度；前缀恢复策略与 DFlash 相同 |

后端选择顺序为 `backend=` 参数、`NANOVLLM_BACKEND` 环境变量、自动探测。可选值为 `auto`、`cuda`、`corex`、`ascend310p`。

Ascend 310P 的推荐起步命令：

```bash
python3 -m scripts.benchmark /data/Qwen3-0.6B \
    --backend ascend310p --gpu-mem-util 0.5 \
    --max-model-len 512 --max-num-seqs 1 \
    --warmup 1 --num-prompts 3 --max-tokens 64 \
    --no-prefix-cache

# Qwen3 + DFlash（draft block=16 时自动使用 k=15）
python3 -m scripts.benchmark /data/Qwen3-8B \
    --backend ascend310p --spec-model /data/Qwen3-8B-DFlash \
    --gpu-mem-util 0.80 --max-model-len 2048 --max-num-seqs 1 \
    --no-think --max-tokens 64

# DeepSeek Qwen3-8B DSpark（draft block=7 时默认 k=7；先在 CUDA/CoreX 验收，
# 再在 310P 做 fp16/NZ/SplitFuse/NPUGraph 对照）
python3 -m scripts.benchmark /data/Qwen3-8B \
    --spec-model /data/dspark_qwen3_8b_block7 \
    --max-model-len 2048 --max-num-seqs 1 --no-think --max-tokens 64
```

`--block-size` / `kvcache_block_size` 是 **target KV block** 的 token 数，不是 DFlash/DSpark 一轮草稿长度。配置层要求 ≥16 且为 2 的幂；后端范围如下：

| 后端 | 可配置范围与限制 | 当前验证状态 |
|---|---|---|
| CUDA | 256、512、1024……（2 的幂且为 256 的倍数；FlashAttention paged KV 的约束） | 默认 256 已使用；更大值需按显存/内核实测 |
| CoreX | 配置层允许 16、32、64、128、256……（2 的幂） | 默认 256 已使用；IXFormer 对其余值的接受范围尚需实机确认 |
| Ascend 310P | 仅 64 或 128，且 `block_size × head_dim ≤ 16384` | 256 等其他值会被自动调成 128；DFlash/DSpark 的 query rows 还须 ≤ block size |

块越小，前缀复用粒度越细，但 block table/调度开销增大；块越大，至少要有一个完整共享块才命中。空板卡建议使用 `--gpu-mem-util 0.85` 给图捕获与采样峰值留余量。

## 目录结构

```
nanovllm/
├── llm.py                  # 用户入口：LLM 类
├── config.py               # 模型、后端、显存、spec 配置与能力门控
├── sampling_params.py      # temperature / top_p / top_k / max_tokens
├── engine/
│   ├── llm_engine.py       # 主循环：普通解码 / 投机轮、启动摘要与 close()
│   ├── scheduler.py        # 公平轮转的连续批处理、分块预填充、抢占
│   ├── block_manager.py    # KV cache 分页与前缀缓存复用
│   ├── sequence.py         # 单请求状态
│   ├── model_runner.py     # 模型执行、图捕获与投机整轮编排
├── models/
│   ├── qwen3.py            # Qwen3 主模型
│   ├── registry.py         # target 模型适配器注册表
│   ├── eagle3.py           # EAGLE3 草稿模型
│   ├── dflash.py           # DFlash 块草稿模型
│   └── dspark.py           # DSpark Markov 草稿模型
├── speculative/            # spec 方法识别、校验与 draft 构造注册表
├── layers/                 # attention / linear / embedding / norm / RoPE / sampler
├── backends/               # 算子与 Graph executor：cuda / corex / ascend
│   ├── base.py             # 后端能力位与算子契约
│   ├── graph.py            # 后端中立的 decode 图执行器接口
│   └── ascend310p/         # NPUGraph 执行器与 ACLGraph attention
└── utils/                  # 全局上下文与权重加载
scripts/
├── bench_concurrent.py     # 多并发 / 连续批处理 HTTP 压测（纯 stdlib，不需 GPU）
├── benchmark.py            # question.jsonl 串行 TPS / TTFT / spec 统计
├── openai_server.py        # OpenAI 兼容服务与实时日志
└── webui.html              # 内置 Web 控制台（对话 + 性能面板，自包含单文件）
bench_data/
├── question.jsonl          # spec_bench 英文（coding/writing/…）
├── Zh_question.jsonl       # 中文 coding
└── agent_question.jsonl    # Agent 工作流共享前缀
docs/
├── EAGLE3.md               # EAGLE3 算法与实现说明
├── DFlash.md               # DFlash / DSpark 块草稿与多后端说明
├── ASCEND310P_GRAPH_SPEC.md # Ascend 310P 图路径技术规格
└── ASCEND310P_OPS.md       # Ascend 310P 算子清单（与 vLLM-Ascend 对照）
```

## 架构

### 分层总览

六层自上而下单向调用：上层只依赖下一层，调度逻辑不碰硬件细节，硬件差异收在后端层。目标模型与投机方法各自经注册表解析（`models/registry.py`、`speculative/registry.py`），`ModelRunner` 只消费解析出的适配器与统一的提交契约，不按名字硬判断具体类。

```
┌────────────────────────────────────────────────
│ 接口层    LLM
│           generate() / OpenAI-compatible server
├────────────────────────────────────────────────
│ 调度层    LLMEngine
│           Scheduler + BlockManager
│           连续批处理、KV 分页、前缀复用、抢占
├────────────────────────────────────────────────
│ 执行层    ModelRunner
│           普通解码、CUDA/NPUGraph、spec 整轮
├────────────────────────────────────────────────
│ 模型层    TargetModelAdapter / Qwen3
│           EAGLE3 / DFlash / DSpark 草稿模型
├────────────────────────────────────────────────
│ 算子层    Attention / Linear / Embedding
│           RMSNorm / RoPE / Sampler
├────────────────────────────────────────────────
│ 后端层    CUDA ｜ CoreX ｜ Ascend 310P
│           attention kernel、KV 排布、图兼容
└────────────────────────────────────────────────
```

一次普通请求的流程：

1. `generate()` 将 prompt 分词并加入调度队列；
2. **预填充**：在 token 预算内批量处理 prompt；长 prompt 跨步分块并持续写入 KV cache；
3. **解码循环**：把未完成请求组成 batch，每步前向并各产出一个 token；
4. 达到 `max_tokens` 或生成 EOS 后，token 拼回文本返回。

开启 EAGLE3 后，一轮变为「草稿链起草 → target 一次验证 → 接受最长匹配前缀 → 采样 bonus token」；开启 DFlash/DSpark 后则是「上下文追加 → 块草稿 → target 验证 → 接受前缀 + bonus」。提交与调度层保持同一套序列状态，因此被拒绝的草稿 token 不会进入 prefix cache。

Qwen3 + DFlash/DSpark 使用同一套块草稿/verify 机制，差别只在草稿：DFlash 一次非因果前向并行出 `k` 个提案，DSpark 在同样的并行骨干上串接 vanilla Markov head 逐位置加偏置。两者的 aux 层描述都是 0-based 目标层**输出**，而 Qwen3 的钩子取层**输入**，配置会自动 `+1` 换算。

### EAGLE3 的流约定

EAGLE3 使用 shifted draft 流：位置 `p` 的 draft 行使用 `tok@p+1` 的 embedding 与 `aux@p` 条件，输出预测 `tok@p+2`。冷启动首轮使用 eager，之后在 carry 状态完整时进入图轮。draft KV、采样行与接受提交均按 `seq_id` 保存，batch 每轮变化不会破坏跨轮状态。

## Prefix cache 与服务行为

prefix cache 是 prefill 阶段的 block 级 hash 复用：命中块直接复用 KV，未命中部分继续计算；投机提交的 token 不参与 hash，避免被拒草稿污染。`--no-prefix-cache` 会连 hash 计算与命中查找一起关闭，适合做 TTFT / 吞吐 A/B。

- CUDA 默认 block size 256，因此短于 256 token 的共享前缀不命中属于预期；要观察命中，需准备至少一个完整块的固定 system/tools 前缀。
- EAGLE3 复用 target/draft 分页块，末段 prefill 重建跨轮引导状态。Qwen3+DFlash/DSpark 的 CUDA/CoreX 路径会把每个可缓存块的草稿投影 KV 快照到有界主机内存（上限 1 GiB），命中后恢复；快照被淘汰则 target 也视作未命中并正常 prefill。该路径在冷 prefill 增加 D2H 拷贝，需用 `--num-repeats` 对照总收益。
- 310P 的块草稿走独立分页 NZ 草稿 KV、与 target 共用物理块索引；被块查询覆盖的 prompt 末块始终不加入可复用哈希，避免污染后续命中。前提是运行时具备 SplitFuse；否则 prefix cache 仍禁用。
- benchmark 的 `--num-repeats` 会让同一 prompt 相邻重复，第 2 次起用于观察热命中；server 日志中的 `cache` 字段展示请求级命中情况。

服务端会在引擎初始化阶段执行一次 `max_tokens=2` 的 dummy generate，提前触发主要 decode/spec kernel，减少首个真实请求的冷启动延迟。流式请求每步有新 token 即唤醒 HTTP 线程，不再等到请求完成后批量返回。

服务端 `GET /metrics`（或 `/v1/metrics`）提供 Prometheus text 格式的累计 token/请求/步数/耗时、TTFT sum/count、运行/排队请求、KV 使用、prefix 命中块、草稿快照保存/恢复/淘汰以及图回放/回退计数。它读取 host 计数，不触发设备同步。默认监听地址为 `0.0.0.0`，需要在网络边界限制该端点的访问。

## 多卡状态

普通 Qwen3 保留原有的进程内 tensor parallel 路径（列并行/行并行与 all-reduce），但本轮没有扩展其硬件验收范围。投机解码仍明确限制为 TP=1，Ascend 310P 也明确限制为 TP=1；data parallel 与跨后端统一的多卡进程管理尚未加入。后续实现时应把 process group、rank/device 映射和 collective 能力放入独立 parallel runtime，并让 target/spec adapter 分别声明可切分的权重与跨 rank 状态，避免把 DP/TP 分支重新塞进 `ModelRunner`。

## 性能参考

以下数据来自 RTX 4070 Laptop 8GB / WSL2，spec_bench coding 前 10 条，串行、temperature=0、k=3、吞吐含 prefill。具体数值会受显存占用、温度和 thinking 口径影响。

| 配置 | 吞吐（tokens/s） | E |
|---|---:|---:|
| 本引擎 decode | 66.2 | — |
| 本引擎 EAGLE3 k=3 | **108.7** | 2.095 |
| vLLM 0.26 decode | 69.5 | — |
| vLLM 0.26 EAGLE3 k=3 | 125.6 | — |

天数智芯 CoreX 32GB 实测：

| 配置 | 吞吐（tokens/s） |
|---|---:|
| eager decode | 8.1 |
| Graph decode | 76.5 |
| Graph + EAGLE3 k=3 | **104.1** |

对比时必须保持 thinking 状态一致：同一数据集、k=3、temperature=0 下，thinking-on 的 E 可能只有约 1.7--1.8，而历史 `E≈2.1` 锚点使用 `--no-think`。低接受率问答负载上 spec 可能负收益；是否开启应结合 benchmark 输出的 `E`、TTFT 与 TPS 判断。

同一机器冷态与热态的 TPS 可能相差很大。长时间高负载后如果出现 GPU thermal slowdown，应先冷却并确认频率，再判断代码回归。
