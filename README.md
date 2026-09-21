# Nano-vLLM

基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的增强版：一个从零实现的轻量级大模型推理引擎。上游约 1200 行代码，本项目加入 EAGLE3、DFlash、CUDA Graph 与多硬件后端后约6800行。

在上游已有的前缀缓存、张量并行、`torch.compile` 加速、分块预填充基础上，本项目新增：

- **EAGLE3 投机解码**：小草稿模型先猜 `k` 个 token，主模型一次前向批量验证，一轮生成多个 token。
- **Qwen3.5 混合架构 + DFlash 块草稿投机**（当前分支 `qwen35-dflash`）：支持 Gated DeltaNet 线性注意力与门控全注意力混合的 Qwen3.5，以及单次非因果前向起草整块 token 的 DFlash。
- **Qwen3 + DFlash**：复用 Qwen3 主模型接入 DFlash 块草稿路径；CUDA/CoreX 使用稠密 draft KV，Ascend 310P 使用分页 NZ draft KV + 非因果 SplitFuse，并支持整轮图。

Ascend 310P 的 Graph、EAGLE3 与 DFlash 适配细节见
[ASCEND310P_GRAPH_SPEC.md](ASCEND310P_GRAPH_SPEC.md)。
- **CUDA Graph / NPUGraph 加速**：普通解码一步录一张图，投机解码一整轮录一张图；捕获失败自动退回 eager，只影响速度、不影响功能。
- **多硬件后端**：NVIDIA CUDA、天数智芯 CoreX、华为 Ascend 310P。
- **OpenAI 兼容服务**：流式输出、连续批处理、前缀缓存统计与实时 TPS 日志，服务端仅使用 Python 标准库。

> 开发约定、硬件限制与实测数据见 [AGENTS.md](AGENTS.md)。本文档以当前代码与当前分支为准。

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

Qwen3.5 + DFlash（当前仅 CUDA，`k` 自动取 draft block size - 1，通常为 15）：

```bash
python3 -m scripts.openai_server /path/to/Qwen3.5-2B \
    --spec-model /path/to/Qwen3.5-2B-Dflash \
    --backend cuda --max-model-len 2048 \
    --max-num-seqs 2 --port 8000 --no-think
```

常用行为：

- 默认 `max-num-seqs` 为 8；小显存卡可显式降低并发。
- 默认 thinking 交给 chat template（Qwen3 通常为开启）；`--no-think` 显式关闭。单请求可用 `chat_template_kwargs.enable_thinking` 覆盖。
- prompt 超过上下文上限时保留 system 与最近对话，从最旧历史开始淘汰，并为输出预留 `--reserve-output`（默认 512）token。
- 日志使用固定标签：`[HTTP ]`、`[REQ  ]`、`[DONE ]`、`[INFO ]`，异常为 `[WARN ]` / `[ERROR]`。

请求示例：

```bash
curl http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"Qwen3-1.7B","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

### 离线 benchmark

benchmark 使用 `question.jsonl` 串行测量吞吐与首字延迟，默认对齐 `vllm bench serve` 的 spec_bench 口径：seed shuffle、chat template、首条 prompt 预热和单墙钟吞吐。

```bash
# 普通解码
python3 -m scripts.benchmark /path/to/Qwen3-1.7B \
    --questions /path/to/spec_bench/question.jsonl \
    --max-model-len 2048 --max-num-seqs 8 --no-think

# EAGLE3 投机解码
python3 -m scripts.benchmark /path/to/Qwen3-1.7B \
    --questions /path/to/spec_bench/question.jsonl \
    --spec-model /path/to/Qwen3-1.7B-EAGLE3 --spec-num-tokens 3 \
    --max-model-len 2048 --max-num-seqs 8 --no-think

# Qwen3.5 + DFlash；k 自动派生为 draft block size - 1
python3 -m scripts.benchmark /path/to/Qwen3.5-2B \
    --spec-model /path/to/Qwen3.5-2B-Dflash \
    --max-model-len 2048 --max-num-seqs 2 --no-think
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
    --questions ../agent_prefix_bench/question.jsonl \
    --category agent --seed -1 --num-prompts 24 --no-think
```

`--spec-model` 时 benchmark 会额外汇总平均接受长度 `E`、token 接受率和逐位置接受率。历史性能锚点使用 `--no-think`；默认 thinking-on 与 vLLM 对齐，但不能直接和 thinking-off 的 `E` 比较。

### Qwen3.5 / DFlash 校验

`check_qwen35` 的引擎校验命令要分进程执行。WSL 下显存回收有延迟，同一进程连续创建 base 与 spec 引擎可能互相挤占 KV 预算。

```bash
python3 -m scripts.check_qwen35 ref  /path/to/Qwen3.5-2B --out /tmp/ref.json
python3 -m scripts.check_qwen35 base /path/to/Qwen3.5-2B --out /tmp/base.json
python3 -m scripts.check_qwen35 check /tmp/ref.json /tmp/base.json --label ref-vs-base

python3 -m scripts.check_qwen35 spec /path/to/Qwen3.5-2B \
    --draft /path/to/Qwen3.5-2B-Dflash \
    --out /tmp/spec.json --graph
python3 -m scripts.check_qwen35 check /tmp/base.json /tmp/spec.json \
    --label spec-vs-base
```

默认校验使用 eager；`--graph` 才启用图。bf16 下目标模型可能因 batch 形状和归约顺序出现 ULP 级翻转，因此 spec-vs-base 不以所有 prompt 位等作为唯一判据，需结合首个分歧位置与 `E` 一起判断。

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
llm.exit()
```

## 支持矩阵与硬件限制

| 目标模型 / 草稿模型 | 后端 | 主要限制 |
|---|---|---|
| Qwen3 / 无 | CUDA、CoreX、Ascend 310P | Ascend 仅 fp16、TP=1；其余限制见下文 |
| Qwen3 + EAGLE3 | CUDA、CoreX；Ascend 310P 支持 eager spec | spec 当前要求 TP=1 |
| Qwen3.5 + DFlash | CUDA | 仅 TP=1；强制关闭 prefix cache；禁 chunked prefill |
| Qwen3 + DFlash | CUDA、CoreX；Ascend 310P | 310P 使用独立分页 NZ 草稿 KV + 非因果 SplitFuse + 整轮 NPUGraph；TP=1，自动关闭 prefix cache |

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
```

Ascend 的 KV `block_size` 仅支持 64/128，且要求 `block_size * head_dim <= 128*128`；引擎会在必要时将配置覆盖为 128。DFlash 会自动关闭 prefix cache；普通解码和 EAGLE3 在 SplitFuse 可用时仍可启用。

CUDA 走 flash-attn paged kernel，`kvcache_block_size` 必须为 256 的倍数，默认值为 256；因此 CUDA 上小于 256 的 block 不会带来更细的 prefix cache 粒度，反而会触发 kernel 报错。空板卡建议使用 `--gpu-mem-util 0.85` 给 CUDA Graph 与采样峰值留余量。

## 目录结构

```
nanovllm/
├── llm.py                  # 用户入口：LLM 类
├── config.py               # 模型、后端、显存、spec 配置与能力门控
├── sampling_params.py      # temperature / top_p / top_k / max_tokens
├── engine/
│   ├── llm_engine.py       # 主循环：普通解码 / EAGLE3 / DFlash
│   ├── scheduler.py        # 调度、连续批处理、抢占与状态槽生命周期
│   ├── block_manager.py    # KV cache 分页与前缀缓存复用
│   ├── sequence.py         # 单请求状态
│   ├── model_runner.py     # 模型执行、图捕获与投机整轮编排
│   ├── npu_graph_runner.py # Ascend NPUGraph
│   └── aclgraph_attn.py    # Ascend 图内 paged attention 辅助路径
├── models/
│   ├── qwen3.py            # Qwen3 主模型
│   ├── qwen3_5.py          # Qwen3.5 混合架构主模型
│   ├── eagle3.py           # EAGLE3 草稿模型
│   └── dflash.py           # DFlash 块草稿模型
├── layers/                 # attention / linear / embedding / norm / RoPE / sampler
├── backends/               # cuda / corex / ascend310p 后端
└── utils/                  # 全局上下文与权重加载
scripts/
├── benchmark.py            # question.jsonl 串行 TPS / TTFT / spec 统计
├── check_qwen35.py         # Qwen3.5/DFlash 参考、base、spec 校验
└── openai_server.py        # OpenAI 兼容服务与实时日志
```

## 架构

### 分层总览

六层自上而下单向调用：上层只依赖下一层，调度逻辑不碰硬件细节，硬件差异收在后端层。

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
│ 模型层    Qwen3 / Qwen3.5
│           EAGLE3 / DFlash 草稿模型
├────────────────────────────────────────────────
│ 算子层    Attention / Linear / Embedding
│           RMSNorm / RoPE / Sampler / GDN
├────────────────────────────────────────────────
│ 后端层    CUDA ｜ CoreX ｜ Ascend 310P
│           attention kernel、KV 排布、图兼容
└────────────────────────────────────────────────
```

一次普通请求的流程：

1. `generate()` 将 prompt 分词并加入调度队列；
2. **预填充**：整段输入一次性送入模型，写入 KV cache 并采样第一个 token；
3. **解码循环**：把未完成请求组成 batch，每步前向并各产出一个 token；
4. 达到 `max_tokens` 或生成 EOS 后，token 拼回文本返回。

开启 EAGLE3 后，一轮变为「草稿链起草 → target 一次验证 → 接受最长匹配前缀 → 采样 bonus token」；开启 DFlash 后则是「上下文追加 → 非因果块草稿 → target 验证 → 接受前缀 + bonus」。提交与调度层保持同一套序列状态，因此被拒绝的草稿 token 不会进入 prefix cache。

### Qwen3.5 + DFlash

Qwen3.5 是 24 层混合模型：18 层 Gated DeltaNet 线性注意力与 6 层门控全注意力交错排列。当前实现使用纯 PyTorch GDN，文本模型从多模态 checkpoint 中加载 language model 部分。

DFlash 是 5 层、非因果的块草稿模型，checkpoint 不带 embedding 与 lm_head，与 target 共享词表和输出头。draft block size 通常为 16，因此每轮默认起草 `k=15` 个 token。Qwen3.5 的 GDN 状态按序列维护，不能做 block hash 级前缀复用，所以该组合会自动关闭 prefix cache。

Qwen3 + DFlash 使用同一套块草稿/verify 机制，但 Qwen3 的 aux 隐层是层输入口径，配置会自动将 DFlash 的 `target_layer_ids` 换算为目标层索引。

### EAGLE3 的流约定

EAGLE3 使用 shifted draft 流：位置 `p` 的 draft 行使用 `tok@p+1` 的 embedding 与 `aux@p` 条件，输出预测 `tok@p+2`。冷启动首轮使用 eager，之后在 carry 状态完整时进入图轮。draft KV、采样行与接受提交均按 `seq_id` 保存，batch 每轮变化不会破坏跨轮状态。

## Prefix cache 与服务行为

prefix cache 是 prefill 阶段的 block 级 hash 复用：命中块直接复用 KV，未命中部分继续计算；投机提交的 token 不参与 hash，避免被拒草稿污染。`--no-prefix-cache` 会连 hash 计算与命中查找一起关闭，适合做 TTFT / 吞吐 A/B。

- CUDA 默认 block size 256，因此短于 256 token 的共享前缀不命中属于预期；要观察命中，需准备至少一个完整块的固定 system/tools 前缀。
- Qwen3.5 强制关闭 prefix cache；Ascend 310P 的 Qwen3+DFlash 也会自动关闭，因为命中块不会重新生成草稿缓存所需的 aux 隐层。
- CUDA/CoreX 的 Qwen3 + DFlash 当前仍允许 prefix cache，但命中前缀不会重新计算 DFlash 所需的 aux 隐层；共享前缀负载可能使草稿预测退化，建议用 `--no-prefix-cache` 做对照。
- benchmark 的 `--num-repeats` 会让同一 prompt 相邻重复，第 2 次起用于观察热命中；server 日志中的 `cache` 字段展示请求级命中情况。

服务端会在引擎初始化阶段执行一次 `max_tokens=2` 的 dummy generate，提前触发主要 decode/spec kernel，减少首个真实请求的冷启动延迟。流式请求每步有新 token 即唤醒 HTTP 线程，不再等到请求完成后批量返回。

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
