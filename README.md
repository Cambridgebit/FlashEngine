# Nano-vLLM

基于 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 的增强版：一个从零实现的轻量级大模型推理引擎，上游约 1200 行代码，本项目加入 EAGLE3 投机解码后约 2400 行。

在上游已有的前缀缓存、张量并行、torch.compile 加速、分块预填充基础上，本项目新增：

- **EAGLE3 投机解码**：小草稿模型先猜 k 个 token，主模型一次前向批量验证，一轮生成多个 token（原理见「架构」一节）。
- **Qwen3.5 混合架构 + DFlash 块草稿投机**（分支 `qwen35-dflash`）：支持 Gated DeltaNet 线性注意力与门控全注意力混合的 Qwen3.5（纯 torch GDN 实现），DFlash 草稿模型单次非因果前向并行起草整块 16 token，主模型一次验证。
- **CUDA Graph 加速**：解码一步录一张图、投机解码一整轮录一张图；录制失败自动退回逐算子执行，只降速、不影响功能。
- **多硬件后端**：NVIDIA CUDA 与天数智芯 CoreX 均已实机验证，华为 Ascend 310P 代码就绪、待实机验证。
- **OpenAI 兼容服务**：流式输出 + 连续批处理，纯 Python 标准库实现，零新增依赖。

> 开发约定与实测数据明细见 [AGENTS.md](AGENTS.md)。

## 快速开始

安装（模型需使用本地目录，不支持 HuggingFace hub id）：

```bash
pip install -e .
pip install tqdm flash-attn --no-build-isolation   # 这两个依赖未写进 pyproject，需手动安装
```

在线服务（OpenAI 兼容 API，需从仓库根目录以 `-m` 方式启动）：

```bash
python3 -m scripts.openai_server /path/to/Qwen3-1.7B --port 8000 --max-model-len 2048

# 开启投机解码：追加 --spec-model 草稿模型目录
python3 -m scripts.openai_server /path/to/Qwen3-1.7B \
    --spec-model /path/to/Qwen3-1.7B-EAGLE3 --spec-num-tokens 3
```

> 默认最大并发数为 8（小显存卡实测更优），需要更高并发时显式传 `--max-num-seqs`。
> 服务端实时打印日志：`[req ]` 请求到达（token 数 / 采样参数 / 内容预览）、
> `[done]` 请求完成（TTFT / 墙钟 / 单请求 TPS）、`[tps ]` 每秒窗口吞吐。

离线 benchmark（spec_bench question.jsonl 串行基准，吞吐 / 首字延迟）：

```bash
python3 -m scripts.benchmark /path/to/Qwen3-1.7B --spec-model /path/to/Qwen3-1.7B-EAGLE3
```

Python API 离线推理：

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/Qwen3-1.7B",
          spec_model="/path/to/Qwen3-1.7B-EAGLE3",   # 可选：投机解码草稿模型
          spec_num_tokens=3)                          # 每轮猜测的 token 数 k
print(llm.generate(["你好"], SamplingParams(temperature=0, max_tokens=512))[0]["text"])
```

## 目录结构

```
nanovllm/
├── llm.py                  # 用户入口：LLM 类（generate 一键生成）
├── config.py               # 引擎配置项（模型路径、并发数、显存占比等）
├── sampling_params.py      # 采样参数（temperature / top_p / top_k / max_tokens）
├── engine/                 # 引擎核心
│   ├── llm_engine.py       # 主循环：普通解码 / 投机解码两种步进
│   ├── scheduler.py        # 调度器：决定每步跑哪些请求，显存不足时抢占
│   ├── block_manager.py    # KV cache 显存分页分配 + 前缀缓存复用
│   ├── sequence.py         # 单个请求（序列）的状态管理
│   └── model_runner.py     # 执行总指挥：组装输入、调用模型、管理 CUDA Graph 与投机流程
├── models/
│   ├── qwen3.py            # Qwen3 主模型
│   ├── qwen3_5.py          # Qwen3.5 混合架构主模型（GDN + 门控注意力）
│   ├── eagle3.py           # EAGLE3 草稿模型
│   └── dflash.py           # DFlash 块草稿模型
├── layers/                 # 基础算子：attention / linear / embedding / 归一化 / RoPE / 采样器
├── backends/               # 硬件后端：cuda / corex / ascend 三套实现
└── utils/                  # 工具：全局上下文、权重加载
scripts/
├── benchmark.py            # 基准测试：question.jsonl 串行吞吐 / 首字延迟
└── openai_server.py        # OpenAI 兼容服务（请求到达 / 完成 / 实时 TPS 日志）
```

## 架构

### 分层总览

六层自上而下单向调用：上层只依赖下一层，调度逻辑不碰硬件细节，硬件差异全部收在最底层。

```
┌────────────────────────────────────────────────
│ 接口层    LLM
│           用户入口：generate() 一键生成
├────────────────────────────────────────────────
│ 调度层    LLMEngine（主循环）
│           ├─ Scheduler      挑选请求、显存不足时抢占
│           └─ BlockManager   KV cache 分页管理 + 前缀复用
├────────────────────────────────────────────────
│ 执行层    ModelRunner
│           组装输入 → 调用模型 → 写回 KV cache → 读回结果
│           （CUDA Graph 回放、投机解码整轮编排都在这里）
├────────────────────────────────────────────────
│ 模型层    Qwen3 主模型 ＋ EAGLE3 草稿模型（共享词表）
├────────────────────────────────────────────────
│ 算子层    Attention / Linear / Embedding
│           RMSNorm / RoPE / Sampler
├────────────────────────────────────────────────
│ 后端层    CUDA ｜ CoreX ｜ Ascend 310P
│           （attention kernel、KV cache 排布、
│             图捕获兼容——硬件差异全部收在这层）
└────────────────────────────────────────────────
```

各层职责：

- **接口层**：用户入口。把文本分词成 token 交给调度器，生成结束后把 token 拼回文本。
- **调度层**：每一步由调度器决定做什么——新请求做「预填充」（整段输入一次算完），进行中的请求做「解码」（每步产出 token）。显存不足时暂停部分请求，优先保正在生成的。KV cache 显存按固定大小的块分页管理，相同前缀的请求直接复用已有块，省显存也省重复计算。
- **执行层**：引擎里最重的一层。把一批请求组装成模型输入（token id、每个 token 在 KV cache 里的位置、注意力元数据），调用模型，写回 KV cache 并读回采样结果。CUDA Graph 的录制与回放、投机解码一轮的完整流程都由它编排。
- **模型层**：Qwen3 主模型与 EAGLE3 草稿模型的网络结构定义，二者共享词表。
- **算子层**：线性层内置张量并行（多卡时各算一部分再汇总）；各算子通过全局上下文获取当前步的位置等信息，不污染模型函数签名。
- **后端层**：同一套引擎代码跑在不同 GPU 上，只有这层不同——KV cache 的显存排布、attention 用的厂商 kernel、图捕获的兼容处理。CUDA 用 flash-attn + triton，CoreX 用 ixformer，Ascend 310P 用 torch_npu。新增硬件只需实现一个后端类。

### 一次请求的流程

1. `generate()` 把 prompt 分词，加入调度队列；
2. **预填充**：整段输入一次性送入模型，产出第一个 token；
3. **解码循环**：之后每步把所有未完成的请求组成一个 batch，前向一次、各产出一个 token（开投机则一轮产出多个）；
4. 达到 `max_tokens` 或生成出结束符，token 拼回文本返回。

## 性能

测试口径：spec_bench coding 数据集前 10 条，单并发串行，temperature=0，投机深度 k=3，吞吐含预填充。E = 平均每轮投机收下的 token 数（越大说明草稿猜得越准）。

**RTX 4070 Laptop 8GB（WSL2）**

| 配置 | 吞吐（tokens/s） | E |
|---|---|---|
| 本引擎 不开投机 | 66.2 | — |
| 本引擎 投机 k=3 | **108.7** | 2.095 |
| vLLM 0.26 不开投机 | 69.5 | — |
| vLLM 0.26 投机 k=3 | 125.6 | — |

**天数智芯 CoreX（32GB 板卡）**

| 配置 | 吞吐（tokens/s） |
|---|---|
| 逐算子执行（无 CUDA Graph） | 8.1 |
| CUDA Graph | 76.5 |
| **CUDA Graph + 投机 k=3** | **104.1** |
