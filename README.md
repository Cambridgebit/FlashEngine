# FlashEngine V1.0

FlashEngine 是一个从零实现的大模型推理引擎，面向研究与学习：完整覆盖请求调度、KV cache 管理、模型执行与硬件后端适配，同时提供 Python API 与 OpenAI 兼容的 HTTP 服务。

当前支持 Qwen3 / Qwen3 MoE / Qwen3.5（dense 与 MoE）文本模型，运行在 NVIDIA CUDA、天数智芯 CoreX、华为 Ascend 310P 三种后端上，并可选 EAGLE3、DFlash、DSpark 三种投机解码方法与 W8A8 量化推理。

## 设计架构

```text
接入层    Python API (LLM)  ·  OpenAI 兼容 HTTP 服务  ·  统一命令行入口
              ↓
调度层    LLMEngine → Scheduler + BlockManager
              ↓
执行层    ModelRunner  ·  投机轮编排  ·  Graph 捕获与回放
              ↓
模型层    目标模型 / 草稿模型 → 通用算子层 (Attention · Linear · Norm · RoPE · Sampler)
              ↓
后端层    CUDA  ·  CoreX  ·  Ascend 310P（设备、dtype、KV 布局、算子、图能力）
```

四个设计要点：

**统一入口，一套配置。** `RuntimeSettings` 与 `create_engine` 是唯一的装配入口，命令行、JSON 配置、HTTP 服务与离线生成共用同一份配置模型。未知配置项直接报错，不做静默忽略；配置装配只依赖标准库，因此在没有设备运行时的机器上也能读取与校验。

**注册表解耦，扩展不改主循环。** 目标模型与投机方法各有一张注册表：新增模型只需注册一个 adapter 与模型实现，新增投机方法只需注册一个 spec adapter 与草稿实现，`ModelRunner` 与调度器无需改动。

**能力位而非后端名。** 后端以能力声明自身行为（dtype 约束、KV 布局与形状、slot 映射类型、是否支持投机与投机图、是否支持前缀缓存、是否可接管 decode 图、算子与图预热参数等），引擎只消费能力位。新增后端不需要在主流程里写后端名判断。

**图与精度可验证。** Graph 预先捕获固定形状的算子调用以消除调度与 kernel 启动开销，捕获失败按桶或整体优雅回退 eager。310P 上另有图内外数值守卫：捕获时与真实输入时比对 logits，异常即判失效回退，可用环境变量关闭。

调度层采用公平轮转的连续批处理与分块预填充：每一步构建同质批次（prefill 或 decode），长 prompt 按 token 预算切开，KV 不足时按 LIFO 抢占并重新 prefill。执行层在普通解码时每步提交一个 token，投机解码时一轮起草、一次验证、多 token 提交。

## 支持的功能

| 类别 | 支持范围 |
|---|---|
| 目标模型 | Qwen3 dense、Qwen3 MoE、Qwen3.5 dense、Qwen3.5 MoE（文本，无视觉塔与 MTP） |
| 后端 | `cuda`、`corex`、`ascend310p`，可显式指定或自动探测 |
| 投机解码 | EAGLE3（链式，利用目标中间隐层）、DFlash（非因果块草稿并行提案）、DSpark（DFlash 骨干 + Markov head） |
| 量化 | W8A8：310P 为真 int8；CUDA / CoreX 为 Quark 检查点加载期反量化 |
| KV 管理 | Paged KV cache、Prefix cache（块级哈希复用）、连续批处理、分块预填充 |
| 执行加速 | CUDA Graph / NPUGraph（普通解码一步、投机解码整轮）、按 batch 大小分桶、算子融合 |
| 精度与守卫 | fp16/bf16 与量化两条路径互斥；`--fp16-only` 拒绝加载量化检查点；运行期参数 dtype 审计 |
| 服务 | OpenAI 兼容接口（chat/completions、completions、models）、SSE 流式、上下文淘汰与输出夹紧、内置 Web 控制台 |
| 运维 | 统一入口 `python -m flashengine`、服务 start/status/stop 托管（PID 身份校验）、`/healthy` `/ready` `/v1/runtime` `/metrics` |

组合的完整声明（模型 × 后端 × 量化 × 投机）由 `flashengine/support_matrix.py` 描述，启动即校验；文档版见 [支持矩阵](docs/SUPPORT_MATRIX.md)。

已知边界：投机解码当前要求 TP=1；Qwen3.5 仅 fp16/bf16；310P 的 KV block 仅支持 64/128，且当前 TP=1；DSpark 使用固定草稿长度，尚未接入基于 confidence 的动态验证长度。

## 快速开始

需要 Python 3.10–3.12、与硬件匹配的加速卡运行环境，以及**本地模型目录**（不接受 HuggingFace hub ID）。

```bash
# 先安装匹配硬件的 PyTorch，再安装引擎
python3 -m pip install -e .            # 或 pip install -r requirements.txt
python3 -m pip install -e ".[cuda]"    # CUDA：triton + flash-attn
# corex：厂商环境 + ixformer；ascend：CANN + torch-npu（310P 不依赖 vLLM）
```

### 统一入口

```bash
python -m flashengine config   --config configs/default.json --model /data/Qwen3-8B
python -m flashengine generate --config configs/eager.json  --model /data/Qwen3-8B --prompt "你好"
python -m flashengine serve    --config configs/default.json --model /data/Qwen3-8B
python -m flashengine start    --config configs/default.json --model /data/Qwen3-8B
python -m flashengine status
python -m flashengine stop
```

`configs/` 提供三档模板：`default.json`（图模式）、`eager.json`（关闭图，调试用）、`batch.json`（多并发 + 前缀缓存）。模板中的模型路径是占位值，用 `--model` 覆盖即生效，命令行覆盖优先级最高。`python -m scripts.framework` 与 `python -m flashengine` 等价。

### Python API

```python
from flashengine import LLM, SamplingParams

llm = LLM("/data/Qwen3-8B", backend="ascend310p", max_model_len=2048,
          max_num_seqs=1, gpu_memory_utilization=0.5)
print(llm.generate(["用一句话解释快速排序"], SamplingParams(temperature=0, max_tokens=64)))
```

开启投机：向 `LLM` 传 `spec_model="<与目标匹配的草稿目录>"`，草稿长度由 checkpoint 自动派生（块草稿上限为 `block_size-1`）。`enforce_eager=True` 关闭图捕获。

### HTTP 服务

```bash
python -m flashengine serve --config configs/default.json --model /data/Qwen3-8B \
    --host 0.0.0.0 --port 8000 --api-key-env FLASHENGINE_API_KEY
```

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":128,"stream":true}'
```

启动后打开 `http://127.0.0.1:8000/` 即为内置控制台（对话 + 实时性能面板）。三类端点分工明确：`/healthy` 只表示进程存活，`/ready` 返回服务名、版本与 PID 供编排与托管校验，`/v1/runtime` 输出完整运行快照（内存分期、引擎配置、调度、缓存、投机、图与精度审计）。

### 基准

```bash
python -m scripts.benchmark /data/Qwen3-8B \
    --backend ascend310p --no-think --gpu-mem-util 0.5 --max-model-len 2048 \
    --max-num-seqs 1 --warmup 1 --num-prompts 3 --max-tokens 64 \
    --questions bench_data/question.jsonl --no-prefix-cache
```

`--spec-model` 开启投机基准并输出接受长度 E 与逐位置接受率，`--num-repeats` 与 `--no-prefix-cache` 做缓存冷热对照，`--eager` 关闭图。脚本以 `python3 -m scripts.<名称>` 形式从仓库根目录运行。

## 性能结果

口径：`scripts.benchmark`，单序列、greedy（temp=0）、thinking-off、无前缀缓存、每档 3 题 × 64 token、warmup 1；TPS 为稳态解码吞吐（排除 TTFT 与 prefill），E 为投机平均接受长度。表内数字均为实测，非推算。

### Ascend 310P（Atlas 300I Duo，CANN 9.0，fp16）

| 模型 | 无投机 | + 投机 | 加速比 | E |
|---|---:|---:|---:|---:|
| Qwen3-8B | 10.9 | **37.6**（DFlash k15） | 3.45× | 4.878 |
| Qwen3-8B | 10.9 | **40.9**（DSpark k7） | 3.75× | 5.289 |
| Qwen3-8B-W8A8 | 15.4 | — | 1.41×（对比 fp16） | — |
| Qwen3.5-4B | 15.8 | **32.8**（DFlash k7） | 2.08× | 4.283 |
| Qwen3.5-9B | 9.5 | **22.5**（DFlash k15） | 2.37× | 5.100 |
| Qwen3-30B-A3B | 22.2 | **26.4**（DFlash k7） | 1.19× | 3.328 |
| Qwen3-Coder-30B | 22.4 | **38.6**（DFlash k15） | 1.72× | 6.839 |
| Qwen3-Coder-30B-w8a8 | 30.4 | — | 1.36×（对比 fp16） | — |

批量聚合（Qwen3-8B，含 prefill，AGG = Σ输出 token / 墙钟）：bs=1/4/8 分别为 **10.7 / 41.3 / 80.8 tok/s**。

补充参照：Qwen3-8B 在 310P 上 eager 解码为 7.4 tok/s，图解码 10.7–11.0 tok/s，即图在此后端带来约 **1.46×** 收益；投机整轮图回放率 100%，无回退。

### CUDA 与 CoreX（历史结果，不同硬件，仅作量级参考）

| 硬件 / 后端 | Graph 解码 | Graph + EAGLE3(k3) | E |
|---|---:|---:|---:|
| RTX 4070 Laptop 8GB / CUDA | 66.2 | **108.7** | 2.095 |
| 天数智芯 32GB / CoreX | 76.5 | **104.1** | 2.050 |

模型为 Qwen3-1.7B，spec_bench coding 前 10 条，串行 batch=1，greedy，thinking-off，TPS 含 prefill。

### 判读提醒

- **投机收益取决于接受率。** 上表的加速比来自高接受负载（代码类、greedy）。低接受负载（如短问答、散文）下草稿开销可能超过收益，此时应关闭投机。
- **图是 310P 与 CoreX 的刚需。** 这两个后端的 kernel 启动开销远高于 CUDA，eager 与图解码存在数倍差距，除非调试否则不要用 `--eager`。
- **口径不可混用。** 串行进程内基准、批量聚合、HTTP 并发三者的吞吐与延迟定义不同，不能同表比较；TTFT 在进程内测量，不含 HTTP 往返。
- **310P 存在偶发慢态。** 同命令同代码可能出现 2× 以上离群，先复测再记录；持续慢态建议重启板卡。E 逐位不变可作为对照。
- **W8A8 与投机暂不叠加。** W8A8 模型配投机时接受长度会退化到接近 1，上表 w8a8 行均为无投机。

完整表格、逐组命令与判读见 [310P 综合基准](docs/ASCEND310P_BENCHMARK.md) 与 [310P 测试配置](docs/ASCEND310P_BENCH.md)。

## 项目结构

```text
flashengine/
├── llm.py · config.py · sampling_params.py    # API、配置装配、采样参数
├── runtime.py · version.py · precision.py     # 统一入口、发布身份、精度守卫
├── validation.py                              # logits 比对与参数 dtype 审计
├── engine/                                    # 主循环、调度器、KV 分块、执行器、观察接口
├── models/                                    # Qwen3 / MoE / Qwen3.5 与目标模型注册表
├── speculative/                               # 投机方法注册表与运行策略
├── layers/                                    # Attention、Linear、Norm、RoPE、Sampler、量化
├── backends/                                  # CUDA、CoreX、Ascend 310P 及 NPUGraph 图执行器
└── utils/                                     # 执行上下文、权重加载
scripts/                                       # HTTP 服务、Web 控制台、统一入口、服务托管、基准
configs/                                       # 配置模板
bench_data/                                    # 英文、中文、Agent 共享前缀数据集
docs/                                          # 架构、后端、算法与基准说明
```

常用文档入口：[支持矩阵](docs/SUPPORT_MATRIX.md)、[引擎架构](docs/ENGINE_RUNTIME.md)、[310P 算子](docs/ASCEND310P_OPS.md) 与 [310P 图规格](docs/ASCEND310P_GRAPH_SPEC.md)、[Qwen3.5 适配](docs/QWEN35.md)、[EAGLE3](docs/EAGLE3.md)、[DFlash / DSpark](docs/DFlash.md)、[显存管理与 Triton 接口](docs/PYTHON_MEMORY_TRITON.md)。

## 使用须知

本项目是研究与学习工程，不是生产级推理服务。性能数字绑定具体硬件、模型、数据集与配置条件，跨环境不可直接外推；`ascend310p` 与 `corex` 的断言均来自对应真机实测，未测的组合按支持矩阵视为未验证。对外暴露服务时请自行在网络边界限制 `/metrics`、`/v1/stats` 与控制台页面。
