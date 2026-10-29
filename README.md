# FlashEngine V1.1

FlashEngine 是一个从零实现的大模型推理引擎，面向研究与学习：覆盖请求调度、KV cache 管理、模型执行与硬件后端适配，提供 Python API、OpenAI 兼容 HTTP 服务与统一命令行入口。

支持 Qwen3 / Qwen3 MoE / Qwen3.5（dense 与 MoE）文本模型，运行在 NVIDIA CUDA、天数智芯 CoreX、华为 Ascend 310P 三种后端上，可选 EAGLE3、DFlash、DSpark 三种投机解码与 W8A8 量化推理。

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

三个设计要点：

- **统一入口，一套配置。** `RuntimeSettings` 与 `create_engine` 是唯一的装配入口，命令行、JSON、HTTP 服务与基准共用同一份配置模型；未知配置项直接报错，不做静默忽略。
- **注册表解耦。** 目标模型与投机方法各有一张注册表，新增模型或投机方法只需注册 adapter，`ModelRunner` 与调度器不改。
- **能力位而非后端名。** 后端声明自身能力（dtype、KV 布局、是否支持投机图、是否可接管 decode 图等），引擎只消费能力位，新增后端不必改主流程。

调度采用公平轮转的连续批处理与分块预填充，每步构建同质批次（prefill 或 decode），KV 不足时 LIFO 抢占；执行层在普通解码时每步提交一个 token，投机解码时一轮起草、一次验证、多 token 提交。Graph 捕获固定形状以消除调度与 kernel 启动开销，捕获失败按桶或整体优雅回退 eager。

## 支持的功能

| 类别 | 支持范围 |
|---|---|
| 目标模型 | Qwen3 dense、Qwen3 MoE、Qwen3.5 dense、Qwen3.5 MoE（文本） |
| 后端 | `cuda`、`corex`、`ascend310p`，可指定或自动探测 |
| 投机解码 | EAGLE3、DFlash、DSpark |
| 量化 | W8A8：310P 真 int8；CUDA/CoreX 为 Quark 检查点加载期反量化 |
| KV 管理 | Paged KV cache、Prefix cache、连续批处理、分块预填充 |
| 执行加速 | CUDA Graph / NPUGraph（解码一步、投机整轮）、按 batch 分桶、算子融合 |
| 精度守卫 | `--fp16-only` 拒绝加载量化检查点；运行期参数 dtype 审计 |
| 服务 | OpenAI 兼容接口、SSE 流式、上下文淘汰、内置 Web 控制台 |
| 运维 | 七个子命令统一入口、服务 start/status/stop 托管、`/ready` `/v1/runtime` `/metrics` |

已知边界：投机要求 TP=1；Qwen3.5 仅 fp16/bf16；310P 的 KV block 仅 64/128 且 TP=1；DSpark 使用固定草稿长度。完整声明见 [支持矩阵](docs/SUPPORT_MATRIX.md)（代码侧 `flashengine/support_matrix.py`，启动即校验）。

## 快速开始

需要 Python 3.10–3.12、与硬件匹配的加速卡环境，以及**本地模型目录**（不接受 HuggingFace hub ID）。

```bash
python3 -m pip install -e .            # 或 pip install -r requirements.txt
python3 -m pip install -e ".[cuda]"    # CUDA：triton + flash-attn
# corex：厂商环境 + ixformer；ascend：CANN + torch-npu（310P 不依赖 vLLM）
```

七个子命令：

```bash
python -m flashengine config    --config configs/default.json --model /data/Qwen3-8B
python -m flashengine benchmark --config configs/default.json --model /data/Qwen3-8B --num-prompts 3 --max-tokens 64
python -m flashengine generate  --config configs/eager.json   --model /data/Qwen3-8B --prompt "你好"
python -m flashengine serve     --config configs/default.json --model /data/Qwen3-8B
python -m flashengine start     --config configs/default.json --model /data/Qwen3-8B
python -m flashengine status
python -m flashengine stop
```

| 子命令 | 用途 | 加载模型 |
|---|---|:---:|
| `config` | 打印合并后的最终配置，不上设备（可在无 torch 的机器上运行） | 否 |
| `benchmark` | 跑串行基准（封装 `scripts.benchmark`），输出 TPS / TTFT / 接受长度 E | 是 |
| `generate` | 离线生成一次，可选 `--output` 存文本、token 与运行快照 | 是 |
| `serve` | 前台运行 HTTP 服务与 Web 控制台 | 是 |
| `start` | 后台启动服务并等待 `/ready`，状态写入 `runtime/service.json` | 是 |
| `status` | 查看运行状态、就绪情况、PID 与日志路径 | 否 |
| `stop` | 校验进程身份后优雅停止（只管理 `start` 起的进程） | 否 |

`start`/`status`/`stop` 依赖 `/proc` 与 pidfd，**只能在 Linux 开发板或容器内执行**。`python -m scripts.framework` 与 `python -m flashengine` 等价。

### 用不同 JSON 测试不同设置

常改的参数都写在 `configs/*.json` 里，换个文件就是换一套设置；命令行只用于临时覆盖，优先级最高：

| 文件 | 场景 | 关键设置 |
|---|---|---|
| [default.json](configs/default.json) | 图模式基线 | `enforce_eager=false` |
| [eager.json](configs/eager.json) | 关图排障、逐位对照 | `enforce_eager=true` |
| [spec.json](configs/spec.json) | 投机解码 | `spec_model`、`spec_num_tokens` |
| [constrained.json](configs/constrained.json) | 禁词约束示例 | 预设词库，需显式开启 |
| [batch.json](configs/batch.json) | 多并发 + 前缀缓存 | `max_num_seqs=4`、`enable_prefix_cache=true` |

```bash
# 同一模型，四套设置各跑一遍（只改 --config）
python -m flashengine benchmark --config configs/default.json --model /data/Qwen3-8B --num-prompts 3 --max-tokens 64
python -m flashengine benchmark --config configs/eager.json   --model /data/Qwen3-8B --num-prompts 3 --max-tokens 64
python -m flashengine benchmark --config configs/spec.json    --model /data/Qwen3-8B \
    --spec-model /data/Qwen3-8B-Dflash --num-prompts 3 --max-tokens 64
python -m flashengine benchmark --config configs/batch.json   --model /data/Qwen3-8B --num-prompts 3 --max-tokens 64
```

模板里的模型路径是占位值，用 `--model` 覆盖即可。想新增一套设置，复制一个 JSON 改几行就行；写错键名会直接报 `未知配置项`，不会被静默忽略。

### 常用配置项

| 键 | 说明 |
|---|---|
| `model`、`backend`、`precision` | 本地模型目录、后端（`cuda`/`corex`/`ascend310p`/`auto`）、计算类型 |
| `max_model_len`、`max_num_seqs`、`max_num_batched_tokens` | 上下文窗口、并发序列数、每步 prefill 预算 |
| `gpu_memory_utilization`、`kvcache_block_size` | KV 池占整卡比例、KV 块大小（310P 仅 64/128） |
| `enable_prefix_cache`、`enable_chunked_prefill` | 前缀缓存、分块预填充开关 |
| `enforce_eager` | 关闭图捕获（仅调试用） |
| `spec_model`、`spec_num_tokens` | 草稿目录与草稿长度；空串关闭投机 |
| `enable_constraints`、`banned_words` | 禁词约束开关与词库 |
| `temperature`、`top_p`、`top_k`、`max_tokens` | 采样默认值与离线输出上限 |
| `enable_thinking`、`reserve_output` | thinking 开关、上下文淘汰时给输出预留的 token |
| `host`、`port`、`web_ui`、`api_key_env` | 监听地址端口、控制台开关、存放密钥的环境变量名 |

命令行旗标与 JSON 键一一对应（JSON 用下划线，命令行用连字符）。**只有环境变量入口的调优开关**（310P 算子选择、图数值守卫、投机图总开关等）不在配置文件里，见 [310P 算子说明](docs/ASCEND310P_OPS.md) 与 [310P 图规格](docs/ASCEND310P_GRAPH_SPEC.md)。

### 投机推理

在 JSON 里写 `spec_model` 即可，或命令行临时指定：

```bash
python -m flashengine serve --config configs/spec.json --model /data/Qwen3-8B \
    --spec-model /data/Qwen3-8B-Dflash --spec-num-tokens 15
```

草稿长度规则：**EAGLE3** 由 `spec_num_tokens` 直接指定（默认 3）；**DFlash / DSpark** 由草稿 checkpoint 的 `block_size` 自动派生，上限 `block_size-1`，默认值 3 和 4 视为「未显式指定」（8B 草稿 block 16 → k=15，30B 的 block 8 → k=7）。

确认投机确实生效：

```bash
curl -s http://127.0.0.1:8000/v1/runtime | python3 -m json.tool | grep -A3 '"spec"'
```

应看到 `"enabled": true` 与具体方法名，且 `engine.graph_mode` 为 `spec_graph`。请求日志里的 `path=` 字段标出实际执行路径（`spec_graph`/`eager_spec`/`decode_graph`/`eager`），是判断有无掉回 eager 最快的办法。

### 禁词约束

在 [constrained.json](configs/constrained.json) 中设置 `"enable_constraints": true` 和 `"banned_words": ["航空", "秘密"]`；若要同时使用投机解码，设置 `"spec_model": "/data/Qwen3-8B-ds-Dspark"`。也可临时覆盖：

```bash
python -m flashengine serve --config configs/constrained.json --model /data/Qwen3-8B \
    --enable-constraints --spec-model /data/Qwen3-8B-ds-Dspark
```

配置默认关闭约束；关闭时设 `"enable_constraints": false`。默认按输出文本的精确子串匹配；`"constraint_ignore_spaces": true` 可忽略 ASCII 空格、制表符和换行，但不覆盖同义词、全角空格等变体。该功能仍在验证中，目前仅在 310P 的 Qwen3-8B + DSpark 上做过短样本测试，不应将其作为语义级内容过滤保证。

### Python API

```python
from flashengine import LLM, SamplingParams

llm = LLM("/data/Qwen3-8B", backend="ascend310p", max_model_len=2048,
          max_num_seqs=1, gpu_memory_utilization=0.5,
          spec_model="/data/Qwen3-8B-Dflash")          # 不传则关闭投机
print(llm.generate(["用一句话解释快速排序"], SamplingParams(temperature=0, max_tokens=64)))
```

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

启动后打开 `http://127.0.0.1:8000/` 即为内置控制台（对话 + 实时性能面板）。`/healthy` 只表示进程存活，`/ready` 返回服务名、版本与 PID 供编排与托管校验，`/v1/runtime` 输出完整运行快照。

原来的入口同样可用，两者等价：`python -m scripts.openai_server <模型> <参数>`、`python -m scripts.benchmark <模型> <参数>`。

## 性能结果

口径：单序列、greedy（temp=0）、thinking-off、无前缀缓存、每档 3 题 × 64 token、warmup 1；TPS 为稳态解码吞吐（排除 TTFT 与 prefill），E 为投机平均接受长度。

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

批量聚合（Qwen3-8B，含 prefill）：bs=1/4/8 分别为 **10.7 / 41.3 / 80.8 tok/s**。Qwen3-8B eager 解码 7.4 tok/s 对图解码 10.7–11.0 tok/s，图带来约 **1.46×** 收益，投机整轮图回放率 100%、无回退。

### CUDA 与 CoreX（历史结果，不同硬件，仅作量级参考）

| 硬件 / 后端 | Graph 解码 | Graph + EAGLE3(k3) | E |
|---|---:|---:|---:|
| RTX 4070 Laptop 8GB / CUDA | 66.2 | **108.7** | 2.095 |
| 天数智芯 32GB / CoreX | 76.5 | **104.1** | 2.050 |

模型为 Qwen3-1.7B，spec_bench coding 前 10 条，串行 batch=1，greedy，thinking-off，TPS 含 prefill。

### 判读提醒

- **投机收益取决于接受率**：上表加速比来自高接受负载（代码类、greedy）；短问答、散文类负载下草稿开销可能超过收益，此时应关闭投机。
- **图是 310P 与 CoreX 的刚需**：两者 kernel 启动开销远高于 CUDA，eager 与图解码可差数倍。
- **口径不可混用**：串行基准、批量聚合、HTTP 并发三者的吞吐与延迟定义不同，不能同表比较。
- **310P 有偶发慢态**：同命令同代码可出现 2× 以上离群，先复测再记录，E 逐位不变可作对照。

完整表格与逐组命令见 [310P 综合基准](docs/ASCEND310P_BENCHMARK.md) 与 [310P 测试配置](docs/ASCEND310P_BENCH.md)。

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
configs/                                       # 配置模板（default / eager / spec / batch）
bench_data/                                    # 英文、中文、Agent 共享前缀数据集
docs/                                          # 架构、后端、算法与基准说明
```

常用文档：[支持矩阵](docs/SUPPORT_MATRIX.md)、[引擎架构](docs/ENGINE_RUNTIME.md)、[310P 算子](docs/ASCEND310P_OPS.md) 与 [310P 图规格](docs/ASCEND310P_GRAPH_SPEC.md)、[Qwen3.5 适配](docs/QWEN35.md)、[EAGLE3](docs/EAGLE3.md)、[DFlash / DSpark](docs/DFlash.md)、[显存管理与 Triton 接口](docs/PYTHON_MEMORY_TRITON.md)。

## 使用须知

本项目是研究与学习工程，不是生产级推理服务。性能数字绑定具体硬件、模型、数据集与配置条件，跨环境不可直接外推；`ascend310p` 与 `corex` 的断言均来自对应真机实测，未测的组合按支持矩阵视为未验证。对外暴露服务时请自行在网络边界限制 `/metrics`、`/v1/stats` 与控制台页面。
