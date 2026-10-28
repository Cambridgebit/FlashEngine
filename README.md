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

### 统一入口一览

```bash
python -m flashengine config   --config configs/default.json --model /data/Qwen3-8B
python -m flashengine generate --config configs/eager.json  --model /data/Qwen3-8B --prompt "你好"
python -m flashengine serve    --config configs/default.json --model /data/Qwen3-8B
python -m flashengine start    --config configs/default.json --model /data/Qwen3-8B
python -m flashengine status
python -m flashengine stop
```

`configs/` 提供三档模板：`default.json`（图模式）、`eager.json`（关闭图，调试用）、`batch.json`（多并发 + 前缀缓存）。模板中的模型路径是占位值，用 `--model` 覆盖即生效，命令行覆盖优先级最高。`python -m scripts.framework` 与 `python -m flashengine` 等价。每个子命令的职责、与原始入口的关系、以及全部可调参数见下方 [使用方式](#使用方式)。

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

## 使用方式

### 六个子命令的职责

| 子命令 | 做什么 | 加载模型 | 占用设备 |
|---|---|---|:---:|
| `config` | 把 `--config` 加命令行覆盖合并成最终配置，打印 JSON 后退出 | 否 | 否 |
| `generate` | 离线跑一次生成：套 chat 模板 → 推理 → 打印文本，可选 `--output` 存 JSON（含 token、耗时、运行快照），结束后释放设备 | 是 | 是（跑完释放） |
| `serve` | **前台**运行 HTTP 服务与 Web 控制台，Ctrl-C 结束 | 是 | 是 |
| `start` | **后台**启动服务，等到 `/ready` 通过才返回，并把 PID、内核启动计数、cwd、argv 写入 `runtime/service.json` | 是（在子进程） | 是 |
| `status` | 读状态文件、校验进程身份与 `/ready`，打印 `{"running","ready","pid","log"}`；未就绪时退出码非 0 | 否 | 否 |
| `stop` | 重新核对 PID / 启动计数 / cwd / 完整 argv 四项，全部匹配才发 `SIGTERM` 优雅停止 | 否 | 否 |

`config` 只做配置装配与校验，和 `status`/`stop` 一样不加载 torch，在没有设备运行时的机器上也能直接跑——适合先确认配置合法再上设备。`status` 与 `stop` 只读状态文件，不需要再传 `--config` 或 `--model`。`start` 的日志落在 `runtime/server_<随机>.log`，`status` 会打印路径。

`start`/`status`/`stop` 依赖 `/proc`、pidfd 与 `SIGTERM`，**只能在 Linux 开发板或容器内执行**；在 Windows 上会直接报错退出。`config`/`generate`/`serve` 不受此限制。

注意 `stop` **只管理 `start` 自己启动的进程**：用 `serve` 前台起的、或用其它包装脚本起的进程不在其管辖范围，会报 `FRAMEWORK_SERVICE_NOT_RUNNING`。这是刻意的安全边界——宁可拒绝，也不误杀占用端口的其他服务。

### 与原始入口的关系

`serve` 并不是服务的另一份实现，而是**把统一配置翻译成原始服务的命令行**（`server_arguments()` 逐项映射，再调用 `scripts/openai_server.main`）。调度、流式、投机、Web 控制台全部沿用原实现。所以三条路等价，按需要选：

```bash
# A. 原始入口：参数最全（--api-key 明文、--raw-log、--webui、显存预算等）
python -m scripts.openai_server /data/Qwen3-8B --backend ascend310p --port 8000

# B. 统一入口前台：从 JSON 装配，透传 --raw-log / --webui
python -m flashengine serve --config configs/default.json --model /data/Qwen3-8B --raw-log /tmp/raw.jsonl

# C. 统一入口后台托管：可用 status/stop 管理生命周期
python -m flashengine start --config configs/default.json --model /data/Qwen3-8B
```

**`benchmark` 不在统一入口内，用法完全不变。** 这是有意的：基准需要精确控制 seed、warmup、repeats、数据集、输出长度等测量口径，套一层配置反而碍事。`configs/*.json` 是给服务用的，不是给基准用的。

```bash
python -m scripts.benchmark /data/Qwen3-8B \
    --backend ascend310p --no-think --gpu-mem-util 0.5 --max-model-len 2048 \
    --max-num-seqs 1 --warmup 1 --num-prompts 3 --max-tokens 64 \
    --questions bench_data/question.jsonl --no-prefix-cache
```

一句话决策：**跑性能用 `scripts.benchmark`，起服务用 `flashengine serve/start`，只查配置用 `config`，快速验证模型用 `generate`。**

### 投机推理

四个入口都能开，优先级为 命令行 > 配置文件 > 默认值：

```bash
# 1) 统一入口（config / generate / serve / start 都支持）
python -m flashengine serve --config configs/default.json --model /data/Qwen3-8B \
    --spec-model /data/Qwen3-8B-Dflash --spec-num-tokens 7
python -m flashengine serve --config configs/default.json --model /data/Qwen3-8B \
    --spec-model ""                      # 空串 = 关闭

# 2) 写进配置文件
#    { "spec_model": "/data/Qwen3-8B-Dflash", "spec_num_tokens": 7 }

# 3) 基准
python -m scripts.benchmark /data/Qwen3-8B <参数> --spec-model /data/Qwen3-8B-Dflash

# 4) 原始服务入口
python -m scripts.openai_server /data/Qwen3-8B --spec-model /data/Qwen3-8B-Dflash
```

草稿长度 `k` 的规则：**EAGLE3** 由 `--spec-num-tokens` 直接指定（默认 3）；**DFlash / DSpark** 由草稿 checkpoint 的 `block_size` 自动派生，上限 `block_size-1`，且**默认值 3 和 4 视为「未显式指定」**——8B 草稿（block 16）自动跑 k=15，30B（block 8）自动跑 k=7。要强制指定就传别的数字。

确认投机真的在跑，别只看启动日志：

```bash
curl -s http://127.0.0.1:8000/v1/runtime | python3 -m json.tool | grep -A3 '"spec"'
```

应看到 `"enabled": true`、`"method": "dflash"`、`"num_tokens": 15`，同时 `engine.graph_mode` 为 `spec_graph`。请求日志里每条的 `path=` 字段标出实际执行路径（`spec_graph` / `eager_spec` / `decode_graph` / `eager`），这是判断有没有掉回 eager 最快的办法。

### 可控制的参数

配置键名与命令行旗标一一对应（JSON 用下划线，命令行用连字符），未列出的键会被拒绝而不是静默忽略。下表标注每个参数的实际生效范围。

**模型与后端**

| 配置键 / 旗标 | 默认 | 说明 | 生效范围 |
|---|---|---|---|
| `model` / `--model` | 必填 | 本地模型目录，不接受 hub ID | config/generate/serve/start |
| `backend` / `--backend` | `auto` | `cuda` / `corex` / `ascend310p`，`auto` 按运行时探测 | 全部 |
| `precision` / `--precision` | `auto` | `auto`/`float16`/`bfloat16`/`float32`，非 auto 时映射为服务的 `--dtype` | 全部 |
| `quantization` / `--quantization` | `none` | `none`/`w8a8`/`quark_w8a8`。**仅作声明与一致性校验**，实际量化方式以检查点元数据为准，不一致会打印告警 | 全部 |
| `fp16_only` / `--fp16-only` | `false` | 拒绝加载量化检查点，防止 FP16 版误用 W8A8 权重 | 全部 |
| `tensor_parallel_size` / `--tensor-parallel-size` | 1 | TP 规模。HTTP 服务只支持同进程 TP=1，传非 1 会直接报错 | 仅 generate |
| `device_ids` / `--device-ids` | 空 | 逗号分隔的设备索引；服务取首个作为 `--device` | 全部 |

**显存与 KV cache**

| 配置键 / 旗标 | 默认 | 说明 |
|---|---|---|
| `max_model_len` / `--max-model-len` | 4096 | 上下文窗口上限，超出按淘汰策略裁剪 |
| `max_num_seqs` / `--max-num-seqs` | 8 | 同时调度的序列数；310P 建议 1–8 |
| `max_num_batched_tokens` / `--max-num-batched-tokens` | 16384 | 每步 prefill 的 token 预算 |
| `max_num_partial_prefills` / `--max-num-partial-prefills` | 1 | 未完成 prefill 的并发上限 |
| `enable_chunked_prefill` / `--chunked-prefill` / `--no-chunked-prefill` | true | 分块预填充开关 |
| `enable_prefix_cache` / `--prefix-cache` / `--no-prefix-cache` | true | 前缀缓存开关；310P 缺 SplitFuse 时自动关闭 |
| `gpu_memory_utilization` / `--gpu-mem-util` | 0.9 | KV 池按整卡比例分配；空卡建议 0.5–0.85 留出图捕获余量 |
| `kvcache_block_size` / `--block-size` | 256 | KV 块大小；310P 仅 64 或 128，CUDA 需 256 的倍数 |
| `num_kvcache_blocks` / `--kv-blocks` | -1 | 显式指定 KV 块数，`-1` 为按预算自动计算 |
| `recurrent_prefix_cache_bytes` / `--recurrent-prefix-cache-mb` | 256 MiB | Qwen3.5 前充缓存（GDN conv/SSM 快照）的主机内存预算，0 关闭 |
| `memory_budget_bytes` / `--memory-budget-mib` | 0 | 引擎显存预算，0 表示只按整卡比例规划 |
| `device_memory_reserve_bytes` / `--device-reserve-mib` | 0 | 为其他模型与工作区预留的显存 |
| `memory_workspace_bytes` / `--memory-workspace-mib` | 256 MiB | 运行工作区最低预留 |
| `graph_memory_reserve_bytes` / `--graph-reserve-mib` | 256 MiB | Graph 额外预留，eager 不使用 |

**投机、执行与图**

| 配置键 / 旗标 | 默认 | 说明 |
|---|---|---|
| `spec_model` / `--spec-model` | 空 | 与目标匹配的草稿目录，空串关闭投机 |
| `spec_num_tokens` / `--spec-num-tokens` | 4 | 草稿长度；块草稿自动派生，3/4 视为未指定 |
| `enforce_eager` / `--eager` / `--no-eager` | false | 关闭图捕获；仅调试用，310P/CoreX 上代价极大 |
| `decode_linear_backend` / `--decode-linear-backend` | `functional` | `functional` 或 `selective`，310P 解码线性层实现 |
| `greedy_backend` / `--greedy-backend` | `npu` | `npu` 或 `cpu`，greedy 归约位置 |
| `cpu_threads` / `--cpu-threads` | 4 | 主机侧线程数 |
| `pa_graph_update` / `--pa-graph-update` | `auto` | `auto`/`0`/`1`，NPU 图内 PA 元数据更新；310P 多数驱动不支持 |

**采样默认值与服务**

| 配置键 / 旗标 | 默认 | 说明 |
|---|---|---|
| `temperature` / `--temperature` | 0 | 请求未显式传 temperature 时使用；0 为 greedy |
| `top_p` / `--top-p` | 1.0 | 随机采样 top-p |
| `top_k` / `--top-k` | 0 | 随机采样 top-k，0 关闭 |
| `max_tokens` / `--max-tokens` | 128 | 离线生成的输出上限；HTTP 请求由请求体决定 |
| `enable_thinking` / `--thinking` / `--no-thinking` | false | chat 模板是否开启 thinking（原始服务与基准里的对应旗标是 `--no-think`） |
| `reserve_output` / `--reserve-output` | 512 | 上下文淘汰时为输出预留的 token 数 |
| `host` / `--host`、`port` / `--port` | 0.0.0.0:8000 | 监听地址与端口 |
| `stats_interval` / `--stats-interval` | 10.0 | `[INFO]` 聚合周期秒数，0 关闭 |
| `web_ui` / `--ui` / `--no-ui` | true | 内置 Web 控制台开关 |
| `api_key_env` / `--api-key-env` | `FLASHENGINE_API_KEY` | 存放密钥的**环境变量名**，密钥本身不进 argv 与状态文件 |

**环境变量。** 以下变量由配置自动写入（`apply_environment()`），也可以在外部直接设置；其余开关只有环境变量入口：

| 变量 | 取值 | 作用 |
|---|---|---|
| `FLASHENGINE_BACKEND` | `cuda`/`corex`/`ascend310p` | 后端选择 |
| `LITEQ_DECODE_LINEAR_BACKEND` | `functional`/`selective` | 310P 解码线性层实现 |
| `LITEQ_GREEDY_BACKEND` | `npu`/`cpu` | greedy 归约位置 |
| `LITEQ_CPU_THREADS` | 整数 | 主机侧线程数 |
| `FLASHENGINE_ACLGRAPH_PA_UPDATE` | `auto`/`0`/`1` | 图内 PA 元数据更新 |
| `FLASHENGINE_GRAPH_NUMERIC_GUARD` | `0` 关闭 | 310P 图内外 logits 数值守卫，默认开 |
| `FLASHENGINE_SPEC_GRAPH` | `0`/`1` | 投机整轮图总开关 |
| `FLASHENGINE_GRAPH_SAMPLE` | `0`/`1` | 图内采样开关 |
| `FLASHENGINE_310P_FLASH_OP` | `auto`/`legacy`/`v3`/`atb_v2` | 310P attention 算子选择 |
| `FLASHENGINE_310P_DFLASH_PAGED` | `auto`/`1`/`0` | 310P 草稿 KV 的分页/稠密形态 |
| `FLASHENGINE_CUDA_ATTENTION` | `flash_attn`/`flashinfer` | CUDA 注意力库选择 |
| `FLASHENGINE_VLLM_GEMM` | `0`/`1` | 310P 是否注册 vllm_ascend GEMM |
| `FLASHENGINE_RUNTIME_DIR` | 路径 | 服务状态/日志目录，默认源码根下 `runtime/` |

**两处需要留意的不对称**：`scripts.benchmark` 没有 `--decode-linear-backend`、`--greedy-backend`、`--fp16-only` 这几个旗标，要测它们请用上表的环境变量（走同一段代码）；`--api-key` 明文只存在于原始服务入口，统一入口只接受环境变量名。

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
