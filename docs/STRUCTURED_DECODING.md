# 结构化约束解码与工具调用

agent（DSH 等）要能把本地模型当 OpenAI 工具调用后端用，必须满足两件事：

1. 模型输出**真正被约束**成合法结构（JSON Schema 或工具调用协议），而不是"希望它听话"；
2. 服务端把模型原生格式翻译成 OpenAI 的 `tools` / `tool_calls` / `response_format`。

本仓此前的"约束"只有禁词掩码（`flashengine/constraints.py`），对 JSON/工具调用无能为力。
本轮加了两层：**引擎侧的字节级语法掩码** + **服务侧的协议中间层**。

## 1. 分层

```
OpenAI 请求 ──► flashengine/protocol.py（中间协议：探测 / messages 规整 / 模板渲染 / 输出解析）
                    │  tools= 交给 chat template
                    ▼
              flashengine/structured/（语法：JSON Schema → 字节状态机；工具协议 → 序列语法）
                    │  PathConstraint
                    ▼
引擎：add_request(..., constraint=) → 每步 mask(state) 给 logits 落 -inf → 采样 → advance(state)
```

| 模块 | 职责 |
|---|---|
| `flashengine/protocol.py` | 协议探测（按 chat template 特征）、OpenAI messages 规整（保留 `tool_calls`/`tool_call_id`/`reasoning_content`，`arguments` 字符串→dict）、`apply_chat_template` 渲染、thinking 拆分、输出解析成 `content`/`tool_calls` |
| `flashengine/structured/vocab.py` | 词表字节表（ByteLevel BPE 还原）、首字节分桶、自由文本区"整 token 放行"位图（UTF-8 完整且非 special token） |
| `flashengine/structured/machines.py` | `Segment`（Literal/AltLiteral/FreeText/NumberText）+ `Grammar`（候选集合状态机）+ 掩码缓存 |
| `flashengine/structured/json_schema.py` | JSON Schema 实用子集 → 字节级生成状态机 |
| `flashengine/structured/protocols.py` | 三种工具调用协议 + `response_format` 的语法图 |

## 2. 状态机怎么工作

- **状态 = 候选集合** `frozenset{(segment_id, segment_state)}`。一个字节同时喂给所有候选，
  失败的候选掉队——"或"分支（多个可选续写：多个工具、参数可选跳过、调用结束 vs 再来一个）
  由候选集合天然表达，不需要额外 NFA/DFA 结构。
- 段 `step()` 返回：下一个状态 / `None`（非法）/ `DONE`（消费本字节后结束）/
  `PENDING`（本段可结束但字节未消费，交给后继重试）/ `DoneWith(succs)`（按匹配分支指定后继）。
  数字与字面量的"结束不消费字节"走 `PENDING`，`</param>` 这类终止符由值段自己吃掉。
- `END` 伪段：出现在候选集合里就表示"语法现在可以结束"→ EOS 进掩码；**其它状态 EOS 一律
  禁止**。自由文本区的整 token 位图会把 ASCII 形态的特殊 token（`</s>`、`<|im_end|>`）算进去，
  掩码最后统一清掉这些位。
- **性能**：自由文本区（JSON 字符串内容、XML 参数值）占词表绝大多数，这类区域里"合法 token
  推进后状态不变"，因此把"UTF-8 完整且不含特殊字节"的词表位图一次性算好整 token 放行；
  只有含引号/反斜杠/控制符/非完整 UTF-8 的少数 token 才逐字节精算。掩码按状态缓存
  （默认 64 MiB），词表视图按 tokenizer 缓存，语法按 `(协议, 工具定义, 词表)` 缓存 16 份。
  实测冷启动 ~2.6 ms/步，热路径 0 ms（`scripts/check_structured.py` 有断言）。

## 3. 支持的协议与 schema 子集

协议（`--tool-protocol auto|xml_function|hermes|deepseek_v3`，auto 按 chat template 探测）：

| 协议 | 适用模型 | 格式 |
|---|---|---|
| `xml_function` | MiniCPM5 系（模板里有 `<function name="…">`） | `<function name="f"><param name="p">v</param></function>`，多调用换行分隔 |
| `hermes` | Qwen3 / Hermes 系（模板里有 `<tool_call>`） | `<tool_call>\n{"name":"f","arguments":{…}}\n</tool_call>` |
| `deepseek_v3` | DeepSeek V3.1+ 专用 token | `<｜｜tool▁calls▁begin｜｜><｜｜tool▁call▁begin｜｜>function<｜｜tool▁sep｜｜>f\n```json\n{…}\n```<｜｜tool▁call▁end｜｜>…` |

JSON Schema 子集（编译期明确报错，不会生成期静默放宽）：

- `type`：`object` / `array` / `string` / `number` / `integer` / `boolean` / `null`；
- `object`：按 `properties` 声明顺序出键，`required` 必须出现、可选键可跳过；
  `additionalProperties` 只支持 `false`/缺省（=不生成额外键）；
- `array`：`items` 必填，`minItems`/`maxItems` 生效（默认上限 32，硬上限 128）；
- `enum` / `const`、`anyOf`/`oneOf`（**各分支首字节必须互不相交**）、`allOf`（对象合并）、
  本地 `$ref`（`#/$defs/...`，不支持递归）；
- 忽略 `pattern`/`minLength`/`format`/`minimum` 等内容约束（不影响 JSON 合法性）。

明确不支持：任意 JSON（空 schema）、`additionalProperties: true`、递归 `$ref`、
`anyOf` 首字节重叠、元组式 `items`、需要转义的属性名、`response_format=json_object`
（没有 schema 就无法约束，直接 400 让客户端改用 `json_schema`）。

后端无关：掩码在主机侧算好再与设备 logits 相加，CUDA/CoreX/310P 通用；要求 TP=1。

## 4. 与投机解码的关系（vLLM 对齐）

约束请求**参加投机轮**。整轮 spec 图把采样/接受也捕获了，动态语法掩码塞不进图，所以
`ModelRunner.spec_round` 检测到 `path_constraint` 时改走 eager spec 轮；算法与 vLLM 一致：

1. 草稿链在设备侧产 k 个草稿 token；
2. verify logits 出图后，按"已提交前缀 + d1..d{j-1}"**乐观推进**语法得到 k+1 行状态
   （语法状态不可变，回滚就是丢掉推演状态，不需要 vLLM 的 `grammar.rollback`）；
3. 主机侧把每行打包成 int32 位图（`ceil(vocab/32)` 个字，bit=1 允许，与 xgrammar 同约定），
   一次 H2D；
4. 图外 apply：CUDA 用 Triton kernel `_apply_bitmask_kernel` 把 0 位写 `-inf`，
   其它后端退回 torch 展开；
5. 采样/接受/提交照旧；真正提交的 token 由 `_sync_path_states` 推进语法（只认已接受 token）。

无约束请求继续走整轮 spec 图（`graph_replay` 正常增长）。位图按状态缓存（13 万词表约
16KB/状态，比原来的 bool 掩码小 8 倍）。

开启 `enable_path_constraints` 会同时关掉**图内采样**（decode graph 只出 logits，掩码在主机侧落），
因此**不使用工具调用的纯性能场景建议用 `--no-path-constraints`**（工具请求退化为"不约束、只解析"）。

## 5. 用法

离线（引擎级）：

```python
from flashengine.structured import tool_constraint, json_constraint

prompt_ids = protocol.render_chat(tokenizer, messages, tools=tools, enable_thinking=False)
constraint = tool_constraint(
    tools, "xml_function", tokenizer, engine.config.eos_ids,
    vocab_size=engine.config.model_config.vocab_size,
)
engine.generate([prompt_ids], SamplingParams(temperature=0.0, max_tokens=256),
                use_tqdm=False, constraint=constraint)
# 约束对象也可以按请求挂到异步路径：engine.add_request(ids, params, constraint=...)
```

构造引擎时需要 `enable_path_constraints=True`（server 默认开启）。

服务端：

```bash
~/vllm-env/bin/python3 -m scripts.openai_server /path/to/MiniCPM5-2B \
    --host 127.0.0.1 --port 8000 --max-model-len 1024 --max-num-seqs 1 \
    --gpu-mem-util 0.95 --eager [--no-think] [--tool-protocol auto] [--no-path-constraints]
```

请求体走标准 OpenAI 字段：`tools`（≤`--max-tools`）、`tool_choice`
（`auto`/`none`/`required`/`{"type":"function","function":{"name":...}}`）、
`parallel_tool_calls`、`response_format={"type":"json_schema",...}`；
响应 `message.tool_calls` + `finish_reason="tool_calls"`；带 tools/response_format 的流式请求
整段缓冲后发 SSE（role → tool_calls 分片 → finish），普通请求保持原逐 token 流式。

## 6. 验证

CPU 自检（不需要 GPU/权重，只要 tokenizer 目录）：

```bash
~/vllm-env/bin/python3 -m scripts.check_structured --model /path/to/tokenizer_dir
```

覆盖：三种协议的合法输出逐 token 通过、非法开头被掩码挡住、不支持 schema 编译期报错、
掩码缓存命中耗时。

实机（2026-11-12，RTX 4070 Laptop 8GB / WSL2，MiniCPM5-2B bf16 eager）：

- 原生协议探测：MiniCPM5 模板 → `xml_function`；模型自己就输出
  `<function name="get_weather"><param name="city">Paris</param></function>`
  （`<function`/`</function>`/`<param`/`</param>` 各是单 token），解析后
  `city="Paris"`、`finish_reason=tool_calls`；
- 约束解码 16 token 约 1.0–1.4s；掩码冷启动 2.6ms/步、热路径 0ms，无约束路径行为与
  逐字节输出不变；
- DSpark 引擎（MiniCPM5-2B + DSpark k=7）：约束请求走 eager spec 轮（`constrained_rounds`
  增长），**E=2.143**（7 轮接受 8 token），输出与普通约束解码逐字一致；长工具调用、
  JSON Schema 约束、混合批（约束 + 自由请求同批）均通过；CUDA 上走 Triton 位图 apply
  （`triton=True`）。无约束请求仍走整轮 spec 图。

端到端（对着已启动服务，stdlib 客户端，含流式）：

```bash
~/vllm-env/bin/python3 -m scripts.toolcall_bench http://127.0.0.1:8000 --model MiniCPM5-2B
~/vllm-env/bin/python3 -m scripts.toolcall_bench http://127.0.0.1:8000 --mode all --stream
```

实测（MiniCPM5-2B，`--max-model-len 1024 --gpu-mem-util 0.95 --eager`）：非流式与流式
的 tool/json 全部通过；稳态单次工具调用约 0.55–0.6s（首请求含掩码冷启动约 1.4–2.2s）；
`auto` 模式下模型原生 XML 调用被正确解析成 `tool_calls`，纯知识问题正常答正文；
`tool_choice=none` 不注入工具定义、不解析调用；400 分支（required 无 tools / json_object /
json_schema+required / thinking 冲突 / completions 带 tools / 非法 tool_choice）文案正确；
普通 chat 流式仍逐 token（35 个 content chunk），没有被缓冲。

实机（2026-11-23，Ascend 310P / Atlas 300I Duo，容器 `vllm018` / CANN 9.0.T3，融合 LoRA 的
Qwen3.5-9B + DFlash k=7、fp16、`--max-model-len 2048 --max-num-seqs 8`）：

- 协议探测：Qwen3.5 模板 → `hermes`；启动摘要打印 `Tool calls: protocol=hermes
  (requested=auto), path_constraints=on, max_tools=64`；
- 端到端四档全过：`--mode tool` 2/2（`get_weather({"city": "Paris"})`、`read_file({"path":
  "src/main.py"})`，单请求 5.3–7.9s）、`--mode tool --stream` 2/2、`--mode json` 与
  `--mode json --stream` 各 1/1（`{"city": "Paris", "temperature_c": 14, "condition":
  "cloudy"}`）；
- 约束请求确实走 eager spec 轮：`[DONE ]` 里 `spec_graph=0.0%(0/N)`、`E=1.58–1.91`；代价是每
  token 190–217ms（同机无约束图路径 55ms），与 CUDA 侧「约束轮比整轮 spec 图慢」一致；
- 310P 特有 bug（已修）：`flashengine/structured/bitmask.py` 的非 Triton 回退用
  `packed.unsqueeze(-1) >> shifts` 广播展开位图，310P 的 `aclnnRightShift` 不接受该广播
  （`AclNN_Parameter_Error(EZ1001): Shape of out should be ...`，错误码 161002，整个请求 500）。
  改成主机侧 `unpack_mask` 展开成 bool 允许表（`self._allowed_host` / `self._allowed_device`，
  只在 `not triton_ready` 时分配），`apply()` 回退分支变成一次 `masked_fill(~allowed, -inf)`；
  CUDA/Triton 路径不变；
- 约束一致性：同一 prompt、同一 schema、`enforce_eager` 下 spec 与无 spec 两次生成**逐 token
  一致**（`[4754, 8656, 3147, 57590, 2129, 33413, 648, 763, 16, 19, 1288, 8783, 3147, 5619,
  391, 14388, 88, 8934, 248046]`，输出 `{"city":"Paris","temperature_c":14,"condition":
  "Partly Cloudy"}`）；
- 零回归（同一台板、同一模型、`--max-model-len 2048 --max-num-seqs 8`、c1、128 token、
  8 条）：业务长提示 13.924 TPS / TPOT p50 58.23ms（合并前 13.683 / 58.33），英文 coding
  25.443 / 35.23ms（合并前 25.049 / 35.29）——约束开关默认关，热路径无额外分支；
- CPU 自检在 310P 树上一并跑通（`scripts.check_structured --model /data/Qwen3-8B`：三协议 ok、
  json_schema 4 项 ok、`pack/unpack 往返一致（151669 token）`），注意容器里需
  `TORCH_DEVICE_BACKEND_AUTOLOAD=0`，否则 `torch_npu` 扩展加载失败会掩盖成
  `ModuleNotFoundError: Could not import module 'AutoTokenizer'`。

## 7. 在 DSH 里调用

`dsh headless` 的实测（2026-11-12）：用一个临时 overlay 把默认模型换成 FlashEngine 服务，
agent 完整跑通「模型 → tool_calls → DSH 执行工具 → 工具结果回填 → 模型作答」：

```bash
# 1) 起服务（agent 场景建议 16k 上下文；8GB 卡需要 --gpu-mem-util 0.97，KV 92×256=23552 tok）
~/vllm-env/bin/python3 -m scripts.openai_server /path/to/MiniCPM5-2B \
    --host 127.0.0.1 --port 8000 --max-model-len 16384 --max-num-seqs 1 \
    --gpu-mem-util 0.97 --eager --no-think

# 2) overlay（临时文件，不改 profile）：llm-pi-ai 增加 flashengine provider +
#    agent-default-model 指向 MiniCPM5-2B；provider 需要 apiKeyEnv，值随意（服务未开鉴权）
# 3) 跑
FLASHENGINE_API_KEY=sk-local dsh --profile headless \
    --patch /path/to/flashengine.patch.yml --approval-policy never \
    "Use the read tool (tool name: read) to read note.txt, then tell me its exact content."
```

overlay 内容：

```yaml
- id: llm-pi-ai
  config:
    providers:
      flashengine:
        displayName: FlashEngine (local)
        apiKeyEnv: FLASHENGINE_API_KEY
        api: openai-completions
        baseURL: http://127.0.0.1:8000/v1
        defaultContextWindow: 16384
        defaultMaxTokens: 512
        models:
          - id: MiniCPM5-2B
            name: MiniCPM5-2B · FlashEngine
            contextWindow: 16384
            maxTokens: 512
            input: [text]
- id: agent-default-model
  config:
    provider: flashengine
    model: MiniCPM5-2B
```

实测结果：简单任务输出 `HELLO_FLASHENGINE`；读文件任务的事件流是
`tool_call(read) → tool_result → 最终文本`，服务端日志对应两步
`finish=tool_calls`（16 tok / 带 tools=23 / tool_choice=auto）→ 第二步
`finish=stop`（prompt 7162 tok，prefix cache 100% 命中 27/27 块），单步 E2E 约 1.1–1.2s。
Web GUI 用同一段 provider 配置（加在 `$DSH_HOME/profiles/<profile>/cordis.patch.yml` 里
`llm-pi-ai.config.providers` 下），重启 DSH 后在 Models 页给该 provider 填任意 API key
（服务未开 `--api-key` 时不校验）、模型选择器里选 `MiniCPM5-2B · FlashEngine`。

注意：`tool_choice=auto`（DSH 默认）走「不约束、只解析」路径，靠模型原生格式；要强制
约束用 `tool_choice=required`（`scripts/toolcall_bench.py` 默认就是 required）。

## 8. 已知限制

- 约束请求走 eager spec 轮（见 §4），比整轮 spec 图慢；无约束请求仍走最快路径；
- `auto` 模式不约束、只解析：模型没调工具/格式不完整时服务端不做补救
  （`tool_choice=required` 才强制约束）；
- 自由文本区禁止输出"非完整 UTF-8/特殊 token"的碎片，模型偶尔会选到更生僻的 token，
  但不会产出非法字节；
- `deepseek_v3` 协议需要目标 tokenizer 自带对应特殊 token（本机无该系模型，仅实现 + CPU 校验）；
- agent 场景下 2B 模型对 23 个工具的选择能力有限（实测简单 shell 任务会直接文字作答），
  这是模型能力而非协议问题；换更大目标模型（Qwen3-8B/30B 等）或减小工具集更稳。
