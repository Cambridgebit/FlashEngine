# Qwen3-ASR-1.7B 语音转写（实验性）

Qwen3-ASR-1.7B 是 Qwen 官方语音识别模型（30 种语言 + 22 种中文方言），
本仓通过目标模型适配器 `qwen3_asr` 接入其 Thinker（文本解码）+ 音频塔（mel → 音频隐层）
结构，支持**离线命令行转写**与 **OpenAI 兼容的 `/v1/audio/transcriptions` 端点**。

当前支持面（与 [`docs/SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md) 一致）：

- 目标模型 `qwen3_asr`，`spec_method = none`（**不接受** `--spec-model`）。
- 量化：仅 `quantization = none`（fp16/bf16；W8A8/Quark 未接入，启动即拒绝）。
- 后端：`cuda` / `corex` / `ascend310p` 三后端同一套接口。
- 张量并行恒为 TP=1（`audio_input` 目标在 `config.py` 强制；传 `tensor_parallel_size>1` 报错）。
- 一次请求一条音频；音频请求**按请求跳过前缀缓存**（`skip_prefix_cache`，音频嵌入不能跨请求复用）。

> 实机验收状态以本文件末尾「验证记录」为准；未跑过的组合明确标注「未验证」，
> 不因代码路径存在而视为已验收。

## 1. 模型准备

直接用官方原始 checkpoint 目录，**不需要转换脚本、不能用 HuggingFace hub id**：

```bash
/mnt/d/Materials/AVIC/models/Qwen3-1.7B-ASR/
├── config.json              # model_type=qwen3_asr, architectures=[Qwen3ASRForConditionalGeneration]
├── tokenizer.json / tokenizer_config.json / chat_template.json
├── thinker/...              # 文本解码器权重（thinker.* 键名由适配器消费）
└── audio_tower/...          # 音频塔权重
```

- 语音前端（mel 滤波器、STFT、重采样、chat template 渲染、`<|audio_pad|>` 展开）由
  `flashengine.asr_processor.AsrProcessor` 在 **CPU** 上完成，不依赖 transformers 的
  `Qwen3ASRModel`（推理路径只用本仓引擎）。
- 模型目录必须是本地路径；`--model` / server 的位置参数都传目录。
- CUDA 上 bf16 是默认计算 dtype（flash-attn 支持 bf16/fp16）；310P 由后端强制 fp16。

CPU 侧接口自检（不加载引擎，秒级）：

```bash
cd /path/to/FlashEngine
PYTHONPATH=$PWD python3 - <<'PY'
from flashengine.asr_processor import AsrProcessor
p = AsrProcessor.from_pretrained("/mnt/d/Materials/AVIC/models/Qwen3-1.7B-ASR")
b = p.build("bench_data/asr/asr_zh.wav")           # 也可传 np.float32 16k 单声道波形
print(len(b["input_ids"]), b["audio_span"], b["num_audio_tokens"], b["mel"].shape, b["valid_frames"])
print(p.parse("language Chinese<asr_text>你好，世界"))
PY
```

## 2. 命令行转写：`scripts/asr_benchmark.py`

单条（默认 greedy、`max_tokens=256`、WSL 8GB 显存友好的 `max_model_len=2048`）：

```bash
cd /mnt/d/Materials/AVIC/FlashEngine
PYTHONPATH=$PWD ~/vllm-env/bin/python3 -m scripts.asr_benchmark \
  /mnt/d/Materials/AVIC/models/Qwen3-1.7B-ASR \
  --backend cuda --gpu-mem-util 0.85 --max-model-len 2048 \
  --wav bench_data/asr/asr_en.wav bench_data/asr/asr_zh.wav
```

目录批量 + 强制语种 + 与参考转写算 CER：

```bash
PYTHONPATH=$PWD ~/vllm-env/bin/python3 -m scripts.asr_benchmark \
  /mnt/d/Materials/AVIC/models/Qwen3-1.7B-ASR \
  --backend cuda --wav-dir bench_data/asr \
  --reference-jsonl bench_data/asr/reference.jsonl \
  --language Chinese --save-jsonl /tmp/asr_result.jsonl
```

多条音频并发（`--batch-size` 同时入队；显存不够时降到 1）：

```bash
PYTHONPATH=$PWD ~/vllm-env/bin/python3 -m scripts.asr_benchmark MODEL \
  --backend cuda --batch-size 4 --max-num-seqs 4 \
  --gpu-mem-util 0.85 --max-model-len 2048 \
  --wav a.wav b.wav c.wav d.wav
```

主要参数：

| 参数 | 说明 |
|---|---|
| `--wav` / `--wav-dir` | 音频文件（可多个）或目录顶层 `*.wav`；`--num-prompts` 截断 |
| `--language` | 强制语种（如 `Chinese`/`English`）；缺省让模型自动检测 |
| `--system-prompt` | 可选 system 提示（热词/上下文） |
| `--batch-size` | 每批并发请求数，默认 1；`--max-num-seqs` 缺省跟随它 |
| `--warmup` | 预热批次数（默认 1，不计指标） |
| `--max-tokens` | 每条音频输出上限，默认 256（官方推荐） |
| `--reference-jsonl` | 参考转写；缺省自动用 `bench_data/asr/reference.jsonl`（存在时） |
| `--save-jsonl` | 把每条转写/指标落盘 |
| `--eager` | 关闭 CUDA Graph（小显存排障用） |
| `--dtype` | `bfloat16` / `float16`；缺省跟随后端 |
| `--backend` | `auto` / `cuda` / `corex` / `ascend310p` |
| `--gpu-mem-util` | 默认 0.85（本机 8GB 卡贴地分配易 OOM） |
| `--max-model-len` | 默认 2048；音频 prompt 通常 100~400 token，长音频务必分块 |

### 指标口径

- **TTFT**：从请求提交到**第一个生成 token** 被调度完成的时间（本脚本手动
  `add_request` + 循环 `step()` 采点，含 prefill，不含 HTTP 传输）。
- **总墙钟 / Request wall**：该请求从提交到 finish 的墙钟。
- **RTF** = 请求墙钟 / 音频秒数（<1 表示快于实时）；同时报逐请求平均与
  `Σwall / Σaudio` 聚合口径。
- **TPS** = 生成 token 数 / 请求墙钟（含 prefill 与首 token；与 `benchmark.py` 的
  "Output token throughput" 同口径），逐请求平均之外另报聚合值（Σtoken / 整批墙钟）。
- **CER**：自实现字符级 Levenshtein（不引依赖）。归一化 = `NFKC` → `casefold`
  → 去掉所有空白与 Unicode 标点（`P*`）→ 对**字符**计算编辑距离 / 参考字符数；
  中文、英文统一同一口径（英文按字符而非词，报告里注明）。参考为空时记 `n/a`。
  输出若带 `language <X><asr_text>` 前缀，先由 `AsrProcessor.parse()` 剥掉再算。

- **音频解码/重采样**统一走 `flashengine.asr_processor` 的 `load_audio(path)` /
  `decode_wav_bytes(data)`：带抗混叠低通的多相重采样（48k→16k 直接抽点会把 8kHz 以上
  能量折叠回带内，mel 最大差 0.1 量级）。脚本内保留纯标准库实现仅作导入兜底；
  server 端 multipart 的 `file` 字节走同一个 `decode_wav_bytes`，两边口径一致。

### 参考转写 JSONL

```json
{"file": "bench_data/asr/asr_en.wav", "text": "the reference transcript ...", "language": "English"}
{"file": "bench_data/asr/asr_zh.wav", "text": "参考转写文本", "language": "Chinese"}
```

`file` 为相对仓库根（或绝对）路径；也兼容 `path`/`wav`/`audio` 键与
`text`/`transcription`/`reference` 键。只要求条目数与 `--wav` 序列能对上：
先按完整路径、再按文件名匹配，最后仅在条目数相同或条目无 `file` 键时按顺序兜底。
缺参考文件时脚本照常转写，只是不打印 CER。

## 3. OpenAI 兼容端点

```bash
cd /mnt/d/Materials/AVIC/FlashEngine
PYTHONPATH=$PWD ~/vllm-env/bin/python3 -m scripts.openai_server \
  /mnt/d/Materials/AVIC/models/Qwen3-1.7B-ASR \
  --backend cuda --gpu-mem-util 0.85 --max-model-len 2048 \
  --port 8000 --api-key sk-test
```

启动日志会有一行（目标模型带 `audio_input` 时才初始化 processor）：

```
[INFO ] ASR transcription enabled: POST /v1/audio/transcriptions (max_tokens=256, upload<=32MB, response_format=json|text)
```

curl（JSON 默认）：

```bash
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-test" \
  -F file=@bench_data/asr/asr_zh.wav \
  -F model=Qwen3-ASR-1.7B \
  -F language=Chinese \
  -F response_format=json
# {"text": "..."}
```

纯文本返回（`response_format=text`）：

```bash
curl -s http://127.0.0.1:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-test" \
  -F file=@bench_data/asr/asr_en.wav -F response_format=text
```

Python OpenAI SDK（与官方 vLLM 转写端点同形）：

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="sk-test")
with open("bench_data/asr/asr_zh.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="Qwen3-ASR-1.7B", file=f).text)
```

- `multipart/form-data` 只允许标准库解析（`email.parser.BytesParser`；Python 3.12
  已移除 `cgi.FieldStorage`）。字段：`file`（必填 wav）、`model`（校验存在但忽略）、
  `language`（可选，`Chinese`/`English`/语种名）、`response_format`（`json` 默认 / `text`）。
- 请求体上限默认 32MB（`--asr-max-upload-mb`），输出上限 `--asr-max-tokens`（默认 256）。
- 与 chat 共用同一条引擎单线程驱动、同一 `--api-key` 鉴权、同样的 `[HTTP ]`/`[REQ  ]`/`[DONE ]`
  日志；`/metrics` 与 `/v1/stats` 新增
  `flashengine_transcription_requests_total`、`flashengine_transcription_errors_total`、
  `flashengine_transcription_audio_seconds_total`。
- 不支持：流式转写（`stream=true` 忽略，整段返回）、时间戳/词级对齐、多音频单请求、
  `prompt` 热词（用 `language` 与命令行 `--system-prompt` 代替）、`logprobs`。
- 非 ASR 模型启动时端点仍在，但返回 400 并提示目标模型没有音频输入。

**实测（2026-11-04，CUDA / RTX 4070 8GB，统一重采样后的最终版本）**：

| 请求 | 结果 |
|---|---|
| zh + `language=Chinese` + `response_format=json` | 200 `{"text": "甚至出现交易几乎停滞的情况。"}` |
| zh + `response_format=text` | 200 纯文本 `甚至出现交易几乎停滞的情况。` |
| en 不传 `language`（自动检测） | 200 整句英文（detected=English，188 字符） |
| 不带 `Authorization` | **401** `{"error": {"message": "invalid api key", "type": "invalid_request_error"}}` |
| 对 Qwen3-1.7B 文本模型打同一端点 | **400** `current model does not support audio transcription (only Qwen3-ASR models expose /v1/audio/transcriptions)`；同进程 chat 仍 200 |

`/metrics` 计数（3 次成功转写后）：`flashengine_transcription_requests_total 3`、
`flashengine_transcription_errors_total 0`、
`flashengine_transcription_audio_seconds_total 23.459125`。`[DONE ]` 行里首个转写请求
TTFT 3830ms（引擎冷启动，含首请求 kernel 初始化），其后 zh 59ms（25.3 tok/s）、
en 74ms（29.8 tok/s）；`[INFO ]` 行会带 `asr audio=..s lang=.. detected=.. chars=..`。
注意 `--no-ui` 会连同 `/v1/stats` 一起关闭（与 chat 路径相同），此时 `curl /v1/stats`
为空属预期，不是转写端点的问题。

## 4. 三后端 Runbook

### 4.1 CUDA（本机 WSL2 / RTX 4070 Laptop 8GB）——✅ 已实测（2026-11-04）

前置：`nvidia-smi` 正常、`~/vllm-env` 内 torch 带 CUDA、flash-attn 可用；模型在
`/mnt/d/Materials/AVIC/models/Qwen3-1.7B-ASR`。

```bash
cd /mnt/d/Materials/AVIC/FlashEngine
# 单条/批量转写 + CER（本次验收用的就是这条命令）
PYTHONPATH=$PWD ~/vllm-env/bin/python3 -m scripts.asr_benchmark \
  /mnt/d/Materials/AVIC/models/Qwen3-1.7B-ASR \
  --backend cuda --eager --gpu-mem-util 0.85 --max-model-len 1024 --max-num-seqs 1 \
  --wav bench_data/asr/asr_zh.wav bench_data/asr/asr_en.wav --max-tokens 256 \
  --reference-jsonl bench_data/asr/reference.jsonl

# 服务端 curl（--no-ui 会连同 /v1/stats 一起关；要看 stats 去掉 --no-ui）
PYTHONPATH=$PWD ~/vllm-env/bin/python3 -m scripts.openai_server \
  /mnt/d/Materials/AVIC/models/Qwen3-1.7B-ASR \
  --backend cuda --eager --gpu-mem-util 0.85 --max-model-len 1024 --max-num-seqs 1 \
  --port 8000 --api-key sk-test --no-ui &
curl -s -H "Authorization: Bearer sk-test" http://127.0.0.1:8000/v1/audio/transcriptions \
  -F file=@bench_data/asr/asr_zh.wav -F language=Chinese
```

实测结果（eager / `max_model_len=1024` / `max_num_seqs=1` / bf16；两次独立运行）：

| 音频 | 时长 | 生成 tok | TTFT | 单条墙钟 | RTF | TPS | 转写 | CER |
|---|---|---|---|---|---|---|---|---|
| asr_zh.wav | 4.20s | 10 | 67.5–89.6ms | 0.415–0.484s | 0.099–0.115 | 20.7–24.1 | `甚至出现交易几乎停滞的情况。` | 0.0% |
| asr_en.wav | 15.05s | 48 | 90.2–92.0ms | 1.78–2.08s | 0.118–0.139 | 23.0–27.0 | `Uh huh. Oh yeah, yeah. He wasn't even that big when I started listening to him. But and his solo music didn't do overly well. But he did very well when he started writing for other people.` | 0.0% |
| 合计 | 19.26s | 58 | mean 78.8–90.8 / p95 89.0–91.9ms | mean 1.10–1.28s | **0.114–0.133**（聚合） | 22.6–26.4（聚合） | — | **0.0% / 0.0%**（mean/corpus） |

- RTF 是「墙钟 / 音频秒数」；小音频（zh 4.2s）被固定开销拉高，长音频（en 15s）更接近稳态。
- **精简后复测**（删掉验收脚本与兜底分支，同一命令）：zh TTFT 82.3ms / RTF 0.1156 / TPS 20.58；en TTFT 109.6ms / RTF 0.1477 / TPS 21.59；合计 TTFT mean 96.0ms / p95 108.3ms、RTF(聚合) 0.1407、TPS(聚合) 21.41、CER 0.0% / 0.0%——转写文本与精简短一致，CER 不变。
  区间来自两次独立运行的 GPU 频率/热态波动（4070 Laptop 持续负载会降频，属预期，勿当回归）。
- 非 eager（`--eager` 去掉、`max-model-len 2048`）同机稳态：zh TTFT 100.6ms / RTF 0.0588 /
  TPS 40.5；en TTFT 81.2ms / RTF 0.0522 / TPS 61.1——但首个请求含图捕获/冷启动（wall 6.48s），
  先跑一条 warmup 再采数。
- 参考转写来源 = transformers 自带的 Qwen3ASR 实现（`thinker.*` 权重映射后 greedy 解码），与引擎输出只差标点
  （引擎 en 用 `.`、HF 参考用 `,`），归一化去标点后 CER=0。
- prompt/token 规模：zh prompt 70 tok（音频 token 55）、en prompt 211 tok（音频 token 196）；
  每 1 秒音频 ≈ 12.5 个音频 token，故 30s 音频 ≈ 375 音频 token + 模板，仍在 1024 窗口内。

8GB 显存注意：默认 `--gpu-mem-util 0.85 --max-model-len 2048 --max-num-seqs 1`；
OOM 时依次降 `--gpu-mem-util 0.8`、加 `--eager`、`--batch-size 1`。桌面/WSL 显存波动
导致 KV 块数漂移属预期（本次 eager 配置 KV 55×256、非 eager 46×256）。

### 4.2 Ascend 310P——**代码路径已具备，未在真机验证**

用户已明确 **310P 与 CoreX 当前都不可用**，本轮只保证代码路径齐全、不做真机验收；
因此本节不给任何主机名/路径占位，也不声称已跑过。就代码而言：

- `qwen3_asr` 的音频塔是**纯 torch 前向**（不调用 NPU 专用算子），mel/prompt 由 CPU 侧
  `asr_processor` 生成；引擎侧每请求一次音频塔前向 + 普通文本 decode，与后端无关。
- 整条链路不依赖 flash-attn / ixformer / SDPA 等特定 attention 实现，也没有新增第三方
  依赖（asr_processor 顶层只 import numpy）；后端差异全部走既有能力位。
- 310P 的既有硬约束照旧适用于该请求：仅 fp16；`kvcache_block_size` 只接受 {64,128}
  （脚本默认 256，后端 `normalize_config` 会覆盖并打印）；TP=1；显存按 util 贴地分配，
  需要给图捕获留余量。
- ASR 无投机、无量化，不需要 `--spec-model` / W8A8 相关参数；前缀缓存对该请求被
  `skip_prefix_cache` 关闭，因此不依赖 SplitFuse 前缀命中路径。
- 机器可用时直接跑同一条 `scripts.asr_benchmark`（模型目录换 Qwen3-ASR、加
  `--backend ascend310p` 和 `--wav`，其余参数不变）；**本轮未执行**。

**状态：未在真机验证**（用户已确认本轮 310P/CoreX 都不可用，只加代码路径；文档不填未产生过的数字）。

### 4.3 CoreX（Iluvatar）——**代码路径已具备，未在真机验证**

同样按「只加代码」处理：

- 音频塔为纯 torch 前向、mel/prompt 在 CPU 侧生成；CoreX 上 eager decode 慢、图是刚需，
  但 ASR 的音频塔只做一次 prefill 前向，不受 decode 图收益影响，文本 decode 仍走通用图/eager 路径。
- 前置条件与 Qwen3 文本模型一致：Iluvatar 驱动 + 厂商提供的 `ixformer`（仓库 `corex` extra）、
  Python 3.10 环境、32GB 级板卡；ASR 本身不调用 ixformer 算子、无需额外依赖或量化支持。
- 机器可用时直接跑同一条 `scripts.asr_benchmark`（`--backend corex --gpu-mem-util 0.85`，
  模型目录换 Qwen3-ASR、再加 `--wav`）；**本轮未执行**。

**状态：未在真机验证**（无 CoreX 机器；代码路径已具备，文档不填未产生过的数字）。

## 5. 已知限制

1. **单条音频/请求**：一次 `build()` 只处理一条波形；README 音频作为一段连续 span 注入。
2. **30 秒分块**：模型按 30s 窗口设计；超长音频请自行切段（建议 ≤30s/段，静音处切、
   重叠 0.2~0.5s），逐段转写后拼接。脚本与 server 在 prompt 超出 `--max-model-len`
   时会 400/报错提示分块，不会静默截断。分块示例：

   ```python
   import numpy as np, soundfile as sf  # 仅示意；仓库脚本自带 stdlib wav 解码
   from flashengine.asr_processor import AsrProcessor, load_audio
   p = AsrProcessor.from_pretrained(MODEL)
   wav = load_audio("long.wav")               # np.float32 16k mono
   for i in range(0, len(wav), 16000 * 30):
       seg = wav[i:i + 16000 * 30]
       req = p.build(seg)
       # 交给 llm.generate([req["input_ids"]], sp, audio=[{k: req[k] for k in
       #   ("mel", "valid_frames", "audio_span", "num_audio_tokens")}])
   ```

3. **TP=1**：音频塔与解码器均不做张量并行。
4. **前缀缓存不适用于音频**：音频请求带独立 `mel`/embedding，引擎对该请求
   `skip_prefix_cache`；对同一条音频重复请求也不会命中。
5. **投机/量化不支持**：`spec_method` 仅 `none`，量化仅 `none`；显式开 `--spec-model`
   或用量化检查点会在启动时被干净拒绝。
6. **无语种概率/时间戳/多说话人**；`language` 只做强制/自动检测二选一。
7. **server 无流式转写**：`/v1/audio/transcriptions` 为非流式整段返回（与官方 vLLM
   行为一致）；chat 端点的 SSE 不受影响。
8. 输出可能带 `language <X><asr_text>` 前缀，命令行与 server 均会剥离后再返回正文
   （`AsrProcessor.parse`）；若解析失败则原样返回并打 `[WARN ]`。

## 6. 排障

| 现象 | 处理 |
|---|---|
| CUDA OOM / KV 块分不出来 | 降 `--gpu-mem-util 0.8`、`--max-model-len 1024`、`--eager`、`--batch-size 1` |
| 310P 报 block size / dtype 不支持 | 后端会把 256 覆盖为 128；确认模型为 fp16 |
| 转写为空/乱码 | 确认 wav 为单声道/16k（脚本会重采样）、时长 ≤30s、`--language` 是否强制错 |
| server 400 "does not support audio" | 目标模型不是 `qwen3_asr`（无 `audio_input`），换 ASR 模型启动 |
| server 400 "prompt 超长" | 音频太长，按 30s 分块或调大 `--max-model-len`（8GB 卡不建议 >2048） |
| CER 偏高 | 先确认语言是否自动检测正确（`--language` 强制），再看是否需要文本归一化口径 |

## 7. 验证记录

| 项目 | 状态 | 说明 |
|---|---|---|
| CPU processor `build()`（asr_en/asr_zh） | ✅ 已自测 | prompt/span/num_tokens/mel 形状与 pad run 均符合契约（2026-11-04） |
| CUDA 端到端转写（asr_en/asr_zh） | ✅ 已实测 | 2026-11-04 本机 4070 8GB：TTFT mean 78.8–96.0ms、RTF 聚合 0.114–0.141、TPS 聚合 21.4–26.4、**CER 0.0%/0.0%（mean/corpus）**；数字与口径见 §4.1 |
| CUDA server `/v1/audio/transcriptions` curl | ✅ 已实测 | json/text/自动检测/401 鉴权/非 ASR 模型 400 全通过；`flashengine_transcription_requests_total 3`、`errors 0`、`audio_seconds_total 23.459125` |
| Ascend 310P | ❌ 未在真机验证 | 代码路径已具备（纯 torch 音频塔 + 通用引擎路径）；310P/CoreX 本轮不可用，未产生任何真机数字 |
| CoreX | ❌ 未在真机验证 | 代码路径已具备（同上）；无 CoreX 机器，未产生任何真机数字 |

详细设计（processor 契约、引擎集成点、分工）见
[`docs/QWEN3_ASR_DESIGN.md`](QWEN3_ASR_DESIGN.md)；支持矩阵见
[`docs/SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md)。
