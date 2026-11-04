"""Qwen3-ASR 转写基准 / 批量转写脚本。

目标模型：Qwen3-ASR-1.7B（``models/registry.py`` 注册为 ``qwen3_asr``，只支持
``spec=none`` + ``quantization=none`` + CUDA / CoreX / Ascend 310P 三后端）。

口径（与 ``scripts/benchmark.py`` 的扁平 CLI 风格一致）：

* **TTFT** = 从提交请求（``engine.add_request`` 调用前一刻）到该请求首个新 token
  被本进程 ``engine.step()`` 收上来的时刻。粒度 = 一个调度步，含排队与 prefill，
  不含 HTTP / tokenize / 音频 mel 计算（那部分在计时窗之前完成）。
* **单条总耗时** = 该请求从提交到 ``finish`` 的墙钟；批量时各请求并发推进，
  所以单条耗时彼此重叠。
* **RTF** = 墙钟 / 音频秒数。逐条报 ``RTF(mean)``（先算每条 RTF 再平均），批量再报
  ``RTF(aggregate) = 整批墙钟 / Σ音频秒``——批内并发会让它显著小于单条 RTF，
  这是吞吐视角，不能当成"单请求延迟"。
* **TPS** = 该请求生成 token 数 / 该请求墙钟；``TPS(aggregate)`` = Σ生成 token /
  整批墙钟。

音频读取统一走 ``flashengine.asr_processor`` 的 ``load_audio`` / ``decode_wav_bytes``
（抗混叠多相重采样到 16 kHz 单声道，与 HF feature extractor 同口径）；输入只支持
RIFF/WAVE PCM（8/16/24/32-bit），其它格式请先用 ffmpeg 转成 wav。

CER：自实现编辑距离，不依赖 jiwer。归一化 = NFKC → casefold → 去掉所有空白与
Unicode 标点符号；中文按字符计数，英文按字符（空格被删掉，因此 "hello world"
与 "helloworld" 视为同一串）。CER = 编辑距离 / 参考串字符数；参考串为空时记
``None``（不参与汇总）。

用法::

    # 单条
    python3 -m scripts.asr_benchmark /path/to/Qwen3-ASR-1.7B \\
        --wav bench_data/asr/asr_en.wav
    # 目录 + 强制语言 + 参考转写
    python3 -m scripts.asr_benchmark /path/to/Qwen3-ASR-1.7B \\
        --wav-dir bench_data/asr --language Chinese \\
        --reference-jsonl bench_data/asr/reference.jsonl --max-tokens 256
"""

import argparse
import json
import os
import sys
import time
import unicodedata

import numpy as np

AUDIO_SAMPLE_RATE = 16000

# 参考转写 JSONL：每行一个 {"path"/"file"/"wav": <路径或文件名>, "text"/"transcription": <参考>,
# 可选 "language": <语言名>}。路径先按原样匹配，再按 basename 匹配，最后退化为
# 「与 wav 列表同序」——后者只在条数完全一致时启用。
DEFAULT_REFERENCE_JSONL = os.path.join("bench_data", "asr", "reference.jsonl")


# ----------------------------------------------------------------------------
# 音频读取（统一走 flashengine.asr_processor）
# ----------------------------------------------------------------------------
def load_wav(path):
    """读本地 wav → (float32 单声道 16k 波形, 秒数)。"""
    from flashengine.asr_processor import load_audio

    data = np.ascontiguousarray(load_audio(path), dtype=np.float32)
    return data, len(data) / float(AUDIO_SAMPLE_RATE)


def decode_audio_bytes(data):
    """读内存里的 wav 字节（HTTP multipart 上传）→ float32 单声道 16k。"""
    from flashengine.asr_processor import decode_wav_bytes

    return np.ascontiguousarray(decode_wav_bytes(data), dtype=np.float32)


# ----------------------------------------------------------------------------
# 文本归一化 / CER（自实现，不引 jiwer）
# ----------------------------------------------------------------------------
def normalize_text(text):
    """ASR 评分归一化：NFKC、小写、去掉空白与标点（保留字母/数字/汉字）。

    Qwen3-ASR 输出可能带 ``language Chinese<asr_text>`` 标记；调用方应先用
    ``AsrProcessor.parse`` 取 transcription 再送进来。这里额外兜底：若文本里
    还有 ``<asr_text>``，只取它后面的部分。
    """
    if not text:
        return ""
    lower = text.lower()
    marker = "<asr_text>"
    if marker in lower:
        text = text[lower.rindex(marker) + len(marker):]
    text = unicodedata.normalize("NFKC", text).casefold()
    out = []
    for ch in text:
        if ch.isspace():
            continue
        if unicodedata.category(ch).startswith("P"):
            continue
        out.append(ch)
    return "".join(out)


def edit_distance(ref, hyp):
    """Levenshtein 距离，滚动一维数组，O(len(ref)*len(hyp)) 时间 / O(len(hyp)) 空间。"""
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    # 让 hyp 是较短的一维，省内存。
    if len(hyp) > len(ref):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1]


def character_error_rate(reference, hypothesis):
    """参考串为空的返回 None；否则返回 编辑距离/参考字符数（已归一化）。"""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return None
    return edit_distance(ref, hyp) / float(len(ref))


# ----------------------------------------------------------------------------
# processor.build() → 引擎 audio 参数
# ----------------------------------------------------------------------------
def audio_kwargs(built):
    """从 ``AsrProcessor.build()`` 的返回里取出引擎 ``audio=`` 需要的字段。

    设计文档 §3.1/§3.3：build() 返回 input_ids / audio_span / num_audio_tokens /
    mel / valid_frames / language；引擎侧只认去掉 ``input_ids`` 的那份，且会校验
    mel 是二维、valid_frames>=1、audio_span 落在 prompt 内、num_audio_tokens 与
    span 长度一致。这里显式挑字段（而不是 del），避免顺手把 build 新增的调试字段
    传进引擎。
    """
    needed = ("mel", "valid_frames", "audio_span", "num_audio_tokens")
    missing = [k for k in needed if k not in built]
    if missing:
        raise ValueError(f"AsrProcessor.build() 返回缺少字段: {missing}")
    span = tuple(built["audio_span"])
    if len(span) != 2:
        raise ValueError(f"audio_span 应为 (start, end)，实际 {built['audio_span']!r}")
    audio = {"mel": built["mel"],
             "valid_frames": int(built["valid_frames"]),
             "audio_span": span,
             "num_audio_tokens": int(built["num_audio_tokens"])}
    if audio["num_audio_tokens"] != span[1] - span[0]:
        raise ValueError("num_audio_tokens 与 audio_span 长度不一致: "
                         f"{audio['num_audio_tokens']} vs {span[1] - span[0]}")
    return audio


def parse_transcription(processor, text):
    """优先用 processor.parse()（官方输出格式），失败则退回原文。

    返回值 (language, transcription)。processor.parse 契约见设计文档 §3.1。
    """
    if processor is not None:
        try:
            parsed = processor.parse(text)
            lang = parsed.get("language")
            trans = parsed.get("transcription", "")
            if trans is None:
                trans = ""
            return lang, str(trans).strip()
        except Exception as exc:  # parse 是纯字符串处理，出错说明输出格式异常
            print(f"[WARN ] AsrProcessor.parse() 失败，按原文计: {exc}", flush=True)
    return None, (text or "").strip()


# ----------------------------------------------------------------------------
# 参考转写加载
# ----------------------------------------------------------------------------
def _ref_key(entry):
    return entry.get("path") or entry.get("file") or entry.get("wav") or entry.get("audio")


def _ref_text(entry):
    for key in ("text", "transcription", "reference"):
        if key in entry and entry[key] is not None:
            return str(entry[key])
    return None


def load_references(path, wav_paths):
    """读参考 JSONL，返回 {wav 绝对路径: {"text":..., "language":...}}。

    匹配优先级：完整路径 → 规范化路径 → basename（重名取第一个）→ 同序号兜底。
    找不到参考的 wav 不打 CER，只打转写文本。
    """
    refs = {}
    with open(path, "r", encoding="utf-8") as fh:
        rows = []
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} 不是合法 JSON: {exc}") from exc

    by_full, by_base = {}, {}
    for entry in rows:
        key = _ref_key(entry)
        text = _ref_text(entry)
        if not key or text is None:
            continue
        item = {"text": text, "language": entry.get("language")}
        by_full[os.path.normpath(str(key))] = item
        base = os.path.basename(str(key).replace("\\", "/"))
        by_base.setdefault(base, item)

    ordered = list(rows)
    for idx, wav in enumerate(wav_paths):
        norm = os.path.normpath(wav)
        base = os.path.basename(norm.replace("\\", "/"))
        if norm in by_full:
            refs[wav] = by_full[norm]
        elif base in by_base:
            refs[wav] = by_base[base]
        elif idx < len(ordered):
            key = _ref_key(ordered[idx])
            text = _ref_text(ordered[idx])
            # 同序号兜底只在「参考行没有指向别的文件」或条数一致时使用
            if text is not None and (not key or len(rows) == len(wav_paths)):
                refs[wav] = {"text": text, "language": ordered[idx].get("language")}
    return refs


def collect_wavs(args):
    """合并 --wav 与 --wav-dir（目录按文件名排序、递归一层），去重后截断。"""
    wavs = []
    for item in args.wav or []:
        wavs.append(item)
    if args.wav_dir:
        if not os.path.isdir(args.wav_dir):
            raise SystemExit(f"--wav-dir 不存在或不是目录: {args.wav_dir}")
        for name in sorted(os.listdir(args.wav_dir)):
            path = os.path.join(args.wav_dir, name)
            if os.path.isfile(path) and name.lower().endswith((".wav", ".wave")):
                wavs.append(path)
    if not wavs:
        raise SystemExit("必须提供 --wav 或 --wav-dir（至少一个 wav 文件）")

    seen, unique = set(), []
    for path in wavs:
        norm = os.path.normpath(path)
        if norm in seen:
            continue
        seen.add(norm)
        unique.append(path)
    missing = [p for p in unique if not os.path.isfile(p)]
    if missing:
        raise SystemExit("音频文件不存在:\n  " + "\n  ".join(missing))
    if args.num_prompts and args.num_prompts > 0:
        unique = unique[:args.num_prompts]
    return unique


# ----------------------------------------------------------------------------
# 引擎批跑（手动 step，才能拿到逐请求 TTFT）
# ----------------------------------------------------------------------------
def run_batch(engine, requests, sampling_params):
    """提交一批请求并驱动到全部完成。

    requests: [(prompt_ids, audio_dict, item), ...]
    返回 [(generated_ids, ttft_seconds|None, wall_seconds), ...]，顺序与输入一致。

    提交走 ``engine.add_request``（= EngineRequests.add_request）：它会做
    audio 字段校验（mel 二维 / span 在 prompt 内 / num_audio_tokens 对齐）并给
    序列打上 ``skip_prefix_cache``，比直接 ``scheduler.add`` 安全。计时从第一次
    add_request 之前开始，因此同一批里后提交的请求 TTFT 含等待前序提交的时间
    （可忽略）。
    """
    t0 = time.perf_counter()
    seqs = []
    for prompt_ids, audio, _item in requests:
        seqs.append(engine.add_request(prompt_ids, sampling_params, audio=audio))

    first = {seq.seq_id: None for seq in seqs}
    seen = {seq.seq_id: 0 for seq in seqs}
    finished = {seq.seq_id: None for seq in seqs}
    while not engine.is_finished():
        engine.step()
        now = time.perf_counter()
        for seq in seqs:
            n = len(seq.completion_token_ids)
            if n > seen[seq.seq_id]:
                if first[seq.seq_id] is None:
                    first[seq.seq_id] = now - t0
                seen[seq.seq_id] = n
            if finished[seq.seq_id] is None and seq.is_finished:
                finished[seq.seq_id] = now - t0

    eos = engine.scheduler.eos
    out = []
    for seq in seqs:
        t_done = finished[seq.seq_id] or (time.perf_counter() - t0)
        tokens = [t for t in seq.completion_token_ids if t != eos]
        out.append((tokens, first[seq.seq_id], t_done))
    return out


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(
        description="Qwen3-ASR 转写 / 基准（TTFT、RTF、TPS、CER）。"
                    "输入只支持 PCM WAV；脚本按模块方式从仓库根目录运行。")
    parser.add_argument("model", help="本地 Qwen3-ASR 模型目录")
    parser.add_argument("--wav", nargs="+", default=None, metavar="PATH",
                        help="一个或多个 wav 文件")
    parser.add_argument("--wav-dir", default=None, metavar="DIR",
                        help="wav 目录（只扫顶层 *.wav/*.wave，按文件名排序）")
    parser.add_argument("--num-prompts", type=int, default=0,
                        help="最多跑前 N 条音频，0 = 全部（默认）")
    parser.add_argument("--language", default=None,
                        help='强制语言，如 "Chinese" / "English" / "Japanese"（默认自动识别）')
    parser.add_argument("--system-prompt", default=None,
                        help="可选 system prompt（原样传给 AsrProcessor.build）")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="每批并发条数，默认 1（8GB 显存建议保持 1-2）")
    parser.add_argument("--warmup", type=int, default=1,
                        help="预热条数（取前 N 条跑一遍并丢弃统计），默认 1；0 关闭")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="每条最多生成 token 数，默认 256（官方推荐，长音频可调大）")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--reference-jsonl", default=None,
                        help=f"参考转写 JSONL（默认尝试 {DEFAULT_REFERENCE_JSONL}）")
    parser.add_argument("--save-jsonl", default=None,
                        help="把逐条结果（含 CER/RTF/TTFT）写到该 JSONL")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex", "ascend310p"])
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"),
                        help="固定计算类型，默认跟随模型/后端")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="上下文长度，默认 2048（8GB 显存保守值）")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="引擎并发上限，默认 = max(1, --batch-size)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85,
                        help="KV/权重显存占比，默认 0.85（WSL 桌面占 ~1GB 需留余量）")
    parser.add_argument("--no-prefix-cache", action="store_true",
                        help="关闭前缀缓存（音频请求本身会 skip prefix cache）")
    from scripts.memory_options import add_memory_arguments
    add_memory_arguments(parser)
    return parser


def _format_pct(value):
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)


    from scripts.bench_utils import percentile

    wav_paths = collect_wavs(args)
    references = {}
    ref_source = args.reference_jsonl
    if ref_source is None and os.path.isfile(DEFAULT_REFERENCE_JSONL):
        ref_source = DEFAULT_REFERENCE_JSONL
    if ref_source:
        if not os.path.isfile(ref_source):
            raise SystemExit(f"--reference-jsonl 不存在: {ref_source}")
        references = load_references(ref_source, wav_paths)

    # 先给一行反馈：torch/torch_npu 导入在板卡上需 ~20s，期间没有任何输出。
    print("[FlashEngine] Importing runtime (torch / torch_npu / transformers / audio tower, "
          "may take ~20s on Ascend boards) ...", flush=True)
    from flashengine import LLM, SamplingParams
    from flashengine.asr_processor import AsrProcessor

    processor = AsrProcessor.from_pretrained(args.model)

    # 先把所有音频过一遍 processor（mel + prompt），计时窗只包住引擎。
    items = []
    for path in wav_paths:
        waveform, seconds = load_wav(path)
        if seconds <= 0:
            raise SystemExit(f"音频时长为 0: {path}")
        built = processor.build(waveform, language=args.language,
                                system_prompt=args.system_prompt)
        audio = audio_kwargs(built)
        ref = references.get(path)
        items.append({"path": path, "waveform": waveform, "seconds": seconds,
                      "built": built, "audio": audio,
                      "ref": ref["text"] if ref else None,
                      "ref_language": ref.get("language") if ref else None})

    max_num_seqs = args.max_num_seqs or max(1, args.batch_size)
    kwargs = dict(enforce_eager=args.eager, tensor_parallel_size=1, backend=args.backend,
                  max_model_len=args.max_model_len, max_num_seqs=max_num_seqs,
                  max_num_batched_tokens=args.max_num_batched_tokens,
                  enable_chunked_prefill=True, max_num_partial_prefills=1,
                  kvcache_block_size=args.block_size,
                  gpu_memory_utilization=args.gpu_mem_util,
                  enable_prefix_cache=not args.no_prefix_cache)
    from scripts.memory_options import memory_kwargs
    kwargs.update(memory_kwargs(args))
    if args.dtype:
        kwargs["dtype"] = args.dtype
    engine = LLM(args.model, **kwargs)

    sampling = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                              top_k=args.top_k, max_tokens=args.max_tokens)

    print(f"[Config] backend={args.backend} eager={args.eager} "
          f"max_model_len={args.max_model_len} max_num_seqs={max_num_seqs} "
          f"batch_size={args.batch_size} language={args.language or 'auto'} "
          f"max_tokens={args.max_tokens} wavs={len(items)}", flush=True)

    # 预热：复用第一条，让图捕获 / 权重加载后的首个 kernel 冷启动落在统计外。
    if args.warmup > 0 and items:
        warm = items[:max(1, min(args.warmup, len(items)))]
        requests = [(it["built"]["input_ids"], it["audio"], it) for it in warm]
        run_batch(engine, requests, sampling)
        print(f"[Warmup] {len(warm)} 条完成（不计入统计）", flush=True)

    per_item, batch_walls, batch_audio, batch_tokens = [], [], [], []
    cursor = 0
    batch_index = 0
    while cursor < len(items):
        batch = items[cursor:cursor + args.batch_size]
        cursor += len(batch)
        batch_index += 1
        requests = [(it["built"]["input_ids"], it["audio"], it) for it in batch]
        t_batch = time.perf_counter()
        outputs = run_batch(engine, requests, sampling)
        batch_wall = time.perf_counter() - t_batch
        batch_walls.append(batch_wall)
        batch_audio.append(sum(it["seconds"] for it in batch))
        batch_tokens.append(sum(len(tokens) for tokens, _t, _w in outputs))

        print(f"\n--- batch {batch_index}: {len(batch)} 条, 墙钟 {batch_wall:.3f}s ---")
        for it, (tokens, ttft, wall) in zip(batch, outputs):
            text = engine.tokenizer.decode(tokens, skip_special_tokens=True)
            language, transcription = parse_transcription(processor, text)
            ref = it["ref"]
            cer = character_error_rate(ref, transcription) if ref is not None else None
            audio_s = it["seconds"]
            rtf = wall / audio_s if audio_s > 0 else float("nan")
            tps = len(tokens) / wall if wall > 0 else float("nan")
            ttft_ms = ttft * 1000.0 if ttft is not None else None
            row = {"path": it["path"], "audio_seconds": audio_s,
                   "prompt_tokens": len(it["built"]["input_ids"]),
                   "generated_tokens": len(tokens), "ttft_ms": ttft_ms,
                   "wall_seconds": wall, "rtf": rtf, "tps": tps,
                   "text": text, "transcription": transcription,
                   "language": language, "reference": ref, "cer": cer}
            per_item.append(row)
            ttft_text = f"{ttft_ms:.1f}ms" if ttft_ms is not None else "n/a"
            print(f"[{len(per_item)}/{len(items)}] {it['path']}  audio={audio_s:.2f}s "
                  f"prompt={row['prompt_tokens']}tok gen={row['generated_tokens']}tok "
                  f"TTFT={ttft_text} wall={wall:.3f}s RTF={rtf:.4f} TPS={tps:.2f}")
            print(f"  lang: {language or 'n/a'}"
                  + (f" (ref={it['ref_language']})" if it["ref_language"] else ""))
            print(f"  text: {transcription!r}")
            if ref is not None:
                print(f"  ref : {ref!r}")
                print(f"  CER : {_format_pct(cer)}")

    # ---- 汇总 ----
    ttfts = [row["ttft_ms"] for row in per_item]
    walls = [row["wall_seconds"] for row in per_item]
    rtfs = [row["rtf"] for row in per_item]
    tps = [row["tps"] for row in per_item]
    cers = [row["cer"] for row in per_item if row["cer"] is not None]
    total_wall = sum(batch_walls)
    total_audio = sum(batch_audio)
    total_tokens = sum(batch_tokens)
    corpus_edits = corpus_ref = 0
    for row in per_item:
        if row["reference"] is None:
            continue
        ref_norm = normalize_text(row["reference"])
        if ref_norm:
            corpus_ref += len(ref_norm)
            corpus_edits += edit_distance(ref_norm, normalize_text(row["transcription"]))

    def _fmt_p(values, fmt, suffix=""):
        values = [v for v in values if v is not None]
        if not values:
            return "n/a"
        return (f"mean={format(_mean(values), fmt)}{suffix} "
                f"p50={format(percentile(values, 50), fmt)}{suffix} "
                f"p95={format(percentile(values, 95), fmt)}{suffix}")

    print("\n===== ASR Benchmark Result =====")
    print(f"model={args.model} backend={args.backend} eager={args.eager} "
          f"batch_size={args.batch_size} language={args.language or 'auto'} "
          f"items={len(per_item)} batches={len(batch_walls)}")
    print(f"Total audio={total_audio:.2f}s  total wall={total_wall:.3f}s  "
          f"generated tokens={total_tokens}")
    print(f"TTFT: {_fmt_p(ttfts, '.1f', 'ms')}")
    print(f"Request wall: {_fmt_p(walls, '.3f', 's')}")
    print(f"RTF(mean per request)={_format_pct(_mean(rtfs))}  "
          f"RTF(aggregate = Σwall/Σaudio)={total_wall / total_audio if total_audio else float('nan'):.4f}")
    print(f"TPS(mean per request)={_mean(tps) if tps else float('nan'):.2f}  "
          f"TPS(aggregate)={total_tokens / total_wall if total_wall else float('nan'):.2f}")
    if cers:
        print(f"CER(mean per item)={_format_pct(_mean(cers))}  "
              f"CER(corpus)={_format_pct(corpus_edits / corpus_ref if corpus_ref else None)} "
              f"on {len(cers)} 条有参考的音频")
    elif references:
        print("CER: 参考 JSONL 已加载，但没有条目匹配到本次音频（检查 path/file 字段）")
    else:
        print("CER: 无参考转写（--reference-jsonl），跳过")

    if args.save_jsonl:
        with open(args.save_jsonl, "w", encoding="utf-8") as fh:
            for row in per_item:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[Saved] {args.save_jsonl}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
