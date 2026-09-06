#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SSE 流式推理服务基准客户端（仅标准库，无第三方依赖）。

对比多个 OpenAI 兼容 /chat/completions 流式服务的吞吐与首字延迟。
统计口径：
- TTFB：请求发出 → 第一个「非空」content 增量（仅含 role 的首块不计入）。
- wall：第一个非空 content 增量 → 收到 "data: [DONE]"。
- TPS：usage.completion_tokens / wall；usage 缺失则该请求记为失败。

用法示例：
    python3 -m scripts.stream_bench --service vllm=http://127.0.0.1:8000/v1 \
        --service nano=http://127.0.0.1:8001/v1 --rounds 2 --warmup --json-out out.json
离线自测（无需任何服务）：
    python3 -m scripts.stream_bench --selftest
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import time
import urllib.request
from typing import Any, Callable, Iterable, List, Optional, Tuple

SYSTEM_PROMPT = "你是一个简洁专业的中文技术助手。"

# 三条固定的中文提示词，均预期产出 300-500 token 的内容性回答
PROMPTS = [
    "请写一篇约 400 字的技术短文，主题是「大模型推理为什么需要连续批处理（continuous batching）」："
    "它解决了静态批处理的什么问题、对吞吐和首字延迟分别有什么影响、实现上有哪几个关键机制。",
    "请解释投机解码（speculative decoding）的原理：draft 模型如何起草 k 个 token、"
    "目标模型如何用一次前向并行验证、接受率与加速比的关系，以及为什么验证阶段一次处理多个 token 不破坏因果性。",
    "请用 Python 写一个快速排序函数（带中文注释），然后逐步解释分区过程、平均与最坏时间复杂度，并和归并排序做取舍对比。",
]

WARMUP_MAX_TOKENS = 32  # 热身请求的 max_tokens（结果不计入统计）


class RequestAbort(Exception):
    """单次请求因响应不符合口径而中止（usage 缺失 / 无内容增量 / 流未正常结束）。"""


# ---------------- SSE 解析 ----------------

def consume_sse(lines: Iterable[Any], t0: Optional[float] = None,
                clock: Callable[[], float] = time.perf_counter) -> dict:
    """逐行解析 OpenAI 兼容 SSE 流。

    lines 为字节行迭代（HTTP 响应 / io.BytesIO 均可）；clock 可注入假时钟供离线自测。
    返回 dict(content, usage, done, ttft_s, wall_s, n_chunks)。
    """
    if t0 is None:
        t0 = clock()
    parts: List[str] = []
    usage: Optional[dict] = None
    done = False
    t_first: Optional[float] = None
    t_done: Optional[float] = None
    n_chunks = 0
    for raw in lines:
        line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
        line = line.rstrip("\r\n")
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload:
            continue
        if payload == "[DONE]":
            t_done = clock()
            done = True
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue  # 非法行跳过，不影响后续事件
        n_chunks += 1
        now = clock()
        if chunk.get("usage"):
            usage = chunk["usage"]
        choices = chunk.get("choices") or []
        if choices:
            piece = (choices[0].get("delta") or {}).get("content")
            if isinstance(piece, str) and piece:
                parts.append(piece)
                if t_first is None:
                    t_first = now
    return {
        "content": "".join(parts),
        "usage": usage,
        "done": done,
        "ttft_s": None if t_first is None else t_first - t0,
        "wall_s": None if (t_first is None or t_done is None) else t_done - t_first,
        "n_chunks": n_chunks,
    }


def check_and_metrics(streamed: dict) -> dict:
    """校验口径并计算 ttft/wall/TPS；不符合则抛 RequestAbort（该请求记为失败）。"""
    usage = streamed["usage"]
    if not usage or usage.get("completion_tokens") is None:
        raise RequestAbort("缺少 usage.completion_tokens（服务端未按 stream_options.include_usage 返回用量）")
    if streamed["ttft_s"] is None:
        raise RequestAbort("流中没有非空 content 增量")
    if not streamed["done"] or streamed["wall_s"] is None:
        raise RequestAbort("流未以 data: [DONE] 结束")
    completion_tokens = int(usage["completion_tokens"])
    if streamed["wall_s"] <= 0:
        raise RequestAbort(f"wall 时间非正（{streamed['wall_s']!r}）")
    return {
        "ttft_ms": streamed["ttft_s"] * 1000.0,
        "wall_s": streamed["wall_s"],
        "completion_tokens": completion_tokens,
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "tps": completion_tokens / streamed["wall_s"],
    }


# ---------------- 请求 ----------------

def stream_request(url: str, model: str, prompt: str, *, temperature: float,
                   max_tokens: int, timeout: int) -> Tuple[dict, int]:
    """发一次流式 chat completion；返回 (metrics, content_chars)，失败抛异常。"""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + "/chat/completions", data=body, method="POST",
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    t_start = time.perf_counter()  # TTFB 从连接发起前计起
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        streamed = consume_sse(resp, t0=t_start)
    return check_and_metrics(streamed), len(streamed["content"])


def fetch_model_name(url: str, timeout: int) -> Optional[str]:
    """GET {url}/models → data[0].id；探测失败一律返回 None（回退 --model）。"""
    req = urllib.request.Request(url.rstrip("/") + "/models",
                                 headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("data") or []
        return models[0].get("id") if models else None
    except Exception:  # 探测失败不算致命，交给 --model 回退
        return None


# ---------------- 单请求执行与统计 ----------------

def _record_base(name: str, url: str, model: Optional[str],
                 rnd: int, idx: int, warmup: bool) -> dict:
    return {"service": name, "url": url, "model": model, "round": rnd,
            "prompt_index": idx, "warmup": warmup, "error": None,
            "ttft_ms": None, "wall_s": None, "tps": None,
            "completion_tokens": None, "prompt_tokens": None,
            "content_chars": None, "elapsed_s": None}


def run_one(name: str, url: str, model: str, rnd: int, idx: int,
            prompt: str, args: argparse.Namespace) -> dict:
    """执行单个请求：任何异常都记录在记录里，不中断其余请求。"""
    rec = _record_base(name, url, model, rnd, idx, False)
    t0 = time.perf_counter()
    try:
        metrics, chars = stream_request(
            url, model, prompt, temperature=args.temperature,
            max_tokens=args.max_tokens, timeout=args.timeout)
        rec.update(metrics, content_chars=chars)
    except Exception as exc:  # 基准客户端：单请求失败仅记录并继续
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["elapsed_s"] = time.perf_counter() - t0
    return rec


def summarize(name: str, url: str, model: Optional[str], records: List[dict]) -> dict:
    """按口径汇总单个服务：只统计非热身且成功的请求。"""
    ok = [r for r in records if not r["warmup"] and r["error"] is None]
    tps = [r["tps"] for r in ok]
    return {
        "name": name, "url": url, "model": model,
        "n_ok": len(ok),
        "n_err": sum(1 for r in records if r["error"] is not None),
        "median_tps": statistics.median(tps) if tps else None,
        "min_tps": min(tps) if tps else None,
        "max_tps": max(tps) if tps else None,
        "mean_ttft_ms": statistics.mean(r["ttft_ms"] for r in ok) if ok else None,
        "total_completion_tokens": sum(r["completion_tokens"] for r in ok),
    }


def print_report(stats_list: List[dict]) -> None:
    print("\n" + "=" * 72)
    print("== 各服务明细 ==")
    for s in stats_list:
        print(f"[{s['name']}] {s['url']}  model={s['model'] or '?'}")
        if not s["n_ok"]:
            print(f"  无成功请求（失败 {s['n_err']} 次）")
            continue
        print(f"  成功 {s['n_ok']} / 失败 {s['n_err']}  "
              f"TPS 中位 {s['median_tps']:.1f}（min {s['min_tps']:.1f} / max {s['max_tps']:.1f}）  "
              f"平均 TTFT {s['mean_ttft_ms']:.0f} ms  "
              f"completion tokens 合计 {s['total_completion_tokens']}")
    ranked = sorted((s for s in stats_list if s["median_tps"] is not None),
                    key=lambda s: s["median_tps"], reverse=True)
    if ranked:
        print("\n== 对比表（按中位 TPS 降序）==")
        print(f"{'排名':>4}  {'服务':<12} {'中位TPS':>8} {'minTPS':>8} {'maxTPS':>8} "
              f"{'TTFT ms':>8} {'tokens':>8} {'错误':>4}")
        for i, s in enumerate(ranked, 1):
            print(f"{i:>4}  {s['name']:<12} {s['median_tps']:>8.1f} {s['min_tps']:>8.1f} "
                  f"{s['max_tps']:>8.1f} {s['mean_ttft_ms']:>8.0f} "
                  f"{s['total_completion_tokens']:>8} {s['n_err']:>4}")


# ---------------- 离线自测 ----------------

def _canned_sse(events: List[str]) -> bytes:
    """把 SSE 事件列表编成字节流（每事件一个 data 行 + 空行）。"""
    return "".join(f"data: {e}\n\n" for e in events).encode("utf-8")


# 内置样例流：role 块 → 内容增量 → finish → usage 块 → [DONE]
CANNED_MAIN = _canned_sse([
    '{"choices":[{"delta":{"role":"assistant","content":""}}]}',
    '{"choices":[{"delta":{"content":"快速"}}]}',
    '{"choices":[{"delta":{"content":"排序测试"}}]}',
    '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
    '{"choices":[],"usage":{"prompt_tokens":50,"completion_tokens":100,"total_tokens":150}}',
    "[DONE]",
])
CANNED_NO_USAGE = _canned_sse([  # 缺 usage 块 → 必须中止
    '{"choices":[{"delta":{"role":"assistant","content":""}}]}',
    '{"choices":[{"delta":{"content":"快速"}}]}',
    '{"choices":[{"delta":{},"finish_reason":"stop"}]}',
    "[DONE]",
])
CANNED_NO_CONTENT = _canned_sse([  # 有 usage 但无内容增量 → 必须中止
    '{"choices":[{"delta":{"role":"assistant","content":""}}]}',
    '{"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":0,"total_tokens":1}}',
    "[DONE]",
])


def selftest() -> int:
    """离线自测：内置样例流走同一解析器，校验 TTFB/wall/TPS 数学口径。"""
    print("== selftest ==")
    clock_vals = iter([0.0, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50])
    streamed = consume_sse(io.BytesIO(CANNED_MAIN), clock=lambda: next(clock_vals))
    metrics = check_and_metrics(streamed)
    assert streamed["content"] == "快速排序测试", streamed
    # role 块不计入 TTFB：首个非空 content 增量在 0.10s
    assert abs(metrics["ttft_ms"] - 100.0) < 1e-9, metrics
    # wall = 首个 content 增量(0.10) → [DONE](0.50)
    assert abs(metrics["wall_s"] - 0.40) < 1e-9, metrics
    # TPS = 100 tokens / 0.40 s = 250
    assert abs(metrics["tps"] - 250.0) < 1e-9, metrics
    assert metrics["completion_tokens"] == 100 and metrics["prompt_tokens"] == 50, metrics
    print(f"  正常流    ttft_ms={metrics['ttft_ms']:.1f} wall_s={metrics['wall_s']:.2f} "
          f"tps={metrics['tps']:.1f} content={streamed['content']!r}")
    for name, canned, vals in (
            ("缺 usage 块", CANNED_NO_USAGE, [0.0, 0.0, 0.1, 0.2, 0.3]),
            ("无内容增量", CANNED_NO_CONTENT, [0.0, 0.0, 0.1, 0.2])):
        try:
            check_and_metrics(consume_sse(
                io.BytesIO(canned), clock=lambda v=iter(vals): next(v)))
        except RequestAbort as exc:
            print(f"  {name}  正确中止：{exc}")
        else:
            raise AssertionError(f"{name} 未触发 RequestAbort")
    print("SELFTEST_OK")
    return 0


# ---------------- CLI ----------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python3 -m scripts.stream_bench",
        description="SSE 流式推理服务基准：对比 OpenAI 兼容 /chat/completions 流式服务的 "
                    "TPS（completion_tokens/wall）与 TTFT（首个非空 content 增量）。仅标准库。",
        epilog="示例：\n"
               "  python3 -m scripts.stream_bench --service vllm=http://127.0.0.1:8000/v1 "
               "--service nano=http://127.0.0.1:8001/v1 --rounds 2 --warmup\n"
               "  python3 -m scripts.stream_bench --selftest   # 离线自测，无需服务",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--service", action="append", metavar="NAME=URL",
                   help="被测服务（可重复）：名称=OpenAI 兼容 base URL，如 vllm=http://127.0.0.1:8000/v1")
    p.add_argument("--rounds", type=int, default=2, help="每条提示词的轮数（默认 2）")
    p.add_argument("--max-tokens", type=int, default=512, help="每次请求 max_tokens（默认 512）")
    p.add_argument("--temperature", type=float, default=0.7, help="采样温度（默认 0.7）")
    p.add_argument("--timeout", type=int, default=600, help="socket 超时秒数（默认 600）")
    p.add_argument("--model", default=None,
                   help="模型名覆盖；默认自动 GET {URL}/models 取 data[0].id")
    p.add_argument("--warmup", action="store_true",
                   help="每个服务先发 1 次热身请求（max_tokens=32，不计入统计）")
    p.add_argument("--json-out", metavar="PATH", default=None,
                   help="把逐请求原始记录写入 JSON 文件")
    p.add_argument("--selftest", action="store_true",
                   help="离线自测：内置 SSE 样例验证解析与 TTFB/TPS 口径，打印 SELFTEST_OK")
    return p


def _parse_services(parser: argparse.ArgumentParser, specs: List[str]) -> List[Tuple[str, str]]:
    services = []
    for spec in specs:
        if "=" not in spec:
            parser.error(f"--service 需要 NAME=URL 格式，收到：{spec!r}")
        name, url = (part.strip() for part in spec.split("=", 1))
        if not name or not url.startswith(("http://", "https://")):
            parser.error(f"--service 非法（URL 需以 http(s):// 开头）：{spec!r}")
        services.append((name, url.rstrip("/")))
    return services


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.service:
        parser.error("至少提供一个 --service NAME=URL（或使用 --selftest 离线自测）")
    services = _parse_services(parser, args.service)

    all_records: List[dict] = []
    stats_list: List[dict] = []
    for name, url in services:
        print(f"\n== 测试服务 {name}（{url}）==", flush=True)
        model = args.model or fetch_model_name(url, args.timeout)
        if not model:
            print(f"[错误] 无法从 {url}/models 解析模型名且未提供 --model，该服务记为失败")
            records = [_record_base(name, url, None, rnd, idx, False)
                       | {"error": "模型名解析失败且未提供 --model"}
                       for rnd in range(1, args.rounds + 1) for idx in range(len(PROMPTS))]
            all_records.extend(records)
            stats_list.append(summarize(name, url, None, records))
            continue
        print(f"model={model}  rounds={args.rounds}×{len(PROMPTS)} 条提示词  "
              f"max_tokens={args.max_tokens}  temperature={args.temperature}")
        records: List[dict] = []
        if args.warmup:
            wrec = _record_base(name, url, model, 0, 0, True)
            t0 = time.perf_counter()
            try:
                stream_request(url, model, PROMPTS[0], temperature=args.temperature,
                               max_tokens=WARMUP_MAX_TOKENS, timeout=args.timeout)
                print(f"热身完成（耗时 {time.perf_counter() - t0:.1f}s，不计入统计）")
            except Exception as exc:  # 热身失败仅警告，不中断
                wrec["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[警告] 热身失败：{wrec['error']}")
            records.append(wrec)
        for rnd in range(1, args.rounds + 1):
            for idx, prompt in enumerate(PROMPTS):
                rec = run_one(name, url, model, rnd, idx, prompt, args)
                records.append(rec)
                if rec["error"]:
                    print(f"  轮 {rnd} 提示词 {idx}  失败：{rec['error']}")
                else:
                    print(f"  轮 {rnd} 提示词 {idx}  TPS={rec['tps']:.1f}  "
                          f"TTFT={rec['ttft_ms']:.0f}ms  tokens={rec['completion_tokens']}")
        all_records.extend(records)
        stats_list.append(summarize(name, url, model, records))

    print_report(stats_list)
    if args.json_out:
        payload = {
            "meta": {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "argv": sys.argv[1:], "rounds": args.rounds,
                "max_tokens": args.max_tokens, "temperature": args.temperature,
                "timeout": args.timeout, "model_override": args.model,
                "warmup": args.warmup,
            },
            "records": all_records,
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        print(f"\n逐请求原始记录已写入：{args.json_out}")
    return 0 if any(s["n_ok"] for s in stats_list) else 1


if __name__ == "__main__":
    sys.exit(main())
