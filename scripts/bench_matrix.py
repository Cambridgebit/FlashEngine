"""引擎内并发压测：四种解码方式 × 前缀缓存开关，输出 JSON/CSV/SVG/HTML。"""

import argparse
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time
import traceback
import uuid


METHODS = ("base", "eagle3", "dflash", "dspark")
COLORS = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#d97706", "#0891b2", "#475569", "#db2777")
METRICS = (("ttft_p50_ms", "TTFT p50 (ms)"),
           ("tpot_p50_ms", "TPOT p50 (ms/token)"),
           ("tps", "Aggregate TPS (token/s)"))


def save_json(path, value):
    """先写临时文件再替换，保留中断前已完成的数据。"""
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def percentile(values, percent):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * percent / 100
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def load_prompts(args):
    """沿用仓库 question.jsonl 的 turns[0] 与类别过滤。"""
    prompts = []
    with open(args.questions, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            if args.category != "all" and item.get("category") not in (None, args.category):
                continue
            prompt = item["turns"][0]
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("数据集 turns[0] 必须是非空文本")
            prompts.append(prompt)
    if not prompts:
        raise ValueError("数据集在指定类别下没有请求")
    if args.seed >= 0:
        random.Random(args.seed).shuffle(prompts)
    # 各 batch 使用相同的题集；末批循环补齐，结果记录实际请求数。
    return prompts[:args.num_prompts]


def render_prompt(tokenizer, system, prompt, thinking):
    kwargs = {} if thinking else {"enable_thinking": False}
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    return tokenizer.encode(text, add_special_tokens=False)


def shared_system(tokenizer, scope, target):
    """实际按 tokenizer 扩充共享文本，不按字符数估算，也不截 ChatML 模板。"""
    text = f"Benchmark context {scope}.\n"
    filler = "Shared reference: use clear explanations, check assumptions, and follow the task instructions.\n"
    while len(tokenizer.encode(text, add_special_tokens=False)) < target:
        text += filler
    return text


def common_blocks(prompts, block_size):
    common = 0
    for tokens in zip(*prompts):
        if len(set(tokens)) != 1:
            break
        common += 1
    return common // block_size


def measure_batch(engine, seqs, clock=time.perf_counter):
    """按每个请求自己的末次 token 到达时间计算 TPOT，保留逐请求数据。"""
    engine.model_runner.backend.synchronize()
    first = [None] * len(seqs)
    last = [None] * len(seqs)
    seen = [0] * len(seqs)
    start = clock()
    for seq in seqs:
        engine.scheduler.add(seq)
    while not engine.is_finished():
        engine.step()
        now = clock()
        for i, seq in enumerate(seqs):
            count = seq.num_completion_tokens
            if count > seen[i]:
                if first[i] is None:
                    first[i] = now - start
                last[i] = now - start
                seen[i] = count
    wall = clock() - start
    requests = []
    for i, seq in enumerate(seqs):
        count = seq.num_completion_tokens
        requests.append({
            "prompt_tokens": seq.num_prompt_tokens, "output_tokens": count,
            "ttft_ms": first[i] * 1000 if first[i] is not None else None,
            "last_token_ms": last[i] * 1000 if last[i] is not None else None,
            "tpot_ms": (last[i] - first[i]) * 1000 / (count - 1) if count > 1 else None,
        })
    return requests, wall


def graph_stats(runner):
    """兼容两仓库的统计接口；缺失字段不伪造为零。"""
    snapshot = getattr(runner, "runtime_snapshot", None)
    if snapshot is not None:
        return snapshot()["graph"]
    name = "spec_graph_stats" if runner.spec_enabled else "decode_graph_stats"
    getter = getattr(runner, "get_" + name, None)
    if getter is not None:
        return getter()
    return dict(getattr(runner, name, {}))


def graph_delta(before, after):
    return {key: value - before.get(key, 0) for key, value in after.items()
            if isinstance(value, (int, float)) and key not in ("capture_success", "capture_failure")}


def worker(job):
    # 父进程、合并和绘图均只依赖标准库；仅此处加载设备与模型依赖。
    from flashengine import LLM, SamplingParams
    from flashengine.engine.sequence import Sequence

    settings = job["settings"]
    kwargs = {
        "backend": job["backend"], "spec_model": job["draft"],
        "spec_num_tokens": settings["spec_num_tokens"],
        "enable_prefix_cache": job["prefix_cache"],
        "max_num_seqs": settings["max_num_seqs"],
        "max_model_len": settings["max_model_len"],
        "max_num_batched_tokens": settings["max_num_batched_tokens"],
        "gpu_memory_utilization": settings["gpu_mem_util"],
        "enforce_eager": settings["eager"],
    }
    if settings["block_size"] is not None:
        kwargs["kvcache_block_size"] = settings["block_size"]
    engine = None
    rows = []
    try:
        engine = LLM(job["model"], **kwargs)
        runner = engine.model_runner
        config = runner.config
        actual = {
            "backend": runner.backend.name, "method": config.spec_method or "base",
            "prefix_cache": config.enable_prefix_cache,
            "block_size": config.kvcache_block_size,
            "spec_num_tokens": config.spec_num_tokens if runner.spec_enabled else 0,
            "max_model_len": config.max_model_len, "max_num_seqs": config.max_num_seqs,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "graph_mode": runner.graph_mode(), "dtype": str(runner.dtype),
            "kv_blocks": len(engine.scheduler.block_manager.blocks),
            "backend_runtime": runner.backend.runtime_info(),
        }
        if actual["method"] != job["method"]:
            raise ValueError(f"草稿类型不匹配：请求 {job['method']}，实际 {actual['method']}")
        if actual["prefix_cache"] != job["prefix_cache"]:
            raise ValueError("引擎自动关闭了 prefix-cache，此组合不受支持，不能作为开启组出图")

        stats = {"prefill_batch_peak": 0, "decode_batch_peak": 0,
                 "preemptions": 0, "spec_rounds": 0, "spec_committed": 0}
        schedule = engine.scheduler.schedule
        def scheduled():
            seqs, prefill = schedule()
            key = "prefill_batch_peak" if prefill else "decode_batch_peak"
            # 投机 schedule 后还会做容量检查/抢占，真实批大小由 spec_round 记录。
            if prefill or not runner.spec_enabled:
                stats[key] = max(stats[key], len(seqs))
            return seqs, prefill
        engine.scheduler.schedule = scheduled
        preempt = engine.scheduler.preempt
        def preempted(seq):
            stats["preemptions"] += 1
            return preempt(seq)
        engine.scheduler.preempt = preempted
        if runner.spec_enabled:
            spec_round = runner.spec_round
            def committed(seqs):
                stats["decode_batch_peak"] = max(stats["decode_batch_peak"], len(seqs))
                result = spec_round(seqs)
                stats["spec_rounds"] += len(result)
                stats["spec_committed"] += sum(map(len, result))
                return result
            runner.spec_round = committed

        sampling = SamplingParams(temperature=0.0, max_tokens=settings["max_tokens"],
                                  ignore_eos=not settings["respect_eos"])
        def sequences(ids):
            return [Sequence(tokens, sampling) for tokens in ids]
        def checked(system, prompts):
            ids = [render_prompt(engine.tokenizer, system, p, settings["thinking"]) for p in prompts]
            # 预留一整轮草稿空间，旧仓库没有上下文硬停，也不会因此越界。
            margin = actual["spec_num_tokens"] + 1 if runner.spec_enabled else 0
            if any(len(tokens) + settings["max_tokens"] + margin > config.max_model_len for tokens in ids):
                raise ValueError("prompt + max_tokens + 投机末轮余量超过 max_model_len；请增大窗口或缩短前缀/输出")
            return ids

        for batch in settings["batch_sizes"]:
            try:
                warm_ids = checked(shared_system(engine.tokenizer, f"warmup-{batch}", settings["shared_prefix_tokens"]),
                                   [job["prompts"][i % len(job["prompts"])] for i in range(batch)])
                for _ in range(settings["warmup"]):
                    measure_batch(engine, sequences(warm_ids))
                bm = engine.scheduler.block_manager
                all_requests, walls, prefix_hits, hit_blocks, shared_counts = [], [], 0, 0, []
                for key in stats:
                    stats[key] = 0
                measured_graph = {}
                groups = math.ceil(settings["num_prompts"] / batch)
                for repeat in range(settings["repeats"]):
                    for group in range(groups):
                        # 组号放在最前面，跨组/档位没有整块可复用；八组合使用相同文本。
                        scope = f"b{batch}-r{repeat}-g{group}"
                        system = shared_system(engine.tokenizer, scope, settings["shared_prefix_tokens"])
                        prompts = [job["prompts"][(group * batch + i) % len(job["prompts"])] for i in range(batch)]
                        seed_ids = checked(system, ["Prepare the shared context."])
                        ids = checked(system, prompts)
                        count = common_blocks(seed_ids + ids, config.kvcache_block_size)
                        if count < 1:
                            raise ValueError("共享前缀不足一个 KV block；请增大 --shared-prefix-tokens")
                        shared_counts.append(count)
                        # 开/关两组都跑相同 seed；seed 时间与 token 不计入指标。
                        saved = dict(stats)
                        measure_batch(engine, sequences(seed_ids))
                        stats.update(saved)
                        before = bm.get_prefix_stats()
                        before_graph = graph_stats(runner)
                        requests, wall = measure_batch(engine, sequences(ids))
                        for key, value in graph_delta(before_graph, graph_stats(runner)).items():
                            measured_graph[key] = measured_graph.get(key, 0) + value
                        after = bm.get_prefix_stats()
                        prefix_hits += after["prefix_hits"] - before["prefix_hits"]
                        hit_blocks += after["cached_blocks"] - before["cached_blocks"]
                        for i, request in enumerate(requests):
                            request.update(repeat=repeat, group=group, index=i)
                        all_requests.extend(requests)
                        walls.append(wall)
                ttfts = [r["ttft_ms"] for r in all_requests if r["ttft_ms"] is not None]
                tpots = [r["tpot_ms"] for r in all_requests if r["tpot_ms"] is not None]
                out_tokens = sum(r["output_tokens"] for r in all_requests)
                warnings = []
                if job["prefix_cache"] and not hit_blocks:
                    warnings.append("缓存开启但主跑未命中；请检查 KV 容量/后端前缀支持")
                if stats["decode_batch_peak"] < batch:
                    warnings.append("实际 decode 批大小未达到目标 batch")
                if stats["preemptions"]:
                    warnings.append("发生 KV 抢占，结果包含重预填充成本")
                if len(ttfts) != len(all_requests):
                    warnings.append("部分请求未输出 token")
                if not settings["respect_eos"] and any(r["output_tokens"] != settings["max_tokens"] for r in all_requests):
                    warnings.append("部分请求未达到固定输出长度")
                row = {**job["identity"], "backend": actual["backend"],
                       "method": job["method"], "prefix_cache": job["prefix_cache"], "batch": batch,
                       "status": "warning" if warnings else "ok", "warnings": warnings,
                       "requests": len(all_requests), "output_tokens": out_tokens,
                       "wall_s": sum(walls), "tps": out_tokens / sum(walls),
                       "ttft_p50_ms": percentile(ttfts, 50), "ttft_p95_ms": percentile(ttfts, 95),
                       "tpot_p50_ms": percentile(tpots, 50), "tpot_p95_ms": percentile(tpots, 95),
                       "prefix_hit_requests": prefix_hits, "prefix_hit_blocks": hit_blocks,
                       "shared_blocks_min": min(shared_counts), **stats,
                       "acceptance_length": stats["spec_committed"] / stats["spec_rounds"] if stats["spec_rounds"] else None,
                       "actual": actual, "graph_measured": measured_graph, "graph_total": graph_stats(runner),
                       "request_results": all_requests}
                rows.append(row)
                save_json(job["result"], rows)
                print(f"{actual['backend']} {job['method']} prefix={job['prefix_cache']} batch={batch}: "
                      f"TTFT={row['ttft_p50_ms']}ms TPOT={row['tpot_p50_ms']}ms TPS={row['tps']:.2f} "
                      f"peak={stats['decode_batch_peak']} hits={hit_blocks}", flush=True)
            except Exception:
                rows.append({**job["identity"], "backend": job["backend"], "method": job["method"],
                             "prefix_cache": job["prefix_cache"], "batch": batch, "status": "failed",
                             "error": traceback.format_exc()})
                save_json(job["result"], rows)
                # 设备错误后不再复用引擎，余下档位由父进程标记失败。
                raise
    finally:
        if engine is not None:
            engine.close()


def svg_chart(path, rows, metric, title, series_key):
    """标准库生成可缩放趋势图；空指标断线，警告点用空心圆标识。"""
    series = {}
    for row in rows:
        series.setdefault(series_key(row), {})[row["batch"]] = row
    batches = sorted({row["batch"] for row in rows})
    valid = [r[metric] for r in rows if r.get(metric) is not None and r.get("status") != "failed"]
    top = max(valid, default=1) * 1.15 or 1
    width, height = 1040, max(390, 90 + len(series) * 23)
    left, right, bottom, upper = 85, 660, 315, 55
    x = lambda b: left + (batches.index(b) / max(1, len(batches) - 1)) * (right - left)
    y = lambda v: bottom - v / top * (bottom - upper)
    esc = lambda value: html.escape(str(value), quote=True)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<g font-family="Arial,sans-serif" font-size="12" fill="#334155">',
             f'<text x="{left}" y="25" font-size="16">{esc(title)}</text>']
    for i in range(6):
        value = top * i / 5
        yy = y(value)
        parts.extend([f'<path d="M{left},{yy} H{right}" stroke="#e2e8f0"/>',
                      f'<text x="{left-10}" y="{yy+4}" text-anchor="end">{value:.2f}</text>'])
    for batch in batches:
        parts.append(f'<text x="{x(batch)}" y="338" text-anchor="middle">{batch}</text>')
    parts.append('<text x="370" y="362" text-anchor="middle">Batch / concurrent requests</text>')
    for i, (label, points) in enumerate(sorted(series.items())):
        color = COLORS[i % len(COLORS)]
        dash = "6 4" if i >= len(COLORS) else "none"
        segment = []
        def flush():
            if segment:
                parts.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>')
                segment.clear()
        circles = []
        for batch in batches:
            row = points.get(batch, {})
            value = row.get(metric)
            if value is None or row.get("status") == "failed":
                flush()
                continue
            xx, yy = x(batch), y(value)
            segment.append(f"{xx},{yy}")
            fill = "white" if row.get("status") == "warning" else color
            circles.append(f'<circle cx="{xx}" cy="{yy}" r="4" fill="{fill}" stroke="{color}" stroke-width="2"><title>{esc(label)} batch={batch}: {value:.3f}</title></circle>')
        flush()
        parts.extend(circles)
        yy = 58 + i * 23
        parts.extend([f'<path d="M700,{yy} h25" stroke="{color}" stroke-width="2" stroke-dasharray="{dash}"/>',
                      f'<text x="735" y="{yy+4}">{esc(label)}</text>'])
    parts.append('</g></svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def report(document, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "results.json", document)
    rows = document["rows"]
    fields = ("run_id", "project", "workload", "backend", "method", "prefix_cache", "batch", "status",
              "requests", "output_tokens", "wall_s", "ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "tps",
              "prefill_batch_peak", "decode_batch_peak", "preemptions", "prefix_hit_requests", "prefix_hit_blocks", "acceptance_length", "warnings", "error")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row["workload"], row["backend"]), []).append(row)
    panels = []
    counter = 0
    def panel(title, data, key):
        nonlocal counter
        counter += 1
        panels.append(f"<h2>{html.escape(title)}</h2><div class='charts'>")
        for metric, label in METRICS:
            name = f"trend_{counter}_{metric}.svg"
            svg_chart(output / name, data, metric, label, key)
            panels.append(f"<img src='{name}' alt='{html.escape(label)}'>")
        panels.append("</div>")
    combo = lambda r: f"{r['method']} / prefix={'on' if r['prefix_cache'] else 'off'}"
    run_label = lambda r: f"{r['project']}:{r['run_id'][:6]}"
    for (workload, backend), data in sorted(groups.items()):
        panel(f"后端 {backend} · 工作负载 {workload}", data,
              lambda r: f"{combo(r)} / {run_label(r)}")
    # 合并后对每个功能组合单独画后端/项目比较，避免 24/48 条线挤在同一图里。
    for workload in sorted({r["workload"] for r in rows}):
        data = [r for r in rows if r["workload"] == workload]
        if len({(r["run_id"], r["backend"]) for r in data}) > 1:
            for method in METHODS:
                for cache in (False, True):
                    selected = [r for r in data if r["method"] == method and r["prefix_cache"] == cache]
                    if selected:
                        panel(f"跨后端/项目 · {method} · prefix={'on' if cache else 'off'} · {workload}", selected,
                              lambda r: f"{r['backend']} / {run_label(r)}")
    table = []
    for row in rows:
        values = [row.get(k) for k in ("project", "backend", "method", "prefix_cache", "batch", "status", "ttft_p50_ms", "tpot_p50_ms", "tps", "decode_batch_peak", "prefix_hit_blocks", "acceptance_length")]
        values.append("; ".join(row.get("warnings", [])) or row.get("error", ""))
        table.append("<tr>" + "".join(f"<td>{html.escape(f'{v:.3f}' if isinstance(v, float) else str(v) if v is not None else '—')}</td>" for v in values) + "</tr>")
    headings = ("项目", "后端", "解码", "Prefix", "Batch", "状态", "TTFT p50/ms", "TPOT p50/ms", "TPS", "decode峰值", "命中块", "E", "备注")
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>FlashEngine 压测趋势</title>
<style>body{font:14px system-ui;margin:24px;background:#f4f5f7;color:#172033}h2{font-size:18px}.charts{display:grid;gap:12px}img{width:100%;background:white;border:1px solid #ddd;border-radius:8px}table{border-collapse:collapse;background:white}td,th{border:1px solid #ddd;padding:8px;white-space:pre-wrap}td:last-child{max-width:500px}.table{overflow:auto}details{margin:12px 0}pre{white-space:pre-wrap}</style>
<h1>FlashEngine 并发压测</h1><p>进程内口径；TTFT 含引擎排队，TPOT=(末 token 时间−首 token 时间)/(输出 token 数−1)，TPS=主跑输出 token 总数/主跑批次墙钟总和。投机提交多个 token 使用 token 总数，不使用轮数或 SSE chunk 数。空心点表示有警告；失败点断线。</p>
<p>缓存组每批先运行同一共享前缀的种子请求，种子与预热不计指标。不同组使用不同前缀，防止上一档缓存污染。E 含 bonus，末轮截断前略高估。不同工作负载不连线；合并仅比较同工作负载哈希，仍须核对模型权重与硬件环境。</p>
<p><a href="results.json">原始数据 JSON</a> · <a href="summary.csv">汇总 CSV</a></p>"""
    for run in document["runs"]:
        page += f"<details><summary>运行元数据 {html.escape(run['run_id'])}</summary><pre>{html.escape(json.dumps(run, ensure_ascii=False, indent=2))}</pre></details>"
    page += "".join(panels) + "<div class='table'><table><tr>" + "".join(f"<th>{h}</th>" for h in headings) + "</tr>" + "".join(table) + "</table></div></html>"
    (output / "report.html").write_text(page, encoding="utf-8")


def parser():
    ap = argparse.ArgumentParser(add_help=False, description="四种解码 × prefix-cache 开关的八组合并发压测与趋势图（从仓库根目录 -m 运行）")
    ap.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    ap.add_argument("--model", help="目标模型本地目录")
    for method in METHODS[1:]:
        ap.add_argument(f"--{method}-model", help=f"{method} 草稿模型本地目录")
    ap.add_argument("--methods", default=",".join(METHODS), help="待测方法，默认 base,eagle3,dflash,dspark")
    ap.add_argument("--backends", default="auto", help="本机后端列表：auto/cuda/corex/ascend310p；跨机器分别跑后合并")
    ap.add_argument("--batch-sizes", default="1,2,4,8", help="并发 batch 列表")
    ap.add_argument("--questions", default="bench_data/question.jsonl", help="question.jsonl 数据集")
    ap.add_argument("--category", default="coding", help="类别；all 表示全部")
    ap.add_argument("--seed", type=int, default=0, help="题集随机种子，-1 保持文件序")
    ap.add_argument("--num-prompts", type=int, default=16, help="每档每次重复的请求数，向上补齐为 batch 整数倍")
    ap.add_argument("--repeats", type=int, default=3, help="主跑重复次数")
    ap.add_argument("--warmup", type=int, default=2, help="每档预热批次数，至少 1 次")
    ap.add_argument("--max-tokens", type=int, default=128, help="每请求输出 token 数，默认忽略 EOS 以固定长度")
    ap.add_argument("--respect-eos", action="store_true", help="允许 EOS 提前结束，批次可能缩小")
    ap.add_argument("--thinking", action="store_true", help="启用模板默认 thinking；默认关闭")
    ap.add_argument("--shared-prefix-tokens", type=int, default=512, help="共享 system 文本的最小实际 token 数")
    ap.add_argument("--spec-num-tokens", type=int, default=3, help="EAGLE3 的 k；DFlash/DSpark 的 3/4 会由引擎按 checkpoint 自动派生")
    ap.add_argument("--max-model-len", type=int, default=2048, help="上下文上限，超长 prompt 报错，不裁切模板")
    ap.add_argument("--max-num-seqs", type=int, help="各档保持相同引擎容量，默认最大 batch")
    ap.add_argument("--max-num-batched-tokens", type=int, default=16384, help="prefill token 预算")
    ap.add_argument("--gpu-mem-util", type=float, default=0.85, help="设备显存预算比例")
    ap.add_argument("--block-size", type=int, help="KV block 大小，默认交由引擎/后端决定")
    ap.add_argument("--eager", action="store_true", help="关闭图捕获，结果记录实际图路径")
    ap.add_argument("--cooldown", type=float, default=5, help="组合间冷却秒数")
    ap.add_argument("--worker-timeout", type=float, default=3600, help="每组合子进程超时秒数")
    ap.add_argument("--output", default="bench_results/matrix", help="JSON/CSV/HTML/SVG 与日志输出目录")
    ap.add_argument("--merge", nargs="+", help="合并已有 results.json 并重画图，不导入 torch，不要求模型")
    ap.add_argument("--dry-run", action="store_true", help="校验参数并保存执行计划，不加载模型")
    ap.add_argument("--worker-job", help=argparse.SUPPRESS)
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    if args.worker_job:
        worker(json.loads(Path(args.worker_job).read_text(encoding="utf-8")))
        return
    if args.merge:
        runs, rows = {}, {}
        for path in args.merge:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            if document.get("schema_version") != 1:
                ap.error(f"无法识别结果格式：{path}")
            for run in document["runs"]:
                runs[run["run_id"]] = run
            for row in document["rows"]:
                rows[(row["run_id"], row["backend"], row["method"], row["prefix_cache"], row["batch"])] = row
        report({"schema_version": 1, "runs": list(runs.values()), "rows": list(rows.values())}, args.output)
        print(f"合并报告：{Path(args.output).resolve() / 'report.html'}")
        return
    try:
        args.batch_sizes = sorted(set(int(x) for x in args.batch_sizes.split(",")))
        methods = list(dict.fromkeys(args.methods.split(",")))
        backends = list(dict.fromkeys(args.backends.split(",")))
        if not args.batch_sizes or min(args.batch_sizes) < 1:
            raise ValueError("batch 必须为正整数")
        if not methods or any(m not in METHODS for m in methods):
            raise ValueError("--methods 只能含 base,eagle3,dflash,dspark")
        if any(b not in ("auto", "cuda", "corex", "ascend310p") for b in backends):
            raise ValueError("未知后端")
        if "auto" in backends and len(backends) > 1:
            raise ValueError("auto 不得与显式后端同时测试")
        for key in ("num_prompts", "repeats", "warmup", "max_tokens", "shared_prefix_tokens", "spec_num_tokens", "max_model_len", "max_num_batched_tokens"):
            if getattr(args, key) < 1:
                raise ValueError(f"--{key.replace('_', '-')} 必须 >= 1")
        if args.cooldown < 0 or args.worker_timeout <= 0 or not 0 < args.gpu_mem_util <= 1:
            raise ValueError("cooldown >= 0，worker-timeout > 0，gpu-mem-util 在 (0,1] 内")
        if args.max_num_seqs is None:
            args.max_num_seqs = max(args.batch_sizes)
        if args.max_num_seqs < max(args.batch_sizes):
            raise ValueError("max-num-seqs 必须覆盖最大 batch，防止测成排队")
        required = [args.model] + [getattr(args, f"{m}_model") for m in methods if m != "base"]
        if any(not p or not Path(p).is_dir() for p in required):
            raise ValueError("请传入目标模型与所选方法的草稿模型本地目录")
        prompts = load_prompts(args)
    except (ValueError, OSError, KeyError, IndexError) as exc:
        ap.error(str(exc))
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # 工作负载身份不使用绝对路径，便于不同服务器合并；权重仍需人工确保同版本。
    settings = {key: getattr(args, key) for key in (
        "batch_sizes", "num_prompts", "repeats", "warmup", "max_tokens", "respect_eos", "thinking",
        "shared_prefix_tokens", "spec_num_tokens", "max_model_len", "max_num_seqs",
        "max_num_batched_tokens", "gpu_mem_util", "block_size", "eager")}
    model_ids = {"target": Path(args.model).resolve().name}
    model_ids.update({m: Path(getattr(args, f"{m}_model")).resolve().name for m in methods if m != "base"})
    config_hashes = {}
    for name, folder in [("target", args.model)] + [(m, getattr(args, f"{m}_model")) for m in methods if m != "base"]:
        config_path = Path(folder) / "config.json"
        config_hashes[name] = hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path.exists() else None
    workload = hashlib.sha256(json.dumps([model_ids, config_hashes, prompts, settings], sort_keys=True).encode()).hexdigest()[:12]
    run_id = uuid.uuid4().hex
    identity = {"run_id": run_id, "project": root.name, "workload": workload}
    run = {**identity, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "model": str(Path(args.model).resolve()),
           "model_ids": model_ids, "config_hashes": config_hashes, "settings": settings, "backends": backends, "methods": methods,
           "dataset": str(Path(args.questions).resolve()), "seed": args.seed, "category": args.category,
           "prompts": prompts, "python": sys.version, "platform": platform.platform(),
           "environment": {k: v for k, v in os.environ.items() if k.startswith("FLASHENGINE_")},
           "metric_scope": "engine_in_process"}
    document = {"schema_version": 1, "runs": [run], "rows": []}
    jobs = []
    for backend in backends:
        for method in methods:
            for cache in (False, True):
                tag = f"{backend}_{method}_prefix_{int(cache)}"
                job = {"identity": identity, "model": run["model"], "draft": str(Path(getattr(args, f"{method}_model")).resolve()) if method != "base" else "",
                       "backend": backend, "method": method, "prefix_cache": cache,
                       "settings": settings, "prompts": prompts,
                       "result": str(output / f"{run_id}_{tag}_rows.json")}
                path = output / f"{run_id}_{tag}_job.json"
                save_json(path, job)
                jobs.append((path, job, tag))
    save_json(output / "plan.json", {"run": run, "jobs": [str(path) for path, _, _ in jobs]})
    if args.dry_run:
        print(f"计划已保存：{output / 'plan.json'}；{len(jobs)} 个独立进程，{len(jobs)*len(args.batch_sizes)} 个数据点")
        return
    for index, (path, job, tag) in enumerate(jobs):
        print(f"[{index+1}/{len(jobs)}] {tag}，batch={args.batch_sizes}", flush=True)
        log = output / f"{run_id}_{tag}.log"
        error = None
        try:
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run([sys.executable, "-u", "-m", "scripts.bench_matrix", "--worker-job", str(path)],
                                        cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=args.worker_timeout, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            if result.returncode:
                error = f"子进程退出码 {result.returncode}；详见 {log.name}"
        except (subprocess.TimeoutExpired, OSError) as exc:
            error = f"{type(exc).__name__}: {exc}；详见 {log.name}"
        rows = json.loads(Path(job["result"]).read_text(encoding="utf-8")) if Path(job["result"]).exists() else []
        completed = {r["batch"] for r in rows}
        for batch in args.batch_sizes:
            if batch not in completed:
                rows.append({**identity, "backend": job["backend"], "method": job["method"],
                             "prefix_cache": job["prefix_cache"], "batch": batch,
                             "status": "failed", "error": error or "子进程未生成该档结果"})
        document["rows"].extend(rows)
        report(document, output)
        for row in rows:
            print(f"  batch={row['batch']} {row['status']} TPS={row.get('tps')} "
                  f"{'; '.join(row.get('warnings', [])) or row.get('error', '')}", flush=True)
        if index + 1 < len(jobs):
            time.sleep(args.cooldown)
    print(f"报告：{output / 'report.html'}", flush=True)
    if any(r["status"] == "failed" for r in document["rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
