"""使用同一冻结 token 输入压测任一 OpenAI-compatible 服务；不将 SSE chunk 数当 token。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import urllib.request

from flashengine.deployment import read_json, sha256, write_json
from scripts.bench_concurrent import PeakSampler, delta
from scripts.bench_utils import percentile
from scripts.native_bench_common import validate_workload


def check_server_config(info, cfg):
    """检查服务的实际配置，不以命令行参数代替 resolved config。"""
    for key, value in cfg.items():
        actual = info.get(key)
        if key == "dtype":
            actual = str(actual).removeprefix("torch.")
        if actual != value:
            raise ValueError(f"HTTP resolved config 不一致：{key}={actual!r}，期望 {value!r}")
    if info.get("enable_prefix_cache") is not False or info.get("tensor_parallel_size") != 1:
        raise ValueError("HTTP 基准必须关闭 prefix cache 且 TP=1")


def latency_percentile(values, percent):
    """缺少延迟样本时保留 null，不把缺测值写成零毫秒。"""
    return percentile(values, percent) if values else None


def run_http(base, workload, levels, repeats=1, warmup=2, timeout=120):
    """按波次同步发车；测量从发车前到该档所有请求完成。"""
    def get(path):
        with urllib.request.urlopen(base + path, timeout=5) as response:
            return json.load(response)
    model = get("/v1/models")["data"][0]["id"]
    info = get("/v1/stats")["info"]
    check_server_config(info, workload["config"])
    def call(item, barrier=None):
        if barrier:
            barrier.wait(timeout=timeout)
        body = dict(model=model, prompt=item["prompt_token_ids"], max_tokens=item["max_tokens"], temperature=0, stream=True, stream_options=dict(include_usage=True))
        request = urllib.request.Request(base + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        start, first, text, usage, done, finish = time.perf_counter(), None, [], None, False, None
        with urllib.request.urlopen(request, timeout=timeout) as response:
            for raw in response:
                if not raw.startswith(b"data:"):
                    continue
                payload = raw[5:].decode().strip()
                if payload == "[DONE]":
                    done = True
                    break
                obj = json.loads(payload)
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    if "text" not in choice or "delta" in choice:
                        raise RuntimeError("Completions SSE 必须使用 choices[].text，不能使用 Chat delta")
                    content = choice["text"]
                    if not isinstance(content, str):
                        raise RuntimeError("Completions SSE text 必须是字符串")
                    if content:
                        if first is None:
                            first = time.perf_counter() - start
                        text.append(content)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
        wall = time.perf_counter() - start
        if not done or not usage or type(usage.get("completion_tokens")) is not int or finish is None:
            raise RuntimeError("SSE 缺少 DONE/usage/finish，不能用 chunk 数替代")
        tokens = usage["completion_tokens"]
        if tokens > 0 and first is None:
            raise RuntimeError("输出 token 非零但没有文本，TTFT 不可测；拒绝记为零")
        if usage.get("prompt_tokens") != len(item["prompt_token_ids"]):
            raise RuntimeError("服务裁剪了冻结 prompt")
        return dict(input_id=item["id"], text="".join(text), output_tokens=tokens, prompt_tokens=usage["prompt_tokens"], finish_reason=finish, ttft_seconds=first, wall_seconds=wall, tpot_seconds=(wall-first)/(tokens-1) if first is not None and tokens > 1 else None)
    for i in range(warmup):
        call(workload["requests"][i % len(workload["requests"])])
    result = dict(protocol="pretokenized-http-sse-v1", measurement_unix_start=time.time(), info=info, levels=[])
    for level in levels:
        requests = workload["requests"] * repeats
        # 整数波次确保两端每档完全相同的请求数量，尾部不足时从头补齐。
        while len(requests) % level:
            requests.append(workload["requests"][len(requests) % len(workload["requests"])])
        before = get("/v1/stats")
        sampler = PeakSampler(base, SimpleNamespace(api_key=""), interval=.1)
        sampler.start()
        rows, errors, start, begin = [], [], time.perf_counter(), time.time()
        try:
            for offset in range(0, len(requests), level):
                barrier = threading.Barrier(level)
                with ThreadPoolExecutor(max_workers=level) as pool:
                    futures = [pool.submit(call, item, barrier) for item in requests[offset:offset + level]]
                    for future in futures:
                        try:
                            rows.append(future.result())
                        except Exception as error:
                            errors.append(repr(error))
        finally:
            wall, end = time.perf_counter() - start, time.time()
            sampler.stop_sampling()
        after = get("/v1/stats")
        d = delta(before["metrics"], after["metrics"])
        output = sum(row["output_tokens"] for row in rows)
        ttfts = [row["ttft_seconds"] * 1000 for row in rows if row["ttft_seconds"] is not None]
        tpots = [row["tpot_seconds"] * 1000 for row in rows if row["tpot_seconds"] is not None]
        result["levels"].append(dict(concurrency=level, wall_seconds=wall, output_tokens=output, output_tps=output/wall, ttft_p50_ms=latency_percentile(ttfts, 50), ttft_p95_ms=latency_percentile(ttfts, 95), tpot_p50_ms=latency_percentile(tpots, 50), peak=sampler.snapshot(), metrics_delta=d, server_before=before, server_after=after, server_counters_match=d.get("out")==output, errors=errors, rows=rows, measurement_unix_start=begin, measurement_unix_end=end))
    result["measurement_unix_end"] = time.time()
    result["passed"] = all(not level["errors"] and level["server_counters_match"] for level in result["levels"])
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base_url")
    p.add_argument("workload")
    p.add_argument("output")
    p.add_argument("--concurrency", default="1,4,8")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    args = p.parse_args()
    if Path(args.output).exists():
        raise ValueError("拒绝覆盖 HTTP 结果")
    workload = read_json(args.workload)
    validate_workload(workload)
    if any(row["ignore_eos"] for row in workload["requests"]):
        raise ValueError("HTTP workload 必须由 --respect-eos 生成；不静默忽略冻结采样语义")
    levels = [int(x) for x in args.concurrency.split(",")]
    if any(x < 1 for x in levels) or args.repeats < 1 or args.warmup < 0:
        raise ValueError("非法 HTTP 基准参数")
    result = run_http(args.base_url.rstrip("/"), workload, levels, args.repeats, args.warmup)
    result["workload_sha256"] = sha256(args.workload)
    result["eos_policy"] = "遵循服务 EOS；不发送未支持的 ignore_eos 字段"
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("passed", "protocol")}, ensure_ascii=False))
    raise SystemExit(0 if result["passed"] else 2)


if __name__ == "__main__":
    main()
