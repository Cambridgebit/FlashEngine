"""串行调度 Python/C++ 冻结输入基准；ABBA 热态顺序，严格正确性与性能分开报告。"""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

from flashengine.deployment import read_json, sha256, write_json
from scripts.native_bench_common import compare_rows, gpu_sample, summary, validate_workload


def run_job(command, output, native, timeout):
    """任何失败都保留命令、日志和设备状态，不并发启动两个 GPU worker。"""
    samples, stop = [], threading.Event()
    def sample():
        while not stop.is_set():
            samples.append(gpu_sample())
            stop.wait(.5)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.monotonic()
    record = dict(command=command, exit_code=None, stdout="", stderr="")
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        record = dict(command=command, exit_code=process.returncode, stdout=process.stdout, stderr=process.stderr)
        result = json.loads(process.stdout) if native and process.returncode == 0 else read_json(output) if not native and output.exists() else None
        if native and result:
            result.update(summary(result["rows"], result["wall_seconds"]))
            write_json(output, result)
    except subprocess.TimeoutExpired as error:
        def as_text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        result = None
        record.update(error=repr(error), timed_out=True, stdout=as_text(error.stdout), stderr=as_text(error.stderr))
    except (OSError, ValueError) as error:
        # JSON 解析失败时保留子进程原始输出，不能丢掉崩溃/协议错误现场。
        result = None
        record["error"] = repr(error)
    finally:
        stop.set()
        thread.join(timeout=5)
    record.update(process_wall_seconds=time.monotonic() - start, gpu_samples=samples)
    if result:
        record["measurement_gpu_samples"] = [s for s in samples if result["measurement_unix_start"] <= s["unix_time"] <= result["measurement_unix_end"]]
        record["result_sha256"] = sha256(output)
    write_json(output.with_suffix(".process.json"), record)
    return result, record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("native")
    p.add_argument("model")
    p.add_argument("bundle")
    p.add_argument("workload")
    p.add_argument("output_dir")
    p.add_argument("--draft")
    p.add_argument("--modes", default="eager,graph")
    p.add_argument("--batch-sizes", default="1,4,8")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--rounds", type=int, default=2, help="每种实现的测量次数；2 对应 ABBA")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--trace-rounds", action="store_true", help="严格验证时使用；开启后不作为性能数字")
    p.add_argument("--attention", choices=("flash_attn", "flashinfer"), default="flash_attn")
    p.add_argument("--timeout", type=int, default=600)
    args = p.parse_args()
    workload = read_json(args.workload)
    validate_workload(workload)
    modes, batches = args.modes.split(","), [int(b) for b in args.batch_sizes.split(",")]
    if set(modes) - {"eager", "graph"} or not batches or any(b < 1 or b > workload["config"]["max_num_seqs"] for b in batches) or args.rounds < 1 or args.warmup < 0 or args.repeats < 1:
        raise ValueError("非法基准矩阵")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    native = str(Path(args.native).resolve())
    report = dict(protocol="pretokenized-grouped-engine-v1", workload_sha256=sha256(args.workload), binary_sha256=sha256(native), python_attention=args.attention, config=workload["config"], synthetic_model=workload.get("synthetic_model", False), trace_rounds=args.trace_rounds, scope="严格验证" if args.trace_rounds else "分组引擎基准；batch=1 为串行，不能与 HTTP 吞吐混表", runs=[], comparisons=[], aggregates=[], release_qualified=False)
    lookup = {}
    for mode in modes:
        for batch in batches:
            for iteration in range(args.rounds):
                order = ("python", "cpp") if iteration % 2 == 0 else ("cpp", "python")
                for implementation in order:
                    output = root / f"{mode}-b{batch}-r{iteration}-{implementation}.json"
                    common = ["--batch-size", str(batch), "--warmup", str(args.warmup), "--repeats", str(args.repeats)]
                    if mode == "graph":
                        common.append("--graph")
                    if args.trace_rounds:
                        common.append("--trace-rounds")
                    if implementation == "python":
                        command = [sys.executable, "-m", "scripts.bench_python_native", args.model, args.bundle, args.workload, str(output), "--attention", args.attention, *common]
                        if args.draft:
                            command += ["--draft", args.draft]
                    else:
                        command = [native, "benchmark", args.bundle, "--workload", args.workload, *common]
                    print(f"开始 {implementation} {mode} batch={batch} 轮次={iteration}", flush=True)
                    result, process = run_job(command, output, implementation == "cpp", args.timeout)
                    row = dict(implementation=implementation, mode=mode, batch_size=batch, iteration=iteration, result=str(output), exit_code=process["exit_code"])
                    if result:
                        lookup[(mode, batch, iteration, implementation)] = result
                        row.update(output_tps=result["output_tps"], ttft_p50_ms=result["ttft_p50_ms"], output_tokens=result["output_tokens"])
                    report["runs"].append(row)
                    write_json(root / "progress.json", report)
            for iteration in range(args.rounds):
                a, b = lookup.get((mode, batch, iteration, "python")), lookup.get((mode, batch, iteration, "cpp"))
                result = compare_rows(a["rows"], b["rows"], args.trace_rounds) if a and b else dict(passed=False, reason="worker 执行失败")
                report["comparisons"].append(dict(mode=mode, batch_size=batch, iteration=iteration, **result))
            a = [lookup[(mode, batch, i, "python")] for i in range(args.rounds) if (mode, batch, i, "python") in lookup]
            b = [lookup[(mode, batch, i, "cpp")] for i in range(args.rounds) if (mode, batch, i, "cpp") in lookup]
            if a and b:
                py_tps, cpp_tps = statistics.median(r["output_tps"] for r in a), statistics.median(r["output_tps"] for r in b)
                py_ttft, cpp_ttft = statistics.median(r["ttft_p50_ms"] for r in a), statistics.median(r["ttft_p50_ms"] for r in b)
                report["aggregates"].append(dict(mode=mode, batch_size=batch, python_tps=py_tps, cpp_tps=cpp_tps, tps_ratio=cpp_tps / py_tps, python_ttft_p50_ms=py_ttft, cpp_ttft_p50_ms=cpp_ttft, ttft_ratio=cpp_ttft / py_ttft, observed_threshold_passed=cpp_tps >= py_tps * .95 and cpp_ttft <= py_ttft * 1.1))
    report["mode_consistency"] = []
    if "eager" in modes and "graph" in modes:
        for batch in batches:
            for implementation in ("python", "cpp"):
                a, b = lookup.get(("eager", batch, 0, implementation)), lookup.get(("graph", batch, 0, implementation))
                report["mode_consistency"].append(dict(implementation=implementation, batch_size=batch, **(compare_rows(a["rows"], b["rows"], args.trace_rounds) if a and b else dict(passed=False))))
    report["execution_passed"] = len(lookup) == len(modes) * len(batches) * args.rounds * 2
    report["correctness_passed"] = all(c["passed"] for c in report["comparisons"] + report["mode_consistency"])
    write_json(root / "summary.json", report)
    print(json.dumps({key: report[key] for key in ("execution_passed", "correctness_passed", "aggregates")}, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["execution_passed"] and report["correctness_passed"] else 2)


if __name__ == "__main__":
    main()
