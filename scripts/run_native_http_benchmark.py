"""在同一 CUDA 机器顺序启动 Python/C++ HTTP 服务，并按 ABBA 顺序压测。"""
import argparse
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time
import urllib.request

from flashengine.deployment import read_json, sha256, source_state, write_json
from scripts.native_bench_common import validate_workload
from scripts.run_native_benchmark import run_job


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("native")
    p.add_argument("model")
    p.add_argument("bundle")
    p.add_argument("workload")
    p.add_argument("output_dir")
    p.add_argument("--draft")
    p.add_argument("--concurrency", default="1,4,8")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--eager", action="store_true")
    args = p.parse_args()
    workload = read_json(args.workload)
    validate_workload(workload)
    bundle = Path(args.bundle)
    if sha256(bundle / "bundle.json") != workload["bundle_sha256"]:
        raise ValueError("workload 与 bundle 不一致")
    if bool(args.draft) != bool(read_json(bundle / "engine/config.json")["draft"]):
        raise ValueError("Python 草稿参数与 bundle 不一致")
    for name, item in read_json(bundle / "bundle.json")["files"].items():
        for kind, directory in (("target", args.model), ("draft", args.draft)):
            if directory and name.startswith(f"weights/{kind}/") and sha256(Path(directory) / Path(name).name) != item["sha256"]:
                raise ValueError("Python 权重不匹配：" + name)
    if args.rounds < 1 or args.warmup < 0:
        raise ValueError("非法轮数")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    cfg = workload["config"]
    report = dict(protocol="pretokenized-http-sse-v1", workload_sha256=sha256(args.workload), binary_sha256=sha256(args.native), oracle_source=source_state(), config=cfg, graph=not args.eager, runs=[], aggregates=[], release_qualified=False)
    results = {"python": [], "cpp": []}
    for iteration in range(args.rounds):
        for impl in (("python", "cpp") if iteration % 2 == 0 else ("cpp", "python")):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            base = f"http://127.0.0.1:{port}"
            shared = ["--host", "127.0.0.1", "--port", str(port), "--dtype", cfg["dtype"], "--no-think", "--max-model-len", str(cfg["max_model_len"]), "--max-num-seqs", str(cfg["max_num_seqs"]), "--max-num-batched-tokens", str(cfg["max_num_batched_tokens"]), "--max-num-partial-prefills", str(cfg["max_num_partial_prefills"]), "--kv-blocks", str(cfg["num_kvcache_blocks"])]
            if impl == "python":
                command = [sys.executable, "-m", "scripts.openai_server", args.model, "--no-prefix-cache", "--stats-interval", "0", "--block-size", str(cfg["kvcache_block_size"]), *shared]
                if args.eager:
                    command += ["--eager"]
                if args.draft:
                    dc = read_json(Path(args.bundle) / "engine/config.json")["draft"]
                    command += ["--spec-model", args.draft, "--spec-num-tokens", str(dc["num_tokens"])]
            else:
                command = [str(Path(args.native).resolve()), "serve", args.bundle, "--kv-block-size", str(cfg["kvcache_block_size"]), "--eager" if args.eager else "--graph", *shared]
            output = root / f"r{iteration}-{impl}.json"
            item = dict(implementation=impl, iteration=iteration, command=command, result=str(output), passed=False)
            print(f"启动 HTTP {impl} 轮次={iteration}", flush=True)
            with output.with_suffix(".server.log").open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        if process.poll() is not None:
                            raise RuntimeError(f"服务启动失败，退出={process.returncode}")
                        try:
                            with urllib.request.urlopen(base + "/v1/models", timeout=1):
                                break
                        except OSError:
                            time.sleep(.2)
                    else:
                        raise TimeoutError("服务启动超时")
                    client = [sys.executable, "-m", "scripts.bench_frozen_http", base, args.workload, str(output), "--concurrency", args.concurrency, "--warmup", str(args.warmup)]
                    result, record = run_job(client, output, False, 600)
                    if result:
                        results[impl].append(result)
                        item["passed"] = result["passed"]
                        item["result_sha256"] = sha256(output)
                    item["client_exit_code"] = record["exit_code"]
                except Exception as error:
                    item["error"] = repr(error)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    item["shutdown_exit_code"] = process.returncode
                    item["passed"] = item["passed"] and process.returncode in ((0, -15) if impl == "python" else (0,))
            report["runs"].append(item)
            write_json(root / "progress.json", report)
    for concurrency in [int(x) for x in args.concurrency.split(",")]:
        pair = {impl: [level for run in runs for level in run["levels"] if level["concurrency"] == concurrency] for impl, runs in results.items()}
        if all(pair.values()) and all(row["ttft_p50_ms"] is not None for rows in pair.values() for row in rows):
            py = statistics.median(row["output_tps"] for row in pair["python"])
            cpp = statistics.median(row["output_tps"] for row in pair["cpp"])
            report["aggregates"].append(dict(concurrency=concurrency, python_tps=py, cpp_tps=cpp, tps_ratio=cpp/py, python_ttft_p50_ms=statistics.median(row["ttft_p50_ms"] for row in pair["python"]), cpp_ttft_p50_ms=statistics.median(row["ttft_p50_ms"] for row in pair["cpp"])))
    report["execution_passed"] = all(run["passed"] for run in report["runs"])
    # HTTP 性能单独报告；此处不以文本相同取代 token/逐轮正确性门。
    write_json(root / "summary.json", report)
    print(report["aggregates"])
    raise SystemExit(0 if report["execution_passed"] else 2)


if __name__ == "__main__":
    main()
