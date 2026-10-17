"""审计 HTTP 基准的配置、计数、首字延迟和测量期图命中；不授予正确性或 production。"""
import argparse
from pathlib import Path

from flashengine.deployment import read_json, sha256, write_json
from scripts.bench_frozen_http import check_server_config


def audit_level(level, graph):
    before, after = level.get("server_before", {}), level.get("server_after", {})
    old, new = before.get("metrics", {}), after.get("metrics", {})
    names = ("decode", "spec", "draft", "verify")
    replays = {kind: new.get(f"flashengine_{kind}_graph_replays_total", 0) - old.get(f"flashengine_{kind}_graph_replays_total", 0) for kind in names}
    graph_ok = bool(before and after) and (sum(replays.values()) > 0 if graph else sum(replays.values()) == 0)
    rows = level["rows"]
    latency_ok = bool(rows) and all(row["output_tokens"] > 0 and row["ttft_seconds"] is not None and 0 < row["ttft_seconds"] <= row["wall_seconds"] and row["text"] for row in rows)
    count_ok = level["server_counters_match"] and sum(row["output_tokens"] for row in rows) == level["output_tokens"]
    kv_ok = after.get("metrics", {}).get("flashengine_kv_blocks_used") == 0
    return dict(concurrency=level["concurrency"], passed=graph_ok and latency_ok and count_ok and kv_ok and not level["errors"], graph_gate_passed=graph_ok, graph_replays=replays, latency_valid=latency_ok, counts_match=count_ok, kv_released=kv_ok, peak=level["peak"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("output")
    args = p.parse_args()
    if Path(args.output).exists():
        raise ValueError("拒绝覆盖 HTTP 审计结果")
    report = read_json(Path(args.run_dir) / "summary.json")
    rows = []
    for run in report["runs"]:
        path = Path(run["result"])
        if not path.is_file():
            rows.append(dict(result=str(path), passed=False, reason="缺少结果"))
            continue
        result = read_json(path)
        try:
            check_server_config(result["info"], report["config"])
            config_ok = True
        except ValueError:
            config_ok = False
        process = read_json(path.with_suffix(".process.json"))
        hash_ok = sha256(path) == process.get("result_sha256") == run.get("result_sha256")
        levels = [audit_level(level, report["graph"]) for level in result["levels"]]
        rows.append(dict(result=str(path), config_match=config_ok, result_hash_match=hash_ok, levels=levels, passed=run["passed"] and config_ok and hash_ok and all(level["passed"] for level in levels)))
    passed = bool(rows) and report["execution_passed"] and all(row["passed"] for row in rows)
    write_json(args.output, dict(runtime_gates_passed=passed, rows=rows, token_correctness_assessed=False, release_qualified=False))
    print(dict(runtime_gates_passed=passed, token_correctness_assessed=False))
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
