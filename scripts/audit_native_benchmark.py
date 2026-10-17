"""审计基准结果的实际配置、Graph 回放和 token 正确性；不授予 production。"""
import argparse
from pathlib import Path

from flashengine.deployment import read_json, sha256, write_json


def graph_progress(result):
    """只用测量窗口计数，预热回放和成功捕获不算稳态命中。"""
    before, after = result["runtime_before"], result["runtime_after"]
    if result["implementation"] == "cpp":
        parts = {kind: after.get(kind + "_graph_replays", 0) - before.get(kind + "_graph_replays", 0) for kind in ("decode", "draft", "verify")}
        return dict(replays=sum(parts.values()), parts=parts, buckets=after.get("graph_buckets", []))
    old, new = before.get("graph", {}), after.get("graph", {})
    return dict(replays=new.get("replay", 0) - old.get("replay", 0), mode=after.get("mode"), capture_failures=new.get("capture_failure", 0))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("output")
    args = p.parse_args()
    if Path(args.output).exists():
        raise ValueError("拒绝覆盖审计结果")
    root = Path(args.run_dir)
    benchmark = read_json(root / "summary.json")
    rows = []
    for run in benchmark["runs"]:
        path = Path(run["result"])
        if not path.exists():
            rows.append(dict(result=str(path), passed=False, reason="worker 无结果"))
            continue
        result = read_json(path)
        cfg = result["resolved_config"]
        same = all(cfg.get(key) == value for key, value in benchmark["config"].items())
        progress = graph_progress(result)
        graph_ok = progress["replays"] > 0 if run["mode"] == "graph" else progress["replays"] == 0
        if run["mode"] == "graph":
            graph_ok = graph_ok and progress.get("capture_failures", 0) == 0
            if result["implementation"] == "cpp" and cfg.get("spec_num_tokens", 0):
                graph_ok = graph_ok and all(progress["parts"][kind] > 0 for kind in ("draft", "verify"))
            if result["implementation"] == "python":
                graph_ok = graph_ok and progress.get("mode") == ("spec_graph" if cfg.get("spec_num_tokens", 0) else "decode_graph")
        hash_ok = sha256(path) == read_json(path.with_suffix(".process.json")).get("result_sha256")
        rows.append(dict(result=str(path), passed=same and graph_ok and hash_ok, config_match=same, result_hash_match=hash_ok, graph_gate_passed=graph_ok, graph=progress))
    write_json(args.output, dict(execution_passed=benchmark["execution_passed"], correctness_passed=benchmark["correctness_passed"], runtime_gates_passed=all(row["passed"] for row in rows), rows=rows, release_qualified=False))
    print({"runtime_gates_passed": all(row["passed"] for row in rows), "correctness_passed": benchmark["correctness_passed"]})


if __name__ == "__main__":
    main()
