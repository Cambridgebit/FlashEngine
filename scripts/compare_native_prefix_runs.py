"""比较两个原生 Prefix 压力矩阵的最终 token；保留输入报告与逐项哈希。"""
import argparse
from pathlib import Path

from flashengine.deployment import read_json, sha256, write_json
from scripts.native_bench_common import compare_rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("left")
    p.add_argument("right")
    p.add_argument("output")
    args = p.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("拒绝覆盖差分报告")
    checks, inputs = [], {}
    for side in (args.left, args.right):
        root = Path(side)
        summary = read_json(root / "summary.json")
        if not summary["passed"]:
            raise ValueError("输入压力测试尚未通过：" + side)
        inputs[side] = dict(summary_sha256=sha256(root / "summary.json"), binary_sha256=summary["binary_sha256"], bundle_sha256=summary["bundle_sha256"])
    if inputs[args.left]["binary_sha256"] != inputs[args.right]["binary_sha256"]:
        raise ValueError("待比较矩阵必须来自同一二进制")
    for tag in ("off-eager", "off-graph", "on-eager", "on-graph"):
        paths = [Path(root) / (tag + ".result.json") for root in (args.left, args.right)]
        values = [read_json(path) for path in paths]
        checks.append(dict(test=tag, result_sha256=[sha256(path) for path in paths], **compare_rows(values[0]["rows"], values[1]["rows"])))
    passed = all(check["passed"] for check in checks)
    write_json(output, dict(passed=passed, release_qualified=False, scope="只验证最终输出，不证明训练草稿的逐轮一致性或性能", inputs=inputs, checks=checks))
    print(output)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
