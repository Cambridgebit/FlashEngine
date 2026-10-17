"""运行原生差分并保存可追溯报告；不自动豁免任何 token/投机轮分歧。"""
import argparse
import json
from pathlib import Path
import subprocess
import time

from flashengine.deployment import read_json, sha256, write_json


def parse_native_output(stdout):
    """崩溃、非 JSON 输出也必须保留为失败报告，而不是丢失现场。"""
    try:
        result = json.loads(stdout)
        if not isinstance(result, dict):
            raise ValueError("输出必须为 JSON 对象")
        return result, None
    except (ValueError, TypeError) as error:
        return {}, str(error)


def compare(golden, result):
    """只比较共同前缀上的 logits；分叉后的 logits 不能再直接对表。"""
    diagnostics = []
    for index, (expected, actual) in enumerate(zip(golden["cases"], result.get("cases", [])), 1):
        a, b = expected["output_token_ids"], actual["token_ids"]
        prefix = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
        info = dict(case=index, expected_tokens=len(a), actual_tokens=len(b), common_prefix=prefix, passed=actual["passed"])
        if prefix < max(len(a), len(b)):
            info["first_difference"] = dict(index=prefix, expected=a[prefix] if prefix < len(a) else None, actual=b[prefix] if prefix < len(b) else None)
            for key, case in (("oracle", expected), ("native", actual)):
                samples = case.get("samples", [])
                if prefix < len(samples):
                    sample = samples[prefix]
                    values = sample["values"]
                    if values and isinstance(values[0], list):
                        # 投机 samples 按轮/批次行组织，不能冒充全局第 prefix 个 token。
                        info["first_difference"][key] = dict(note="投机 logits 按轮/行索引，见首轮差异；未按全局 token 位置对齐")
                    elif len(values) >= 2:
                        info["first_difference"][key] = dict(sample, top2_margin=values[0] - values[1])
        expected_rounds, actual_rounds = expected.get("rounds", []), actual.get("rounds", [])
        info["round_counts"] = [len(expected_rounds), len(actual_rounds)]
        for step, (reference, observed) in enumerate(zip(expected_rounds, actual_rounds)):
            if reference != observed:
                info["first_round_difference"] = dict(round=step, oracle=reference, native=observed)
                for key, case in (("oracle", expected), ("native", actual)):
                    samples = case.get("samples", [])
                    if len(samples) > step * 2 + 1:
                        info["first_round_difference"][key + "_logits"] = samples[step * 2:step * 2 + 2]
                break
        diagnostics.append(info)
    return diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native")
    parser.add_argument("bundle")
    parser.add_argument("oracle")
    parser.add_argument("output")
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("报告已存在，拒绝覆盖")
    extra = args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args
    command = [str(Path(args.native).resolve()), "replay", args.bundle, "--oracle", args.oracle, *extra]
    start = time.monotonic()
    timed_out = False
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=600)
        stdout, stderr, exit_code = process.stdout, process.stderr, process.returncode
    except subprocess.TimeoutExpired as error:
        timed_out = True
        def as_text(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        stdout, stderr, exit_code = as_text(error.stdout), as_text(error.stderr), None
    result, parse_error = parse_native_output(stdout)
    golden = read_json(args.oracle)
    complete = bool(golden["cases"]) and len(result.get("cases", [])) == len(golden["cases"])
    passed = exit_code == 0 and not timed_out and not parse_error and complete and result.get("passed", False)
    diagnostic_error = None
    try:
        diagnostics = compare(golden, result)
    except (KeyError, TypeError, IndexError, ValueError) as error:
        diagnostics, diagnostic_error, passed = [], repr(error), False
    report = dict(passed=bool(passed), command=command, elapsed_seconds=time.monotonic() - start, exit_code=exit_code, timed_out=timed_out, parse_error=parse_error, diagnostic_error=diagnostic_error, binary_sha256=sha256(args.native), oracle_sha256=sha256(args.oracle), bundle_sha256=sha256(Path(args.bundle) / "bundle.json"), result=result, stderr=stderr, diagnostics=diagnostics)
    if parse_error:
        report["invalid_stdout"] = stdout
    write_json(args.output, report)
    print(json.dumps(dict(passed=bool(passed), diagnostics=report["diagnostics"], runtime=result.get("runtime"), report=args.output), ensure_ascii=False, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
