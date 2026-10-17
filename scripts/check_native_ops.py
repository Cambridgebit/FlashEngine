"""归档 C++ 基础算子与实际 Python 编译结果的逐元素差分。"""
import argparse
from pathlib import Path
import subprocess
import time

from flashengine.deployment import sha256, write_json
from scripts.check_native_replay import parse_native_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_tests", help="C++ fe-gpu-ops-tests 可执行文件")
    parser.add_argument("oracle", help="export_native_ops_oracle 导出的 safetensors")
    parser.add_argument("output", help="新的差分报告")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("拒绝覆盖算子报告")
    command = [str(Path(args.native_tests).resolve()), str(Path(args.oracle).resolve())]
    start = time.monotonic()
    process = subprocess.run(command, capture_output=True, text=True, timeout=120)
    result, error = parse_native_output(process.stdout)
    passed = process.returncode == 0 and result.get("passed", False) and not error
    report = dict(passed=bool(passed), command=command, elapsed_seconds=time.monotonic() - start, binary_sha256=sha256(args.native_tests), oracle_sha256=sha256(args.oracle), exit_code=process.returncode, result=result, parse_error=error, stderr=process.stderr)
    if error:
        report["invalid_stdout"] = process.stdout
    write_json(args.output, report)
    print(result or report)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
