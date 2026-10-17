"""使用 Compute Sanitizer 检查指定原生 CUDA 命令，保存完整失败现场。"""
import argparse
from pathlib import Path
import shutil
import subprocess
import time

from flashengine.deployment import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("native", help="待测 ELF，部署程序或独立设备测试程序")
    p.add_argument("output")
    p.add_argument("native_args", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if Path(args.output).exists():
        raise ValueError("拒绝覆盖 Compute Sanitizer 报告")
    tool = shutil.which("compute-sanitizer")
    if tool is None and Path("/usr/local/cuda/bin/compute-sanitizer").is_file():
        tool = "/usr/local/cuda/bin/compute-sanitizer"
    report = dict(available=tool is not None, binary_sha256=sha256(args.native), passed=False, scope="仅覆盖本报告 command 中实际执行的路径，不代替全部模型/并发内存验收")
    if tool:
        extra = args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args
        command = [tool, "--tool", "memcheck", "--error-exitcode", "86", str(Path(args.native).resolve()), *extra]
        report["command"] = command
        start = time.monotonic()
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            report.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
            report["passed"] = result.returncode == 0 and "ERROR SUMMARY: 0 errors" in result.stdout + result.stderr
        except subprocess.TimeoutExpired as error:
            report.update(timed_out=True, stdout=(error.stdout or b"").decode(errors="replace"), stderr=(error.stderr or b"").decode(errors="replace"))
        except OSError as error:
            report["error"] = repr(error)
        report["elapsed_seconds"] = time.monotonic() - start
    write_json(args.output, report)
    print({key: report[key] for key in ("available", "passed", "scope")})
    raise SystemExit(0 if report["passed"] else 2)


if __name__ == "__main__":
    main()
