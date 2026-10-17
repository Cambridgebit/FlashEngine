"""重复在生成中及半包连接存在时停机，检查线程和网络对象生命周期。"""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

from flashengine.deployment import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("native")
    parser.add_argument("bundle")
    parser.add_argument("output")
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.runs < 1 or Path(args.output).exists():
        raise ValueError("非法次数或输出已存在")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    extra = args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args
    report = dict(binary_sha256=sha256(args.native), runs=[], passed=False)
    for trial in range(args.runs):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        command = [str(Path(args.native).resolve()), "serve", args.bundle, "--host", "127.0.0.1", "--port", str(port), "--served-model-name", "shutdown-test", "--no-think", *extra]
        sockets, response = [], None
        item = dict(trial=trial, command=command, passed=False)
        with output.with_suffix(f".{trial}.log").open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                for _ in range(180):
                    if process.poll() is not None:
                        raise RuntimeError(f"启动失败：{process.returncode}")
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=1):
                            break
                    except OSError:
                        time.sleep(.2)
                else:
                    raise TimeoutError("启动超时")
                body = json.dumps(dict(model="shutdown-test", prompt="Count to one thousand:", max_tokens=128, stream=True)).encode()
                response = urllib.request.urlopen(urllib.request.Request(base + "/v1/completions", body, {"Content-Type": "application/json"}), timeout=30)
                for line in response:
                    if line.startswith(b"data: "):
                        break
                # 留下正在读取的半包，以及尚在接入/排队的连接。
                for _ in range(8):
                    pending = socket.create_connection(("127.0.0.1", port), timeout=2)
                    sockets.append(pending)
                    pending.sendall(b"POST /v1/completions HTTP/1.1\r\nHost: local\r\nContent-Length: 9999\r\n\r\n{")
                process.terminate()
                process.wait(timeout=30)
                item["exit_code"] = process.returncode
                item["passed"] = process.returncode == 0
            except Exception as error:
                item["error"] = repr(error)
            finally:
                if response is not None:
                    response.close()
                for pending in sockets:
                    pending.close()
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=10)
                report["runs"].append(item)
        if not item["passed"]:
            break
    report["passed"] = len(report["runs"]) == args.runs and all(r["passed"] for r in report["runs"])
    write_json(output, report)
    print(json.dumps(dict(passed=report["passed"], runs=len(report["runs"]), report=str(output))))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
