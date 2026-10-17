"""原生 HTTP/SSE 实机回归：协议、并发、文本 stop、断连回收与正常停机。"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from flashengine.deployment import sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native")
    parser.add_argument("bundle")
    parser.add_argument("output")
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("报告已存在，拒绝覆盖")
    output.parent.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    headers = {"Authorization": "Bearer native-test", "Content-Type": "application/json"}
    extra = args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args
    command = [str(Path(args.native).resolve()), "serve", args.bundle, "--host", "127.0.0.1", "--port", str(port), "--api-key", "native-test", "--served-model-name", "flashengine", "--no-think", *extra]
    checks = []

    def check(value, name):
        if not value:
            raise AssertionError(name)
        checks.append(name)

    def request(path, body=None, authorized=True):
        payload = json.dumps(body).encode() if body is not None else None
        return urllib.request.urlopen(urllib.request.Request(base + path, payload, headers if authorized else {}), timeout=60)

    def get(path):
        with request(path) as response:
            return json.load(response)

    def post(body, chat=False):
        with request("/v1/chat/completions" if chat else "/v1/completions", body) as response:
            return json.load(response)

    def sse(body, chat=False):
        parts, chunks, done = [], [], False
        with request("/v1/chat/completions" if chat else "/v1/completions", dict(body, stream=True, stream_options={"include_usage": True})) as response:
            check(response.headers.get_content_type() == "text/event-stream", "SSE content-type")
            for raw in response:
                if not raw.startswith(b"data: "):
                    continue
                data = raw[6:].decode().strip()
                if data == "[DONE]":
                    done = True
                    break
                chunk = json.loads(data)
                chunks.append(chunk)
                for choice in chunk["choices"]:
                    parts.append(choice["delta"].get("content", "") if chat else choice["text"])
        check(done, "SSE DONE")
        check(chunks[-1].get("usage", {}).get("completion_tokens", 0) > 0, "SSE usage")
        return "".join(parts), chunks

    report = dict(command=command, binary_sha256=sha256(args.native), bundle_sha256=sha256(Path(args.bundle) / "bundle.json"), checks=checks, passed=False)
    with output.with_suffix(".log").open("x", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            for _ in range(180):
                if process.poll() is not None:
                    raise RuntimeError(f"原生服务启动失败：{process.returncode}；参见 {log.name}")
                try:
                    if get("/health")["status"] == "ok":
                        break
                except (OSError, urllib.error.URLError):
                    time.sleep(.5)
            else:
                raise TimeoutError("服务启动超时")
            check(get("/v1/models")["data"][0]["id"] == "flashengine", "models")
            try:
                request("/v1/models", authorized=False)
                raise AssertionError("未鉴权请求被接受")
            except urllib.error.HTTPError as error:
                check(error.code == 401, "authentication 401")
            body = dict(model="flashengine", prompt="The capital of France is", max_tokens=8, temperature=0)
            ordinary = post(body)
            check(ordinary["usage"]["completion_tokens"] == 8, "nonstream usage")
            streamed, _ = sse(body)
            check(streamed == ordinary["choices"][0]["text"], "SSE/nonstream identical")
            stop = streamed[1:5]
            stopped, chunks = sse(dict(body, stop=stop))
            check(stopped == streamed.split(stop)[0], "cross-token text stop")
            check(chunks[-2]["choices"][0]["finish_reason"] == "stop", "stop finish reason")
            chat = dict(model="flashengine", messages=[dict(role="system", content="Be concise."), dict(role="user", content="你好")], max_tokens=8, chat_template_kwargs=dict(enable_thinking=False))
            chat_result = post(chat, True)
            chat_stream, _ = sse(chat, True)
            check(chat_stream == chat_result["choices"][0]["message"]["content"], "chat SSE/nonstream identical")
            for invalid in (dict(n=2), dict(temperature=.1), dict(max_tokens=1.5), dict(tools=[]), dict(prompt=[-1]), dict(chat_template_kwargs=dict(unknown=True))):
                try:
                    post(dict(body, **invalid))
                    raise AssertionError(f"非法请求被接受：{invalid}")
                except urllib.error.HTTPError as error:
                    check(error.code == 400, f"invalid 400: {next(iter(invalid))}")
            before = get("/v1/stats")["metrics"]
            barrier = threading.Barrier(8)

            def concurrent(index):
                barrier.wait(timeout=10)
                result = post(dict(body, prompt=f"Count from {index} to 100:", max_tokens=32))
                return result["usage"]["completion_tokens"]

            peaks = dict(running=0, waiting=0, active=0)
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(concurrent, index) for index in range(8)]
                while not all(f.done() for f in futures):
                    metrics = get("/v1/stats")["metrics"]
                    for key in peaks:
                        peaks[key] = max(peaks[key], metrics[f"flashengine_{key}_requests"])
                    time.sleep(.01)
                total = sum(f.result() for f in futures)
            after = get("/v1/stats")["metrics"]
            check(after["flashengine_generated_tokens_total"] - before["flashengine_generated_tokens_total"] == total, "concurrent token counters")
            check(peaks["active"] >= 2, "continuous batching observed")
            report["concurrency_peak"] = peaks
            cancelled_before = after["flashengine_cancelled_requests_total"]
            response = request("/v1/completions", dict(body, stream=True, max_tokens=128))
            for line in response:
                if line.startswith(b"data: "):
                    break
            response.close()
            for _ in range(100):
                metrics = get("/v1/stats")["metrics"]
                if metrics["flashengine_active_requests"] == 0 and metrics["flashengine_waiting_requests"] == 0:
                    break
                time.sleep(.02)
            check(metrics["flashengine_cancelled_requests_total"] > cancelled_before, "disconnect cancellation")
            check(metrics["flashengine_kv_blocks_used"] == 0, "KV recycled")
            with request("/metrics") as response:
                prom = dict(line.split() for line in response.read().decode().splitlines())
            check(float(prom["flashengine_generated_tokens_total"]) == metrics["flashengine_generated_tokens_total"], "stats/metrics same snapshot")
            report["snapshot"] = get("/v1/stats")
            report["passed"] = True
        except Exception as error:
            report["error"] = repr(error)
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            report["shutdown_exit_code"] = process.returncode
            report["passed"] = report["passed"] and process.returncode == 0
            write_json(output, report)
            print(json.dumps(dict(passed=report["passed"], checks=len(checks), error=report.get("error"), report=str(output)), ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
