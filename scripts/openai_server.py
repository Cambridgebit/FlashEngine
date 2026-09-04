"""OpenAI 兼容 HTTP server（给 Cherry Studio 等客户端用）。

- 只使用 nano-vLLM 公开/半公开 API（LLMEngine.add_request/step/is_finished、
  engine.tokenizer / engine.scheduler），不修改任何库文件。
- 纯 Python stdlib，零新依赖（无需 fastapi / uvicorn）。
- 真逐 token SSE 流式：每 decode step 后从存活 seq 的 completion 里取出
  新增 token，decode 后以文本 diff 形式推送（兼容 CJK 多字节切分）。
- 并发模型：全局锁串行化（单引擎、单 GPU，个人使用足够）。

用法：
    python3 scripts/openai_server.py /path/to/Qwen3-1.7B [--port 8000]
                  [--api-key xxx] [--eager] [--max-model-len 2048]

Cherry Studio 配置：OpenAI 兼容 provider，
    Base URL = http://127.0.0.1:8000/v1
    API Key  = 任意（或 --api-key 指定的值）
"""
import argparse
import json
import threading
import time

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nanovllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# 全局引擎 + 串行锁
# ---------------------------------------------------------------------------

engine = None
lock = threading.Lock()
_default_temperature = 0.6
_api_key = None
_eos = None


def _drain_request(seq, max_tokens):
    """驱动引擎直到该 seq 结束。由调用方持有 lock。"""
    while not seq.is_finished:
        engine.step()
    return


def _run(seq):
    """跑完一个请求，返回 (text, prompt_len, completion_len)。

    seq 已在 add_request 后捕获（Sequence 对象在队列移除后仍存活）。
    eos token 不进入输出文本（与 OpenAI 语义一致）。
    """
    full_ids = []
    prev_text = ""
    while not seq.is_finished:
        engine.step()
        comp = seq.completion_token_ids
        for tok in comp[len(full_ids):]:
            if tok != _eos:
                full_ids.append(tok)
        text = engine.tokenizer.decode(full_ids, skip_special_tokens=True)
        if len(text) > len(prev_text):
            prev_text = text
    prompt_len = seq.num_prompt_tokens
    return prev_text, prompt_len, len(full_ids)


def _run_streaming(seq, emit):
    """流式跑完一个请求；emit(delta_text) 每有新 token 调用一次。

    客户端断连后 emit 会抛异常 —— 捕获后继续把请求跑完（不推数据），
    避免把引擎留在半途状态。
    """
    full_ids = []
    prev_text = ""
    dead = False
    while not seq.is_finished:
        engine.step()
        comp = seq.completion_token_ids
        for tok in comp[len(full_ids):]:
            if tok != _eos:
                full_ids.append(tok)
        text = engine.tokenizer.decode(full_ids, skip_special_tokens=True)
        delta = text[len(prev_text):]
        prev_text = text
        if delta and not dead:
            try:
                emit(delta)
            except (BrokenPipeError, ConnectionResetError, OSError):
                dead = True
    return prev_text, seq.num_prompt_tokens, len(full_ids)


# ---------------------------------------------------------------------------
# OpenAI 协议辅助
# ---------------------------------------------------------------------------

def _chat_prompt(messages):
    return engine.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )


def _openai_body(req_id, model, content, finish_reason, usage=None):
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
        }],
        "usage": usage or {},
    }


def _chunk(req_id, model, delta_text, finish_reason=None):
    delta = {"content": delta_text} if delta_text else {}
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish_reason,
        }],
    }


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):

    server_version = "nanovllm-openai/0.1"

    def log_message(self, fmt, *args):
        print("[http]", self.address_string(), fmt % args, flush=True)

    # ---- CORS / 预检 ----

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    # ---- 工具 ----

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, message, code=500):
        self._json({"error": {"message": message, "type": "invalid_request_error"}}, code)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _check_auth(self):
        if not _api_key:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {_api_key}"

    # ---- 路由 ----

    def do_GET(self):
        if self.path in ("/v1/models", "/models"):
            self._json({
                "object": "list",
                "data": [{
                    "id": engine_model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nanovllm",
                }],
            })
            return
        self._error(f"unknown path: {self.path}", 404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._error(f"unknown path: {self.path}", 404)
            return
        if not self._check_auth():
            self._error("invalid api key", 401)
            return
        try:
            body = self._read_body()
        except Exception:
            self._error("invalid json body", 400)
            return

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._error("'messages' is required", 400)
            return

        model = body.get("model", engine_model_name)
        stream = bool(body.get("stream", False))
        max_tokens = int(body.get("max_tokens") or 64)
        temperature = body.get("temperature")
        if temperature is None:
            temperature = _default_temperature
        temperature = max(float(temperature), 1e-6)    # nano 禁止 0：等效 greedy

        try:
            prompt_ids = _chat_prompt(messages)
        except Exception as exc:
            self._error(f"chat template failed: {exc}", 400)
            return

        req_id = f"chatcmpl-{time.time_ns()}"
        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": None,
            "total_tokens": None,
        }

        # ---- 全局串行化：单引擎一次只服务一个请求 ----
        acquired = lock.acquire(timeout=1)
        if not acquired:
            self._error("server busy (previous request still running)", 429)
            return
        try:
            engine.add_request(prompt_ids, SamplingParams(
                temperature=temperature, max_tokens=max_tokens))
            seq = engine.scheduler.waiting[0]    # 串行下唯一 seq，捕获对象引用

            if not stream:
                text, pt, ct = _run(seq)
                usage.update(completion_tokens=ct, total_tokens=pt + ct)
                self._json(_openai_body(req_id, model, text, "stop", usage))
                return

            # ---- SSE 流式 ----
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self._cors()
            self.end_headers()

            def emit(delta_text):
                payload = json.dumps(
                    _chunk(req_id, model, delta_text), ensure_ascii=False
                ).encode("utf-8")
                self.wfile.write(b"data: " + payload + b"\n\n")
                self.wfile.flush()

            text, pt, ct = _run_streaming(seq, emit)

            emit_done = json.dumps(
                _chunk(req_id, model, "", "stop"), ensure_ascii=False
            ).encode("utf-8")
            self.wfile.write(b"data: " + emit_done + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            usage.update(completion_tokens=ct, total_tokens=pt + ct)
            # OpenAI 不在 SSE 里带 usage（此处保持简单，可忽略）
        except BrokenPipeError:
            pass    # 客户端断开：请求已在 _run_streaming 内跑完
        except Exception as exc:
            try:
                self._error(f"generation failed: {exc}", 500)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        finally:
            lock.release()


def main():
    global engine, _api_key, _eos, engine_model_name

    parser = argparse.ArgumentParser(description="nano-vLLM OpenAI-compatible server (TP=1)")
    parser.add_argument("model", help="model directory path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--eager", action="store_true",
                        help="enforce_eager=True（跳过 CUDA Graph capture，启动更快）")
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()

    global _default_temperature
    _default_temperature = args.temperature
    _api_key = args.api_key

    print(f"loading model {args.model} ...", flush=True)
    engine = LLM(
        args.model,
        enforce_eager=args.eager,
        tensor_parallel_size=1,    # server 只支持 TP=1（引擎同进程驱动）
        max_model_len=args.max_model_len,
    )
    _eos = engine.scheduler.eos
    engine_model_name = args.model.rstrip("/").rsplit("/", 1)[-1]
    print(f"model '{engine_model_name}' ready", flush=True)
    print(f"eos={_eos} | eager={args.eager}", flush=True)

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"OpenAI server on http://{args.host}:{args.port}/v1", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
