"""OpenAI 兼容 HTTP server（给 Cherry Studio / vLLM client 用）。

特性：
- 并发请求 + continuous batching：后台引擎驱动线程统一调度所有排队请求，
  不再全局串行 —— 多客户端 / 多会话可同时请求（vLLM 式）。
- 纯 Python stdlib，零新依赖（无需 fastapi / uvicorn）。
- 真逐 token SSE 流式：每 decode step 后取新增 token，decode 后按文本 diff
  推送（兼容 CJK 多字节切分）。
- 标准 OpenAI 端点：/v1/models、/v1/chat/completions、/v1/completions、/health。
- 参数：max_tokens / temperature / top_p（转发给 SamplingParams）/ stop（文本级截断并中止生成）/
  n=1 / stream / stream_options.include_usage。
- 只使用 nano-vLLM 公开 API（LLM、Sequence、SamplingParams、scheduler），不修改库文件。

用法（必须从仓库根目录以 -m 方式运行，直跑 scripts/X.py 会 ModuleNotFoundError）：
    python3 -m scripts.openai_server /path/to/Qwen3-1.7B [--port 8000]
                  [--api-key xxx] [--eager] [--max-model-len 4096]
                  [--temperature 0.7] [--backend auto]

Cherry Studio 配置：OpenAI 兼容 provider，
    Base URL = http://127.0.0.1:8000/v1
    API Key  = 任意（或 --api-key 指定的值）
"""
import argparse
import json
import threading
import time

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus


# ---------------------------------------------------------------------------
# 全局引擎 + 并发控制
# ---------------------------------------------------------------------------

engine = None
_tokenizer = None
_eos = None
_model_name = "nanovllm"
_api_key = None
_default_temperature = 0.7
_max_model_len = 4096

engine_lock = threading.RLock()      # 保护 add / step / abort / _active
_wakeup = threading.Event()          # 有请求加入 / 中止时唤醒 driver
_shutdown = threading.Event()
_active = {}                         # seq_id -> _Request


class _Request:
    """单个请求的状态：driver 写入，HTTP 线程读取。"""

    def __init__(self, seq, max_tokens):
        self.seq = seq
        self.max_tokens = max_tokens
        self.prompt_tokens = seq.num_prompt_tokens
        self.tokens = deque()        # driver 追加已生成的 token（eos 已剔除）
        self.n_seen = 0
        self.event = threading.Event()
        self.finished = False
        self.finish_reason = None
        self.error = None


def _drain(queue):
    out = []
    while queue:
        out.append(queue.popleft())
    return out


def _abort(seq):
    """立即终止一个 seq：标记 FINISHED 并释放其 KV blocks、移出调度队列。"""
    with engine_lock:
        seq.status = SequenceStatus.FINISHED
        scheduler = engine.scheduler
        scheduler.block_manager.deallocate(seq)
        try:
            scheduler.running.remove(seq)
        except ValueError:
            pass
        try:
            scheduler.waiting.remove(seq)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# 后台引擎驱动线程（continuous batching）
# ---------------------------------------------------------------------------

def _engine_loop():
    """循环调用 engine.step()，把新增 token / 结束事件分发给各请求。"""
    while not _shutdown.is_set():
        with engine_lock:
            if _active and not engine.is_finished():
                try:
                    engine.step()   # 活跃期连续步进：不等待 _wakeup（旧的 50ms 等待门
                                    # 会让每个 decode 步白等 50ms，流式 TPS 被压到 ~14）
                except Exception as exc:    # 引擎异常：让所有存活请求报错退出
                    for st in list(_active.values()):
                        _abort(st.seq)
                        st.error = str(exc)
                        st.finished = True
                        st.event.set()
                    _active.clear()
                    continue
                done = []
                for st in list(_active.values()):
                    seq = st.seq
                    comp = seq.completion_token_ids
                    if len(comp) > st.n_seen:
                        for tok in comp[st.n_seen:]:
                            if tok != _eos:
                                st.tokens.append(tok)
                        st.n_seen = len(comp)
                    if seq.is_finished:
                        st.finish_reason = (
                            "length" if seq.num_completion_tokens >= st.max_tokens
                            else "stop"
                        )
                        st.finished = True
                        st.event.set()
                        done.append(seq.seq_id)
                for seq_id in done:
                    _active.pop(seq_id, None)
                continue
            # 空转前清扫残留：stop 命中 / 客户端断连等路径会把 seq 置为 FINISHED，
            # 但不经过上面的 fanout 分支（调度器已空时 is_finished() 恒真，分支不再
            # 进入），残留的 _active 条目会造成 20Hz 空转轮询与小泄漏——统一在此清理。
            done = [seq_id for seq_id, st in _active.items() if st.seq.is_finished]
            for seq_id in done:
                _active.pop(seq_id, None)
        _wakeup.wait(timeout=0.05)   # 仅空转时等待唤醒
        _wakeup.clear()


# ---------------------------------------------------------------------------
# 生成辅助
# ---------------------------------------------------------------------------

def _apply_stop(text, stop):
    """在文本里找第一个 stop 串出现位置；未命中返回原文本。"""
    if not stop:
        return text, False
    p = -1
    for s in stop:
        q = text.find(s)
        if q != -1 and (p == -1 or q < p):
            p = q
    return (text[:p], True) if p != -1 else (text, False)


def _run_llm(st, stop, emit):
    """驱动到结束；返回 (text, finish_reason, completion_tokens)。

    emit(delta_text) 在流式下逐 delta 调用；None 表示非流式。
    客户端断连（emit 抛 IOError）后继续跑完请求但不推送。
    增量 detokenization：每次唤醒只解码新增 token 中能完整落在 UTF-8 字符
    边界上的前缀（pending 携带跨字节残留），不再对全量 token 列表重解码
    （旧实现每次唤醒 decode 整个 gen，O(n²)/请求，且在引擎线程释放窗口内
    长时间持有 GIL）。finish 时仍做一次全量解码兜底，保证最终文本与旧
    实现逐字节一致。
    """
    if emit is None:    # 非流式：吞掉增量推送，只累积最终文本
        emit = lambda _delta: None
    gen = []
    pending = []        # 尚不能安全解码的 token 尾部（UTF-8 多字节边界残留）
    prev_text = ""
    final = ""
    reason = "stop"
    dead = False

    def send(delta_text):
        nonlocal dead
        if delta_text and not dead:
            try:
                emit(delta_text)
            except (BrokenPipeError, ConnectionResetError, OSError):
                dead = True

    while True:
        if st.finished:
            gen.extend(_drain(st.tokens))
            final, _ = _apply_stop(
                _tokenizer.decode(gen, skip_special_tokens=True), stop)
            reason = st.finish_reason or "stop"
            send(final[len(prev_text):])
            prev_text = final
            break
        if st.event.wait(timeout=0.05):
            st.event.clear()
        new = _drain(st.tokens)
        if not new:
            continue    # 无新 token：跳过解码（上一轮文本已推送）
        gen.extend(new)
        pending.extend(new)
        # 边界回退策略：整段 pending 解码若以替换符 \ufffd 结尾，说明末尾
        # token 可能只覆盖了某个多字节字符的前半部分 —— 从尾部回退一个
        # token 再试；单 token 仍不完整时整体留到下次唤醒（finish 全量
        # 解码兜底）。字节级 BPE 下，安全边界两侧的 decode 结果与全量
        # decode 逐字节一致（utf-8 自同步，errors=replace 只在非法边界生效）。
        piece = ""
        keep = len(pending)
        while keep > 0:
            text = _tokenizer.decode(pending[:keep], skip_special_tokens=True)
            if text.endswith("\ufffd"):
                keep -= 1    # 末尾 token 疑似跨字节字符的前半，回退
                continue
            piece = text
            break
        if not piece:
            continue    # 全部残留：本轮无可安全推送的文本
        del pending[:keep]
        full = prev_text + piece
        final, hit = _apply_stop(full, stop)
        if hit:
            _abort(st.seq)
            st.finished = True
            reason = "stop"
            send(final[len(prev_text):])
            prev_text = final
            break
        send(final[len(prev_text):])
        prev_text = final

    if st.error:
        raise RuntimeError(st.error)
    return prev_text, reason, len(gen)


def _chat_prompt_ids(messages):
    normalized = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            content = "".join(parts)
        if content is None:
            content = ""
        normalized.append({"role": role, "content": content})
    return _tokenizer.apply_chat_template(
        normalized, tokenize=True, add_generation_prompt=True)["input_ids"]


def _completion_prompt_ids(body):
    prompt = body.get("prompt")
    if prompt is None:
        raise ValueError("'prompt' is required")
    if isinstance(prompt, str):
        return _tokenizer.encode(prompt)
    if isinstance(prompt, list) and prompt:
        if all(isinstance(x, int) for x in prompt):
            return list(prompt)
        if all(isinstance(x, str) for x in prompt):
            ids = []
            for s in prompt:
                ids.extend(_tokenizer.encode(s))
            return ids
    raise ValueError("invalid 'prompt'")


def _openai_body(req_id, model, mode, content, finish_reason, usage):
    if mode == "chat":
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
            "usage": usage,
        }
    return {
        "id": req_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "text": content,
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }


def _chunk(req_id, model, mode, delta_text, finish_reason=None, role=False):
    if role:
        delta = {"role": "assistant"}
    else:
        delta = {"text": delta_text} if mode == "completion" and delta_text else \
                {"content": delta_text} if mode == "chat" and delta_text else {}
    return {
        "id": req_id,
        "object": "chat.completion.chunk" if mode == "chat" else "text_completion",
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

    server_version = "nanovllm-openai/0.2"

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
        self._json(
            {"error": {"message": message, "type": "invalid_request_error"}}, code)

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
                    "id": _model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "nanovllm",
                }],
            })
            return
        if self.path in ("/health", "/v1/health"):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"OK")
            return
        self._error(f"unknown path: {self.path}", 404)

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/v1/completions"):
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
        mode = "chat" if self.path == "/v1/chat/completions" else "completion"
        self._generate(body, mode)

    # ---- 生成 ----

    def _generate(self, body, mode):
        model = body.get("model", _model_name)
        stream = bool(body.get("stream", False))
        n = int(body.get("n", 1))
        if n != 1:
            self._error("'n' != 1 is not supported", 400)
            return
        max_tokens = int(body.get("max_tokens") or 512)
        if max_tokens < 1:
            max_tokens = 1
        temperature = body.get("temperature")
        if temperature is None:
            temperature = _default_temperature
        temperature = max(float(temperature), 0.0)    # 0 = greedy（引擎已支持）
        top_p = body.get("top_p")
        if top_p is None:
            top_p = 1.0
        top_p = min(max(float(top_p), 1e-6), 1.0)
        top_k = body.get("top_k")
        top_k = max(int(top_k), 0) if top_k is not None else 0
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        elif not isinstance(stop, list):
            stop = []
        stop = [s for s in stop if isinstance(s, str) and s]
        include_usage = bool((body.get("stream_options") or {}).get("include_usage", False))

        try:
            prompt_ids = (_chat_prompt_ids(body["messages"]) if mode == "chat"
                          else _completion_prompt_ids(body))
        except Exception as exc:
            self._error(f"prompt build failed: {exc}", 400)
            return

        if len(prompt_ids) >= _max_model_len:
            print(f"[warn] prompt truncated to {_max_model_len - 1} tokens", flush=True)
            prompt_ids = prompt_ids[:_max_model_len - 1]

        req_id = f"chatcmpl-{time.time_ns()}" if mode == "chat" else f"cmpl-{time.time_ns()}"

        seq = Sequence(prompt_ids, SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k))
        with engine_lock:
            engine.scheduler.add(seq)
            st = _Request(seq, max_tokens)
            _active[seq.seq_id] = st
            _wakeup.set()

        usage = {
            "prompt_tokens": st.prompt_tokens,
            "completion_tokens": None,
            "total_tokens": None,
        }

        def emit(delta_text):
            payload = json.dumps(
                _chunk(req_id, model, mode, delta_text), ensure_ascii=False
            ).encode("utf-8")
            self.wfile.write(b"data: " + payload + b"\n\n")
            self.wfile.flush()

        try:
            if not stream:
                text, reason, ct = _run_llm(st, stop, None)
                usage.update(completion_tokens=ct, total_tokens=st.prompt_tokens + ct)
                self._json(_openai_body(req_id, model, mode, text, reason, usage))
                return

            # ---- SSE 流式 ----
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self._cors()
            self.end_headers()

            if mode == "chat":
                first = json.dumps(_chunk(req_id, model, mode, "", role=True),
                                   ensure_ascii=False).encode("utf-8")
                self.wfile.write(b"data: " + first + b"\n\n")
                self.wfile.flush()

            text, reason, ct = _run_llm(st, stop, emit)

            done = json.dumps(_chunk(req_id, model, mode, "", reason),
                              ensure_ascii=False).encode("utf-8")
            self.wfile.write(b"data: " + done + b"\n\n")

            if include_usage:
                usage.update(completion_tokens=ct, total_tokens=st.prompt_tokens + ct)
                usage_chunk = {
                    "id": req_id,
                    "object": "chat.completion.chunk" if mode == "chat" else "text_completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [],
                    "usage": usage,
                }
                payload = json.dumps(usage_chunk, ensure_ascii=False).encode("utf-8")
                self.wfile.write(b"data: " + payload + b"\n\n")

            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            pass    # 客户端断开：请求已在 _run_llm 内跑完/中止
        except Exception as exc:
            try:
                self._error(f"generation failed: {exc}", 500)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass


def main():
    global engine, _tokenizer, _eos, _model_name, _api_key
    global _default_temperature, _max_model_len

    parser = argparse.ArgumentParser(description="nano-vLLM OpenAI-compatible server (TP=1)")
    parser.add_argument("model", help="model directory path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--eager", action="store_true",
                        help="enforce_eager=True（跳过 CUDA Graph capture，启动更快）")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile 编译模型（CoreX 上约 8% prefill 提速）")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None,
                        help="KV block size（2 的幂，越小 prefix cache 粒度越细，需 CoreX kernel 支持）")
    parser.add_argument("--spec-model", default="",
                        help="EAGLE3 draft 模型目录（启用投机解码，需与 target 同词表的官方权重）")
    parser.add_argument("--spec-num-tokens", type=int, default=4,
                        help="每轮起草 token 数 k（draft depth）")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    _api_key = args.api_key
    _default_temperature = args.temperature

    kwargs = dict(
        model=args.model,
        enforce_eager=args.eager,
        compile=args.compile,
        tensor_parallel_size=1,       # server 只支持 TP=1（引擎同进程驱动）
        max_model_len=args.max_model_len,
        backend=args.backend,
        spec_model=args.spec_model,
        spec_num_tokens=args.spec_num_tokens,
    )
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    else:
        # 8GB 卡上大 max_num_seqs 的宽桶/宽 buffer 拖慢单步回放，实测 nospec
        # 512→8 = 59.0→66.8 tok/s；需要更高并发的用户显式传 --max-num-seqs。
        # spec 模式此前已默认 8（捕获池随 max_bs 增长），现统一。
        kwargs["max_num_seqs"] = 8
    if args.block_size is not None:
        kwargs["kvcache_block_size"] = args.block_size

    print(f"loading model {args.model} ...", flush=True)
    engine = LLM(**kwargs)
    _tokenizer = engine.tokenizer
    _eos = engine.scheduler.eos
    _max_model_len = engine.model_runner.config.max_model_len
    _model_name = args.model.rstrip("/").rsplit("/", 1)[-1]
    print(f"model '{_model_name}' ready", flush=True)
    spec_info = f"ON(k={args.spec_num_tokens}, {args.spec_model.rstrip('/').rsplit('/', 1)[-1]})" if args.spec_model else "OFF"
    ms_info = args.max_num_seqs if args.max_num_seqs is not None else 8
    print(f"eos={_eos} | eager={args.eager} | spec={spec_info} | max_model_len={_max_model_len} | max_num_seqs={ms_info}", flush=True)

    threading.Thread(target=_engine_loop, daemon=True).start()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"OpenAI server on http://{args.host}:{args.port}/v1", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown.set()
        _wakeup.set()
        httpd.server_close()


if __name__ == "__main__":
    main()