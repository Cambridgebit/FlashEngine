"""OpenAI 兼容 HTTP server（continuous batching + SSE 流式，纯 stdlib 零新依赖）。

端点：/v1/models、/v1/chat/completions、/v1/completions、/health（含无 /v1 前缀别名）。
参数：max_tokens（未传 = 生成到上下文顶）/ temperature / top_p / top_k / stop（文本级截断并中止生成）/
stream / stream_options.include_usage；n 仅支持 1。
上下文淘汰：chat 请求 prompt 超出 max_model_len 时，保留 system + 最近轮次、从最旧历史起逐条丢弃
（不再砍尾巴吞掉最新提问）；为客户未指定 max_tokens 的请求预留 --reserve-output（默认 512）token
作为输出预算。completions 超长时保留尾部（最新内容）。
并发模型：后台引擎驱动线程统一 step() 驱动所有排队请求（continuous batching），
HTTP 线程只负责收发与流式推送。

服务端日志（固定 5 字符标签 + 毫秒时间戳；TTY 下自动上色；rid 贯穿 HTTP/REQ/DONE）：
- [HTTP ] 传输层：rid/method/path/status/client/响应字节/耗时；
- [REQ  ] 引擎受理（首次被调度时）：rid/端点/stream/prompt/max_out/采样/thinking/淘汰数/引擎调度等待 wait；
- [DONE ] 单请求完成：rid/in(输入 prompt token)/out(输出 token)/E(平均接受长度)/TTFT(实际模型处理)/TPS/ctx(上下文占用)/cache(本请求命中)/finish；
- [INFO ] 周期聚合（每 TPS_INTERVAL 秒，仅模型推理相关）：winTPS/active/cache_hit/avgE；
- [WARN ] / [ERROR] 异常（同宽度）。

用法（必须从仓库根目录以 -m 方式运行，直跑 scripts/X.py 会 ModuleNotFoundError）：
    python3 -m scripts.openai_server /path/to/Qwen3-1.7B [--port 8000]
        [--spec-model <draft 目录>] [--spec-num-tokens k] [--eager] [--no-think] [--reserve-output N] [--api-key xxx]
"""
import argparse
import json
import sys
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
# chat template 的 enable_thinking 默认：None = 交给模板默认（Qwen3 = on）；
# False = 显式关（--no-think，对齐 benchmark 历史 E 锚点口径）。每请求可用
# chat_template_kwargs.enable_thinking 覆盖。
_default_thinking = None
# 上下文淘汰：客户端未显式给 max_tokens 时，为输出预留的 token 数（--reserve-output）。
# prompt（system + 历史 + 最新）超出 max_model_len - reserve 时，从最旧历史起淘汰。
_default_reserve = 512

# ---- 日志（固定宽度标签 + 毫秒时间戳 + TTY 上色）----
_TAG_W = 5
_TAG_COLORS = {
    "HTTP": "\033[90m", "REQ": "\033[94m", "DONE": "\033[92m",
    "INFO": "\033[96m", "WARN": "\033[93m", "ERROR": "\033[91m",
}
_RESET = "\033[0m"
_use_color = sys.stdout.isatty()


def _ts():
    return f"{time.strftime('%H:%M:%S')}.{(time.time_ns() // 1_000_000) % 1000:03d}"


def _log(tag, msg):
    """统一日志：`HH:MM:SS.mmm [TAG  ] msg`；标签内容固定 _TAG_W 宽，TTY 上色。"""
    label = f"[{tag:<{_TAG_W}}]"
    if _use_color:
        color = _TAG_COLORS.get(tag)
        if color:
            label = f"{color}{label}{_RESET}"
    print(f"{_ts()} {label} {msg}", flush=True)


class _CountingWriter:
    """包一层 wfile 以统计每次 HTTP 响应的字节数（[HTTP] 行用）。"""

    def __init__(self, wfile, box):
        self._wfile = wfile
        self._box = box

    def write(self, data):
        self._box["n"] = self._box.get("n", 0) + len(data)
        return self._wfile.write(data)

    def flush(self):
        return self._wfile.flush()

    def writelines(self, lines):
        for line in lines:
            self.write(line)

    def __getattr__(self, name):
        return getattr(self._wfile, name)   # 透传 closed 等底层属性


# 投机解码累计统计（[INFO] 的平均 E）；非 spec 恒 0。
_spec_rounds = 0
_spec_committed = 0

engine_lock = threading.RLock()      # 保护 add / step / abort / _active
_wakeup = threading.Event()          # 有请求加入 / 中止时唤醒 driver
_shutdown = threading.Event()
_active = {}                         # seq_id -> _Request

TPS_INTERVAL = 1.0                   # [INFO] 周期聚合打印间隔（秒）
_gen_lock = threading.Lock()
_gen_total = 0                       # 服务启动以来累计生成 token 数
_gen_window = 0                      # 当前 TPS 统计窗口内的生成 token 数


class _Request:
    """单个请求的状态：driver 写入，HTTP 线程读取。"""

    def __init__(self, seq, max_tokens, rid):
        self.seq = seq
        self.max_tokens = max_tokens
        self.rid = rid
        self.prompt_tokens = seq.num_prompt_tokens
        self.t_add = time.perf_counter()   # 入队（引擎受理）时刻
        self.t_started = None              # 首次被引擎调度（开始处理）时刻
        self.t_first = None                # 首个 token 产出时刻（driver 写入）
        self.t_done = None                 # 完成时刻
        self.stats_before = None     # 入队时 BlockManager 统计快照（算本请求缓存命中用）
        self.tokens = deque()        # driver 追加已生成的 token（eos 已剔除）
        self.n_seen = 0
        self.event = threading.Event()
        self.finished = False
        self.finish_reason = None
        self.error = None
        # 投机解码 per-request 统计（spec_round 埋点累计；非 spec 恒 0）
        self.spec_rounds = 0
        self.spec_committed = 0
        # [REQ] 展示用元数据（_generate 填充）
        self.endpoint = ""
        self.stream = False
        self.client = ""
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 0
        self.thinking = "on"
        self.evicted = 0
        self.preview = ""


def _bump_tokens(n):
    """driver 每产出 n 个 token 调用一次，供 [INFO] 聚合线程统计。"""
    global _gen_total, _gen_window
    with _gen_lock:
        _gen_total += n
        _gen_window += n


def _tps_loop():
    """[INFO] 周期聚合（每 TPS_INTERVAL 秒，仅模型推理相关；窗口无生成则跳过）。"""
    global _gen_window
    while not _shutdown.wait(TPS_INTERVAL):
        with _gen_lock:
            win, total = _gen_window, _gen_total
            _gen_window = 0
            rounds, committed = _spec_rounds, _spec_committed
        if win == 0:
            continue
        with engine_lock:
            n_active = len(_active)
            s = engine.scheduler.block_manager.stats
            total_blocks = s["cached_blocks"] + s["new_blocks"]
        cache = f"{s['cached_blocks'] / total_blocks * 100:.0f}%" if total_blocks else "-"
        avg_e = f"{committed / rounds:.2f}" if rounds else "-"
        _log("INFO", f"winTPS={win / TPS_INTERVAL:.0f} active={n_active} "
                     f"cache_hit={cache} avgE={avg_e}")


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


def _log_request(st):
    """[REQ] 引擎首次调度该请求时打印：含引擎调度等待（受理→被调度）。"""
    wait_ms = (st.t_started - st.t_add) * 1000
    _log("REQ", f"{st.rid} {st.endpoint} stream={'on' if st.stream else 'off'} "
                f"client={st.client} prompt={st.prompt_tokens}tok max_out={st.max_tokens}tok "
                f"temp={st.temperature:g} top_p={st.top_p:g} top_k={st.top_k} "
                f"think={st.thinking} evict={st.evicted} wait={wait_ms:.0f}ms | {st.preview}")


def _log_done(st, ct, reason):
    """[DONE] 单请求完成：TTFT 为实际模型处理时间（被调度→首 token），不含引擎调度等待。"""
    now = time.perf_counter()
    st.t_done = now
    prefill_ms = (st.t_first - st.t_started) * 1000 if (st.t_first and st.t_started) else 0.0
    wall = now - st.t_add
    tps = ct / wall if wall > 0 else 0.0
    used = st.seq.num_tokens
    used_pct = used / _max_model_len * 100 if _max_model_len else 0.0
    e_str = f"{st.spec_committed / st.spec_rounds:.2f}" if st.spec_rounds else "-"
    cache = "-"
    if st.stats_before is not None:
        s = engine.scheduler.block_manager.stats
        cached = s["cached_blocks"] - st.stats_before["cached_blocks"]
        total = cached + (s["new_blocks"] - st.stats_before["new_blocks"])
        if total > 0:
            cache = f"{cached / total * 100:.0f}%({cached}/{total}blk)"
    _log("DONE", f"{st.rid} in={st.prompt_tokens}tok out={ct}tok E={e_str} "
                 f"TTFT={prefill_ms:.0f}ms TPS={tps:.1f} ctx={used}/{_max_model_len}"
                 f"({used_pct:.0f}%) cache={cache} finish={reason}")


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
                    # 用 num_tokens（O(1)）算增量，只切新增尾部——避免每步全量拷贝
                    # completion_token_ids（随输出长度线性增长的 host 开销）。
                    n_comp = seq.num_tokens - seq.num_prompt_tokens
                    if n_comp > st.n_seen:
                        new_tokens = [t for t in seq.token_ids[
                            seq.num_prompt_tokens + st.n_seen: seq.num_tokens]
                            if t != _eos]
                        if new_tokens:
                            if st.t_first is None:
                                st.t_first = time.perf_counter()
                            st.tokens.extend(new_tokens)
                            _bump_tokens(len(new_tokens))
                            st.event.set()   # 立即唤醒 HTTP 线程（原先仅 finish 时 set，
                                             # 生成期 _run_llm 只能 50ms 轮询 → 流式延迟地板）
                        st.n_seen = n_comp
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
    边界上的前缀（pending 携带跨字节残留）；finish 时全量解码兜底，保证最终
    文本与全量解码逐字节一致。
    """
    if emit is None:    # 非流式：吞掉增量推送，只累积最终文本
        emit = lambda _delta: None
    gen = []
    pending = []        # 尚不能安全解码的 token 尾部（UTF-8 多字节边界残留）
    prev_text = ""
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
        # 解码兜底）。
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


def _normalize_messages(messages):
    """OpenAI messages → 模板可用的 {role, content} 列表（多模态 content 取文本）。"""
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
    return normalized


def _template_ids(normalized, enable_thinking=None):
    """套 chat template 并 encode。enable_thinking=None 时不传该 kwarg（tokenize 结果
    与 benchmark 默认逐字节一致）；False 时显式关 thinking（模板插入空
    <think></think> 直答）。"""
    think_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    out = _tokenizer.apply_chat_template(
        normalized, tokenize=True, add_generation_prompt=True, **think_kwargs)
    # 兼容旧 transformers（<4.44 return_dict=False 直接返回 list）
    return out if isinstance(out, (list, tuple)) else out["input_ids"]


def _chat_prompt_ids(messages, enable_thinking=None):
    return _template_ids(_normalize_messages(messages), enable_thinking)


def _chat_prompt_ids_fit(messages, enable_thinking, prompt_cap):
    """聊天感知的上下文淘汰：保留开头连续的 system 消息 + 最近的消息，从**最旧的
    非 system 消息**开始逐条丢弃，直到 token 数 <= prompt_cap；始终保留最后一条
    消息（最新提问）。若 system + 最新一条仍超预算（极端），保留 token 尾部（最新
    内容），确保最新提问不被吞。返回 (prompt_ids, 丢弃的消息数)。"""
    msgs = _normalize_messages(messages)
    n_sys = 0
    while n_sys < len(msgs) and msgs[n_sys].get("role") == "system":
        n_sys += 1
    system, convo = msgs[:n_sys], msgs[n_sys:]
    keep = list(convo)
    ids = _template_ids(system + keep, enable_thinking)
    while len(ids) > prompt_cap and len(keep) > 1:
        keep.pop(0)                      # 丢最旧的非 system 消息
        ids = _template_ids(system + keep, enable_thinking)
    if len(ids) > prompt_cap:
        ids = ids[-prompt_cap:]          # 极端兜底：保尾部（最新提问）
    return ids, len(convo) - len(keep)


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


def _preview(body, mode, limit=40):
    """请求内容预览（日志用）：chat 取最后一条 user 消息，completion 取 prompt。"""
    if mode == "chat":
        content = next((m.get("content") for m in reversed(body.get("messages") or [])
                        if m.get("role") == "user"), "")
    else:
        content = body.get("prompt") or ""
    if isinstance(content, list):
        content = "".join(i.get("text", "") for i in content if isinstance(i, dict))
    if not isinstance(content, str):
        return ""
    flat = " ".join(content.split())
    return flat[:limit] + ("…" if len(flat) > limit else "")


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

    def setup(self):
        super().setup()
        self._box = {"n": 0}
        self.wfile = _CountingWriter(self.wfile, self._box)
        self._t0 = time.perf_counter()
        self._n0 = 0
        self._status = 200
        self._rid = None

    def log_message(self, fmt, *args):
        pass    # 默认访问日志由 [HTTP] 行取代

    def send_response(self, code, message=None):
        self._status = code
        super().send_response(code, message)

    def _http_begin(self):
        self._t0 = time.perf_counter()
        self._n0 = self._box["n"]
        self._status = 200
        self._rid = None

    def _http_end(self):
        dur_ms = (time.perf_counter() - self._t0) * 1000
        nbytes = self._box["n"] - self._n0
        _log("HTTP", f"{self._rid or '-'} {self.command} {self.path} {self._status} "
                     f"client={self.client_address[0]} {nbytes}B {dur_ms:.0f}ms")

    # ---- CORS / 预检 ----

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self._http_begin()
        try:
            self.send_response(204)
            self._cors()
            self.end_headers()
        finally:
            self._http_end()

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
        self._http_begin()
        try:
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
        finally:
            self._http_end()

    def do_POST(self):
        self._http_begin()
        try:
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
        finally:
            self._http_end()

    # ---- 生成 ----

    def _generate(self, body, mode):
        model = body.get("model", _model_name)
        stream = bool(body.get("stream", False))
        n = int(body.get("n", 1))
        if n != 1:
            self._error("'n' != 1 is not supported", 400)
            return
        max_tokens = int(body.get("max_tokens") or 0)    # 未传 / 0 / 负数 = 生成到上下文顶
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

        # 上下文淘汰：先为输出预留 budget（客户端显式 max_tokens 优先，否则用
        # --reserve-output），把 prompt 压进 max_model_len 的剩余窗口。
        reserve = max_tokens if max_tokens >= 1 else _default_reserve
        reserve = min(max(reserve, 1), _max_model_len - 1)
        prompt_cap = _max_model_len - reserve
        dropped = 0
        try:
            # thinking 口径优先级：请求 chat_template_kwargs.enable_thinking /
            # 顶层 enable_thinking > --no-think > 模板默认（on）。
            ctk = body.get("chat_template_kwargs") or {}
            enable_thinking = ctk.get("enable_thinking", body.get("enable_thinking",
                                                                  _default_thinking))
            if mode == "chat":
                # 保 system + 最近轮次，丢最旧中间历史（不再砍尾巴吞掉最新提问）。
                prompt_ids, dropped = _chat_prompt_ids_fit(
                    body["messages"], enable_thinking, prompt_cap)
                if dropped:
                    _log("WARN", f"context eviction: dropped {dropped} oldest message(s) "
                                 f"(prompt<={prompt_cap}tok reserve={reserve} "
                                 f"max_model_len={_max_model_len})")
            else:
                prompt_ids = _completion_prompt_ids(body)
                if len(prompt_ids) > prompt_cap:
                    dropped = len(prompt_ids) - prompt_cap
                    prompt_ids = prompt_ids[-prompt_cap:]    # 保尾部（最新内容）
                    _log("WARN", f"completion prompt truncated {dropped} oldest tokens "
                                 f"(prompt<={prompt_cap}tok)")
        except Exception as exc:
            self._error(f"prompt build failed: {exc}", 400)
            return

        # 输出预算：绝不超过剩余上下文（旧实现允许客户端 max_tokens 与历史叠加越界）。
        remaining = _max_model_len - len(prompt_ids)
        max_tokens = min(max_tokens, remaining) if max_tokens >= 1 else remaining
        max_tokens = max(1, max_tokens)    # 长文本：到上下文顶为止（vLLM 语义）

        req_id = f"chatcmpl-{time.time_ns()}" if mode == "chat" else f"cmpl-{time.time_ns()}"
        rid = req_id[-6:]
        self._rid = rid    # [HTTP] 行关联

        seq = Sequence(prompt_ids, SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=top_p, top_k=top_k))
        with engine_lock:
            engine.scheduler.add(seq)
            st = _Request(seq, max_tokens, rid)
            st.endpoint = self.path
            st.stream = stream
            st.client = self.client_address[0]
            st.temperature = temperature
            st.top_p = top_p
            st.top_k = top_k
            st.thinking = "off" if enable_thinking is False else "on"
            st.evicted = dropped
            st.preview = _preview(body, mode)
            _active[seq.seq_id] = st
            st.stats_before = dict(engine.scheduler.block_manager.stats)
            _wakeup.set()
        # [REQ] 行由 scheduler.schedule 钩子在首次被调度时打印（含引擎调度等待 wait=）。

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
                _log_done(st, ct, reason)
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
            _log_done(st, ct, reason)

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
            _log("ERROR", f"{rid} generation failed: {exc}")
            try:
                self._error(f"generation failed: {exc}", 500)
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass


def _install_schedule_hook():
    """挂 scheduler.schedule：记录每个请求首次被引擎调度的时刻，并在此时打 [REQ]
    行（含引擎调度等待 wait=受理→被调度）。"""
    sched = engine.scheduler
    orig = sched.schedule

    def wrapped():
        seqs, is_prefill = orig()
        now = time.perf_counter()
        for seq in seqs:
            st = _active.get(seq.seq_id)
            if st is not None and st.t_started is None:
                st.t_started = now
                _log_request(st)
        return seqs, is_prefill

    sched.schedule = wrapped


def _install_spec_stats():
    """包 model_runner.spec_round：累计每请求/全局的投机轮数与提交 token 数（accept/E）。
    非 spec 直通（rounds 恒 0 → DONE/INFO 显示 -）。"""
    mr = engine.model_runner
    if not getattr(mr, "spec_enabled", False):
        return
    orig = mr.spec_round

    def wrapped(seqs):
        out = orig(seqs)
        global _spec_rounds, _spec_committed
        with _gen_lock:
            for seq, toks in zip(seqs, out):
                n = len(toks)
                st = _active.get(seq.seq_id)
                if st is not None:
                    st.spec_rounds += 1
                    st.spec_committed += n
                _spec_rounds += 1
                _spec_committed += n
        return out

    mr.spec_round = wrapped


def main():
    global engine, _tokenizer, _eos, _model_name, _api_key
    global _default_temperature, _max_model_len, _default_thinking, _default_reserve

    parser = argparse.ArgumentParser(description="nano-vLLM OpenAI-compatible server (TP=1)")
    parser.add_argument("model", help="model directory path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--eager", action="store_true",
                        help="enforce_eager=True（跳过 CUDA Graph capture，启动更快）")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile 编译模型（CoreX 上约 8% prefill 提速）")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex", "ascend310p"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None,
                        help="KV block size（2 的幂，越小 prefix cache 粒度越细，需 CoreX kernel 支持）")
    parser.add_argument("--spec-model", default="",
                        help="EAGLE3 draft 模型目录（启用投机解码，需与 target 同词表的官方权重）")
    parser.add_argument("--spec-num-tokens", type=int, default=4,
                        help="每轮起草 token 数 k（draft depth）")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--no-think", action="store_true",
                        help="chat template 显式 enable_thinking=False（关闭 thinking，"
                             "对齐 benchmark --no-think 的历史 E 锚点口径；默认交给模板"
                             "=thinking on）。每请求可用 chat_template_kwargs.enable_thinking 覆盖")
    parser.add_argument("--reserve-output", type=int, default=512,
                        help="上下文淘汰时为客户未指定 max_tokens 的请求预留的输出 token 数"
                             "（默认 512）：prompt（system+历史+最新）超出 max_model_len 减去"
                             "该值后，从最旧历史起逐条淘汰")
    # KV 池按 (total*util - used) 贴地分配；空板卡（无桌面占用）需调低 util
    # 给图捕获/运行峰值留余量（32GB CoreX 卡贴地后仅剩 ~112MB，spec 图捕获必 OOM 降级）。
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    args = parser.parse_args()

    _api_key = args.api_key
    _default_temperature = args.temperature
    _default_thinking = False if args.no_think else None
    _default_reserve = max(1, args.reserve_output)

    kwargs = dict(
        model=args.model,
        enforce_eager=args.eager,
        compile=args.compile,
        tensor_parallel_size=1,       # server 只支持 TP=1（引擎同进程驱动）
        max_model_len=args.max_model_len,
        backend=args.backend,
        spec_model=args.spec_model,
        spec_num_tokens=args.spec_num_tokens,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    else:
        # 8GB 卡上大 max_num_seqs 的宽桶/宽 buffer 拖慢单步回放，实测 nospec
        # 512→8 = 59.0→66.8 tok/s；需要更高并发的用户显式传 --max-num-seqs。
        kwargs["max_num_seqs"] = 8
    if args.block_size is not None:
        kwargs["kvcache_block_size"] = args.block_size

    print(f"loading model {args.model} ...", flush=True)
    engine = LLM(**kwargs)
    _tokenizer = engine.tokenizer
    _eos = engine.scheduler.eos
    _max_model_len = engine.model_runner.config.max_model_len
    _model_name = args.model.rstrip("/").rsplit("/", 1)[-1]
    _install_schedule_hook()
    _install_spec_stats()
    spec_info = f"ON(k={args.spec_num_tokens}, {args.spec_model.rstrip('/').rsplit('/', 1)[-1]})" \
        if args.spec_model else "OFF"
    print(f"model '{_model_name}' ready | eos={_eos} | eager={args.eager} | spec={spec_info} | "
          f"max_model_len={_max_model_len} | max_num_seqs={kwargs['max_num_seqs']} | "
          f"thinking={'off' if args.no_think else 'on'}", flush=True)

    threading.Thread(target=_engine_loop, daemon=True).start()
    threading.Thread(target=_tps_loop, daemon=True).start()

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
