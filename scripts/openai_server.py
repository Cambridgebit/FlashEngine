import argparse
import json
import pathlib
import sys
import threading
import time
from urllib.parse import urlsplit

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from flashengine.engine.observer import EngineObserver



engine = None
_tokenizer = None
_eos = None
_model_name = "flashengine"
_api_key = None
_default_temperature = 0.0
_max_model_len = 4096
_default_thinking = None
_default_reserve = 512
_ui_enabled = True
_ui_html_path = None
_ui_html_cache = None
_ui_html_failed = False


def _webui_html():
    global _ui_html_cache, _ui_html_failed
    if _ui_html_cache is None and not _ui_html_failed:
        path = (pathlib.Path(_ui_html_path) if _ui_html_path
                else pathlib.Path(__file__).with_name("webui.html"))
        try:
            _ui_html_cache = path.read_bytes()
        except OSError as exc:
            _ui_html_failed = True
            _log("WARN", f"Web 控制台不可用（读不到 {path}：{exc}）")
    return _ui_html_cache

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
    label = f"[{tag:<{_TAG_W}}]"
    if _use_color:
        color = _TAG_COLORS.get(tag)
        if color:
            label = f"{color}{label}{_RESET}"
    print(f"{_ts()} {label} {msg}", flush=True)


class _CountingWriter:

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


engine_lock = threading.RLock()      # 保护 add / step / abort / _active
_wakeup = threading.Event()          # 有请求加入 / 中止时唤醒 driver
_shutdown = threading.Event()
_active = {}                         # seq_id -> _Request

TPS_INTERVAL = 10.0                  # [INFO] 周期聚合打印间隔（秒，vLLM 默认风格）
_gen_lock = threading.Lock()
_window = dict(gen=0, prompt=0, steps=0, prefill_steps=0, decode_steps=0,
               busy=0.0, spec_rounds=0, spec_committed=0)
_warned_execution_paths = set()
_metrics_started = time.time()
_QUIET_HTTP_PATHS = frozenset({"/v1/stats", "/stats"})
_ui_polls = 0
_requests_recent = deque(maxlen=200)
_engine_info_cache = None
_metrics_total = {
    "generated_tokens": 0,
    "prompt_tokens_computed": 0,
    "engine_steps": 0,
    "prefill_steps": 0,
    "decode_steps": 0,
    "engine_busy_seconds": 0.0,
    "spec_rounds": 0,
    "spec_committed_tokens": 0,
    "spec_accepted_tokens": 0,
    "requests_finished": 0,
    "request_errors": 0,
    "ttft_seconds_sum": 0.0,
    "ttft_seconds_count": 0,
}


class _Request:

    def __init__(self, seq, max_tokens, rid, t_http=None, auto_continue=False):
        self.seq = seq
        self.max_tokens = max_tokens
        self.auto_continue = auto_continue
        self.segment = 0
        self.rid = rid
        self.prompt_tokens = seq.num_prompt_tokens
        self.t_add = time.perf_counter()   # 入队（引擎受理）时刻
        self.t_http = self.t_add if t_http is None else t_http
        self.t_started = None              # 首次被引擎调度（开始处理）时刻
        self.t_first = None                # 首个 token 产出时刻（driver 写入）
        self.t_done = None                 # 完成时刻
        self.cached_prompt_tokens = 0
        self.tokens = deque()        # driver 追加已生成的 token（eos 已剔除）
        self.recent_tokens = deque(maxlen=256)
        self.generated_tokens = 0    # host-side hard limit, independent of engine path
        self.n_seen = 0
        self.event = threading.Event()
        self.finished = False
        self.finish_reason = None
        self.error = None
        self.spec_rounds = 0
        self.spec_committed = 0
        self.endpoint = ""
        self.stream = False
        self.client = ""
        self.temperature = 0.0
        self.top_p = 1.0
        self.top_k = 0
        self.thinking = "on"
        self.evicted = 0
        self.preview = ""
        self.execution_path = "eager"
        self.preempt0 = 0


def _bump_tokens(n):
    with _gen_lock:
        _window["gen"] += n
        _metrics_total["generated_tokens"] += n


def _record_engine_step(num_tokens, elapsed):
    with _gen_lock:
        _window["steps"] += 1
        _window["busy"] += elapsed
        _metrics_total["engine_steps"] += 1
        _metrics_total["engine_busy_seconds"] += elapsed
        if num_tokens > 0:
            _window["prompt"] += num_tokens
            _window["prefill_steps"] += 1
            _metrics_total["prompt_tokens_computed"] += num_tokens
            _metrics_total["prefill_steps"] += 1
        elif num_tokens < 0:
            _window["decode_steps"] += 1
            _metrics_total["decode_steps"] += 1


def _metrics_values():
    with engine_lock:
        with _gen_lock:
            totals = _metrics_total.copy()
        runtime = engine.runtime_snapshot()
        scheduler_stats = runtime["scheduler"]
        cache = runtime["cache"]
        graph_kind, graph = _graph_stats()
        values = {
            "flashengine_start_time_seconds": _metrics_started,
            **{f"flashengine_{key}_total": value for key, value in totals.items()
               if not key.startswith("ttft_")},
            "flashengine_ttft_seconds_sum": totals["ttft_seconds_sum"],
            "flashengine_ttft_seconds_count": totals["ttft_seconds_count"],
            "flashengine_running_requests": scheduler_stats["running"],
            "flashengine_waiting_requests": scheduler_stats["waiting"],
            "flashengine_active_requests": scheduler_stats["active"],
            "flashengine_partial_prefills": scheduler_stats["partial_prefills"],
            "flashengine_scheduler_chunked_prefill_steps_total": (
                scheduler_stats["chunked_prefill_steps"]
            ),
            "flashengine_scheduler_preemptions_total": scheduler_stats["preemptions"],
            "flashengine_kv_blocks_used": cache["used_blocks"],
            "flashengine_kv_blocks_total": cache["total_blocks"],
            "flashengine_prefix_requests_total": cache["requests"],
            "flashengine_prefix_hit_requests_total": cache["prefix_hits"],
            "flashengine_prefix_hit_blocks_total": cache.get(
                "prefix_hit_blocks", cache["cached_blocks"]),
            "flashengine_prefix_cached_blocks_total": cache["cached_blocks"],
            "flashengine_prefix_new_blocks_total": cache["new_blocks"],
        }
        for key, value in runtime["prefill"].items():
            values[f"flashengine_prefill_{key}_total"] = value
        draft_prefix = runtime["draft_prefix"]
        values["flashengine_draft_prefix_snapshots_saved_total"] = draft_prefix["saved"]
        values["flashengine_draft_prefix_snapshots_restored_total"] = draft_prefix["restored"]
        values["flashengine_draft_prefix_snapshots_evicted_total"] = draft_prefix["evicted"]
        values["flashengine_draft_prefix_snapshot_bytes"] = draft_prefix["bytes"]
        if graph_kind and graph is not None:
            values[f"flashengine_{graph_kind}_graph_replays_total"] = graph.get("replay", 0)
            values[f"flashengine_{graph_kind}_graph_fallbacks_total"] = graph.get(
                "eager_rounds", graph.get("fallback", 0))
    return values


def _prometheus_metrics():
    return "".join(
        f"{key} {value}\n" for key, value in _metrics_values().items()
    ).encode("ascii")


def _engine_info():
    global _engine_info_cache
    if _engine_info_cache is not None:
        return _engine_info_cache
    runtime = engine.runtime_snapshot()
    core = runtime["engine"]
    spec = runtime["spec"]
    cfg = engine.config
    info = {
        "model": _model_name,
        "model_path": cfg.model,
        "backend": core["backend"],
        "backend_info": core["backend_info"],
        "graph_mode": core["graph_mode"],
        "enforce_eager": core["enforce_eager"],
        "spec_enabled": spec["enabled"],
        "spec_method": spec["method"],
        "spec_model": core["spec_model"],
        "spec_num_tokens": spec["num_tokens"],
        "max_model_len": cfg.max_model_len,
        "max_num_seqs": cfg.max_num_seqs,
        "max_num_batched_tokens": cfg.max_num_batched_tokens,
        "enable_chunked_prefill": bool(cfg.enable_chunked_prefill),
        "max_num_partial_prefills": cfg.max_num_partial_prefills,
        "enable_prefix_cache": bool(cfg.enable_prefix_cache),
        "kvcache_block_size": cfg.kvcache_block_size,
        "num_kvcache_blocks": runtime["cache"]["total_blocks"],
        "dtype": core["dtype"],
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "default_temperature": _default_temperature,
        "default_thinking": "off" if _default_thinking is False else "template",
        "reserve_output": _default_reserve,
        "execution_path": _execution_path(_default_temperature),
    }
    _engine_info_cache = info
    return info


def _stats_json():
    with _gen_lock:
        requests = list(_requests_recent)
    return {
        "info": _engine_info(),
        "metrics": _metrics_values(),
        "requests": requests,
        "now": time.time(),
    }


def _graph_stats():
    runtime = engine.runtime_snapshot()
    kind = "spec" if runtime["spec"]["enabled"] else "decode"
    graph = runtime["graph"]
    return (kind, graph) if graph else (None, None)


def _execution_path(temperature):
    return engine.execution_path(temperature)


def _tps_loop():
    global _ui_polls
    last_t = time.perf_counter()
    last_graph_replay = 0
    last_graph_eager = 0
    while not _shutdown.wait(TPS_INTERVAL):
        now = time.perf_counter()
        elapsed = max(now - last_t, 1e-9)
        last_t = now
        with _gen_lock:
            current = _window.copy()
            for key in _window:
                _window[key] = 0.0 if key == "busy" else 0
        gen, prompt, steps = (current[key] for key in ("gen", "prompt", "steps"))
        prefill_steps, decode_steps, busy = (
            current[key] for key in ("prefill_steps", "decode_steps", "busy"))
        rounds, committed = (current[key]
                             for key in ("spec_rounds", "spec_committed"))
        with engine_lock:
            n_active = len(_active)
            n_running = len(engine.scheduler.running)
            n_waiting = len(engine.scheduler.waiting)
            scheduler_stats = engine.scheduler.get_stats()
            n_partial = scheduler_stats["partial_prefills"]
            bm = engine.scheduler.block_manager
            used_blocks = len(bm.used_block_ids)
            all_blocks = len(bm.blocks)
            s = engine.scheduler.block_manager.stats
            total_blocks = s["cached_blocks"] + s["new_blocks"]
            graph_kind, graph = _graph_stats()
        if not (gen or prompt or steps or n_active or n_waiting):
            continue    # 空闲窗口不打印；_ui_polls 不在此清零，留到下次有输出时汇总
        with _gen_lock:
            polls, _ui_polls = _ui_polls, 0
        cache = f"{s['cached_blocks'] / total_blocks * 100:.1f}%" if total_blocks else "0.0%"
        kv_usage = used_blocks / all_blocks * 100 if all_blocks else 0.0
        avg_e = f"{committed / rounds:.2f}" if rounds else "-"
        graph_text = ""
        if graph is not None:
            replay = graph.get("replay", 0)
            eager = graph.get("eager_rounds", graph.get("fallback", 0))
            d_replay = max(replay - last_graph_replay, 0)
            d_eager = max(eager - last_graph_eager, 0)
            last_graph_replay, last_graph_eager = replay, eager
            graph_total = d_replay + d_eager
            graph_rate = d_replay / graph_total * 100 if graph_total else 0.0
            graph_text = f" {graph_kind}_graph={graph_rate:.1f}%({d_replay}/{graph_total})"
        _log(
            "INFO",
            f"Avg prompt throughput: {prompt / elapsed:.1f} tok/s, "
            f"Avg generation throughput: {gen / elapsed:.1f} tok/s, "
            f"Running: {n_running} reqs, Waiting: {n_waiting} reqs "
            f"(partial_prefill={n_partial}), "
            f"KV cache usage: {kv_usage:.1f}%, Prefix cache hit rate: {cache}, "
            f"Engine busy: {min(busy / elapsed * 100, 100.0):.1f}%, "
            f"Steps: {steps} (prefill={prefill_steps}, decode={decode_steps}), "
            f"Avg E: {avg_e}{graph_text}"
            + (f" ui_polls={polls}" if polls else ""),
        )


def _drain(queue):
    out = []
    while queue:
        out.append(queue.popleft())
    return out


def _abort(seq):
    with engine_lock:
        seq.status = SequenceStatus.FINISHED
        scheduler = engine.scheduler
        if scheduler.on_preempt is not None:
            scheduler.on_preempt(seq)
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
    wait_ms = (st.t_started - st.t_add) * 1000
    max_out = "auto(EOS)" if st.auto_continue else f"{st.max_tokens}tok"
    _log("REQ", f"{st.rid} {st.endpoint} stream={'on' if st.stream else 'off'} "
                f"client={st.client} prompt={st.prompt_tokens}tok max_out={max_out} "
                f"temp={st.temperature:g} top_p={st.top_p:g} top_k={st.top_k} "
                f"think={st.thinking} path={st.execution_path} evict={st.evicted} "
                f"queue={wait_ms:.0f}ms | {st.preview}")


def _log_done(st, ct, reason):
    now = st.t_done if st.t_done is not None else time.perf_counter()
    st.t_done = now
    prefill_ms = (st.t_first - st.t_started) * 1000 if (st.t_first and st.t_started) else 0.0
    queue_ms = (st.t_started - st.t_add) * 1000 if st.t_started else 0.0
    model_wall = now - st.t_started if st.t_started else now - st.t_add
    e2e_wall = now - st.t_http
    tps = ct / model_wall if model_wall > 0 else 0.0
    tpot_ms = ((now - st.t_first) / (ct - 1) * 1000
               if st.t_first is not None and ct > 1 else 0.0)
    with _gen_lock:
        _metrics_total["requests_finished"] += 1
        if st.t_first is not None and st.t_started is not None:
            _metrics_total["ttft_seconds_sum"] += st.t_first - st.t_started
            _metrics_total["ttft_seconds_count"] += 1
    used = st.seq.num_tokens
    used_pct = used / _max_model_len * 100 if _max_model_len else 0.0
    preempts = engine.scheduler.stats["preemptions"] - st.preempt0
    e_str = f"{st.spec_committed / st.spec_rounds:.2f}" if st.spec_rounds else "-"
    cacheable = max((st.prompt_tokens - 1) // Sequence.block_size, 0)
    cached = st.cached_prompt_tokens // Sequence.block_size
    cache = f"{cached / cacheable * 100:.0f}%({cached}/{cacheable}blk)" if cacheable else "-"
    seq = st.seq
    hit_blocks = seq.prefix_cache_hit_blocks
    query_tokens = seq.prefill_query_tokens
    splitfuse = seq.prefill_splitfuse_used
    recomputed = seq.prefill_prefix_recomputed
    with _gen_lock:
        _requests_recent.appendleft({
            "t": time.time(),
            "rid": st.rid,
            "endpoint": st.endpoint,
            "stream": bool(st.stream),
            "path": st.execution_path,
            "prompt_tokens": st.prompt_tokens,
            "completion_tokens": ct,
            "ttft_ms": round(prefill_ms, 1),
            "tpot_ms": round(tpot_ms, 2),
            "tps": round(tps, 2),
            "queue_ms": round(queue_ms, 1),
            "e2e_ms": round(e2e_wall * 1000, 1),
            "ctx_tokens": used,
            "spec_rounds": st.spec_rounds,
            "spec_committed": st.spec_committed,
            "e": round(st.spec_committed / st.spec_rounds, 3) if st.spec_rounds else None,
            "prefix_hit_blocks": cached,
            "prefix_cacheable_blocks": cacheable,
            "prefill_cache_hit_blocks": hit_blocks,
            "prefill_query_tokens": query_tokens,
            "prefill_splitfuse": bool(splitfuse),
            "prefix_recomputed": bool(recomputed),
            "temperature": st.temperature,
            "evicted": st.evicted,
            "finish": reason,
            "preview": st.preview,
        })
    _log("DONE", f"{st.rid} in={st.prompt_tokens}tok out={ct}tok E={e_str} "
                 f"TTFT={prefill_ms:.0f}ms TPOT={tpot_ms:.1f}ms/tok TPS={tps:.1f}tok/s "
                 f"queue={queue_ms:.0f}ms E2E={e2e_wall * 1000:.0f}ms "
                 f"ctx={used}/{_max_model_len}({used_pct:.0f}%) cache={cache} "
                 f"prefill_hit={hit_blocks}blk query={query_tokens}tok "
                 f"splitfuse={'yes' if splitfuse else 'no'} "
                 f"prefix_recompute={'yes' if recomputed else 'no'} "
                 f"path={st.execution_path} finish={reason}"
                 + (f" preempt={preempts}" if preempts else ""))
    if used_pct >= 95 or preempts:
        _log("WARN", f"{st.rid} 逼近上下文上限：ctx={used}/{_max_model_len}"
                     f"({used_pct:.0f}%) preempt={preempts} finish={reason}——"
                     f"更早轮次已被滑动窗口移出窗口（可持续续聊）；若要更长窗口请增大 "
                     f"--max-model-len，KV 池不足则调低 --gpu-mem-util 或 --max-num-seqs")



def _roll_auto_context(st):
    old_seq = st.seq
    max_len = engine.config.max_model_len
    reserve = min(max(_default_reserve, 1), max_len - 1)
    prompt_cap = max(1, max_len - reserve)
    prompt_ids = old_seq.token_ids[-prompt_cap:]
    if not prompt_ids:
        return False
    segment_max = max_len - len(prompt_ids)
    if segment_max < 1:
        return False

    new_seq = Sequence(prompt_ids, SamplingParams(
        temperature=old_seq.temperature,
        max_tokens=segment_max,
        top_p=old_seq.top_p,
        top_k=old_seq.top_k,
        ignore_eos=old_seq.ignore_eos,
    ))
    new_seq.prefix_cache_hit_blocks = old_seq.prefix_cache_hit_blocks
    new_seq.prefill_query_tokens = old_seq.prefill_query_tokens
    new_seq.prefill_splitfuse_used = old_seq.prefill_splitfuse_used
    new_seq.prefill_prefix_recomputed = old_seq.prefill_prefix_recomputed

    scheduler = engine.scheduler
    if scheduler.on_preempt is not None:
        scheduler.on_preempt(old_seq)
    scheduler.block_manager.deallocate(old_seq)
    try:
        scheduler.running.remove(old_seq)
    except ValueError:
        pass
    try:
        scheduler.waiting.remove(old_seq)
    except ValueError:
        pass
    scheduler.add(new_seq)
    _active.pop(old_seq.seq_id, None)
    st.seq = new_seq
    st.n_seen = 0
    st.segment += 1
    _active[new_seq.seq_id] = st
    return True


def _has_repetition_loop(tokens):
    values = list(tokens)
    if len(values) < 32:
        return False
    max_period = min(64, len(values) // 4)
    for period in range(1, max_period + 1):
        tail = values[-period * 4:]
        if (tail[:period] == tail[period:2 * period]
                == tail[2 * period:3 * period]
                == tail[3 * period:]):
            return True
    return False

def _engine_loop():
    while not _shutdown.is_set():
        with engine_lock:
            if _active and not engine.is_finished():
                try:
                    step_t0 = time.perf_counter()
                    _, num_tokens = engine.step()
                    step_elapsed = time.perf_counter() - step_t0
                    _record_engine_step(num_tokens, step_elapsed)
                except Exception as exc:    # 引擎异常：让所有存活请求报错退出
                    with _gen_lock:
                        _metrics_total["request_errors"] += len(_active)
                    for st in list(_active.values()):
                        _abort(st.seq)
                        st.error = str(exc)
                        st.finished = True
                        st.event.set()
                    _active.clear()
                    continue
                done = []
                step_new_tokens = 0
                for st in list(_active.values()):
                    seq = st.seq
                    n_comp = seq.num_tokens - seq.num_prompt_tokens
                    if n_comp > st.n_seen:
                        new_tokens = [t for t in seq.token_ids[
                            seq.num_prompt_tokens + st.n_seen: seq.num_tokens]
                            if t != _eos]
                        if not st.auto_continue:
                            remaining = st.max_tokens - st.generated_tokens
                            new_tokens = new_tokens[:max(remaining, 0)]
                        if new_tokens:
                            if st.t_first is None:
                                st.t_first = time.perf_counter()
                            st.tokens.extend(new_tokens)
                            st.generated_tokens += len(new_tokens)
                            if st.auto_continue:
                                st.recent_tokens.extend(new_tokens)
                            step_new_tokens += len(new_tokens)
                            st.event.set()   # 立即唤醒 HTTP 线程（原先仅 finish 时 set，
                        st.n_seen = n_comp
                        if (not st.auto_continue
                                and st.generated_tokens >= st.max_tokens):
                            _abort(seq)
                            st.t_done = time.perf_counter()
                            st.finish_reason = "length"
                            st.finished = True
                            st.event.set()
                            done.append(seq.seq_id)
                            continue
                        if (st.auto_continue
                                and _has_repetition_loop(st.recent_tokens)):
                            _abort(seq)
                            st.t_done = time.perf_counter()
                            st.finish_reason = "stop"
                            st.finished = True
                            st.event.set()
                            _log("WARN", f"{st.rid} auto continuation stopped by "
                                 "repetition guard; set max_tokens explicitly or "
                                 "use a clean chat-template boundary")
                            done.append(seq.seq_id)
                            continue
                    if seq.is_finished:
                        if (st.auto_continue and
                                (not _eos or seq.last_token != _eos) and
                                seq.num_tokens >= _max_model_len):
                            if _roll_auto_context(st):
                                continue
                        st.t_done = time.perf_counter()
                        if st.auto_continue:
                            st.finish_reason = "stop" if _eos and seq.last_token == _eos else "length"
                        else:
                            st.finish_reason = (
                                "length" if seq.num_completion_tokens >= st.max_tokens
                                else "stop"
                            )
                        st.finished = True
                        st.event.set()
                        done.append(seq.seq_id)
                if step_new_tokens:
                    _bump_tokens(step_new_tokens)
                for seq_id in done:
                    _active.pop(seq_id, None)
                continue
            done = [seq_id for seq_id, st in _active.items() if st.seq.is_finished]
            for seq_id in done:
                _active.pop(seq_id, None)
        _wakeup.wait(timeout=0.05)   # 仅空转时等待唤醒
        _wakeup.clear()



def _apply_stop(text, stop):
    if not stop:
        return text, False
    p = -1
    for s in stop:
        q = text.find(s)
        if q != -1 and (p == -1 or q < p):
            p = q
    return (text[:p], True) if p != -1 else (text, False)


def _run_llm(st, stop, emit):
    if emit is None and not stop:
        while not st.finished:
            if st.event.wait(timeout=0.05):
                st.event.clear()
        gen = _drain(st.tokens)
        if st.error:
            raise RuntimeError(st.error)
        text = _tokenizer.decode(gen, skip_special_tokens=True)
        return text, st.finish_reason or "stop", len(gen)

    if emit is None:    # non-streaming with stop strings still needs incremental checks
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
    think_kwargs = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    out = _tokenizer.apply_chat_template(
        normalized, tokenize=True, add_generation_prompt=True, **think_kwargs)
    return out if isinstance(out, (list, tuple)) else out["input_ids"]


def _truncate_last(messages, enable_thinking, prompt_cap):
    fitted = list(messages)
    last = dict(fitted[-1])
    content_ids = _tokenizer.encode(last["content"] or "")
    allow = len(content_ids)
    while allow >= 0:
        content = _tokenizer.decode(content_ids[-allow:]) if allow else ""
        fitted[-1] = dict(last, content=content)
        ids = _template_ids(fitted, enable_thinking)
        if len(ids) <= prompt_cap:
            return fitted, ids
        allow -= max(1, len(ids) - prompt_cap)
        content_ids = _tokenizer.encode(fitted[-1]["content"])
    return fitted, ids


def _chat_prompt_ids_fit(messages, enable_thinking, prompt_cap):
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
    if len(ids) > prompt_cap and keep:
        fitted, ids = _truncate_last(system + keep, enable_thinking, prompt_cap)
        keep[-1] = fitted[-1]
    if len(ids) > prompt_cap and (keep or system):
        keep = [keep[-1] if keep else system[-1]]
        system = []
        keep, ids = _truncate_last(keep, enable_thinking, prompt_cap)
    if len(ids) > prompt_cap:
        raise ValueError(
            f"上下文窗口不足：system + 最新提问裁剪后仍需 {len(ids)} token，可用 "
            f"{prompt_cap}（请增大 --max-model-len，或调小 --reserve-output / 请求的 "
            f"max_tokens）")
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



class Handler(BaseHTTPRequestHandler):

    server_version = "flashengine-openai/0.2"

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
        global _ui_polls
        dur_ms = (time.perf_counter() - self._t0) * 1000
        nbytes = self._box["n"] - self._n0
        if self.command == "GET" and self.path.split("?", 1)[0] in _QUIET_HTTP_PATHS:
            with _gen_lock:
                _ui_polls += 1
            return
        _log("HTTP", f"{self._rid or '-'} {self.command} {self.path} {self._status} "
                     f"client={self.client_address[0]} {nbytes}B {dur_ms:.0f}ms")


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


    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self._bytes(body, "application/json", code)

    def _bytes(self, body, content_type, code=200, headers=()):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in headers:
            self.send_header(key, value)
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


    def do_GET(self):
        self._http_begin()
        try:
            path = urlsplit(self.path).path
            if path.startswith("/v1/") and not self._check_auth():
                self._error("invalid api key", 401)
                return
            if _ui_enabled and path in ("/", "/index.html", "/ui", "/webui"):
                body = _webui_html()
                if body is None:
                    self._error("webui.html is missing next to scripts/openai_server.py", 404)
                    return
                self._bytes(body, "text/html; charset=utf-8",
                            headers=(("Cache-Control", "no-store"),))
                return
            if _ui_enabled and path in ("/v1/stats", "/stats"):
                self._json(_stats_json())
                return
            if path in ("/v1/models", "/models"):
                self._json({
                    "object": "list",
                    "data": [{
                        "id": _model_name,
                        "object": "model",
                        "created": 0,
                        "owned_by": "flashengine",
                    }],
                })
                return
            if path in ("/health", "/v1/health"):
                self._bytes(b"OK", "text/plain")
                return
            if path in ("/metrics", "/v1/metrics"):
                self._bytes(_prometheus_metrics(),
                            "text/plain; version=0.0.4; charset=utf-8")
                return
            self._error(f"unknown path: {path}", 404)
        finally:
            self._http_end()

    def do_POST(self):
        self._http_begin()
        try:
            path = urlsplit(self.path).path
            if path not in ("/v1/chat/completions", "/v1/completions"):
                self._error(f"unknown path: {path}", 404)
                return
            if not self._check_auth():
                self._error("invalid api key", 401)
                return
            try:
                body = self._read_body()
            except Exception:
                self._error("invalid json body", 400)
                return
            mode = "chat" if path == "/v1/chat/completions" else "completion"
            self._generate(body, mode)
        finally:
            self._http_end()


    def _generate(self, body, mode):
        if not isinstance(body, dict):
            self._error("JSON body must be an object", 400)
            return
        try:
            model = body.get("model", _model_name)
            stream = bool(body.get("stream", False))
            n = int(body.get("n", 1))
            raw_max_tokens = body.get(
                "max_tokens", body.get("max_completion_tokens"))
            auto_continue = raw_max_tokens is None
            requested_max_tokens = None if auto_continue else int(raw_max_tokens)
        except (TypeError, ValueError) as exc:
            self._error(f"invalid request parameters: {exc}", 400)
            return
        if n != 1:
            self._error("'n' != 1 is not supported", 400)
            return
        if requested_max_tokens is not None and requested_max_tokens < 1:
            self._error("'max_tokens' must be a positive integer when supplied; "
                        "omit it for automatic EOS continuation", 400)
            return
        try:
            temperature = body.get("temperature")
            if temperature is None:
                temperature = _default_temperature
            temperature = max(float(temperature), 0.0)    # 0 = greedy
            top_p = body.get("top_p")
            if top_p is None:
                top_p = 1.0
            top_p = min(max(float(top_p), 1e-6), 1.0)
            top_k = body.get("top_k")
            top_k = max(int(top_k), 0) if top_k is not None else 0
        except (TypeError, ValueError) as exc:
            self._error(f"invalid sampling parameters: {exc}", 400)
            return
        stop = body.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        elif not isinstance(stop, list):
            stop = []
        stop = [s for s in stop if isinstance(s, str) and s]
        stream_options = body.get("stream_options") or {}
        if not isinstance(stream_options, dict):
            self._error("'stream_options' must be an object", 400)
            return
        include_usage = bool(stream_options.get("include_usage", False))

        reserve = requested_max_tokens if requested_max_tokens is not None else _default_reserve
        reserve = min(max(reserve, 1), _max_model_len - 1)
        prompt_cap = _max_model_len - reserve
        dropped = 0
        try:
            ctk = body.get("chat_template_kwargs") or {}
            enable_thinking = ctk.get("enable_thinking", body.get("enable_thinking",
                                                                  _default_thinking))
            if mode == "chat":
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
        if not prompt_ids:
            self._error("prompt must contain at least one token", 400)
            return

        remaining = _max_model_len - len(prompt_ids)
        max_tokens = (min(requested_max_tokens, remaining)
                      if requested_max_tokens is not None else remaining)
        max_tokens = max(1, max_tokens)

        req_id = f"chatcmpl-{time.time_ns()}" if mode == "chat" else f"cmpl-{time.time_ns()}"
        rid = req_id[-6:]
        self._rid = rid    # [HTTP] 行关联

        seq = Sequence(prompt_ids, SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k))
        with engine_lock:
            engine.scheduler.add(seq)
            st = _Request(seq, None if auto_continue else max_tokens, rid,
                          t_http=self._t0, auto_continue=auto_continue)
            st.endpoint = self.path
            st.stream = stream
            st.client = self.client_address[0]
            st.temperature = temperature
            st.top_p = top_p
            st.top_k = top_k
            st.thinking = "off" if enable_thinking is False else "on"
            st.evicted = dropped
            st.preview = _preview(body, mode)
            st.execution_path = _execution_path(temperature)
            st.preempt0 = engine.scheduler.stats["preemptions"]
            _active[seq.seq_id] = st
            _wakeup.set()
            if st.execution_path == "eager_spec(sampling)" and st.execution_path not in _warned_execution_paths:
                _warned_execution_paths.add(st.execution_path)
                _log(
                    "WARN",
                    "Speculative Graph requires greedy sampling on this backend; "
                    "this request keeps its stochastic sampling semantics and falls back to eager spec. "
                    "Use temperature=0 for the benchmark-equivalent Graph path.",
                )

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


class _ServerObserver(EngineObserver):
    def on_schedule(self, seqs, is_prefill, timestamp):
        for seq in seqs:
            st = _active.get(seq.seq_id)
            if st is not None and st.t_started is None:
                st.t_started = timestamp
                st.cached_prompt_tokens = seq.num_cached_tokens
                _log_request(st)

    def on_commit(self, seqs, committed, timestamp):
        with _gen_lock:
            for seq, toks in zip(seqs, committed):
                n = len(toks)
                st = _active.get(seq.seq_id)
                if st is not None:
                    st.spec_rounds += 1
                    st.spec_committed += n
                _window["spec_rounds"] += 1
                _window["spec_committed"] += n
                _metrics_total["spec_rounds"] += 1
                _metrics_total["spec_committed_tokens"] += n
                _metrics_total["spec_accepted_tokens"] += max(n - 1, 0)


def main():
    global engine, _tokenizer, _eos, _model_name, _api_key
    global SamplingParams, Sequence, SequenceStatus
    global _default_temperature
    global _max_model_len, _default_thinking, _default_reserve
    global TPS_INTERVAL, _ui_html_path

    parser = argparse.ArgumentParser(description="FlashEngine OpenAI-compatible server (TP=1)")
    parser.add_argument("model")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex", "ascend310p"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--max-num-partial-prefills", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--spec-model", default="")
    parser.add_argument("--spec-num-tokens", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--reserve-output", type=int, default=512)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--stats-interval", type=float, default=10.0)
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument(
        "--webui", default=None, metavar="HTML",
        help="Web 控制台 HTML 文件路径（默认 scripts/webui.html；"
             "内置可选 scripts/webui-aero.html）")
    args = parser.parse_args()

    from flashengine import LLM, SamplingParams
    from flashengine.engine.sequence import Sequence, SequenceStatus

    global _ui_enabled
    _api_key = args.api_key
    _default_temperature = args.temperature
    _default_thinking = False if args.no_think else None
    _default_reserve = max(1, args.reserve_output)
    TPS_INTERVAL = max(0.0, args.stats_interval)
    _ui_enabled = not args.no_ui
    _ui_html_path = args.webui or None

    kwargs = dict(
        model=args.model,
        enforce_eager=args.eager,
        tensor_parallel_size=1,       # server 只支持 TP=1（引擎同进程驱动）
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_chunked_prefill=not args.disable_chunked_prefill,
        max_num_partial_prefills=args.max_num_partial_prefills,
        backend=args.backend,
        spec_model=args.spec_model,
        spec_num_tokens=args.spec_num_tokens,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs
    else:
        kwargs["max_num_seqs"] = 8
    if args.block_size is not None:
        kwargs["kvcache_block_size"] = args.block_size

    engine = LLM(**kwargs)
    _tokenizer = engine.tokenizer
    _eos = engine.scheduler.eos
    _max_model_len = engine.config.max_model_len
    _model_name = args.model.rstrip("/").rsplit("/", 1)[-1]
    engine.observer = _ServerObserver()
    engine.reset_measurement_stats()
    engine_thread = threading.Thread(
        target=_engine_loop, name="flashengine-engine", daemon=True)
    engine_thread.start()
    stats_thread = None
    if TPS_INTERVAL > 0:
        stats_thread = threading.Thread(
            target=_tps_loop, name="flashengine-stats", daemon=True)
        stats_thread.start()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    _log("INFO", f"Starting OpenAI-compatible API server on http://{args.host}:{args.port}/v1")
    _log("INFO", f"Served model: {_model_name} (source={args.model})")
    if _ui_enabled:
        view_host = "127.0.0.1" if args.host in ("0.0.0.0", "::", "") else args.host
        _log("INFO",
             f"Web console (chat + live metrics): http://{view_host}:{args.port}/"
             f"{'  [api-key required by the page prompt]' if _api_key else ''}")
    else:
        _log("INFO", "Web console disabled (--no-ui); /metrics still available.")
    _log(
        "INFO",
        f"Request defaults: temperature={_default_temperature:g}, thinking="
        f"{'off' if _default_thinking is False else 'on(template default)'}, "
        f"reserve_output={_default_reserve}tok, max_num_seqs="
        f"{engine.config.max_num_seqs}, chunked_prefill="
        f"{'on' if engine.config.enable_chunked_prefill else 'off'}"
        f"(budget={engine.config.max_num_batched_tokens}, partial="
        f"{engine.config.max_num_partial_prefills}), stats_interval="
        f"{TPS_INTERVAL:g}s",
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown.set()
        _wakeup.set()
        with engine_lock:
            for st in list(_active.values()):
                _abort(st.seq)
                st.error = "server is shutting down"
                st.finished = True
                st.event.set()
            _active.clear()
        httpd.server_close()
        engine_thread.join()
        if stats_thread is not None:
            stats_thread.join()
        if engine is not None:
            engine.close()
            engine = None


if __name__ == "__main__":
    main()
