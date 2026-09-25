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
- [INFO ] 周期聚合（默认每 10 秒）：prompt/generation throughput、running/waiting、
  KV cache、prefix hit、engine busy、Graph replay rate 与 speculative E；
- [WARN ] / [ERROR] 异常（同宽度）。

用法（必须从仓库根目录以 -m 方式运行，直跑 scripts/X.py 会 ModuleNotFoundError）：
    python3 -m scripts.openai_server /path/to/Qwen3-1.7B [--port 8000]
        [--spec-model <draft 目录>] [--spec-num-tokens k] [--eager] [--no-think] [--reserve-output N] [--api-key xxx]
"""
import argparse
import json
import pathlib
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
_default_temperature = 0.0
_max_model_len = 4096
# chat template 的 enable_thinking 默认：None = 交给模板默认（Qwen3 = on）；
# False = 显式关（--no-think，对齐 benchmark 历史 E 锚点口径）。每请求可用
# chat_template_kwargs.enable_thinking 覆盖。
_default_thinking = None
# 上下文淘汰：客户端未显式给 max_tokens 时，为输出预留的 token 数（--reserve-output）。
# prompt（system + 历史 + 最新）超出 max_model_len - reserve 时，从最旧历史起淘汰。
_default_reserve = 512
# 内置 Web 控制台（对话 + 性能面板）：随 API 同端口挂在 / 上，页面本身不鉴权，
# 但其中所有 /v1/* 调用仍受 --api-key 约束；--no-ui 连 /v1/stats 一起关闭。
_ui_enabled = True
_ui_html_cache = None
_ui_html_failed = False


def _webui_html():
    """读取并缓存 scripts/webui.html（自包含单文件：零外部依赖、离线可用）。"""
    global _ui_html_cache, _ui_html_failed
    if _ui_html_cache is None and not _ui_html_failed:
        path = pathlib.Path(__file__).with_name("webui.html")
        try:
            _ui_html_cache = path.read_bytes()
        except OSError as exc:
            _ui_html_failed = True
            _log("WARN", f"Web 控制台不可用（读不到 {path}：{exc}）")
    return _ui_html_cache

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


# 投机解码窗口统计（[INFO] 的平均 E）；非 spec 恒 0。
_spec_rounds_window = 0
_spec_committed_window = 0

engine_lock = threading.RLock()      # 保护 add / step / abort / _active
_wakeup = threading.Event()          # 有请求加入 / 中止时唤醒 driver
_shutdown = threading.Event()
_active = {}                         # seq_id -> _Request

TPS_INTERVAL = 10.0                  # [INFO] 周期聚合打印间隔（秒，vLLM 默认风格）
_gen_lock = threading.Lock()
_gen_window = 0                      # 当前 TPS 统计窗口内的生成 token 数
_prompt_window = 0                   # 当前窗口实际计算的 prompt token 数（cache hit 不重复计）
_step_window = 0
_prefill_steps_window = 0
_decode_steps_window = 0
_engine_busy_window = 0.0
_warned_execution_paths = set()
_metrics_started = time.time()
# Web 控制台的高频轮询端点：不逐条打 [HTTP ]，改为计数并进 [INFO ] 聚合行。
_QUIET_HTTP_PATHS = frozenset({"/v1/stats", "/stats"})
_ui_polls = 0
# 前端性能面板的数据源：最近完成的请求明细（环形缓冲，host 侧，无设备同步）。
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
    """单个请求的状态：driver 写入，HTTP 线程读取。"""

    def __init__(self, seq, max_tokens, rid, t_http=None):
        self.seq = seq
        self.max_tokens = max_tokens
        self.rid = rid
        self.prompt_tokens = seq.num_prompt_tokens
        self.t_add = time.perf_counter()   # 入队（引擎受理）时刻
        self.t_http = self.t_add if t_http is None else t_http
        self.t_started = None              # 首次被引擎调度（开始处理）时刻
        self.t_first = None                # 首个 token 产出时刻（driver 写入）
        self.t_done = None                 # 完成时刻
        self.cached_prompt_tokens = 0
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
        self.execution_path = "eager"
        # 诊断：该请求存活期间发生过多少次 KV 抢占（scheduler.stats 计数差值）
        self.preempt0 = 0


def _bump_tokens(n):
    """driver 每产出 n 个 token 调用一次，供 [INFO] 聚合线程统计。"""
    global _gen_window
    with _gen_lock:
        _gen_window += n
        _metrics_total["generated_tokens"] += n


def _record_engine_step(num_tokens, elapsed):
    """Accumulate low-overhead engine metrics for the periodic vLLM-style log."""
    global _prompt_window, _step_window, _prefill_steps_window
    global _decode_steps_window, _engine_busy_window
    with _gen_lock:
        _step_window += 1
        _engine_busy_window += elapsed
        _metrics_total["engine_steps"] += 1
        _metrics_total["engine_busy_seconds"] += elapsed
        if num_tokens > 0:
            _prompt_window += num_tokens
            _prefill_steps_window += 1
            _metrics_total["prompt_tokens_computed"] += num_tokens
            _metrics_total["prefill_steps"] += 1
        elif num_tokens < 0:
            _decode_steps_window += 1
            _metrics_total["decode_steps"] += 1


def _metrics_values():
    """累计计数器与实时仪表（JSON-safe 扁平字典）。

    /metrics（Prometheus 文本）与 /v1/stats（前端面板）共用同一份取值，保证两个
    端点永不漂移；只读 host 计数，不触发设备同步。
    """
    with engine_lock:
        with _gen_lock:
            totals = _metrics_total.copy()
        scheduler = engine.scheduler
        scheduler_stats = scheduler.get_stats()
        bm = scheduler.block_manager
        cache = bm.get_prefix_stats()
        graph_kind, graph = _graph_stats()
        values = {
            "nanovllm_start_time_seconds": _metrics_started,
            **{f"nanovllm_{key}_total": value for key, value in totals.items()
               if not key.startswith("ttft_")},
            "nanovllm_ttft_seconds_sum": totals["ttft_seconds_sum"],
            "nanovllm_ttft_seconds_count": totals["ttft_seconds_count"],
            "nanovllm_running_requests": len(scheduler.running),
            "nanovllm_waiting_requests": len(scheduler.waiting),
            "nanovllm_active_requests": scheduler_stats["active"],
            "nanovllm_partial_prefills": scheduler_stats["partial_prefills"],
            "nanovllm_scheduler_chunked_prefill_steps_total": (
                scheduler_stats["chunked_prefill_steps"]
            ),
            "nanovllm_scheduler_preemptions_total": scheduler_stats["preemptions"],
            "nanovllm_kv_blocks_used": len(bm.used_block_ids),
            "nanovllm_kv_blocks_total": len(bm.blocks),
            "nanovllm_prefix_requests_total": cache["requests"],
            "nanovllm_prefix_hit_requests_total": cache["prefix_hits"],
            "nanovllm_prefix_cached_blocks_total": cache["cached_blocks"],
            "nanovllm_prefix_new_blocks_total": cache["new_blocks"],
        }
        runner = engine.model_runner
        if hasattr(runner, "df_prefix_snapshot_stats"):
            snapshot = runner.df_prefix_snapshot_stats
            values["nanovllm_draft_prefix_snapshots_saved_total"] = snapshot["saved"]
            values["nanovllm_draft_prefix_snapshots_restored_total"] = snapshot["restored"]
            values["nanovllm_draft_prefix_snapshots_evicted_total"] = snapshot["evicted"]
            values["nanovllm_draft_prefix_snapshot_bytes"] = runner._df_prefix_snapshot_bytes
        if graph_kind and graph is not None:
            values[f"nanovllm_{graph_kind}_graph_replays_total"] = graph.get("replay", 0)
            values[f"nanovllm_{graph_kind}_graph_fallbacks_total"] = graph.get(
                "eager_rounds", graph.get("fallback", 0))
    return values


def _prometheus_metrics():
    """Prometheus text exposition of the cumulative counters and live gauges."""
    return "".join(
        f"{key} {value}\n" for key, value in _metrics_values().items()
    ).encode("ascii")


def _engine_info():
    """启动后不再变化的引擎画像（供前端标题栏/面板一次性展示）。"""
    global _engine_info_cache
    if _engine_info_cache is not None:
        return _engine_info_cache
    mr = engine.model_runner
    cfg = mr.config
    backend_info = {}
    try:
        backend_info = dict(mr.backend.runtime_info())
    except Exception:     # 画像只是展示用途，后端实现差异不应拖垮面板
        pass
    info = {
        "model": _model_name,
        "model_path": cfg.model,
        "backend": getattr(mr.backend, "name", "?"),
        "backend_info": backend_info,
        "graph_mode": mr.graph_mode(),
        "enforce_eager": bool(mr.enforce_eager),
        "spec_enabled": bool(getattr(mr, "spec_enabled", False)),
        "spec_method": getattr(cfg, "spec_method", "") or "",
        "spec_model": getattr(cfg, "spec_model", "") or "",
        "spec_num_tokens": getattr(cfg, "spec_num_tokens", 0),
        "max_model_len": cfg.max_model_len,
        "max_num_seqs": cfg.max_num_seqs,
        "max_num_batched_tokens": cfg.max_num_batched_tokens,
        "enable_chunked_prefill": bool(cfg.enable_chunked_prefill),
        "max_num_partial_prefills": cfg.max_num_partial_prefills,
        "enable_prefix_cache": bool(cfg.enable_prefix_cache),
        "kvcache_block_size": cfg.kvcache_block_size,
        "num_kvcache_blocks": len(engine.scheduler.block_manager.blocks),
        "dtype": str(mr.dtype),
        "tensor_parallel_size": getattr(cfg, "tensor_parallel_size", 1),
        "default_temperature": _default_temperature,
        "default_thinking": "off" if _default_thinking is False else "template",
        "reserve_output": _default_reserve,
        "execution_path": _execution_path(_default_temperature),
    }
    _engine_info_cache = info
    return info


def _stats_json():
    """/v1/stats：引擎画像 + 与 /metrics 同名计数器 + 最近请求明细。

    时序曲线由前端对累计量做差得到（服务端不额外起采样线程，避免与 [INFO ] 的
    窗口排他读取打架）。
    """
    with _gen_lock:
        requests = list(_requests_recent)
    return {
        "info": _engine_info(),
        "metrics": _metrics_values(),
        "requests": requests,
        "now": time.time(),
    }


def _graph_stats():
    """Return the active graph runtime counters and a short graph kind."""
    mr = engine.model_runner
    if getattr(mr, "spec_enabled", False):
        return "spec", mr.get_spec_graph_stats()
    if getattr(mr, "graph_capture_attempted", False) and hasattr(mr, "get_decode_graph_stats"):
        return "decode", mr.get_decode_graph_stats()
    return None, None


def _execution_path(temperature):
    """Describe the path this request can take without changing sampling semantics."""
    mr = engine.model_runner
    if mr.enforce_eager:
        return "eager"
    if getattr(mr, "spec_enabled", False):
        if not getattr(mr, "spec_graphs", None):
            return "eager_spec"
        sampling_compatible = bool(mr.graph_sample or temperature <= 1e-10)
        return "spec_graph" if sampling_compatible else "eager_spec(sampling)"
    if getattr(mr, "graph_runner", None) is not None or getattr(mr, "graphs", None) is not None:
        return "decode_graph"
    return "eager"


def _tps_loop():
    """Print vLLM-style rolling engine metrics using the real elapsed interval."""
    global _gen_window, _prompt_window, _step_window, _prefill_steps_window
    global _decode_steps_window, _engine_busy_window
    global _spec_rounds_window, _spec_committed_window
    global _ui_polls
    last_t = time.perf_counter()
    last_graph_replay = 0
    last_graph_eager = 0
    while not _shutdown.wait(TPS_INTERVAL):
        now = time.perf_counter()
        elapsed = max(now - last_t, 1e-9)
        last_t = now
        with _gen_lock:
            gen = _gen_window
            prompt = _prompt_window
            steps = _step_window
            prefill_steps = _prefill_steps_window
            decode_steps = _decode_steps_window
            busy = _engine_busy_window
            rounds = _spec_rounds_window
            committed = _spec_committed_window
            _gen_window = 0
            _prompt_window = 0
            _step_window = 0
            _prefill_steps_window = 0
            _decode_steps_window = 0
            _engine_busy_window = 0.0
            _spec_rounds_window = 0
            _spec_committed_window = 0
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
                f"think={st.thinking} path={st.execution_path} evict={st.evicted} "
                f"queue={wait_ms:.0f}ms | {st.preview}")


def _log_done(st, ct, reason):
    """[DONE] 单请求完成：TTFT 为实际模型处理时间（被调度→首 token），不含引擎调度等待。"""
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
            "temperature": st.temperature,
            "evicted": st.evicted,
            "finish": reason,
            "preview": st.preview,
        })
    _log("DONE", f"{st.rid} in={st.prompt_tokens}tok out={ct}tok E={e_str} "
                 f"TTFT={prefill_ms:.0f}ms TPOT={tpot_ms:.1f}ms/tok TPS={tps:.1f}tok/s "
                 f"queue={queue_ms:.0f}ms E2E={e2e_wall * 1000:.0f}ms "
                 f"ctx={used}/{_max_model_len}({used_pct:.0f}%) cache={cache} "
                 f"path={st.execution_path} finish={reason}"
                 + (f" preempt={preempts}" if preempts else ""))
    if used_pct >= 95 or preempts:
        _log("WARN", f"{st.rid} 逼近上下文上限：ctx={used}/{_max_model_len}"
                     f"({used_pct:.0f}%) preempt={preempts} finish={reason}——"
                     f"更早轮次已被滑动窗口移出窗口（可持续续聊）；若要更长窗口请增大 "
                     f"--max-model-len，KV 池不足则调低 --gpu-mem-util 或 --max-num-seqs")


# ---------------------------------------------------------------------------
# 后台引擎驱动线程（continuous batching）
# ---------------------------------------------------------------------------

def _engine_loop():
    """循环调用 engine.step()，把新增 token / 结束事件分发给各请求。"""
    while not _shutdown.is_set():
        with engine_lock:
            if _active and not engine.is_finished():
                try:
                    step_t0 = time.perf_counter()
                    _, num_tokens = engine.step()
                    step_elapsed = time.perf_counter() - step_t0
                    _record_engine_step(num_tokens, step_elapsed)
                    # 活跃期连续步进：不等待 _wakeup（旧的 50ms 等待门会让每个
                    # decode 步白等 50ms，流式 TPS 被压到 ~14）。
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
                            step_new_tokens += len(new_tokens)
                            st.event.set()   # 立即唤醒 HTTP 线程（原先仅 finish 时 set，
                                             # 生成期 _run_llm 只能 50ms 轮询 → 流式延迟地板）
                        st.n_seen = n_comp
                    if seq.is_finished:
                        st.t_done = time.perf_counter()
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
    # Common non-streaming requests without textual stop sequences do not need
    # per-step detokenization.  Waiting for completion and decoding once avoids
    # tokenizer/GIL work running in parallel with every accelerator replay.
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
    """聊天感知的滑动窗口淘汰：保留开头连续的 system 消息 + 最近的消息，从**最旧的
    非 system 消息**开始逐条丢弃，直到 token 数 <= prompt_cap；始终保留最后一条
    消息（最新提问）。返回 (prompt_ids, 丢弃的消息数)。

    只剩一条仍放不下时，裁的是这条消息**文本的 token 尾部**并重新套模板——绝不切
    渲染后的 id 流：ChatML 的角色标记与结尾 generation prompt 一旦被截断，模型会
    退化成重复同一个 token（表现为满屏 "1111"/"！！！！"），而不是报上下文超限。

    不变式：返回的 `len(prompt_ids) <= prompt_cap`，否则抛 ValueError（外层转 400）；
    绝不返回超预算的 prompt——那样的话 `max_tokens` 会被夹成 1，表现为「请求秒结束」
    而不是「窗口不够」，且可能越过引擎的上下文硬停边界。
    """
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
        last = dict(keep[-1])
        content_ids = _tokenizer.encode(last["content"] or "")
        allow = prompt_cap - (len(ids) - len(content_ids))
        # 重套模板后的长度不是线性的（BPE 边界合并可能反而变长），因此按实测超出量
        # 精确回退；allow 每轮至少减 1，必然收敛或走到 allow < 8 退出。
        while allow >= 8:
            keep[-1] = dict(last, content=_tokenizer.decode(content_ids[-allow:]))
            ids = _template_ids(system + keep, enable_thinking)
            if len(ids) <= prompt_cap:
                break
            allow -= max(1, len(ids) - prompt_cap)
            content_ids = _tokenizer.encode(keep[-1]["content"])
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
        global _ui_polls
        dur_ms = (time.perf_counter() - self._t0) * 1000
        nbytes = self._box["n"] - self._n0
        # Web 控制台每秒轮询 /v1/stats，逐条打印会刷掉真实请求的 [HTTP ] 行——
        # 静默并计数，由 [INFO ] 聚合行以 ui_polls=N 汇总（其余端点不受影响）。
        if self.command == "GET" and self.path.split("?", 1)[0] in _QUIET_HTTP_PATHS:
            with _gen_lock:
                _ui_polls += 1
            return
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
            if _ui_enabled and self.path in ("/", "/index.html", "/ui", "/webui"):
                body = _webui_html()
                if body is None:
                    self._error("webui.html is missing next to scripts/openai_server.py", 404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self._cors()
                self.end_headers()
                self.wfile.write(body)
                return
            if _ui_enabled and self.path in ("/v1/stats", "/stats"):
                self._json(_stats_json())
                return
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
            if self.path in ("/metrics", "/v1/metrics"):
                body = _prometheus_metrics()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
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
            st = _Request(seq, max_tokens, rid, t_http=self._t0)
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
                st.cached_prompt_tokens = seq.num_cached_tokens
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
        global _spec_rounds_window, _spec_committed_window
        with _gen_lock:
            for seq, toks in zip(seqs, out):
                n = len(toks)
                st = _active.get(seq.seq_id)
                if st is not None:
                    st.spec_rounds += 1
                    st.spec_committed += n
                _spec_rounds_window += 1
                _spec_committed_window += n
                _metrics_total["spec_rounds"] += 1
                _metrics_total["spec_committed_tokens"] += n
                _metrics_total["spec_accepted_tokens"] += max(n - 1, 0)
        return out

    mr.spec_round = wrapped


def main():
    global engine, _tokenizer, _eos, _model_name, _api_key
    global _default_temperature, _max_model_len, _default_thinking, _default_reserve
    global TPS_INTERVAL

    parser = argparse.ArgumentParser(description="nano-vLLM OpenAI-compatible server (TP=1)")
    parser.add_argument("model", help="model directory path")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--eager", action="store_true",
                        help="Set enforce_eager=True and skip graph capture.")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex", "ascend310p"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384,
                        help="Maximum prompt tokens processed in one prefill step.")
    parser.add_argument("--disable-chunked-prefill", action="store_true",
                        help="Require every prompt prefill to fit in one scheduler step.")
    parser.add_argument("--max-num-partial-prefills", type=int, default=1,
                        help="Maximum admitted requests with an unfinished prompt prefill.")
    parser.add_argument("--block-size", type=int, default=None,
                        help="KV-cache block size; must be supported by the selected backend.")
    parser.add_argument("--spec-model", default="",
                        help="Draft model directory; enables speculative decoding.")
    parser.add_argument("--spec-num-tokens", type=int, default=3,
                        help="Number of draft tokens per speculative round.")
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help=("Default request temperature when the JSON body omits it. "
              "The default is greedy (0) to match scripts.benchmark and keep "
              "Ascend speculative decoding on the captured Graph path."),
    )
    parser.add_argument("--no-think", action="store_true",
                        help="Pass enable_thinking=False to the chat template by default.")
    parser.add_argument("--reserve-output", type=int, default=512,
                        help="Output-token reserve used during context eviction.")
    # KV 池按 (total*util - used) 贴地分配；空板卡（无桌面占用）需调低 util
    # 给图捕获/运行峰值留余量（32GB CoreX 卡贴地后仅剩 ~112MB，spec 图捕获必 OOM 降级）。
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument(
        "--stats-interval", type=float, default=10.0,
        help="Seconds between vLLM-style engine statistics logs; 0 disables them.",
    )
    parser.add_argument("--no-ui", action="store_true",
                        help="Disable the built-in web console (/) and /v1/stats.")
    args = parser.parse_args()

    global _ui_enabled
    _api_key = args.api_key
    _default_temperature = args.temperature
    _default_thinking = False if args.no_think else None
    _default_reserve = max(1, args.reserve_output)
    TPS_INTERVAL = max(0.0, args.stats_interval)
    _ui_enabled = not args.no_ui

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
        # 8GB 卡上大 max_num_seqs 的宽桶/宽 buffer 拖慢单步回放，实测 nospec
        # 512→8 = 59.0→66.8 tok/s；需要更高并发的用户显式传 --max-num-seqs。
        kwargs["max_num_seqs"] = 8
    if args.block_size is not None:
        kwargs["kvcache_block_size"] = args.block_size

    engine = LLM(**kwargs)
    _tokenizer = engine.tokenizer
    _eos = engine.scheduler.eos
    _max_model_len = engine.model_runner.config.max_model_len
    _model_name = args.model.rstrip("/").rsplit("/", 1)[-1]
    _install_schedule_hook()
    _install_spec_stats()
    if hasattr(engine.model_runner, "reset_decode_graph_runtime_stats"):
        engine.model_runner.reset_decode_graph_runtime_stats()
    if getattr(engine.model_runner, "spec_enabled", False):
        engine.model_runner.reset_spec_graph_runtime_stats()
    engine_thread = threading.Thread(
        target=_engine_loop, name="nanovllm-engine", daemon=True)
    engine_thread.start()
    stats_thread = None
    if TPS_INTERVAL > 0:
        stats_thread = threading.Thread(
            target=_tps_loop, name="nanovllm-stats", daemon=True)
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
        f"{engine.model_runner.config.max_num_seqs}, chunked_prefill="
        f"{'on' if engine.model_runner.config.enable_chunked_prefill else 'off'}"
        f"(budget={engine.model_runner.config.max_num_batched_tokens}, partial="
        f"{engine.model_runner.config.max_num_partial_prefills}), stats_interval="
        f"{TPS_INTERVAL:g}s",
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _shutdown.set()
        _wakeup.set()
        # Wake request handlers and remove queued/running sequences before the
        # device is torn down. Do not close the engine while its driver thread
        # may still be executing an NPU kernel.
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
