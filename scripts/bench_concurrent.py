import argparse
import json
import math
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from scripts.bench_utils import load_prompts, metric_delta, percentile

DEFAULT_URL = "http://127.0.0.1:8000"
FILLER = "Shared reference context for the task, listing the agreed conventions. "
CHARS_PER_TOKEN = 4.0          # 粗估，只为让 --shared-prefix 有个可操作的量级


def _headers(api_key, content=False):
    h = {"Accept": "application/json"}
    if content:
        h["Content-Type"] = "application/json"
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def _get_json(url, api_key, timeout=10.0):
    req = urllib.request.Request(url, headers=_headers(api_key))
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_messages(prompt, shared_prefix):
    msgs = []
    if shared_prefix > 0:
        reps = max(1, math.ceil(shared_prefix * CHARS_PER_TOKEN / len(FILLER)))
        msgs.append({"role": "system", "content": FILLER * reps})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def send_request(base, model, msgs, args, barrier, probe=None):
    body = {
        "model": model,
        "messages": msgs,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "stream": not args.no_stream,
        "stream_options": {"include_usage": True},
    }
    if args.no_think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                data=json.dumps(body).encode("utf-8"),
                                headers=_headers(args.api_key, content=True))
    res = {"ttft": None, "wall": 0.0, "tpot": None, "out_tok": 0,
           "prompt_tok": None, "finish": "-", "err": None}
    if barrier is not None:
        try:
            barrier.wait(timeout=args.timeout)     # 同批同步发车
        except threading.BrokenBarrierError:
            pass    # 主线程已放弃这一批：照发不误，错误由下面统一收口
    if probe is not None:
        with probe["lock"]:
            probe["cur"] += 1
            probe["max"] = max(probe["max"], probe["cur"])
    t0 = time.perf_counter()
    first = last = None
    n_delta = 0
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            if args.no_stream:
                j = json.loads(resp.read().decode("utf-8"))
                usage = j.get("usage") or {}
                ch = (j.get("choices") or [{}])[0]
                res["out_tok"] = int(usage.get("completion_tokens") or 0)
                res["prompt_tok"] = usage.get("prompt_tokens")
                res["finish"] = ch.get("finish_reason") or "-"
                res["wall"] = res["ttft"] = time.perf_counter() - t0
                return res
            for line in resp:                      # resp 可迭代，逐 SSE 事件已 flush 到达
                s = line.decode("utf-8", "replace").strip()
                if not s.startswith("data:"):
                    continue
                payload = s[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                usage = obj.get("usage") or {}
                if usage:
                    res["out_tok"] = int(usage.get("completion_tokens") or res["out_tok"])
                    res["prompt_tok"] = usage.get("prompt_tokens")
                ch = (obj.get("choices") or [{}])[0]
                delta = (ch.get("delta") or {}).get("content") or ch.get("text") or ""
                if delta:
                    now = time.perf_counter()
                    if first is None:
                        first = now
                    last = now
                    n_delta += 1
                if ch.get("finish_reason"):
                    res["finish"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        res["err"] = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:180]}"
    except threading.BrokenBarrierError:
        res["err"] = "barrier broken (a peer request never started)"
    except Exception as e:                          # 连接失败/读超时/服务端 500
        res["err"] = f"{type(e).__name__}: {e}"
    finally:
        if probe is not None:
            with probe["lock"]:
                probe["cur"] -= 1
    res["wall"] = time.perf_counter() - t0
    res["ttft"] = (first - t0) if first is not None else res["wall"]
    if not res["out_tok"]:                          # 无 usage：退化成 chunk 计数
        res["out_tok"] = n_delta
    if n_delta > 1 and first is not None:
        res["tpot"] = (last - first) / (n_delta - 1)
    return res


def metrics(base, args):
    try:
        return (_get_json(base.rstrip("/") + "/v1/stats", args.api_key) or {}).get("metrics", {})
    except Exception:
        return {}


_DELTA_KEYS = {
    "out": "flashengine_generated_tokens_total",
    "steps": "flashengine_engine_steps_total",
    "prefill_steps": "flashengine_prefill_steps_total",
    "decode_steps": "flashengine_decode_steps_total",
    "busy": "flashengine_engine_busy_seconds_total",
    "preempt": "flashengine_scheduler_preemptions_total",
    "spec_rounds": "flashengine_spec_rounds_total",
    "spec_committed": "flashengine_spec_committed_tokens_total",
    "prefix_req": "flashengine_prefix_requests_total",
    "prefix_hit": "flashengine_prefix_hit_requests_total",
    "hit_blk": "flashengine_prefix_cached_blocks_total",
    "new_blk": "flashengine_prefix_new_blocks_total",
}


def delta(before, after):
    return metric_delta(before, after, _DELTA_KEYS)


class PeakSampler(threading.Thread):

    def __init__(self, base, args, interval=0.25):
        super().__init__(daemon=True)
        self.base, self.args, self.interval = base, args, interval
        self._quit = threading.Event()
        self.running = self.waiting = self.active = 0

    def run(self):
        while not self._quit.is_set():
            m = metrics(self.base, self.args)
            self.running = max(self.running, int(m.get("flashengine_running_requests", 0)))
            self.waiting = max(self.waiting, int(m.get("flashengine_waiting_requests", 0)))
            self.active = max(self.active, int(m.get("flashengine_active_requests", 0)))
            self._quit.wait(self.interval)

    def stop_sampling(self):
        self._quit.set()
        self.join(timeout=2.0)

    def snapshot(self):
        return {"running": self.running, "waiting": self.waiting, "active": self.active}


def run_level(base, model, prompts, level, args):
    probe = {"cur": 0, "max": 0, "lock": threading.Lock()}
    call = lambda msgs, barrier: send_request(base, model, msgs, args, barrier, probe)
    for i in range(args.warmup):                   # 预热：不进指标
        try:
            call(build_messages(prompts[i % len(prompts)], args.shared_prefix), None)
        except Exception as e:
            raise SystemExit(f"Warmup request failed: {e}")
    probe["cur"] = probe["max"] = 0                # 清掉预热期的在飞计数
    batches = max(1, math.ceil(args.num_prompts / level))
    ok = fail = total_out = 0
    ttfts, walls, tpots, tps_one, errs = [], [], [], [], []
    before = metrics(base, args)
    sampler = PeakSampler(base, args)
    sampler.start()
    t0 = time.perf_counter()
    for b in range(batches):
        group = [prompts[(b * level + i) % len(prompts)] for i in range(level)]
        barrier = threading.Barrier(len(group) + 1)
        with ThreadPoolExecutor(max_workers=len(group)) as ex:
            futs = [ex.submit(call, build_messages(p, args.shared_prefix), barrier)
                    for p in group]
            try:
                barrier.wait(timeout=args.timeout)          # 主线程凑齐发车人数
            except threading.BrokenBarrierError:
                barrier.abort()                             # 有 worker 没起来：立刻解开其余
            for r in [f.result() for f in futs]:
                if r["err"]:
                    fail += 1
                    if len(errs) < 3:
                        errs.append(r["err"])
                    continue
                ok += 1
                total_out += r["out_tok"]
                ttfts.append(r["ttft"])
                walls.append(r["wall"])
                if r["tpot"]:
                    tpots.append(r["tpot"])
                if r["wall"] > 0 and r["out_tok"]:
                    tps_one.append(r["out_tok"] / r["wall"])
    wall = time.perf_counter() - t0
    sampler.stop_sampling()
    d = delta(before, metrics(base, args))
    return {"level": level, "requests": ok + fail, "ok": ok, "fail": fail, "wall": wall,
            "out_tok": total_out, "tps": total_out / wall if wall else 0.0,
            "ttft_p50": percentile(ttfts, 50), "ttft_p95": percentile(ttfts, 95),
            "tpot_p50": percentile(tpots, 50), "tok_s_p50": percentile(tps_one, 50),
            "client_peak": probe["max"],
            "peak": sampler.snapshot(), "delta": d, "errs": errs}


def main():
    ap = argparse.ArgumentParser(
        description="Concurrent / continuous-batching load test against scripts.openai_server.")
    ap.add_argument("base_url", nargs="?", default=DEFAULT_URL)
    ap.add_argument("--questions", default="bench_data/question.jsonl")
    ap.add_argument("--category", default="coding")
    ap.add_argument("--concurrency", default="1,4,8")
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--shared-prefix", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()

    levels = [int(x) for x in str(args.concurrency).replace(" ", "").split(",") if x]
    if not levels or min(levels) < 1:
        raise SystemExit("--concurrency must be comma-separated integers >= 1")
    prompts = load_prompts(args.questions, args.category,
                           max(args.num_prompts, max(levels), args.warmup), args.seed)
    try:
        models = _get_json(args.base_url.rstrip("/") + "/v1/models", args.api_key).get("data", [])
    except Exception as e:
        raise SystemExit(f"Cannot reach {args.base_url}/v1/models: {e}\n"
                         f"Start the server first: python3 -m scripts.openai_server "
                         f"/path/to/model --port 8000")
    model = args.model or (models[0]["id"] if models else "flashengine")
    try:
        info = _get_json(args.base_url.rstrip("/") + "/v1/stats", args.api_key).get("info", {})
    except Exception:
        info = {}

    print("=================== bench_concurrent ===================")
    print(f"server = {args.base_url}  model = {model}  backend = {info.get('backend', '?')}  "
          f"graph = {info.get('graph_mode', '?')}  "
          f"spec = {info.get('spec_method') or 'off'}(k={info.get('spec_num_tokens', 0)})")
    print(f"server config: max_model_len={info.get('max_model_len', '?')}  "
          f"max_num_seqs={info.get('max_num_seqs', '?')}  "
          f"prefix_cache={info.get('enable_prefix_cache', '?')}  "
          f"block_size={info.get('kvcache_block_size', '?')}  "
          f"partial_prefills={info.get('max_num_partial_prefills', '?')}")
    print(f"dataset = {args.questions}  category = {args.category}  seed = {args.seed}  "
          f"prompts_available = {len(prompts)}")
    print(f"stream = {'off' if args.no_stream else 'on'}  thinking = "
          f"{'off' if args.no_think else 'on(template)'}  temp = {args.temperature:g}  "
          f"max_tokens = {args.max_tokens}  shared_prefix = {args.shared_prefix}tok(approx)  "
          f"warmup_per_level = {args.warmup}")
    print(f"levels = {levels}  requests_per_level = {args.num_prompts}  "
          f"each batch starts on a Barrier (no head-of-line consumption)")
    print("note: c_pk = client in-flight peak is the exact proof of real concurrency; "
          "server running/waiting are 0.25s samples (queue depth only). Concurrency above "
          "the server's max_num_seqs keeps c_pk at the requested level but queues the extra "
          "requests on the HTTP side\n")

    rows = []
    for lv in levels:
        r = run_level(args.base_url, model, prompts, lv, args)
        rows.append(r)
        d = r["delta"]
        e_val = d["spec_committed"] / d["spec_rounds"] if d.get("spec_rounds") else None
        busy = d.get("busy", 0.0) / r["wall"] * 100 if r["wall"] else 0.0
        print(f"--- concurrency {lv} ---")
        print(f"  requests {r['requests']} (ok {r['ok']} / failed {r['fail']})  "
              f"wall {r['wall']:.2f}s  aggregate {r['tps']:.1f} tok/s"
              f" (client {r['out_tok']}tok / server {d.get('out', '?')}tok)  "
              f"per-request tok/s p50 {r['tok_s_p50']:.1f}")
        print(f"  TTFT p50 {r['ttft_p50'] * 1000:.1f}ms  p95 {r['ttft_p95'] * 1000:.1f}ms  "
              f"TPOT p50 {r['tpot_p50'] * 1000:.2f}ms/tok")
        print(f"  in-flight peak: client={r['client_peak']}  "
              f"server running={r['peak']['running']} waiting={r['peak']['waiting']} "
              f"active={r['peak']['active']}  steps {d.get('steps', '?')}"
              f" (prefill {d.get('prefill_steps', '?')} / decode {d.get('decode_steps', '?')})  "
              f"engine busy {busy:.0f}%")
        print(f"  preemptions {d.get('preempt', 0)}  prefix hits {d.get('prefix_hit', '?')}"
              f"/{d.get('prefix_req', '?')} req (cached {d.get('hit_blk', '?')}blk / "
              f"new {d.get('new_blk', '?')}blk)  avg E "
              + (f"{e_val:.2f}" if e_val else "-"))
        for m in r["errs"]:
            print(f"  [error sample] {m}")
        print()

    base_row = rows[0]
    print("============ summary (speedup vs first level) ============")
    print("conc  reqs  fail   wall  agg_tok/s  speedup  TTFT_p50  TTFT_p95  TPOT_p50 "
          " c_pk  run_pk  wait_pk  preempt")
    for r in rows:
        sp = r["tps"] / base_row["tps"] if base_row["tps"] else 0.0
        print(f"{r['level']:>3}  {r['requests']:>4}  {r['fail']:>4}  {r['wall']:>6.2f}  "
              f"{r['tps']:>9.1f}  {sp:>6.2f}  {r['ttft_p50'] * 1000:>7.1f}ms  "
              f"{r['ttft_p95'] * 1000:>7.1f}ms  {r['tpot_p50'] * 1000:>7.2f}ms  "
              f"{r['client_peak']:>4}  {r['peak']['running']:>6}  "
              f"{r['peak']['waiting']:>7}  {r['delta'].get('preempt', 0):>7}")
    print("===============================================================")


if __name__ == "__main__":
    main()
