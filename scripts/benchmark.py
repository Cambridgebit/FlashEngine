import argparse
import os
import time

from tqdm import tqdm

from scripts.bench_utils import load_prompts, percentile

TEMPERATURE = 0.0


def run_seqs(llm, seqs):
    for s in seqs:
        llm.scheduler.add(s)
    eos = llm.scheduler.eos
    first, seen = {}, {}
    for s in seqs:
        first[s.seq_id], seen[s.seq_id] = None, 0
    t0 = time.perf_counter()
    while not llm.is_finished():
        llm.step()
        now = time.perf_counter()
        for s in seqs:
            if len(s.completion_token_ids) > seen[s.seq_id]:
                if first[s.seq_id] is None:
                    first[s.seq_id] = now - t0
                seen[s.seq_id] = len(s.completion_token_ids)
    wall = time.perf_counter() - t0
    return [[t for t in s.completion_token_ids if t != eos] for s in seqs], \
        [first[s.seq_id] for s in seqs], wall


def request_progress(n, ttft, wall):
    if n >= 2:
        tpot_ms = (wall - ttft) / (n - 1) * 1000
        return f"TPOT={tpot_ms:.1f}ms/tok  TTFT={ttft * 1000:.1f}ms"
    return f"TPOT=n/a  TTFT={ttft * 1000:.1f}ms"


def print_graph_stats(label, stats, fallback_key):
    fallback = stats[fallback_key]
    total = stats["replay"] + fallback
    fields = {"captures_ok": stats["capture_success"],
              "captures_failed": stats["capture_failure"],
              "replays": stats["replay"], fallback_key: fallback,
              "replay_rate": f"{stats['replay'] / total if total else 0:.1%}",
              "bucket_hits": stats["bucket_hits"]}
    if "sampling_fallbacks" in stats:
        fields["sampling_fallbacks"] = stats["sampling_fallbacks"]
    print(f"{label} = " + ", ".join(f"{key}={value}" for key, value in fields.items()))


def main():
    global Sequence
    parser = argparse.ArgumentParser(
        description="Serial TPS/TTFT benchmark compatible with vllm bench serve.")
    parser.add_argument("model")
    parser.add_argument("--questions", default=os.path.join("..", "question.jsonl"))
    parser.add_argument("--category", default="coding")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--num-repeats", type=int, default=1)
    parser.add_argument("--no-prefix-cache", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--spec-model", default="")
    parser.add_argument("--spec-num-tokens", type=int, default=3)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex", "ascend310p"])
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--disable-chunked-prefill", action="store_true")
    parser.add_argument("--max-num-partial-prefills", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    args = parser.parse_args()

    # 先给一行反馈：torch/torch_npu 导入在板卡上需 ~20s，期间没有任何输出。
    print("[FlashEngine] Importing runtime (torch / torch_npu / transformers, "
          "may take ~20s on Ascend boards) ...", flush=True)

    from flashengine import LLM, SamplingParams
    from flashengine.engine.sequence import Sequence

    qs = load_prompts(args.questions, args.category, args.num_prompts, args.seed)
    thinking = "off" if args.no_think else "on(default)"

    kwargs = dict(enforce_eager=args.eager, tensor_parallel_size=1, backend=args.backend,
                  max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                  max_num_batched_tokens=args.max_num_batched_tokens,
                  enable_chunked_prefill=not args.disable_chunked_prefill,
                  max_num_partial_prefills=args.max_num_partial_prefills,
                  kvcache_block_size=args.block_size,
                  gpu_memory_utilization=args.gpu_mem_util,
                  enable_prefix_cache=not args.no_prefix_cache)
    if args.spec_model:
        kwargs.update(spec_model=args.spec_model, spec_num_tokens=args.spec_num_tokens)
    llm = LLM(args.model, **kwargs)
    cfg = llm.config
    pcache = "on" if cfg.enable_prefix_cache else "off"
    spec = (f"spec({cfg.spec_method},k={cfg.spec_num_tokens})"
            if args.spec_model else "decode")
    chunked = (
        f"on(budget={cfg.max_num_batched_tokens},partial={cfg.max_num_partial_prefills})"
        if cfg.enable_chunked_prefill else "off"
    )
    print(f"prompts: {len(qs)} (category={args.category}, seed={args.seed}) "
          f"x repeats={args.num_repeats} | mode={spec} | prefix_cache={pcache} | "
          f"chunked_prefill={chunked} | "
          f"max_tokens={args.max_tokens} | temp={args.temperature:g} | "
          f"top_p={args.top_p:g} | top_k={args.top_k} | thinking={thinking} | "
          f"protocol=vllm-bench-serve-compatible\n",
          flush=True)

    graph_stats = None
    spec_graph_stats = None
    draft_prefix_stats = None
    try:
        sp = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        def chat(q):
            think_kwargs = {"enable_thinking": False} if args.no_think else {}
            text = llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True, tokenize=False, **think_kwargs)
            return llm.tokenizer.encode(text)

        for _ in range(args.warmup):
            run_seqs(llm, [Sequence(chat(qs[0]), sp)])

        llm.reset_measurement_stats()
        spec_stats = None
        k_stat = cfg.spec_num_tokens

        rows = []  # (n, ttft, wall, rep)
        prompt_lens = []
        t_bench = time.perf_counter()
        with tqdm(total=len(qs) * args.num_repeats, desc="bench", unit="req",
                  dynamic_ncols=True) as pbar:
            for q in qs:
                tokens_in = chat(q)
                for rep in range(args.num_repeats):
                    prompt_lens.append(len(tokens_in))
                    tokens, ttfts, wall = run_seqs(llm, [Sequence(tokens_in, sp)])
                    n = len(tokens[0])
                    ttft = ttfts[0] or 0.0
                    rows.append((n, ttft, wall, rep))
                    pbar.set_postfix_str(request_progress(n, ttft, wall))
                    pbar.update(1)
        bench_wall = time.perf_counter() - t_bench
        runtime = llm.runtime_snapshot()
        if runtime["spec"]["enabled"]:
            spec_graph_stats = runtime["graph"]
            spec_stats = {**runtime["spec"], "pos": runtime["spec"]["positions"]}
        else:
            graph_stats = None if cfg.enforce_eager else runtime["graph"]
        draft_prefix_stats = (
            runtime["draft_prefix"] if cfg.spec_is_block_draft else None
        )
        pstats = runtime["cache"]
    finally:
        llm.close()

    total_tok = sum(r[0] for r in rows)
    total_input_tok = sum(prompt_lens)
    request_wall = sum(r[2] for r in rows)
    decode_tok = sum(max(r[0] - 1, 0) for r in rows)
    decode_wall = sum(max(r[2] - r[1], 0.0) for r in rows)
    host_overhead = max(bench_wall - request_wall, 0.0)
    ttfts = [r[1] for r in rows]
    ttft_avg = sum(ttfts) / len(ttfts)
    ttft_med = percentile(ttfts, 50)
    tps = total_tok / bench_wall if bench_wall else 0.0
    tpot_s = [(r[2] - r[1]) / (r[0] - 1) for r in rows if r[0] >= 2]
    tpot_ms_values = [v * 1000 for v in tpot_s]
    tpot_ms = sum(tpot_ms_values) / len(tpot_ms_values) if tpot_ms_values else 0.0
    e2el = [r[2] for r in rows]
    print("\n============ Benchmark Result ============")
    print(f"Successful requests:                 {len(rows)}")
    print(f"Benchmark duration (s):              {bench_wall:.2f}")
    print(f"Total input tokens:                  {total_input_tok}")
    print(f"Total generated tokens:              {total_tok}")
    print(f"Request throughput (req/s):          {len(rows) / bench_wall if bench_wall else 0.0:.2f}")
    print(f"Input token throughput (tok/s):      {total_input_tok / bench_wall if bench_wall else 0.0:.2f}")
    print(f"Output token throughput (tok/s):     {tps:.2f}")
    print(f"Total token throughput (tok/s):      "
          f"{(total_input_tok + total_tok) / bench_wall if bench_wall else 0.0:.2f}")
    print("---------------Time to First Token----------------")
    print(f"Mean TTFT (ms):                      {ttft_avg * 1000:.2f}")
    print(f"Median TTFT (ms):                    {ttft_med * 1000:.2f}")
    print(f"P99 TTFT (ms):                       {percentile(ttfts, 99) * 1000:.2f}")
    print("-----Time per Output Token (excl. 1st token)------")
    print(f"Mean TPOT (ms):                      {tpot_ms:.2f}")
    print(f"Median TPOT (ms):                    {percentile(tpot_ms_values, 50):.2f}")
    print(f"P99 TPOT (ms):                       {percentile(tpot_ms_values, 99):.2f}")
    print("----------------End-to-end Latency-----------------")
    print(f"Mean E2EL (ms):                      {sum(e2el) / len(e2el) * 1000:.2f}")
    print(f"Median E2EL (ms):                    {percentile(e2el, 50) * 1000:.2f}")
    print(f"P99 E2EL (ms):                       {percentile(e2el, 99) * 1000:.2f}")
    print("================ Engine Details ===================")
    print(f"Prompt length (tokens):              avg={total_input_tok / len(prompt_lens):.1f}, "
          f"min={min(prompt_lens)}, max={max(prompt_lens)}")
    print(f"Output length (tokens):              avg={total_tok / len(rows):.1f}, "
          f"min={min(r[0] for r in rows)}, max={max(r[0] for r in rows)}")
    print(
        "Request-wall throughput = "
        f"{total_tok / request_wall if request_wall else 0.0:.1f} tok/s   "
        f"Benchmark host overhead = {host_overhead:.3f} s"
    )
    print(
        "Steady-state decode throughput = "
        f"{decode_tok / decode_wall if decode_wall else 0.0:.1f} tok/s   "
        f"({decode_tok} tokens / {decode_wall:.2f} s after first-token latency)"
    )
    if graph_stats is not None:
        print_graph_stats("Decode graph runtime", graph_stats, "fallback")
    if spec_graph_stats is not None:
        print_graph_stats("Spec graph runtime", spec_graph_stats, "eager_rounds")
    if args.num_repeats > 1:
        cold = [r[1] for r in rows if r[3] == 0]
        warm = [r[1] for r in rows if r[3] > 0]
        if args.warmup > 0 and len(cold) > 1:
            cold = cold[1:]
        if cold:
            print(f"TTFT cold (rep=0, n={len(cold)}) = {sum(cold) / len(cold) * 1000:.1f} ms")
        if warm:
            print(f"TTFT warm (rep>0, n={len(warm)}) = {sum(warm) / len(warm) * 1000:.1f} ms")
    print(f"Prefix cache = {pcache}  "
          f"(requests={pstats['requests']}, prefix_hits={pstats['prefix_hits']}, "
          f"hit_rate={pstats['hit_rate']:.1%}, cached_blocks={pstats['cached_blocks']}, "
          f"new_blocks={pstats['new_blocks']})")
    if draft_prefix_stats is not None:
        print("Draft prefix snapshots = "
              f"saved={draft_prefix_stats['saved']}, "
              f"restored={draft_prefix_stats['restored']}, "
              f"evicted={draft_prefix_stats['evicted']}, "
              f"host_memory={draft_prefix_stats['bytes'] / (1 << 20):.1f} MiB")
    if spec_stats is not None and spec_stats["rounds"]:
        rounds = spec_stats["rounds"]
        accepted = spec_stats["committed"] - rounds   # Σ 每轮接受 draft 数（提交数 - bonus）
        k = k_stat
        print(f"Acceptance length (E) = {spec_stats['committed'] / rounds:.3f}   "
              f"(avg committed/round = 1 bonus + accepted/rounds)")
        print(f"Token acceptance rate = {accepted / (rounds * k):.1%}   "
              f"(accepted {accepted} / drafted {rounds * k})")
        print("Per-position acceptance = "
              + " / ".join(f"{p / rounds:.1%}" for p in spec_stats["pos"]))
    print("===================================================")


if __name__ == "__main__":
    main()
