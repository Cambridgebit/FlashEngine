"""spec_bench question.jsonl 串行 TPS/TTFT 基准（对齐 vllm bench serve 口径）。

口径（与 vllm bench serve 的 spec_bench 数据集逐项对齐，同 --seed 即同 prompt 集）：
- prompt：对 turns[0] 套 chat template（tokenize=False 字符串、不传 kwargs →
  Qwen3 默认 enable_thinking）后 encode；--no-think 显式关 thinking（模板插入
  空 <think></think> 直答，对齐 V1.8/历史 E 锚点口径，用于单变量二分接受率差异）；
- 数据集：random.seed(--seed) 后 shuffle 取前 --num-prompts 条（--seed -1 = 文件序前 N 条）；
- 采样：默认 temperature=0 greedy，可用 --temperature/--top-p/--top-k 对齐在线请求；
  输出上限 --max-tokens（默认 256 = vllm spec_bench output_len）；
- 预热：--warmup 次（不计指标），统一打 shuffle 后首条 prompt（对齐 vllm --num-warmups）；
- 吞吐：Σ输出 token / 主跑总墙钟（单墙钟，含逐条 tokenize/进度条刷新的 host 开销，
  对齐 vllm「Output token throughput」）；TTFT 为进程内口径（vllm 为 client 端，高几 ms）；
- 输出形式：主跑期间用 tqdm 进度条，末端实时显示本次请求的 TPOT 与 TTFT（不再逐条打印）；
  结束后汇总打印 TTFT（ms）/ TPOT（ms/token）/ TPS（tok/s），均带单位。
  TPOT = 逐请求 (wall - TTFT) / (输出 token 数 - 1) 的均值（对齐 vllm bench serve 的
  Mean TPOT）；输出 ≤1 token 的请求无定义，不参与统计；
- 重复请求负载：--num-repeats R 时每条 prompt 连续跑 R 次（相邻重复，第 2..R 次
  全量命中前缀缓存；只 tokenize 一次），配合 --no-prefix-cache 做 TTFT 冷/热 A/B
  （prefix cache 只降 TTFT、不动吞吐的结论即出自该对照）；repeats>1 时汇总额外
  打印 cold（rep=0）/warm（rep>0）分组 TTFT；
- prefix cache 开关：--no-prefix-cache 传 enable_prefix_cache=False，引擎跳过
  block hash 与命中查找；汇总打印 block_manager 命中率统计（预热后清零，仅计主跑）；
- 投机统计：--spec-model 时打印平均接受长度 E / token 接受率 / 逐位置接受率
  （monkey-patch spec_round 读每轮提交数 cnt=m+1；仅计主跑，预热不计；
  EOS/max_tokens 截断发生在 scheduler.commit_spec，图内 cnt 不感知，收尾轮轻微高估）。

用法（必须从仓库根目录以 -m 方式运行，直跑会 ModuleNotFoundError）：
    python3 -m scripts.benchmark /path/to/Qwen3-1.7B [--spec-model <draft 目录>]
        [--questions <spec_bench/question.jsonl>] [--num-prompts 10] [--num-repeats 1]
        [--no-prefix-cache] [--max-tokens 256] [--seed 0] [--warmup 5]
        [--category coding] [--no-think]
"""
import argparse
import json
import os
import random
import time

from tqdm import tqdm

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence

TEMPERATURE = 0.0


def load_questions(path, category, num, seed):
    """question.jsonl → prompts（取 turns[0]；category=all 不过滤，记录无 category 字段也不过滤）。

    seed>=0：复现 vllm bench 口径——random.seed(seed) 后 shuffle 再取前 num 条
    （vllm spec_bench 默认 --seed 0 且必 shuffle，同 seed 即同 prompt 集）。
    seed<0：不 shuffle，文件序前 num 条（历史「前 N 条」口径）。
    """
    qs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cat = d.get("category")
            if category != "all" and cat is not None and cat != category:
                continue
            qs.append(d["turns"][0])
    if not qs:
        raise SystemExit(f"No prompts found in {path} (category={category}).")
    if seed >= 0:
        random.seed(seed)
        random.shuffle(qs)
    return qs[:num]


def run_seqs(llm, seqs):
    """入队并驱动到全部结束，返回 (每 seq 输出 token 列表, TTFT 列表, wall 秒)。"""
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
    """进度条末端文本：本次请求的 TPOT 与 TTFT（输出 ≤1 token 时 TPOT 无定义）。"""
    if n >= 2:
        tpot_ms = (wall - ttft) / (n - 1) * 1000
        return f"TPOT={tpot_ms:.1f}ms/tok  TTFT={ttft * 1000:.1f}ms"
    return f"TPOT=n/a  TTFT={ttft * 1000:.1f}ms"


def percentile(values, p):
    """Linear percentile without adding a NumPy dependency to the benchmark."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def main():
    parser = argparse.ArgumentParser(
        description="Serial TPS/TTFT benchmark compatible with vllm bench serve.")
    parser.add_argument("model")
    parser.add_argument("--questions",
                        default=os.path.join("..", "question.jsonl"),
                        help="Path to question.jsonl (default: ../question.jsonl).")
    parser.add_argument("--category", default="coding",
                        help="SpecBench category filter; use 'all' to disable filtering.")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--num-repeats", type=int, default=1,
                        help="Consecutive runs per prompt; repeats can hit the prefix cache.")
    parser.add_argument("--no-prefix-cache", action="store_true",
                        help="Disable prefix caching and block-hash lookup for A/B testing.")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Maximum output tokens per request; EOS may stop generation earlier.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="Sampling temperature; 0 uses greedy sampling (default: 0).")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0,
                        help="Dataset shuffle seed; -1 preserves file order.")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup iterations excluded from benchmark metrics.")
    parser.add_argument("--spec-model", default="", help="Draft model directory; enables speculative decoding.")
    parser.add_argument("--spec-num-tokens", type=int, default=3)
    parser.add_argument("--no-think", action="store_true",
                        help="Pass enable_thinking=False to the chat template.")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex", "ascend310p"])
    parser.add_argument("--eager", action="store_true", help="Disable graph capture and use eager mode.")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384,
                        help="Maximum prompt tokens processed in one prefill step.")
    parser.add_argument("--disable-chunked-prefill", action="store_true",
                        help="Require every prompt prefill to fit in one scheduler step.")
    parser.add_argument("--max-num-partial-prefills", type=int, default=1,
                        help="Maximum admitted requests with an unfinished prompt prefill.")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9,
                        help="Device-memory utilization reserved for model and KV cache.")
    args = parser.parse_args()

    qs = load_questions(args.questions, args.category, args.num_prompts, args.seed)
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
    pcache = "on" if llm.model_runner.config.enable_prefix_cache else "off"
    # 表头在引擎构造后打印：k 取引擎实际生效值（dflash 按 draft config block_size-1
    # 自动派生，CLI 默认 3 会被覆盖——在派生前打印会显示错误的 k）。
    spec = (f"spec({llm.model_runner.config.spec_method},"
            f"k={llm.model_runner.config.spec_num_tokens})") if args.spec_model else "decode"
    chunked = (
        f"on(budget={llm.model_runner.config.max_num_batched_tokens},"
        f"partial={llm.model_runner.config.max_num_partial_prefills})"
        if llm.model_runner.config.enable_chunked_prefill else "off"
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
            # 对齐 vllm bench spec_bench 口径：CustomDataset.sample 对 prompt 套
            # chat template（tokenize=False 字符串、不传 kwargs → Qwen3 默认
            # enable_thinking）后作为 completions 的 prompt 发送；此处同构生成 id。
            # --no-think：显式关 thinking（模板插入空 <think></think> 直答）；
            # 默认路径不传任何 kwargs，与 vllm bench 逐字节一致。
            think_kwargs = {"enable_thinking": False} if args.no_think else {}
            text = llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True, tokenize=False, **think_kwargs)
            return llm.tokenizer.encode(text)

        # 预热（不计指标）：对齐 vllm bench——就绪检查 + num_warmups 均打同一首条
        # prompt（shuffle 后的 qs[0]）。
        for _ in range(args.warmup):
            run_seqs(llm, [Sequence(chat(qs[0]), sp)])

        # Report graph activity for the measured window only. Keep capture
        # success/failure counters, but exclude engine and benchmark warmups.
        if hasattr(llm.model_runner, "reset_decode_graph_runtime_stats"):
            llm.model_runner.reset_decode_graph_runtime_stats()
        if args.spec_model and hasattr(llm.model_runner, "reset_spec_graph_runtime_stats"):
            llm.model_runner.reset_spec_graph_runtime_stats()

        # 投机解码接受率统计（对齐历史 check_eagle3 的 monkey-patch spec_round 口径，
        # 引擎零改动）：spec_round 每轮返回各 seq 的提交 token 列表，长度 = 接受
        # draft 数 m + 1 bonus（图内 _sg_acceptance 的 cnt 打包，冷/热/图/eager
        # 四路统一从 _readback_spec_commit 出）。预热后安装，仅计主跑。
        spec_stats = None
        if args.spec_model:
            # 统计口径取引擎实际生效的 k（config.spec_num_tokens）：dflash 会按
            # draft config block_size-1 自动派生（CLI 默认 3 被视为未显式指定），
            # 直接用 args 值会按 3 建直方图而在 c>4 时越界；eagle3 下 config 与
            # args 恒等，行为不变（阶段2 最小适配，统计逻辑本身未动）。
            k_stat = llm.model_runner.config.spec_num_tokens
            spec_stats = {"rounds": 0, "committed": 0, "pos": [0] * k_stat}
            orig_spec_round = llm.model_runner.spec_round

            def spec_round_with_stats(seqs):
                out = orig_spec_round(seqs)
                for tokens in out:
                    c = len(tokens)          # 本轮提交数 = 接受 draft 数 + 1 bonus
                    spec_stats["rounds"] += 1
                    spec_stats["committed"] += c
                    for j in range(c - 1):   # 前 c-1 个提案位被接受
                        spec_stats["pos"][j] += 1
                return out

            llm.model_runner.spec_round = spec_round_with_stats

        # 串行正式跑：单墙钟计时——对齐 vllm「Output token throughput = Σ输出
        # token / 主跑总墙钟」口径（含逐条 tokenize/构造/打印的 host 开销）。
        # 重复请求负载：每条 prompt 连续跑 num_repeats 次（相邻重复保证 rep>0
        # 全量命中前缀缓存；同 prompt 只 tokenize 一次）。命中统计在预热后清零。
        llm.scheduler.block_manager.reset_prefix_stats()
        if hasattr(llm.model_runner, "df_prefix_snapshot_stats"):
            for key in llm.model_runner.df_prefix_snapshot_stats:
                llm.model_runner.df_prefix_snapshot_stats[key] = 0
        rows = []  # (n, ttft, wall, rep)
        prompt_lens = []
        t_bench = time.perf_counter()
        # 进度条替代逐条打印：末端实时显示本请求的 TPOT/TTFT；刷新开销计入单墙钟
        # （与「含逐条 host 开销」口径一致）。
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
        if (not llm.model_runner.spec_enabled
                and getattr(llm.model_runner, "graph_capture_attempted", False)
                and hasattr(llm.model_runner, "get_decode_graph_stats")):
            graph_stats = llm.model_runner.get_decode_graph_stats()
        if args.spec_model and hasattr(llm.model_runner, "get_spec_graph_stats"):
            spec_graph_stats = llm.model_runner.get_spec_graph_stats()
        if hasattr(llm.model_runner, "df_prefix_snapshot_stats"):
            draft_prefix_stats = {
                **llm.model_runner.df_prefix_snapshot_stats,
                "bytes": llm.model_runner._df_prefix_snapshot_bytes,
            }
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
    # TPOT：逐请求 (wall - TTFT) / (输出 token 数 - 1) 后取均值（对齐 vllm bench serve
    # 的 Mean TPOT）；输出 ≤1 token 的请求无定义，不参与。串行 bs=1 下与下方
    # 「Steady-state decode throughput」互为倒数（此处为请求均值，后者为 token 加权）。
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
        graph_total = graph_stats["replay"] + graph_stats["fallback"]
        graph_rate = graph_stats["replay"] / graph_total if graph_total else 0.0
        print(
            "Decode Graph runtime = "
            f"captures_ok={graph_stats['capture_success']}, "
            f"captures_failed={graph_stats['capture_failure']}, "
            f"replays={graph_stats['replay']}, "
            f"fallbacks={graph_stats['fallback']}, "
            f"replay_rate={graph_rate:.1%}, "
            f"bucket_hits={graph_stats['bucket_hits']}"
        )
    if spec_graph_stats is not None:
        graph_total = spec_graph_stats["replay"] + spec_graph_stats["eager_rounds"]
        graph_rate = spec_graph_stats["replay"] / graph_total if graph_total else 0.0
        print(
            "Spec graph runtime = "
            f"captures_ok={spec_graph_stats['capture_success']}, "
            f"captures_failed={spec_graph_stats['capture_failure']}, "
            f"replays={spec_graph_stats['replay']}, "
            f"eager_rounds={spec_graph_stats['eager_rounds']}, "
            f"sampling_fallbacks={spec_graph_stats['sampling_fallbacks']}, "
            f"replay_rate={graph_rate:.1%}, "
            f"bucket_hits={spec_graph_stats['bucket_hits']}"
        )
    if args.num_repeats > 1:
        # 冷/热分组（A/B 判据）：rep=0 = 冷跑（全量 prefill），rep>0 = 热跑（前缀
        # 命中，prefill 只重算尾部不足一块）。预热(>0)已把首条 prompt 写入缓存，
        # 其主跑 rep=0 非有效冷样本——开/关缓存两边统一剔除，保证口径一致。
        cold = [r[1] for r in rows if r[3] == 0]
        warm = [r[1] for r in rows if r[3] > 0]
        if args.warmup > 0 and len(cold) > 1:
            cold = cold[1:]
        if cold:
            print(f"TTFT cold (rep=0, n={len(cold)}) = {sum(cold) / len(cold) * 1000:.1f} ms")
        if warm:
            print(f"TTFT warm (rep>0, n={len(warm)}) = {sum(warm) / len(warm) * 1000:.1f} ms")
    pstats = llm.scheduler.block_manager.get_prefix_stats()
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
