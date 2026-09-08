"""spec_bench question.jsonl 串行 TPS/TTFT 基准（对齐 vllm bench serve 口径）。

口径（与 vllm bench serve 的 spec_bench 数据集逐项对齐，同 --seed 即同 prompt 集）：
- prompt：对 turns[0] 套 chat template（tokenize=False 字符串、不传 kwargs →
  Qwen3 默认 enable_thinking）后 encode；--no-think 显式关 thinking（模板插入
  空 <think></think> 直答，对齐 V1.8/历史 E 锚点口径，用于单变量二分接受率差异）；
- 数据集：random.seed(--seed) 后 shuffle 取前 --num-prompts 条（--seed -1 = 文件序前 N 条）；
- 采样：temperature=0 greedy，输出上限 --max-tokens（默认 256 = vllm spec_bench output_len）；
- 预热：--warmup 次（不计指标），统一打 shuffle 后首条 prompt（对齐 vllm --num-warmups）；
- 吞吐：Σ输出 token / 主跑总墙钟（单墙钟，含逐条 tokenize/打印的 host 开销，
  对齐 vllm「Output token throughput」）；TTFT 为进程内口径（vllm 为 client 端，高几 ms）；
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

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence

TEMPERATURE = 0


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
        raise SystemExit(f"问题集为空: {path}（category={category}）")
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


def main():
    parser = argparse.ArgumentParser(
        description="spec_bench question.jsonl 串行 TPS/TTFT 基准（vllm bench serve 口径）")
    parser.add_argument("model")
    parser.add_argument("--questions",
                        default=os.path.join("..", "question.jsonl"),
                        help="question.jsonl 路径（默认仓库上级目录的 spec_bench question.jsonl）")
    parser.add_argument("--category", default="coding",
                        help="spec_bench 类别过滤（默认 coding；all 不过滤；文件无 category 字段时不过滤）")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--num-repeats", type=int, default=1,
                        help="每条 prompt 连续重复次数（>1 构成重复请求负载：第 2..R 次"
                             "全量命中前缀缓存；配合 --no-prefix-cache 做 TTFT A/B）")
    parser.add_argument("--no-prefix-cache", action="store_true",
                        help="关闭前缀缓存（enable_prefix_cache=False，引擎跳过 block hash"
                             "与命中查找），用于 TTFT/吞吐 A/B 对照")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="每条输出上限（对齐 vllm spec_bench 默认 output_len=256；EOS 即停）")
    parser.add_argument("--seed", type=int, default=0,
                        help="数据集 shuffle 种子（对齐 vllm bench --seed 默认 0：同 seed 即同 "
                             "prompt 集；-1=不 shuffle，文件序前 N 条历史口径）")
    parser.add_argument("--warmup", type=int, default=5,
                        help="预热次数（不计指标；对齐 vllm --num-warmups，统一打首条 prompt）")
    parser.add_argument("--spec-model", default="", help="EAGLE3 draft 目录（给出即启用投机）")
    parser.add_argument("--spec-num-tokens", type=int, default=3)
    parser.add_argument("--no-think", action="store_true",
                        help="chat template 显式 enable_thinking=False（Qwen3 模板插入空 "
                             "<think></think>，直答；默认不传 kwargs=vllm bench 对齐口径，"
                             "thinking 默认开启）。用于对齐 V1.8/历史 E 锚点做单变量对照")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex", "ascend310p"])
    parser.add_argument("--eager", action="store_true", help="关闭 CUDA Graph")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9,
                        help="gpu_memory_utilization（默认 0.9；空板卡 KV 贴地需给图/运行峰值留余量时可调低）")
    args = parser.parse_args()

    qs = load_questions(args.questions, args.category, args.num_prompts, args.seed)
    spec = f"spec(k={args.spec_num_tokens})" if args.spec_model else "decode"
    thinking = "off" if args.no_think else "on(default)"
    pcache = "off" if args.no_prefix_cache else "on"
    print(f"prompts: {len(qs)} (category={args.category}, seed={args.seed}) "
          f"x repeats={args.num_repeats} | mode={spec} | prefix_cache={pcache} | "
          f"max_tokens={args.max_tokens} | temp={TEMPERATURE} | thinking={thinking} | "
          f"口径=vllm bench serve 对齐\n",
          flush=True)

    kwargs = dict(enforce_eager=args.eager, tensor_parallel_size=1, backend=args.backend,
                  max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                  kvcache_block_size=args.block_size, compile=args.compile,
                  gpu_memory_utilization=args.gpu_mem_util,
                  enable_prefix_cache=not args.no_prefix_cache)
    if args.spec_model:
        kwargs.update(spec_model=args.spec_model, spec_num_tokens=args.spec_num_tokens)
    llm = LLM(args.model, **kwargs)
    try:
        sp = SamplingParams(temperature=TEMPERATURE, max_tokens=args.max_tokens)

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

        # 投机解码接受率统计（对齐历史 check_eagle3 的 monkey-patch spec_round 口径，
        # 引擎零改动）：spec_round 每轮返回各 seq 的提交 token 列表，长度 = 接受
        # draft 数 m + 1 bonus（图内 _sg_acceptance 的 cnt 打包，冷/热/图/eager
        # 四路统一从 _readback_spec_commit 出）。预热后安装，仅计主跑。
        spec_stats = None
        if args.spec_model:
            k_stat = args.spec_num_tokens
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
        rows = []  # (n, ttft, wall, rep)
        t_bench = time.perf_counter()
        i = 0
        for q in qs:
            tokens_in = chat(q)
            for rep in range(args.num_repeats):
                i += 1
                tokens, ttfts, wall = run_seqs(llm, [Sequence(tokens_in, sp)])
                n = len(tokens[0])
                ttft = ttfts[0] or 0.0
                rows.append((n, ttft, wall, rep))
                print(f"#{i:>3}  rep={rep}  tok={n:>4}  TTFT={ttft * 1000:>7.1f}ms  "
                      f"wall={wall:>6.2f}s", flush=True)
        bench_wall = time.perf_counter() - t_bench
    finally:
        llm.exit()

    total_tok = sum(r[0] for r in rows)
    ttfts = sorted(r[1] for r in rows)
    print(f"\n==== bench (n={len(rows)}) ====")
    print(f"Benchmark duration = {bench_wall:.2f} s   "
          f"Total generated tokens = {total_tok}")
    print(f"Output token throughput = {total_tok / bench_wall if bench_wall else 0.0:.1f} tok/s")
    print(f"TTFT avg = {sum(ttfts) / len(ttfts) * 1000:.1f} ms   "
          f"median = {ttfts[len(ttfts) // 2] * 1000:.1f} ms")
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
    print(f"Prefix cache = {'off' if args.no_prefix_cache else 'on'}  "
          f"(requests={pstats['requests']}, prefix_hits={pstats['prefix_hits']}, "
          f"hit_rate={pstats['hit_rate']:.1%}, cached_blocks={pstats['cached_blocks']}, "
          f"new_blocks={pstats['new_blocks']})")
    if spec_stats is not None and spec_stats["rounds"]:
        rounds = spec_stats["rounds"]
        accepted = spec_stats["committed"] - rounds   # Σ 每轮接受 draft 数（提交数 - bonus）
        k = args.spec_num_tokens
        print(f"Acceptance length (E) = {spec_stats['committed'] / rounds:.3f}   "
              f"(avg committed/round = 1 bonus + accepted/rounds)")
        print(f"Token acceptance rate = {accepted / (rounds * k):.1%}   "
              f"(accepted {accepted} / drafted {rounds * k})")
        print("Per-position acceptance = "
              + " / ".join(f"{p / rounds:.1%}" for p in spec_stats["pos"]))


if __name__ == "__main__":
    main()
