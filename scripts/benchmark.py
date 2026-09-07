"""spec_bench question.jsonl 串行 TPS/TTFT 基准（对齐 vllm bench serve 口径）。

口径（与 vllm bench serve 的 spec_bench 数据集逐项对齐，同 --seed 即同 prompt 集）：
- prompt：对 turns[0] 套 chat template（tokenize=False 字符串、不传 kwargs →
  Qwen3 默认 enable_thinking）后 encode；
- 数据集：random.seed(--seed) 后 shuffle 取前 --num-prompts 条（--seed -1 = 文件序前 N 条）；
- 采样：temperature=0 greedy，输出上限 --max-tokens（默认 256 = vllm spec_bench output_len）；
- 预热：--warmup 次（不计指标），统一打 shuffle 后首条 prompt（对齐 vllm --num-warmups）；
- 吞吐：Σ输出 token / 主跑总墙钟（单墙钟，含逐条 tokenize/打印的 host 开销，
  对齐 vllm「Output token throughput」）；TTFT 为进程内口径（vllm 为 client 端，高几 ms）。

用法（必须从仓库根目录以 -m 方式运行，直跑会 ModuleNotFoundError）：
    python3 -m scripts.benchmark /path/to/Qwen3-1.7B [--spec-model <draft 目录>]
        [--questions <spec_bench/question.jsonl>] [--num-prompts 10]
        [--max-tokens 256] [--seed 0] [--warmup 5] [--category coding]
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
                        default=os.path.expanduser("~/vllm-lab/benchdata/spec_bench/question.jsonl"),
                        help="question.jsonl 路径（默认本机 spec_bench benchdata）")
    parser.add_argument("--category", default="coding",
                        help="spec_bench 类别过滤（默认 coding；all 不过滤；文件无 category 字段时不过滤）")
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="每条输出上限（对齐 vllm spec_bench 默认 output_len=256；EOS 即停）")
    parser.add_argument("--seed", type=int, default=0,
                        help="数据集 shuffle 种子（对齐 vllm bench --seed 默认 0：同 seed 即同 "
                             "prompt 集；-1=不 shuffle，文件序前 N 条历史口径）")
    parser.add_argument("--warmup", type=int, default=5,
                        help="预热次数（不计指标；对齐 vllm --num-warmups，统一打首条 prompt）")
    parser.add_argument("--spec-model", default="", help="EAGLE3 draft 目录（给出即启用投机）")
    parser.add_argument("--spec-num-tokens", type=int, default=3)
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
    print(f"prompts: {len(qs)} (category={args.category}, seed={args.seed}) | mode={spec} | "
          f"max_tokens={args.max_tokens} | temp={TEMPERATURE} | 口径=vllm bench serve 对齐\n",
          flush=True)

    kwargs = dict(enforce_eager=args.eager, tensor_parallel_size=1, backend=args.backend,
                  max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                  kvcache_block_size=args.block_size, compile=args.compile,
                  gpu_memory_utilization=args.gpu_mem_util)
    if args.spec_model:
        kwargs.update(spec_model=args.spec_model, spec_num_tokens=args.spec_num_tokens)
    llm = LLM(args.model, **kwargs)
    try:
        sp = SamplingParams(temperature=TEMPERATURE, max_tokens=args.max_tokens)

        def chat(q):
            # 对齐 vllm bench spec_bench 口径：CustomDataset.sample 对 prompt 套
            # chat template（tokenize=False 字符串、不传 kwargs → Qwen3 默认
            # enable_thinking）后作为 completions 的 prompt 发送；此处同构生成 id。
            text = llm.tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                add_generation_prompt=True, tokenize=False)
            return llm.tokenizer.encode(text)

        # 预热（不计指标）：对齐 vllm bench——就绪检查 + num_warmups 均打同一首条
        # prompt（shuffle 后的 qs[0]）。
        for _ in range(args.warmup):
            run_seqs(llm, [Sequence(chat(qs[0]), sp)])

        # 串行正式跑：单墙钟计时——对齐 vllm「Output token throughput = Σ输出
        # token / 主跑总墙钟」口径（含逐条 tokenize/构造/打印的 host 开销）。
        rows = []  # (n, ttft, wall)
        t_bench = time.perf_counter()
        for i, q in enumerate(qs, 1):
            tokens, ttfts, wall = run_seqs(llm, [Sequence(chat(q), sp)])
            n = len(tokens[0])
            ttft = ttfts[0] or 0.0
            rows.append((n, ttft, wall))
            print(f"#{i:>2}  tok={n:>4}  TTFT={ttft * 1000:>7.1f}ms  wall={wall:>6.2f}s",
                  flush=True)
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


if __name__ == "__main__":
    main()
