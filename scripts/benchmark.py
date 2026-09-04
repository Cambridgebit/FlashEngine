"""Eager vs CUDA Graph 推理性能与正确性基准（真实对话）。

对比两种引擎模式（同一模型、同一批真实对话 prompt）：
- enforce_eager=True   → eager
- enforce_eager=False  → CUDA Graph（FULL_DECODE_ONLY）

指标：
- TTFT：每个请求从加入调度队列到产出第一个输出 token 的时间（含 prefill）
- TPS ：总输出 token 数 / 总墙钟时间
- 正确性：温度取 1e-6（近 greedy）保证 eager / graph 采样可比，
  逐 token 比对两条输出序列，并打印完整回答便于人工核对。

用法：
    python scripts/benchmark.py /path/to/Qwen3-1.7B [--max-tokens 128]
                  [--mode both|eager|graph] [--backend auto|cuda|corex]
                  [--repeat 1] [--max-model-len 4096] [--max-num-seqs 64]
"""
import argparse
import atexit
import gc
import time

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


# 近 greedy：保证 eager / graph 的输出可比（Gumbel-max 在近 one-hot 下确定）
TEMPERATURE = 1e-6

CONVERSATIONS = [
    [
        {"role": "system", "content": "你是一个有用的中文助手。"},
        {"role": "user", "content": "请用三句话解释什么是大语言模型。"},
    ],
    [
        {"role": "user", "content": "What is the capital of France? Answer in one sentence."},
    ],
    [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Explain quantum computing in plain words."},
    ],
    [
        {"role": "user", "content": "写一段 80 字左右的介绍，关于人工智能在医疗领域的应用。"},
    ],
    [
        {"role": "user", "content": "把这句话翻译成英文：今天天气很好，我们去公园散步吧。"},
    ],
    [
        {"role": "user", "content": "用 Python 写一个快速排序函数。"},
    ],
    [
        {"role": "user", "content": "列出深度学习常用的 5 个优化器并简要说明。"},
    ],
    [
        {"role": "user", "content": "Summarize the benefits of CUDA Graphs for LLM inference in 3 bullet points."},
    ],
]


def _add_req(llm, prompt_ids, max_tokens):
    seq = Sequence(prompt_ids, SamplingParams(temperature=TEMPERATURE, max_tokens=max_tokens))
    llm.scheduler.add(seq)
    return seq


def _drive(llm, seqs, max_tokens):
    """驱动引擎直到全部 seq 结束，采集 TTFT / token 计数 / 文本。"""
    eos = llm.scheduler.eos
    first, counts, n_seen = {}, {}, {}
    for seq in seqs:
        first[seq.seq_id] = None
        counts[seq.seq_id] = []
        n_seen[seq.seq_id] = 0

    t0 = time.perf_counter()
    while not llm.is_finished():
        llm.step()
        now = time.perf_counter()
        for seq in seqs:
            comp = seq.completion_token_ids
            seen = n_seen[seq.seq_id]
            if len(comp) > seen:
                new = comp[seen:]
                if first[seq.seq_id] is None and new:
                    first[seq.seq_id] = now - t0
                counts[seq.seq_id].extend(t for t in new if t != eos)
                n_seen[seq.seq_id] = len(comp)
    wall = time.perf_counter() - t0

    total = sum(len(v) for v in counts.values())
    ttfts = [v for v in first.values() if v is not None]
    return {
        "seqs": seqs,
        "first": first,
        "counts": counts,
        "wall": wall,
        "total": total,
        "ttft_avg": sum(ttfts) / len(ttfts) if ttfts else 0.0,
        "ttft_max": max(ttfts) if ttfts else 0.0,
        "tps": total / wall if wall > 0 else 0.0,
        "prompts": [
            llm.tokenizer.decode(seq.prompt_token_ids, skip_special_tokens=True)
            for seq in seqs
        ],
        "texts": {
            seq.seq_id: llm.tokenizer.decode(counts[seq.seq_id], skip_special_tokens=True)
            for seq in seqs
        },
    }


def _report(mode, res):
    print(f"wall time      : {res['wall']:.3f} s")
    print(f"output tokens  : {res['total']}")
    print(f"TPS            : {res['tps']:.2f} tok/s")
    print(f"TTFT avg       : {res['ttft_avg'] * 1000:.1f} ms")
    print(f"TTFT max       : {res['ttft_max'] * 1000:.1f} ms")
    print(f"requests       : {len(res['seqs'])}")
    print("-" * 70)
    for i, seq in enumerate(res["seqs"]):
        print(f"[req {i}] TTFT {res['first'][seq.seq_id] * 1000:7.1f} ms | "
              f"tokens {len(res['counts'][seq.seq_id])}")
        print(f"    prompt: {res['prompts'][i][:100]}")
        print(f"    answer: {res['texts'][seq.seq_id]}")
    print()


def _run_mode(model, mode, args):
    kwargs = dict(
        model=model,
        enforce_eager=(mode == "eager"),
        tensor_parallel_size=1,
        backend=args.backend,
        max_model_len=args.max_model_len,
    )
    if args.max_num_seqs is not None:
        kwargs["max_num_seqs"] = args.max_num_seqs

    tag = "CUDA Graph" if mode == "graph" else "eager"
    print(f"\n===== {mode.upper()} ({tag}) =====", flush=True)

    llm = None
    try:
        llm = LLM(**kwargs)

        # warmup：触发 kernel lazy init / workspace，避免污染首请求计时
        warm_ids = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}], tokenize=True, add_generation_prompt=True)
        warm = _add_req(llm, warm_ids, 4)
        while not llm.is_finished():
            llm.step()

        seqs = []
        for _ in range(args.repeat):
            for conv in CONVERSATIONS:
                prompt_ids = llm.tokenizer.apply_chat_template(
                    conv, tokenize=True, add_generation_prompt=True)
                seqs.append(_add_req(llm, prompt_ids, args.max_tokens))
        res = _drive(llm, seqs, args.max_tokens)
        _report(mode, res)
        return res
    finally:
        if llm is not None:
            atexit.unregister(llm.exit)
            llm.exit()
            del llm
            gc.collect()
            torch.cuda.empty_cache()


def _compare(results):
    eager, graph = results["eager"], results["graph"]
    n = min(len(eager["seqs"]), len(graph["seqs"]))
    ok_count = 0
    print("\n===== CORRECTNESS (eager vs cuda graph, token-level) =====")
    for i in range(n):
        te = eager["counts"][eager["seqs"][i].seq_id]
        tg = graph["counts"][graph["seqs"][i].seq_id]
        ok = te == tg
        ok_count += ok
        print(f"[req {i}] {'PASS' if ok else 'FAIL'}  "
              f"eager={len(te)} tok / graph={len(tg)} tok")
        if not ok:
            print(f"    eager: {eager['texts'][eager['seqs'][i].seq_id]}")
            print(f"    graph: {graph['texts'][graph['seqs'][i].seq_id]}")
    print(f"Overall: {ok_count}/{n} PASS")
    return ok_count == n


def _summary(results):
    print("\n===== PERFORMANCE SUMMARY =====")
    print(f"{'mode':<12}{'TPS(tok/s)':>14}{'TTFT avg(ms)':>16}{'TTFT max(ms)':>16}")
    for mode, res in results.items():
        print(f"{mode:<12}{res['tps']:>14.2f}"
              f"{res['ttft_avg'] * 1000:>16.1f}{res['ttft_max'] * 1000:>16.1f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="nano-vLLM eager vs CUDA Graph benchmark (real conversations)")
    parser.add_argument("model", help="model directory path")
    parser.add_argument("--mode", choices=["both", "eager", "graph"], default="both",
                        help="要跑的引擎模式（默认 both：两个都跑并对比）")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex"])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="每个请求的最大输出 token 数")
    parser.add_argument("--repeat", type=int, default=1,
                        help="重复对话集次数，增加并发负载")
    args = parser.parse_args()

    modes = ["eager", "graph"] if args.mode == "both" else [args.mode]
    results = {}
    for mode in modes:
        results[mode] = _run_mode(args.model, mode, args)
    if args.mode == "both":
        ok = _compare(results)
        _summary(results)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()