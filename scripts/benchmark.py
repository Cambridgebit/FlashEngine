"""Nano-vLLM 性能验证：CUDA Graph 与 Prefix Cache。

子命令 graph  ：不同 batch size 下 eager vs CUDA Graph 的 TPS/TTFT 对比 + token 级正确性
子命令 prefix ：长共享 system prompt 下 prefix cache 冷/热 TTFT 与 wall 时间对比

用法：
    python3.10 -m scripts.benchmark graph  /path/to/Qwen3-1.7B [--batch 1 2 4 8] [--backend corex]
                  [--max-tokens 128] [--block-size 256] [--max-model-len 4096]
    python3.10 -m scripts.benchmark prefix /path/to/Qwen3-1.7B [--backend corex]
                  [--sys-tokens 1200] [--max-tokens 8] [--block-size 64] [--max-model-len 8192]
"""
import argparse
import atexit
import gc
import time

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


TEMPERATURE = 0
PROMPT = "请用三句话解释什么是大语言模型。"
WARM_TOKENS = 1300
WARM_TEXT = "你好，这是用于初始化长序列 prefill kernel 的预热文本。"
TOOL_UNIT = (
    "search_web(url, query): 在互联网搜索并返回结果; "
    "call_api(endpoint, payload): 调用 REST API; "
    "read_file(path): 读取本地文件; "
    "run_shell(cmd): 执行 shell 命令; "
    "write_file(path, content): 写入文件; "
    "analyze(data): 分析结构化数据; "
)
Q1 = "帮我查一下今天的天气并给出穿衣建议。"
Q2 = "总结一下当前目录下的文件结构。"


def new_engine(model, eager, backend, max_model_len, max_num_seqs, block_size, compile=False):
    llm = None
    try:
        llm = LLM(model, enforce_eager=eager, tensor_parallel_size=1, backend=backend,
                  max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                  kvcache_block_size=block_size, compile=compile)
        warm = Sequence(llm.tokenizer.encode("hi"),
                        SamplingParams(temperature=TEMPERATURE, max_tokens=1, ignore_eos=True))
        llm.scheduler.add(warm)
        while not llm.is_finished():
            llm.step()
        ids = []
        while len(ids) < WARM_TOKENS:
            ids.extend(llm.tokenizer.encode(WARM_TEXT))
        warm = Sequence(ids[:WARM_TOKENS],
                        SamplingParams(temperature=TEMPERATURE, max_tokens=1, ignore_eos=True))
        llm.scheduler.add(warm)
        while not llm.is_finished():
            llm.step()
        llm.scheduler.block_manager.reset_prefix_stats()
        return llm
    except BaseException:
        if llm is not None:
            llm.exit()
        raise


def close_engine(llm):
    atexit.unregister(llm.exit)
    llm.exit()
    del llm
    gc.collect()
    torch.cuda.empty_cache()


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
    tokens = [[t for t in s.completion_token_ids if t != eos] for s in seqs]
    return tokens, [first[s.seq_id] for s in seqs], wall


def cmd_graph(args):
    results = {}
    for mode, eager in (("eager", True), ("graph", False)):
        print(f"\n===== {mode.upper()} =====")
        print(f"{'batch':>6}{'TPS':>10}{'TTFT avg(ms)':>14}{'TTFT max(ms)':>14}")
        results[mode] = {}
        for bs in args.batch:
            llm = new_engine(args.model, eager, args.backend,
                             args.max_model_len, bs, args.block_size, args.compile)
            try:
                prompt = llm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": PROMPT}], tokenize=True,
                    add_generation_prompt=True)["input_ids"]
                seqs = [Sequence(prompt, SamplingParams(
                    temperature=TEMPERATURE, max_tokens=args.max_tokens, ignore_eos=True))
                    for _ in range(bs)]
                tokens, ttft, wall = run_seqs(llm, seqs)
                total = sum(len(t) for t in tokens)
                tps = total / wall
                results[mode][bs] = (tokens, tps, sum(ttft) / len(ttft), max(ttft))
                print(f"{bs:>6}{tps:>10.1f}{sum(ttft) / len(ttft) * 1000:>14.1f}"
                      f"{max(ttft) * 1000:>14.1f}", flush=True)
            finally:
                close_engine(llm)

    print(f"\n===== GRAPH vs EAGER (block_size={args.block_size}, max_tokens={args.max_tokens}) =====")
    print(f"{'batch':>6}{'TPS x':>10}{'TTFT x':>10}{'correct':>10}")
    for bs in args.batch:
        te, tg = results["eager"][bs], results["graph"][bs]
        ok = all(a == b for a, b in zip(te[0], tg[0]))
        print(f"{bs:>6}{tg[1] / te[1]:>10.2f}{tg[3] / te[3]:>10.2f}"
              f"{'PASS' if ok else 'FAIL':>10}")


def _build_system(tokenizer, target):
    n = 1
    while len(tokenizer.encode(TOOL_UNIT)) * n < target:
        n += 1
    return TOOL_UNIT * n


def cmd_prefix(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    sys_text = _build_system(tokenizer, args.sys_tokens)
    sys_len = len(tokenizer.encode(sys_text))

    def chat(q):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_text}, {"role": "user", "content": q}],
            tokenize=True, add_generation_prompt=True)["input_ids"]

    p1, p2 = chat(Q1), chat(Q2)
    print(f"sys={sys_len} tok | p1={len(p1)} | p2={len(p2)} | "
          f"block_size={args.block_size} | max_tokens={args.max_tokens}\n")

    for mode, eager in (("EAGER", True), ("CUDA GRAPH", False)):
        llm = new_engine(args.model, eager, args.backend,
                         args.max_model_len, 1, args.block_size, args.compile)
        try:
            _, ttfts, wall_cold = run_seqs(llm, [
                Sequence(p2, SamplingParams(
                    temperature=TEMPERATURE, max_tokens=args.max_tokens, ignore_eos=True))])
            ttft_cold = ttfts[0]
        finally:
            close_engine(llm)

        llm = new_engine(args.model, eager, args.backend,
                         args.max_model_len, 1, args.block_size, args.compile)
        try:
            run_seqs(llm, [Sequence(p1, SamplingParams(
                temperature=TEMPERATURE, max_tokens=args.max_tokens, ignore_eos=True))])
            _, ttfts, wall_warm = run_seqs(llm, [
                Sequence(p2, SamplingParams(
                    temperature=TEMPERATURE, max_tokens=args.max_tokens, ignore_eos=True))])
            ttft_warm = ttfts[0]
            stats = llm.scheduler.block_manager.get_prefix_stats()
        finally:
            close_engine(llm)

        print(f"--- {mode} ---")
        print(f"p2 cold: TTFT={ttft_cold * 1000:7.1f} ms  wall={wall_cold * 1000:7.1f} ms")
        print(f"p2 warm: TTFT={ttft_warm * 1000:7.1f} ms  wall={wall_warm * 1000:7.1f} ms")
        print(f"TTFT speedup = {ttft_cold / ttft_warm:.2f}x   wall speedup = {wall_cold / wall_warm:.2f}x")
        print(f"prefix cache: requests={stats['requests']} hits={stats['prefix_hits']} "
              f"cached={stats['cached_blocks']} new={stats['new_blocks']} "
              f"(hit_rate={stats['hit_rate'] * 100:.0f}%)")
        print()


def main():
    parser = argparse.ArgumentParser(description="nano-vLLM 性能验证 (CUDA Graph / Prefix Cache)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("graph", help="batch size 下 eager vs CUDA Graph TPS/TTFT")
    g.add_argument("model")
    g.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    g.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex"])
    g.add_argument("--compile", action="store_true",
                   help="torch.compile 编译模型（CoreX 上约 8% prefill 提速）")
    g.add_argument("--max-tokens", type=int, default=128)
    g.add_argument("--max-model-len", type=int, default=4096)
    g.add_argument("--block-size", type=int, default=256)
    g.set_defaults(fn=cmd_graph)

    p = sub.add_parser("prefix", help="prefix cache 冷/热 TTFT 对比")
    p.add_argument("model")
    p.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex"])
    p.add_argument("--compile", action="store_true",
                   help="torch.compile 编译模型（CoreX 上约 8% prefill 提速）")
    p.add_argument("--sys-tokens", type=int, default=1200, help="共享 system prompt 目标 token 数")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--block-size", type=int, default=256)
    p.set_defaults(fn=cmd_prefix)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()