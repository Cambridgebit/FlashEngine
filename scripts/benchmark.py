"""Nano-vLLM 性能验证：CUDA Graph 与 Prefix Cache。

子命令 graph  ：不同 batch size 下 eager vs CUDA Graph 的 TPS/TTFT 对比 + token 级正确性
子命令 prefix ：长共享 system prompt 下 prefix cache 冷/热 TTFT 与 wall 时间对比
子命令 jsonl  ：spec_bench 串行 TPS/TTFT 基准（默认对齐 vllm bench serve 口径），可选投机

用法：
    python3.10 -m scripts.benchmark graph  /path/to/Qwen3-1.7B [--batch 1 2 4 8] [--backend corex]
                  [--max-tokens 128] [--block-size 256] [--max-model-len 4096]
    python3.10 -m scripts.benchmark prefix /path/to/Qwen3-1.7B [--backend corex]
                  [--sys-tokens 1200] [--max-tokens 8] [--block-size 64] [--max-model-len 8192]
    python3.10 -m scripts.benchmark jsonl  /path/to/Qwen3-1.7B [--questions <spec_bench/question.jsonl>]
                  [--num-prompts 100] [--max-tokens 256] [--seed 0] [--spec-model <draft 目录>]
"""
import argparse
import atexit
import gc
import json
import os
import random
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


def new_engine(model, eager, backend, max_model_len, max_num_seqs, block_size,
               compile=False, spec_model="", spec_num_tokens=3, gpu_mem_util=0.9):
    llm = None
    try:
        kwargs = dict(enforce_eager=eager, tensor_parallel_size=1, backend=backend,
                      max_model_len=max_model_len, max_num_seqs=max_num_seqs,
                      kvcache_block_size=block_size, compile=compile,
                      gpu_memory_utilization=gpu_mem_util)
        if spec_model:
            kwargs.update(spec_model=spec_model, spec_num_tokens=spec_num_tokens)
        llm = LLM(model, **kwargs)
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
    # 显存回收走 backend 代理（scripts 零硬件直呼：NPU-only 主机上
    # torch.cuda.empty_cache() 会因 CUDA 初始化失败而崩）。
    backend = llm.model_runner.backend
    llm.exit()
    del llm
    gc.collect()
    backend.empty_cache()


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
                             args.max_model_len, bs, args.block_size, args.compile,
                             gpu_mem_util=args.gpu_mem_util)
            try:
                out = llm.tokenizer.apply_chat_template(
                    [{"role": "user", "content": PROMPT}], tokenize=True,
                    add_generation_prompt=True)
                # 兼容旧 transformers（<4.44 return_dict=False 直接返回 list）
                prompt = out if isinstance(out, (list, tuple)) else out["input_ids"]
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
        out = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys_text}, {"role": "user", "content": q}],
            tokenize=True, add_generation_prompt=True)
        # 兼容旧 transformers（<4.44 return_dict=False 直接返回 list）
        return out if isinstance(out, (list, tuple)) else out["input_ids"]

    p1, p2 = chat(Q1), chat(Q2)
    print(f"sys={sys_len} tok | p1={len(p1)} | p2={len(p2)} | "
          f"block_size={args.block_size} | max_tokens={args.max_tokens}\n")

    for mode, eager in (("EAGER", True), ("CUDA GRAPH", False)):
        llm = new_engine(args.model, eager, args.backend,
                         args.max_model_len, 1, args.block_size, args.compile,
                         gpu_mem_util=args.gpu_mem_util)
        try:
            _, ttfts, wall_cold = run_seqs(llm, [
                Sequence(p2, SamplingParams(
                    temperature=TEMPERATURE, max_tokens=args.max_tokens, ignore_eos=True))])
            ttft_cold = ttfts[0]
        finally:
            close_engine(llm)

        llm = new_engine(args.model, eager, args.backend,
                         args.max_model_len, 1, args.block_size, args.compile,
                         gpu_mem_util=args.gpu_mem_util)
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


def load_jsonl_prompts(path, category, num, seed):
    """spec_bench question.jsonl → prompts（取 turns[0]；category=all 不过滤）。

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
            if category != "all" and d.get("category") != category:
                continue
            qs.append(d["turns"][0])
    if not qs:
        raise SystemExit(f"问题集为空: {path}（category={category}）")
    if seed >= 0:
        random.seed(seed)
        random.shuffle(qs)
    return qs[:num]


def cmd_jsonl(args):
    qs = load_jsonl_prompts(args.questions, args.category, args.num_prompts, args.seed)
    spec = f"spec(k={args.spec_num_tokens})" if args.spec_model else "decode"
    print(f"prompts: {len(qs)} (category={args.category}, seed={args.seed}) | mode={spec} | "
          f"max_tokens={args.max_tokens} | temp={TEMPERATURE} | 口径=vllm bench serve 对齐\n",
          flush=True)

    llm = new_engine(args.model, args.eager, args.backend, args.max_model_len,
                     args.max_num_seqs, args.block_size, args.compile,
                     spec_model=args.spec_model, spec_num_tokens=args.spec_num_tokens,
                     gpu_mem_util=args.gpu_mem_util)
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
        close_engine(llm)

    total_tok = sum(r[0] for r in rows)
    ttfts = sorted(r[1] for r in rows)
    print(f"\n==== jsonl bench (n={len(rows)}, vllm bench serve 口径) ====")
    print(f"Benchmark duration = {bench_wall:.2f} s   "
          f"Total generated tokens = {total_tok}")
    print(f"Output token throughput = {total_tok / bench_wall if bench_wall else 0.0:.1f} tok/s")
    print(f"TTFT avg = {sum(ttfts) / len(ttfts) * 1000:.1f} ms   "
          f"median = {ttfts[len(ttfts) // 2] * 1000:.1f} ms（进程内口径；vllm 为 client 端，高几 ms）")


def main():
    parser = argparse.ArgumentParser(description="nano-vLLM 性能验证 (CUDA Graph / Prefix Cache)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("graph", help="batch size 下 eager vs CUDA Graph TPS/TTFT")
    g.add_argument("model")
    g.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    g.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex", "ascend310p"])
    g.add_argument("--compile", action="store_true",
                   help="torch.compile 编译模型（CoreX 上约 8% prefill 提速）")
    g.add_argument("--max-tokens", type=int, default=128)
    g.add_argument("--max-model-len", type=int, default=4096)
    g.add_argument("--block-size", type=int, default=256)
    g.add_argument("--gpu-mem-util", type=float, default=0.9,
                   help="gpu_memory_utilization（默认 0.9；空板卡 KV 贴地需给图捕获留余量时可调低）")
    g.set_defaults(fn=cmd_graph)

    p = sub.add_parser("prefix", help="prefix cache 冷/热 TTFT 对比")
    p.add_argument("model")
    p.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex", "ascend310p"])
    p.add_argument("--compile", action="store_true",
                   help="torch.compile 编译模型（CoreX 上约 8% prefill 提速）")
    p.add_argument("--sys-tokens", type=int, default=1200, help="共享 system prompt 目标 token 数")
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--gpu-mem-util", type=float, default=0.9,
                   help="gpu_memory_utilization（默认 0.9；空板卡 KV 贴地需给图捕获留余量时可调低）")
    p.set_defaults(fn=cmd_prefix)

    j = sub.add_parser("jsonl", help="spec_bench question.jsonl 真实问题集串行 TPS/TTFT")
    j.add_argument("model")
    j.add_argument("--questions",
                   default=os.path.expanduser("~/vllm-lab/benchdata/spec_bench/question.jsonl"),
                   help="question.jsonl 路径（默认本机 spec_bench benchdata）")
    j.add_argument("--category", default="coding",
                   help="spec_bench 类别过滤（默认 coding；all 不过滤）")
    j.add_argument("--num-prompts", type=int, default=10)
    j.add_argument("--max-tokens", type=int, default=256,
                   help="每条输出上限（对齐 vllm spec_bench 默认 output_len=256；EOS 即停）")
    j.add_argument("--seed", type=int, default=0,
                   help="数据集 shuffle 种子（对齐 vllm bench --seed 默认 0：同 seed 即同 "
                        "prompt 集；-1=不 shuffle，文件序前 N 条历史口径）")
    j.add_argument("--spec-model", default="", help="EAGLE3 draft 目录（给出即启用投机）")
    j.add_argument("--spec-num-tokens", type=int, default=3)
    j.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex", "ascend310p"])
    j.add_argument("--eager", action="store_true", help="关闭 CUDA Graph")
    j.add_argument("--compile", action="store_true")
    j.add_argument("--max-model-len", type=int, default=4096)
    j.add_argument("--max-num-seqs", type=int, default=8)
    j.add_argument("--block-size", type=int, default=256)
    j.add_argument("--warmup", type=int, default=5,
                   help="预热次数（不计指标；对齐 vllm --num-warmups，统一打首条 prompt）")
    j.add_argument("--gpu-mem-util", type=float, default=0.9,
                   help="gpu_memory_utilization（默认 0.9；空板卡 KV 贴地需给图/运行峰值留余量时可调低）")
    j.set_defaults(fn=cmd_jsonl)

    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()