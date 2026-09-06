"""Prefix Cache 正确性与命中率验证（agent 工作流场景）。

验证内容：
1. 正确性：同一请求在"命中缓存"与"冷缓存"下输出逐 token 完全一致
   —— 证明共享前缀 KV 复用不会污染结果。
2. 命中率：第二个共享前缀的请求应命中 >=1 个 block，报告复用 block 占比。
3. TTFT 收益：命中缓存的请求首 token 时间应显著低于冷启动。

场景：长 system prompt（工具定义，默认约 1200 token）跨多个请求共享，
模拟 agent 多步工作流（batch=1）。

默认 eager；--graph 额外验证 CUDA Graph 模式并与 eager 交叉比对。

用法：
    python3.10 -m scripts.check_prefix_cache /home/iluvatar/Qwen3-1.7B --backend corex
    python3.10 -m scripts.check_prefix_cache ... --graph --block-size 64
"""
import argparse
import atexit
import gc
import time

import torch
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence


TEMPERATURE = 1e-6
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


def _build_system(tokenizer, target):
    n = 1
    while len(tokenizer.encode(TOOL_UNIT)) * n < target:
        n += 1
    return TOOL_UNIT * n


def _run_seq(llm, prompt_ids, max_tokens):
    seq = Sequence(prompt_ids, SamplingParams(
        temperature=TEMPERATURE, max_tokens=max_tokens, ignore_eos=True))
    llm.scheduler.add(seq)
    t0 = time.perf_counter()
    first_t = None
    tokens = []
    n_seen = 0
    eos = llm.scheduler.eos
    while not seq.is_finished:
        llm.step()
        now = time.perf_counter()
        comp = seq.completion_token_ids
        if len(comp) > n_seen:
            if first_t is None and comp[n_seen:]:
                first_t = now - t0
            tokens = [t for t in comp if t != eos]
            n_seen = len(comp)
    return tokens, first_t, time.perf_counter() - t0


def _run_engine(model, backend, eager, block_size, prompts, max_tokens, max_model_len,
                max_num_seqs, gpu_mem_util):
    llm = LLM(
        model,
        enforce_eager=eager,
        tensor_parallel_size=1,
        backend=backend,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_mem_util,
        kvcache_block_size=block_size,
    )
    try:
        # 引擎级 warmup：kernel lazy init / workspace 分配不计入任何计时，
        # 否则冷缓存请求的 TTFT 会被进程级初始化污染。
        warm_out = llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": "hi"}], tokenize=True,
            add_generation_prompt=True)
        # 兼容旧 transformers（<4.44 return_dict=False 直接返回 list）
        warm_ids = warm_out if isinstance(warm_out, (list, tuple)) else warm_out["input_ids"]
        _run_seq(llm, warm_ids, 4)
        llm.scheduler.block_manager.reset_prefix_stats()
        outs = [_run_seq(llm, p, max_tokens) for p in prompts]
        return outs, llm.scheduler.block_manager.get_prefix_stats()
    finally:
        atexit.unregister(llm.exit)
        # 显存回收走 backend 代理（scripts 零硬件直呼：NPU-only 主机上
        # torch.cuda.empty_cache() 会因 CUDA 初始化失败而崩）。
        backend = llm.model_runner.backend
        llm.exit()
        del llm
        gc.collect()
        backend.empty_cache()


def _report(name, outs, stats, sys_len, p1_len, p2_len):
    r1, r2 = outs
    print(f"--- {name} ---")
    print(f"sys={sys_len} tok | prompt1={p1_len} tok | prompt2={p2_len} tok "
          f"| out={len(r1[0])} tok")
    print(f"R1(cold) TTFT={r1[1] * 1000:6.1f} ms   R2(warm) TTFT={r2[1] * 1000:6.1f} ms")
    print(f"R1 wall  ={r1[2] * 1000:7.1f} ms   R2 wall  ={r2[2] * 1000:7.1f} ms")
    print(f"prefix cache: requests={stats['requests']} hits={stats['prefix_hits']} "
          f"cached_blocks={stats['cached_blocks']} new_blocks={stats['new_blocks']} "
          f"(hit_rate={stats['hit_rate'] * 100:.0f}%)")
    print()


def _assert(name, cond, detail=""):
    print(f"[{name}] {'PASS' if cond else 'FAIL'} {detail}")
    return cond


def main():
    parser = argparse.ArgumentParser(
        description="prefix cache correctness & hit-rate check (agent workflow)")
    parser.add_argument("model", help="model directory path")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "corex", "ascend310p"])
    parser.add_argument("--graph", action="store_true", help="额外验证 CUDA Graph 模式")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--sys-tokens", type=int, default=1200, help="system prompt 目标 token 数")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=8192)
    # 默认 8（对齐 openai_server 的 ms=8 实测结论）：大 ms 放大图桶表与图内采样
    # 峰值显存（bs=512 桶的 argsort 需 ~600MB），空板卡 KV 贴地后必 OOM。
    parser.add_argument("--max-num-seqs", type=int, default=8)
    # KV 池按 (total*util - used) 贴地分配；空板卡（无桌面占用）需调低 util
    # 给图捕获/运行峰值留余量（verify 脚本统一传 0.85）。
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    args = parser.parse_args()

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
    p1_len, p2_len = len(p1), len(p2)
    print(f"prompt1={p1_len} tok, prompt2={p2_len} tok (block_size={args.block_size})\n")

    ok = True

    # ---- EAGER ----
    # 引擎 A：R1 冷 + R2 热（共享 system 前缀）
    warm_outs, warm_stats = _run_engine(
        args.model, args.backend, True, args.block_size, [p1, p2],
        args.max_tokens, args.max_model_len, args.max_num_seqs, args.gpu_mem_util)
    # 引擎 B（全新引擎）：R2 冷
    cold_outs, cold_stats = _run_engine(
        args.model, args.backend, True, args.block_size, [p2],
        args.max_tokens, args.max_model_len, args.max_num_seqs, args.gpu_mem_util)
    _report("EAGER", warm_outs, warm_stats, sys_len, p1_len, p2_len)

    ok &= _assert("eager warm-R2 == cold-R2 (KV 复用无污染)",
                  warm_outs[1][0] == cold_outs[0][0])
    ok &= _assert("eager prefix cache hit (warm engine)",
                  warm_stats["prefix_hits"] >= 1,
                  f"hits={warm_stats['prefix_hits']}")
    ok &= _assert("eager cold engine no hit", cold_stats["prefix_hits"] == 0)
    if warm_stats["cached_blocks"] > 0:
        ttft_speedup = cold_outs[0][1] / warm_outs[1][1]
        print(f"[eager] R2 TTFT speedup (cold/warm) = {ttft_speedup:.1f}x "
              f"({cold_outs[0][1] * 1000:.0f}ms -> {warm_outs[1][1] * 1000:.0f}ms)\n")

    # ---- CUDA GRAPH ----
    if args.graph:
        gwarm_outs, gwarm_stats = _run_engine(
            args.model, args.backend, False, args.block_size, [p1, p2],
            args.max_tokens, args.max_model_len, args.max_num_seqs, args.gpu_mem_util)
        gcold_outs, gcold_stats = _run_engine(
            args.model, args.backend, False, args.block_size, [p2],
            args.max_tokens, args.max_model_len, args.max_num_seqs, args.gpu_mem_util)
        _report("CUDA GRAPH", gwarm_outs, gwarm_stats, sys_len, p1_len, p2_len)

        ok &= _assert("graph warm-R2 == graph cold-R2", gwarm_outs[1][0] == gcold_outs[0][0])
        ok &= _assert("graph prefix cache hit", gwarm_stats["prefix_hits"] >= 1,
                      f"hits={gwarm_stats['prefix_hits']}")
        ok &= _assert("graph R1 == eager R1 (跨后端一致)",
                      gwarm_outs[0][0] == warm_outs[0][0])
        ok &= _assert("graph R2 == eager R2 (跨后端一致)",
                      gwarm_outs[1][0] == warm_outs[1][0])
        if gwarm_stats["cached_blocks"] > 0:
            ttft_speedup = gcold_outs[0][1] / gwarm_outs[1][1]
            print(f"[graph] R2 TTFT speedup (cold/warm) = {ttft_speedup:.1f}x "
                  f"({gcold_outs[0][1] * 1000:.0f}ms -> {gwarm_outs[1][1] * 1000:.0f}ms)\n")

    print("=== RESULT:", "ALL PASS" if ok else "FAIL", "===")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()