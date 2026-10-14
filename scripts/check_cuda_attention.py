"""CUDA 注意力回归：同一模型对比 FlashAttention、FlashInfer eager 与图路径。"""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def compare_tokens(base, other, trace=None):
    if len(base) != len(other):
        raise ValueError("对照结果的场景数量不一致")
    checks = []
    for row_a, row_b in zip(base, other):
        if row_a["label"] != row_b["label"] or len(row_a["token_ids"]) != len(row_b["token_ids"]):
            raise ValueError("对照场景或请求数量不一致")
        if row_a["label"] == "sampling":
            continue
        for seq_index, (a, b) in enumerate(zip(row_a["token_ids"], row_b["token_ids"])):
            result = {"label": row_a["label"], "sequence": seq_index, "exact": a == b}
            if a != b:
                pos = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), None)
                result["first_difference"] = pos
                result["bf16_tie"] = False
                if pos is not None and trace is not None:
                    left = row_a["logits_trace"][seq_index][pos]
                    # 只比较首次分歧；此位置前的输入 token 完全相同。
                    row_trace = next(row for row in trace if row["label"] == row_a["label"])
                    right = row_trace["logits_trace"][seq_index][pos]
                    ids = (a[pos], b[pos])
                    # topk 对平票候选的排序不稳定，候选可能排在第三位。
                    if all(token in left["ids"] and token in right["ids"] for token in ids):
                        lv = [left["logits"][left["ids"].index(token)] for token in ids]
                        rv = [right["logits"][right["ids"].index(token)] for token in ids]
                        ulp = 2 ** (math.floor(math.log2(max(map(abs, lv + rv)))) - 7)
                        result.update(baseline_logits=lv, candidate_logits=rv, bf16_ulp=ulp)
                        result["bf16_tie"] = (
                            max(left["logits"]) - min(lv) <= ulp
                            and max(right["logits"]) - min(rv) <= ulp
                            and max(abs(x - y) for x, y in zip(lv, rv)) <= ulp)
            checks.append(result)
    return checks


def kernel_checks():
    import torch
    from flashengine.backends.flashinfer import FlashInferAttention
    from flashengine.utils.context import Context

    torch.manual_seed(7)
    attention = FlashInferAttention()
    device = "cuda"
    dtype = torch.bfloat16
    heads, kv_heads, dim, page = 16, 8, 128, 64
    scale = dim ** -0.5
    tables, lengths, qlens = [[3, 1], [2, 0, 4]], [70, 129], [3, 2]
    k = torch.randn(5, page, kv_heads, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    q = torch.randn(sum(qlens), heads, dim, device=device, dtype=dtype)

    def reference(query, qs):
        out, offset = [], 0
        for table, length, qlen in zip(tables, lengths, qs):
            keys = k[table].reshape(-1, kv_heads, dim)[:length].repeat_interleave(2, dim=1).float()
            vals = v[table].reshape(-1, kv_heads, dim)[:length].repeat_interleave(2, dim=1).float()
            qi = query[offset:offset + qlen].float()
            logits = torch.einsum("qhd,khd->hqk", qi, keys) * scale
            mask = torch.arange(length, device=device)[None, :] <= (
                torch.arange(qlen, device=device)[:, None] + length - qlen)
            logits.masked_fill_(~mask[None], -float("inf"))
            out.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1), vals))
            offset += qlen
        return torch.cat(out)

    context = Context(is_prefill=True)
    context.block_tables = torch.tensor([[3, 1, -1], [2, 0, 4]], dtype=torch.int32, device=device)
    context.cu_seqlens_q = torch.tensor([0, 3, 5], dtype=torch.int32, device=device)
    context.cu_seqlens_k = torch.tensor([0, 70, 199], dtype=torch.int32, device=device)
    prefill = attention.prefill(q, k, v, context, scale).float()
    expected = reference(q, qlens)
    torch.testing.assert_close(prefill, expected, atol=0.01, rtol=0.03)
    decode_q = q[:2]
    context = Context(block_tables=context.block_tables,
                      context_lens=torch.tensor(lengths, dtype=torch.int32, device=device))
    decoded = attention.decode(decode_q, k, v, context, scale).float()
    expected_decode = reference(decode_q, [1, 1])
    torch.testing.assert_close(decoded, expected_decode, atol=0.01, rtol=0.03)
    # 冷 prefill 使用 ragged K/V，同样验证变长批与底部对齐的因果掩码。
    packed_k = torch.cat([k[t].reshape(-1, kv_heads, dim)[:n] for t, n in zip(tables, lengths)])
    packed_v = torch.cat([v[t].reshape(-1, kv_heads, dim)[:n] for t, n in zip(tables, lengths)])
    context = Context(is_prefill=True,
                      cu_seqlens_q=torch.tensor([0, 3, 5], dtype=torch.int32, device=device),
                      cu_seqlens_k=torch.tensor([0, 70, 199], dtype=torch.int32, device=device))
    ragged = attention.prefill(q, packed_k, packed_v, context, scale).float()
    torch.testing.assert_close(ragged, expected, atol=0.01, rtol=0.03)
    result = {"paged_prefill_max_abs": (prefill - expected).abs().max().item(),
              "decode_max_abs": (decoded - expected_decode).abs().max().item(),
              "ragged_prefill_max_abs": (ragged - expected).abs().max().item()}
    attention.close()
    return result


def worker(args):
    import torch
    from flashengine import LLM, SamplingParams

    torch.manual_seed(0)
    kernels = kernel_checks() if args.attention == "flashinfer" else None
    torch.cuda.empty_cache()
    rows = []
    with LLM(args.model, backend="cuda", enforce_eager=args.eager,
             max_model_len=1024, max_num_seqs=4, max_num_batched_tokens=128,
             num_kvcache_blocks=32, gpu_memory_utilization=0.8,
             kvcache_block_size=args.block_size) as llm:
        probe = None
        if args.eager or args.attention == "flashinfer" or os.getenv("FLASHENGINE_GRAPH_SAMPLE") == "0":
            from scripts.cuda_logits_probe import LogitsProbe
            probe = LogitsProbe(llm)
        questions = ["What is 2 + 2? Answer briefly.", "用一句话介绍北京。", "Name three primary colors.",
                     "Write a Python function that adds two integers."]
        shared = "Reference: this is a concise question answering test. " * 48
        sp = SamplingParams(temperature=0, max_tokens=24)

        def run(label, prompts):
            ids = [llm.tokenizer.encode(llm.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
                for messages in prompts]
            started = time.perf_counter()
            output = llm.generate(ids, sp, use_tqdm=False)
            rows.append({"label": label, "prompt_tokens": list(map(len, ids)),
                         "token_ids": [row["token_ids"] for row in output],
                         "text": [row["text"] for row in output],
                         "wall_s": time.perf_counter() - started,
                         "logits_trace": probe.drain() if probe else None})

        for n in (1, 3, 4):
            run(f"batch{n}", [[{"role": "user", "content": q}] for q in questions[:n]])
        prompts = [[{"role": "system", "content": shared}, {"role": "user", "content": q}]
                   for q in questions[:3]]
        run("prefix_cold", prompts)
        run("prefix_warm", prompts)
        # 跨物理页边界且随机采样，核对图内 logits 仍由正常 sampler 消费。
        sp.temperature = 0.7
        sp.top_p = 0.9
        run("sampling", [[{"role": "user", "content": "Explain why the sky is blue."}]])
        snapshot = llm.runtime_snapshot()
        if probe:
            probe.close()
    Path(args.output).write_text(json.dumps(
        {"attention": args.attention, "eager": args.eager, "block_size": args.block_size,
         "kernels": kernels, "rows": rows, "runtime": snapshot},
        ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="目标模型本地目录")
    parser.add_argument("--output", default="bench_results/flashinfer-check", help="测试结果目录或 worker JSON")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--attention", choices=("flash_attn", "flashinfer"), default="flashinfer")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--compare-only", action="store_true", help="只检查已有结果，不重新加载模型")
    parser.add_argument("--allow-bf16-ties", action="store_true", help="显式允许两侧候选都有 logits 证据的单 ULP 翻转")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    cases = [("flash_attn_eager", "flash_attn", True, 256),
             ("flash_attn_graph", "flash_attn", False, 256),
             ("flash_attn_graph_no_sample", "flash_attn", False, 256),
             ("flashinfer_eager", "flashinfer", True, 256),
             ("flashinfer_graph", "flashinfer", False, 256),
             ("flashinfer_eager64", "flashinfer", True, 64),
             ("flashinfer_graph64", "flashinfer", False, 64)]
    results = {}
    for name, attention, eager, block_size in cases:
        command = [sys.executable, "-m", "scripts.check_cuda_attention", args.model,
                   "--worker", "--attention", attention, "--block-size", str(block_size),
                   "--output", str(destination / f"{name}.json")]
        if eager:
            command.append("--eager")
        env = {**os.environ, "FLASHENGINE_CUDA_ATTENTION": attention,
               "FLASHENGINE_GRAPH_SAMPLE": "0" if name == "flash_attn_graph_no_sample" else "1"}
        if not args.compare_only:
            print(f"运行 {name}", flush=True)
            with (destination / f"{name}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[name] = json.loads((destination / f"{name}.json").read_text(encoding="utf-8"))
    comparisons = {}
    graph_comparisons = {}
    baseline = results["flash_attn_eager"]["rows"]
    # 图内/图外采样的贪心输出须一致，之后才可用图外采样的 logits 审查平票。
    sampler_agreement = compare_tokens(results["flash_attn_graph_no_sample"]["rows"],
                                       results["flash_attn_graph"]["rows"])
    for name, result in results.items():
        eager = (baseline if result["attention"] == "flash_attn" else
                 results["flashinfer_eager64" if result["block_size"] == 64 else "flashinfer_eager"]["rows"])
        trace = result["rows"]
        if name == "flash_attn_graph":
            trace = (results["flash_attn_graph_no_sample"]["rows"]
                     if all(check["exact"] for check in sampler_agreement) else None)
        checks = compare_tokens(baseline, result["rows"], trace)
        comparisons[name] = checks
        print(f"{name}: 严格一致 {sum(check['exact'] for check in checks)}/{len(checks)}，"
              f"已证实单 ULP 翻转 {sum(check.get('bf16_tie', False) for check in checks)}", flush=True)
        if "graph" in name:
            graph_checks = compare_tokens(eager, result["rows"], trace)
            graph_comparisons[name] = graph_checks
            assert result["runtime"]["graph"]["capture_success"] == 3, result["runtime"]
            assert result["runtime"]["graph"]["replay"] > 0, result["runtime"]
            assert result["runtime"]["graph"]["fallback"] == 0, result["runtime"]
        assert result["runtime"]["cache"]["prefix_hits"] > 0, result["runtime"]
    report = {"model": args.model, "allow_bf16_ties": args.allow_bf16_ties,
              "comparisons": comparisons, "graph_vs_eager": graph_comparisons,
              "flash_attn_graph_sampling_agreement": sampler_agreement}
    (destination / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all(check["exact"] for check in sampler_agreement):
        raise SystemExit("FlashAttention 图内/图外采样的贪心输出不一致，请检查 sampler 或输出缓冲。")
    if not all(check["exact"] or (args.allow_bf16_ties and check.get("bf16_tie", False))
               for checks in [*comparisons.values(), *graph_comparisons.values()] for check in checks):
        raise SystemExit("贪心 token 存在差异，请查看 JSON 并核对数值漂移。")
    print("PASS: 注意力数值、贪心输出（精度豁免见 summary）、prefix/chunked prefill、Graph padding 与采样路径。")


if __name__ == "__main__":
    main()
