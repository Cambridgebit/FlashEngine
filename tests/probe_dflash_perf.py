"""DFlash 定点性能探针：长输出、逐 token 工件和可选设备剖析（不替代 benchmark）。"""

import argparse
from functools import wraps
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch


def gpu_status():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True, timeout=5)
        return output.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--draft", default="")
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--gdn", choices=("torch", "triton", "auto"), default="auto")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--profile", action="store_true", help="另跑短请求剖析；不计入吞吐")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sweep", help="串行扫描逗号分隔 k；output 此时为目录")
    parser.add_argument("--verify-logits", action="store_true", help="双跑目标验证诊断 BF16 翻转；此模式的时间不作性能结果")
    args = parser.parse_args()
    if args.sweep:
        for width in args.sweep.split(","):
            command = [sys.executable, "-m", "tests.probe_dflash_perf", args.model,
                "--draft", args.draft, "--k", width, "--gdn", args.gdn,
                "--num-prompts", str(args.num_prompts), "--repeats", str(args.repeats),
                "--max-tokens", str(args.max_tokens), "--output", str(Path(args.output) / f"k{width}.json")]
            command.extend(flag for flag, active in (("--eager", args.eager), ("--profile", args.profile)) if active)
            subprocess.run(command, check=True)
        return
    os.environ["FLASHENGINE_CUDA_GDN"] = args.gdn
    import torch
    from flashengine import LLM, SamplingParams
    from flashengine.engine.sequence import Sequence
    from scripts import benchmark
    from scripts.bench_utils import load_prompts
    benchmark.Sequence = Sequence
    questions = load_prompts("bench_data/question.jsonl", "coding", args.num_prompts, 0)
    result = {"settings": vars(args), "runs": []}
    if args.verify_logits and (not args.eager or not args.draft):
        parser.error("--verify-logits 需要 --eager 和 --draft")
    with LLM(args.model, backend="cuda", spec_model=args.draft, spec_num_tokens=args.k,
             enforce_eager=args.eager, max_model_len=1024, max_num_seqs=1,
             max_num_batched_tokens=128, num_kvcache_blocks=8,
             enable_prefix_cache=False, gpu_memory_utilization=0.85) as llm:
        result["actual_k"] = llm.config.spec_num_tokens if args.draft else 0
        if args.verify_logits:
            runner = llm.model_runner
            backend = runner.backend
            original = runner._sg_verify
            original_sample = runner._sg_sample
            result["verify_differences"] = []
            result["verify_rounds"] = 0
            result["verify_max_logit_error"] = 0.0
            def compare_verify(batch, use_cuda_graph=True):
                runtime = runner.state_runtime
                index = runtime.verify_carry[:batch].long()
                pools = (runtime.conv_pool, runtime.ssm_pool)
                before = [pool.index_select(1, index) for pool in pools]
                capture = []
                def sample_capture(logits, *a):
                    capture.append(logits.detach().clone())
                    return original_sample(logits, *a)
                previous = (backend.supports_fused_gdn, backend.supports_fused_qwen35_norm)
                try:
                    with patch.object(runner, "_sg_sample", sample_capture):
                        backend.supports_fused_gdn = backend.supports_fused_qwen35_norm = True
                        original(batch, False)
                        fast = capture[-1]
                        for pool, values in zip(pools, before):
                            pool.index_copy_(1, index, values)
                        backend.supports_fused_gdn = backend.supports_fused_qwen35_norm = False
                        original(batch, False)
                        reference = capture[-1]
                    error = float((fast.float() - reference.float()).abs().amax())
                    result["verify_max_logit_error"] = max(result["verify_max_logit_error"], error)
                    result["verify_rounds"] += 1
                    width = llm.config.spec_num_tokens
                    rows = width + 1
                    ref_ids = reference.argmax(-1).view(batch, rows).tolist()
                    fast_ids = fast.argmax(-1).view(batch, rows).tolist()
                    drafts = runner._sg["drafted_ids"][:width, :batch].t().tolist()
                    for seq in range(batch):
                        accepted = 0
                        for draft, token in zip(drafts[seq], ref_ids[seq]):
                            if draft != token:
                                break
                            accepted += 1
                        for row in range(accepted + 1):
                            if ref_ids[seq][row] != fast_ids[seq][row]:
                                position = seq * rows + row
                                record = {"round": result["verify_rounds"], "row": row,
                                          "accepted": accepted, "max_error": error}
                                for label, values in (("fast", fast[position]), ("reference", reference[position])):
                                    top_values, ids = values.topk(4)
                                    record[label] = {"ids": ids.tolist(), "logits": top_values.float().tolist()}
                                result["verify_differences"].append(record)
                finally:
                    backend.supports_fused_gdn, backend.supports_fused_qwen35_norm = previous
            runner._sg_verify = compare_verify
        prompts = [llm.tokenizer.encode(llm.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)) for question in questions]
        sample = SamplingParams(temperature=0, max_tokens=args.max_tokens)
        for _ in range(2):
            llm.generate([prompts[0]], sample, use_tqdm=False)
        for rep in range(args.repeats):
            llm.reset_measurement_stats()
            rows = []
            telemetry_before = gpu_status()
            started = time.perf_counter()
            for prompt in prompts:
                tokens, first, wall = benchmark.run_seqs(llm, [Sequence(prompt, sample)])
                rows.append({"tokens": tokens[0], "ttft_s": first[0], "wall_s": wall})
            wall = time.perf_counter() - started
            count = sum(len(row["tokens"]) for row in rows)
            decode_s = sum(row["wall_s"] - row["ttft_s"] for row in rows)
            snapshot = llm.runtime_snapshot()
            assert snapshot["recurrent"]["used_slots"] == 0
            result["runs"].append({"rows": rows, "wall_s": wall, "tokens": count,
                "tps": count / wall, "decode_tps": sum(max(len(row["tokens"]) - 1, 0) for row in rows) / decode_s,
                "runtime": snapshot, "gpu_before": telemetry_before, "gpu_after": gpu_status()})
            print(f"rep={rep} tokens={count} tps={count / wall:.2f} decode_tps={result['runs'][-1]['decode_tps']:.2f}", flush=True)
        if args.profile:
            from contextlib import ExitStack
            runner = llm.model_runner
            result["profile_runtime_before"] = llm.runtime_snapshot()
            def scoped(original, label):
                @wraps(original)
                def call(*a, **kw):
                    with torch.profiler.record_function(label):
                        return original(*a, **kw)
                return call
            with ExitStack() as stack:
                for method, label in (("_df_append_context_graph", "df.context"),
                                      ("_df_draft_block_graph", "df.draft"),
                                      ("_sg_verify", "df.verify"),
                                      ("_df_readback_commit", "df.commit")):
                    if args.draft:
                        stack.enter_context(patch.object(runner, method, scoped(getattr(runner, method), label)))
                for index, layer in enumerate(runner.model.model.gdn_layers):
                    stack.enter_context(patch.object(layer, "_forward_rows", scoped(layer._forward_rows, f"gdn.rows.{index}")))
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA], acc_events=True) as profiler:
                    llm.generate([prompts[0]], SamplingParams(temperature=0, max_tokens=16), use_tqdm=False)
                    torch.cuda.synchronize()
            events = profiler.key_averages()
            result["profile_runtime_after"] = llm.runtime_snapshot()
            cuda_events = [event for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA]
            result["profile_cuda_events"] = len(cuda_events)
            result["profile_device_us"] = sum(event.device_time_total for event in cuda_events)
            result["profile"] = [{"name": event.key, "count": event.count,
                "self_device_us": event.self_device_time_total, "device_us": event.device_time_total}
                for event in sorted(events, key=lambda e: e.self_device_time_total, reverse=True)[:40]]
            result["profile_scopes"] = [{"name": event.key, "count": event.count,
                "device_us": event.device_time_total} for event in events
                if event.key.startswith(("df.", "gdn."))]
            print(events.table(sort_by="self_cuda_time_total", row_limit=15))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
