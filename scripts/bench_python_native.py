"""Python 引擎的冻结 token 基准/严格 trace worker，与原生 benchmark 保持同一测量边界。"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import time

from flashengine.deployment import read_json, sha256, source_state, write_json
from scripts.native_bench_common import summary, validate_workload, checked_greedy_round


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("bundle")
    p.add_argument("workload")
    p.add_argument("output")
    p.add_argument("--draft")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--graph", action="store_true")
    p.add_argument("--trace-rounds", action="store_true")
    p.add_argument("--attention", choices=("flash_attn", "flashinfer"), default="flash_attn")
    args = p.parse_args()
    if Path(args.output).exists():
        raise ValueError("拒绝覆盖 Python 结果")
    workload = read_json(args.workload)
    validate_workload(workload)
    cfg = workload["config"]
    if not 1 <= args.batch_size <= cfg["max_num_seqs"] or args.warmup < 0 or args.repeats < 1:
        raise ValueError("非法基准参数")
    bundle = Path(args.bundle)
    if sha256(bundle / "bundle.json") != workload["bundle_sha256"]:
        raise ValueError("workload 与 bundle 不一致")
    for name, item in read_json(bundle / "bundle.json")["files"].items():
        for kind, directory in (("target", args.model), ("draft", args.draft)):
            if directory and name.startswith(f"weights/{kind}/") and sha256(Path(directory) / Path(name).name) != item["sha256"]:
                raise ValueError("Python 权重不匹配：" + name)
    bc = read_json(bundle / "engine/config.json")
    if bool(args.draft) != bool(bc["draft"]):
        raise ValueError("草稿配置与 bundle 不一致")
    os.environ["FLASHENGINE_BACKEND"] = "cuda"
    os.environ["FLASHENGINE_CUDA_ATTENTION"] = args.attention
    from flashengine import LLM, SamplingParams
    from flashengine.engine.sequence import Sequence
    import torch
    kwargs = dict(cfg, backend="cuda", enforce_eager=not args.graph)
    kwargs.setdefault("enable_prefix_cache",True)
    if args.draft:
        k = bc["draft"]["num_tokens"]
        if k in (3, 4):
            raise ValueError("当前 Python 历史 sentinel 不能严格冻结 k=3/4")
        kwargs.update(spec_model=args.draft, spec_num_tokens=k)
        if bc["draft"].get("spec_method")=="eagle3":kwargs["spec_aux_layers"]=tuple(bc["draft"]["aux_layers"])
    with LLM(args.model, **kwargs) as llm:
        if llm.model_runner.dtype != getattr(torch, cfg["dtype"]) or llm.config.model_config.dtype != llm.model_runner.dtype:
            raise RuntimeError("模型和 planner 的实际 dtype 不一致")
        for key, value in cfg.items():
            if key != "dtype" and getattr(llm.config, key) != value:
                raise RuntimeError(f"Python resolved config 与 workload 不一致：{key}")
        trace = {}
        if args.trace_rounds and args.draft:
            runner=llm.model_runner
            method="_readback_spec_commit" if bc["draft"].get("spec_method")=="eagle3" else "_df_readback_commit"
            original=getattr(runner,method)
            def traced(seqs):
                n, k = len(seqs), bc["draft"]["num_tokens"]
                g = runner._sg
                # EAGLE refill 会覆写 drafted_ids[0]；d_pad 保留接受算子的真实输入。
                drafts = (g["d_pad"][:n, :k] if bc["draft"].get("spec_method")=="eagle3"
                          else g["drafted_ids"][:k, :n].t()).tolist()
                verified = g["v_token_ids"][:n * (k + 1)].view(n, k + 1).tolist()
                committed = original(seqs)
                for seq, d, v, c in zip(seqs, drafts, verified, committed):
                    trace.setdefault(seq.seq_id, []).append(checked_greedy_round(d,v,c,len(seq)-1))
                return committed
            setattr(runner,method,traced)

        def group(requests, repeat):
            seqs, first, last = [], {}, {}
            start = time.perf_counter()
            for request in requests:
                seq = Sequence(request["prompt_token_ids"], SamplingParams(temperature=0, max_tokens=request["max_tokens"], ignore_eos=request["ignore_eos"]))
                seqs.append(seq)
                llm.scheduler.add(seq)
            while not llm.is_finished():
                llm.step()
                now = time.perf_counter() - start
                for seq in seqs:
                    if seq.num_completion_tokens and seq.seq_id not in first:
                        first[seq.seq_id] = now
                    if seq.is_finished and seq.seq_id not in last:
                        last[seq.seq_id] = now
            wall = time.perf_counter() - start
            if llm.runtime_snapshot()["cache"]["used_blocks"]:
                raise RuntimeError("组结束后 KV 未释放")
            rows = []
            for request, seq in zip(requests, seqs):
                tokens = seq.completion_token_ids
                reason = "stop" if not request["ignore_eos"] and tokens[-1] == llm.config.eos else "length"
                row = dict(input_id=request["id"], repeat=repeat, prompt_tokens=len(request["prompt_token_ids"]), token_ids=tokens, finish_reason=reason, ttft_seconds=first[seq.seq_id], e2e_seconds=last[seq.seq_id], group_wall_seconds=wall)
                if args.trace_rounds:
                    row["rounds"] = trace.pop(seq.seq_id, [])
                rows.append(row)
            return rows, wall

        for _ in range(args.warmup):
            group(workload["requests"][:args.batch_size], -1)
        trace.clear()
        before = llm.runtime_snapshot()
        start, begin = time.perf_counter(), time.time()
        rows, group_wall = [], 0.
        for rep in range(args.repeats):
            for offset in range(0, len(workload["requests"]), args.batch_size):
                part, elapsed = group(workload["requests"][offset:offset + args.batch_size], rep)
                rows.extend(part)
                group_wall += elapsed
        wall, end = time.perf_counter() - start, time.time()
        after = llm.runtime_snapshot()
        result = dict(implementation="python", protocol="pretokenized-grouped-engine-v1", workload_sha256=sha256(args.workload), resolved_config=dict(cfg, enforce_eager=not args.graph, backend="cuda", tensor_parallel_size=1, enable_prefix_cache=kwargs["enable_prefix_cache"], spec_num_tokens=bc["draft"]["num_tokens"] if args.draft else 0), batch_size=args.batch_size, warmup=args.warmup, repeats=args.repeats, trace_rounds=args.trace_rounds, measurement_unix_start=begin, measurement_unix_end=end, wall_seconds=wall, group_wall_seconds_sum=group_wall, runtime_before=before, runtime_after=after, rows=rows, environment=dict(torch=torch.__version__, transformers=importlib.metadata.version("transformers"), attention=args.attention, device=torch.cuda.get_device_name()), oracle_source=source_state())
        result.update(summary(rows, wall))
    write_json(args.output, result)
    print({key: result[key] for key in ("output_tps", "ttft_p50_ms", "output_tokens")})


if __name__ == "__main__":
    main()
