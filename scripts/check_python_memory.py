"""Python 引擎的真实设备显存、Prefix、取消与重复加载回归入口。"""

import argparse
import gc
import json
from pathlib import Path
import weakref


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="目标模型本地目录")
    parser.add_argument("--backend", default="auto", choices=("auto", "cuda", "corex", "ascend310p"))
    parser.add_argument("--spec-model", default="", help="可选的本地草稿目录")
    parser.add_argument("--spec-num-tokens", type=int, default=2, help="投机长度")
    parser.add_argument("--eager", action="store_true", help="关闭图捕获")
    parser.add_argument("--no-prefix-cache", action="store_true", help="关闭前缀缓存")
    parser.add_argument("--device", type=int, default=0, help="进程可见设备索引")
    parser.add_argument("--max-model-len", type=int, default=1024, help="最大上下文长度")
    parser.add_argument("--max-num-seqs", type=int, default=2, help="最大活跃序列数")
    parser.add_argument("--max-tokens", type=int, default=16, help="每请求输出上限")
    parser.add_argument("--cycles", type=int, default=2, help="加载、卸载重复次数")
    parser.add_argument("--rounds", type=int, default=3, help="每次加载的并发压力轮数")
    parser.add_argument("--memory-budget-mib", type=int, default=4096, help="本引擎显存预算")
    parser.add_argument("--device-reserve-mib", type=int, default=512, help="为其他模型预留的显存")
    parser.add_argument("--drift-tolerance-mib", type=int, default=16, help="稳定轮次/卸载后的容许漂移")
    parser.add_argument("--runtime-cache-tolerance-mib", type=int, default=32,
                        help="首轮进程级运行库缓存容许量；不适用于模型张量或后续循环")
    parser.add_argument("--output", type=Path, help="保存真实设备测试 JSON 记录")
    args = parser.parse_args()
    if args.cycles < 1 or args.rounds < 2 or args.max_tokens < 1 or args.max_num_seqs < 1:
        parser.error("cycles/max_tokens/max_num_seqs 须为正数，rounds 至少为 2")
    print("Importing runtime ...", flush=True)
    from flashengine import LLM, SamplingParams
    from flashengine.backends import get_backend
    import os
    if args.backend != "auto":
        os.environ["FLASHENGINE_BACKEND"] = args.backend
    backend = get_backend()
    backend.set_device(args.device)
    tolerance = args.drift_tolerance_mib * 2**20
    records = []
    for cycle in range(args.cycles):
        gc.collect()
        backend.empty_cache()
        baseline = backend.memory_stats()["allocated_bytes.all.current"]
        with LLM(args.model, backend=args.backend, device_ids=(args.device,),
                 max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                 max_num_batched_tokens=128, max_num_partial_prefills=1,
                 spec_model=args.spec_model, spec_num_tokens=args.spec_num_tokens,
                 enable_prefix_cache=not args.no_prefix_cache,
                 memory_budget_bytes=args.memory_budget_mib * 2**20,
                 device_memory_reserve_bytes=args.device_reserve_mib * 2**20,
                 enforce_eager=args.eager) as engine:
            watched = [weakref.ref(t) for t in engine.model_runner.model.parameters()]
            watched += [weakref.ref(t) for t in engine.model_runner.model.buffers()]
            if engine.model_runner.spec_enabled:
                watched += [weakref.ref(t) for t in engine.model_runner.draft_model.parameters()]
            if hasattr(engine.model_runner, "kv_cache"):
                watched.append(weakref.ref(engine.model_runner.kv_cache))
            block_size = engine.config.kvcache_block_size
            shared = engine.tokenizer.encode("Explain reliable memory management. " * 100)
            prefix_length = min(block_size + 16, engine.config.max_model_len - args.max_tokens - 2)
            if prefix_length < 1:
                raise ValueError("测试上下文预算不足")
            shared = (shared * (prefix_length // len(shared) + 1))[:prefix_length]
            prompts = [shared + [shared[-1]]] * (args.max_num_seqs + 1)
            sampling = SamplingParams(temperature=0, max_tokens=args.max_tokens, ignore_eos=True)
            allocations = []
            first = None
            for _ in range(args.rounds):
                outputs = engine.generate(prompts, sampling, use_tqdm=False)
                tokens = [o["token_ids"] for o in outputs]
                assert all(len(t) == args.max_tokens for t in tokens), "输出长度异常"
                if first is None:
                    first = tokens
                assert tokens == first, "相同并发负载的重复贪心结果发生漂移"
                assert not engine.scheduler.block_manager.used_block_ids, "完成后 KV 未归还"
                if engine.model_runner.state_runtime:
                    assert not engine.model_runner.state_runtime.slots, "recurrent 槽未归还"
                assert not engine.model_runner.spec_hidden, "投机隐藏状态未释放"
                allocations.append(backend.memory_stats()["allocated_bytes.all.current"])
            assert max(allocations[1:]) - min(allocations[1:]) <= tolerance, "稳定轮次显存持续增长"
            seq = engine.add_request(prompts[0], sampling)
            engine.step()
            assert engine.cancel_request(seq), "分块 prefill 取消失败"
            assert not engine.scheduler.block_manager.used_block_ids, "取消后 KV 未归还"
            if engine.model_runner.state_runtime:
                assert not engine.model_runner.state_runtime.slots, "取消后 recurrent 槽未归还"
            engine.generate([prompts[0]], sampling, use_tqdm=False)
            try:
                engine.add_request(shared, SamplingParams(max_tokens=engine.config.max_model_len))
            except ValueError:
                pass
            else:
                raise AssertionError("超上下文请求未被拒绝")
            try:
                engine.generate([prompts[0], []], sampling, use_tqdm=False)
            except ValueError:
                pass
            else:
                raise AssertionError("无效批请求未被拒绝")
            assert engine.is_finished(), "批提交失败后留下了未完成请求"
            snapshot = engine.runtime_snapshot()
            if not args.no_prefix_cache and prefix_length >= block_size:
                assert snapshot["cache"]["prefix_hit_blocks"] > 0, "未触发真实 Prefix 命中"
            record = {"cycle": cycle, "memory": snapshot["memory"],
                      "graph": snapshot["graph"],
                      "cache": snapshot["cache"], "allocations_bytes": allocations,
                      "execution_path": engine.execution_path(0),
                      "completion_tokens": sum(map(len, first))}
        gc.collect()
        backend.empty_cache()
        after = backend.memory_stats()["allocated_bytes.all.current"]
        assert all(ref() is None for ref in watched), "卸载后仍有模型权重/缓冲或 KV 张量存活"
        record["unload_delta_bytes"] = after - baseline
        allowance = args.runtime_cache_tolerance_mib * 2**20 if cycle == 0 else tolerance
        assert after - baseline <= allowance, f"卸载后显存未回到基线：{after - baseline} 字节"
        records.append(record)
    report = {"status": "passed", "backend": backend.name, "model": args.model,
              "spec_model": args.spec_model, "synthetic_draft_is_quality_evidence": False,
              "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "cycles": records}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
