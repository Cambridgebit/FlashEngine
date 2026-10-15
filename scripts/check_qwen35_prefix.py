"""Qwen3.5 前缀缓存验收：普通/投机、冷/热位等、分块、并发与快照淘汰。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def host_check():
    """用小状态验证末块门控、哈希失效、复制与淘汰，不加载真实模型。"""
    import torch
    from types import SimpleNamespace
    from flashengine.engine.block_manager import BlockManager
    from flashengine.engine.recurrent_state import RecurrentStateRuntime
    from flashengine.engine.sequence import Sequence
    from flashengine.sampling_params import SamplingParams
    from flashengine.utils.context import Context

    layer = SimpleNamespace()
    backbone = SimpleNamespace(gdn_layers=[layer], get_gdn_state_spec=lambda: {
        "conv_shape": (2, 2), "conv_dtype": torch.float32,
        "ssm_shape": (1, 2, 2), "ssm_dtype": torch.float32})
    cfg = SimpleNamespace(backend_impl=SimpleNamespace(device_name="cpu"),
                          max_num_seqs=2, spec_num_tokens=2, spec_model="",
                          enable_prefix_cache=True, recurrent_prefix_cache_bytes=32,
                          kvcache_block_size=4)
    runtime = RecurrentStateRuntime(SimpleNamespace(model=backbone), cfg)
    bm = BlockManager(12, 4)
    bm.prefix_state_validator = runtime.has_prefix
    previous = Sequence.block_size
    Sequence.block_size = 4
    try:
        seq = Sequence(list(range(10)), SamplingParams())
        bm.allocate(seq, bm.can_allocate(seq))
        seq.num_scheduled_tokens = 10
        ctx = Context()
        runtime.bind_prefill(ctx, [seq])
        assert ctx.gdn_prefix_checkpoints == ((8,),), ctx
        runtime.capture_prefix_layer(0, 0, 8, torch.full((2, 2), 8.),
                                     torch.full((1, 2, 2), 80.))
        runtime.finish_prefill()
        bm.hash_blocks(seq)
        runtime.drop(seq.seq_id)
        bm.deallocate(seq)
        warm = Sequence(list(range(10)), SamplingParams())
        assert bm.can_allocate(warm) == 2, "无需首块状态，只需命中末块状态"
        bm.allocate(warm, 2)
        warm.num_scheduled_tokens = 2
        runtime.bind_prefill(Context(), [warm])
        slot = runtime.slots[warm.seq_id]
        assert torch.all(runtime.conv_pool[:, slot] == 8)
        assert torch.all(runtime.ssm_pool[:, slot] == 80)
        runtime.conv_pool[:, slot].zero_()
        assert torch.all(next(iter(runtime.prefix_snapshots.values()))[1] == 8)
        runtime.drop(warm.seq_id)
        bm.deallocate(warm)
        assert bm.can_allocate(Sequence([99, *range(1, 10)], SamplingParams())) == 0
        key = next(iter(runtime.prefix_snapshots))
        saved = runtime.prefix_snapshots.pop(key)
        runtime.prefix_bytes = 0
        assert bm.can_allocate(Sequence(list(range(10)), SamplingParams())) == 0
        runtime.prefix_snapshots[key] = saved
        runtime.prefix_bytes = runtime.prefix_entry_bytes
        runtime.prefix_pending = {(0, 4): (11, 123, saved[1], saved[2], {0})}
        runtime.finish_prefill()
        assert runtime.prefix_stats["evicted"] == 1
        assert runtime.prefix_bytes == 32
        # KV 哈希已经被另一组 token 覆盖时，旧状态不能误命中。
        assert not runtime.has_prefix(11, 124)
        runtime.clear()
        assert not runtime.slots
    finally:
        Sequence.block_size = previous
    print("主机契约通过：末块门控、状态复制、哈希失效、预算与淘汰", flush=True)


def input_prompts(tokenizer, block_size=256, repeats=40):
    fixed = "Reference facts: alpha=1, beta=2. Answer concisely. " * repeats
    def render(role, question):
        messages = ([{"role": "system", "content": fixed},
                     {"role": "user", "content": question}]
                    if role == "system" else
                    [{"role": "user", "content": fixed + "\nQuestion: " + question}])
        return tokenizer.encode(tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    rows = [render("system", "What is alpha?"), render("system", "What is beta?"),
            render("system", "What is alpha?"), render("user", "What is alpha?"),
            render("user", "What is beta?"), render("user", "What is alpha?")]
    short = tokenizer.encode(tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False))
    return rows + [short, short, rows[0][:block_size], rows[0][:block_size]]


def run_requests(llm, prompts, maximum, ignore_eos=False):
    from flashengine.engine.sequence import Sequence
    from flashengine import SamplingParams
    seqs = [Sequence(prompt, SamplingParams(temperature=0, max_tokens=maximum,
                                           ignore_eos=ignore_eos))
            for prompt in prompts]
    for seq in seqs:
        llm.scheduler.add(seq)
    start = time.perf_counter()
    first = {}
    while not llm.is_finished():
        llm.step()
        for seq in seqs:
            if seq.num_completion_tokens and seq.seq_id not in first:
                first[seq.seq_id] = time.perf_counter() - start
    wall = time.perf_counter() - start
    assert llm.runtime_snapshot()["recurrent"]["used_slots"] == 0
    return [{"tokens": seq.completion_token_ids, "prompt_tokens": seq.num_prompt_tokens,
             "hit_blocks": seq.prefix_cache_hit_blocks, "query_tokens": seq.prefill_query_tokens,
             "ttft_ms": first[seq.seq_id] * 1000, "wall_s": wall} for seq in seqs]


def worker(args):
    from flashengine import LLM
    speculative = args.case.startswith("spec") or args.case == "fi_spec"
    with LLM(args.model, backend=args.backend, spec_model=args.draft if speculative else "",
             spec_num_tokens=args.spec_num_tokens,
             enforce_eager=args.case.endswith("eager") or args.case == "fi_spec",
             enable_prefix_cache=args.cache == "on", max_model_len=args.max_model_len,
             max_num_batched_tokens=args.chunk_size, max_num_seqs=args.seqs,
             kvcache_block_size=args.block_size,
             num_kvcache_blocks=args.kv_blocks, gpu_memory_utilization=0.9) as llm:
        assert llm.config.enable_prefix_cache == (args.cache == "on")
        prompts = input_prompts(llm.tokenizer, llm.config.kvcache_block_size, args.prefix_repeats)
        assert max(map(len, prompts)) + args.max_tokens <= llm.config.max_model_len
        rows = [run_requests(llm, [prompt], args.max_tokens)[0] for prompt in prompts]
        if args.cache == "on":
            assert rows[1]["hit_blocks"] >= args.min_hit_blocks and rows[2]["hit_blocks"] >= args.min_hit_blocks, rows
            assert rows[4]["hit_blocks"] >= args.min_hit_blocks and rows[5]["hit_blocks"] >= args.min_hit_blocks, rows
            assert all(not row["hit_blocks"] for row in rows[6:]), rows
            assert llm.runtime_snapshot()["recurrent"]["prefix_cache"]["restored"] >= 4
            if speculative and not llm.config.use_paged_draft:
                assert llm.runtime_snapshot()["draft_prefix"]["restored"] >= 4
        else:
            assert not any(row["hit_blocks"] for row in rows)
        assert rows[0]["tokens"] == rows[2]["tokens"]
        assert rows[3]["tokens"] == rows[5]["tokens"]
        batch = run_requests(llm, prompts[:min(args.seqs, 3)], args.max_tokens)
        extra = []
        if args.cache == "on":
            runtime = llm.model_runner.state_runtime
            runtime.prefix_limit = runtime.prefix_entry_bytes
            runtime.prefix_snapshots.clear()
            runtime.prefix_bytes = 0
            extra.append(run_requests(llm, [prompts[0]], args.max_tokens)[0])
            assert extra[-1]["hit_blocks"] == 0
            extra.append(run_requests(llm, [prompts[0]], args.max_tokens)[0])
            assert extra[-1]["hit_blocks"] >= args.min_hit_blocks
            assert all(row["tokens"] == rows[0]["tokens"] for row in extra)
            assert runtime.prefix_bytes <= runtime.prefix_entry_bytes
            # 整段 prefill 只暂存预算内的末块；用另一条前缀触发真实 LRU 淘汰。
            alternate = run_requests(llm, [prompts[3]], args.max_tokens)[0]
            assert alternate["tokens"] == rows[3]["tokens"]
            assert runtime.prefix_stats["evicted"] > 0
            if speculative and not llm.config.use_paged_draft:
                runner = llm.model_runner
                run_requests(llm, [prompts[0]], args.max_tokens)
                from flashengine.engine.sequence import Sequence
                from flashengine import SamplingParams
                probe = Sequence(prompts[0], SamplingParams(temperature=0))
                assert llm.scheduler.block_manager.can_allocate(probe) >= args.min_hit_blocks
                runner._df_prefix_snapshots.clear()
                runner._df_prefix_snapshot_bytes = 0
                missing = run_requests(llm, [prompts[0]], args.max_tokens)[0]
                assert missing["hit_blocks"] == 0 and missing["tokens"] == rows[0]["tokens"]
            # 模拟 HTTP 取消，使用服务端的同一清理入口。
            from scripts import openai_server
            from flashengine.engine.sequence import Sequence, SequenceStatus
            from flashengine import SamplingParams
            seq = Sequence(prompts[0], SamplingParams(temperature=0, max_tokens=128))
            llm.scheduler.add(seq)
            llm.step()
            openai_server.engine = llm
            openai_server.SequenceStatus = SequenceStatus
            openai_server._abort(seq)
            assert llm.runtime_snapshot()["recurrent"]["used_slots"] == 0
            metrics = openai_server._metrics_values()
            assert metrics["flashengine_recurrent_prefix_restored_total"] > 0
            assert metrics["flashengine_recurrent_prefix_snapshot_bytes"] == runtime.prefix_bytes
            if args.preempt:
                from flashengine.engine.block_manager import BlockManager
                # 仅比单请求占块多一块，短窗口模型也能实际触发抢占。
                block_size = llm.config.kvcache_block_size
                single_blocks = (len(prompts[0]) + 8 + block_size - 1) // block_size
                bm = BlockManager(single_blocks + 1, block_size)
                bm.cache_hit_validator = llm.model_runner.prefix_cache_validator()
                bm.prefix_state_validator = runtime.has_prefix
                llm.scheduler.block_manager = bm
                before = llm.scheduler.stats["preemptions"]
                restarted = run_requests(llm, [prompts[0], prompts[3]], 8, ignore_eos=True)
                assert llm.scheduler.stats["preemptions"] > before
                for restarted_row, reference in zip(restarted, (rows[0], rows[3])):
                    length = min(8, len(reference["tokens"]))
                    assert restarted_row["tokens"][:length] == reference["tokens"][:length]
        snapshot = llm.runtime_snapshot()
        if args.case.endswith("graph"):
            assert snapshot["graph"]["capture_success"] > 0
            assert snapshot["graph"]["replay"] > 0
    return {"case": args.case, "cache": args.cache, "rows": rows, "batch": batch,
            "eviction": extra, "runtime": snapshot}


def http_check(args):
    """临时启动真实服务，验证非流式/流式请求共用前缀与统计。"""
    import signal
    import socket
    from urllib.request import Request, urlopen
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "scripts.openai_server", args.model,
               "--backend", args.backend, "--host", "127.0.0.1", "--port", str(port),
               "--max-model-len", "1024", "--max-num-seqs", "1", "--no-think",
               "--max-num-batched-tokens", "192", "--gpu-mem-util", "0.95",
               "--stats-interval", "0"]
    if args.draft:
        command.extend(["--spec-model", args.draft])
    root = f"http://127.0.0.1:{port}"
    with (output / "http.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 180
            while True:
                if process.poll() is not None:
                    raise RuntimeError("临时 server 提前退出，请查看 http.log")
                try:
                    with urlopen(root + "/health", timeout=1):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("server 初始化超时")
                    time.sleep(0.25)
            body = {"model": Path(args.model).name, "max_tokens": 16, "temperature": 0,
                    "messages": [
                        {"role": "system", "content":
                         "Reference facts: alpha=1, beta=2. Answer concisely. " * 40},
                        {"role": "user", "content": "What is alpha?"}]}
            def request(stream):
                payload = {**body, "stream": stream,
                           "stream_options": {"include_usage": True}}
                return urlopen(Request(root + "/v1/chat/completions",
                                       data=json.dumps(payload).encode(),
                                       headers={"Content-Type": "application/json"}), timeout=120)
            with request(False) as response:
                cold = json.load(response)
            with urlopen(root + "/v1/stats") as response:
                before = json.load(response)["metrics"]
            text, usage = "", None
            with request(True) as response:
                assert response.headers.get_content_type() == "text/event-stream"
                for line in response:
                    if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                        continue
                    chunk = json.loads(line[6:])
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        text += choice.get("delta", {}).get("content", "")
            with urlopen(root + "/v1/stats") as response:
                stats = json.load(response)
            after = stats["metrics"]
            assert text == cold["choices"][0]["message"]["content"]
            assert usage == cold["usage"]
            hits = after["flashengine_prefix_hit_blocks_total"] - before["flashengine_prefix_hit_blocks_total"]
            assert hits >= 2
            assert after["flashengine_recurrent_prefix_restored_total"] > before["flashengine_recurrent_prefix_restored_total"]
            assert after["flashengine_active_requests"] == 0
            (output / "http.json").write_text(json.dumps(
                {"cold": cold, "hot_text": text, "hot_usage": usage, "hit_blocks": hits,
                 "stats": stats}, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"HTTP 严格一致，stream=false→true 命中 {hits} 块，主机统计正常", flush=True)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", help="本地 Qwen3.5 模型目录")
    parser.add_argument("--draft", default="", help="本地 DFlash 模型目录")
    parser.add_argument("--backend", default="cuda", choices=["cuda", "corex", "ascend310p"])
    parser.add_argument("--cases", default="base_eager,base_graph,spec_eager,spec_graph")
    parser.add_argument("--output", default="bench_results/qwen35-check/prefix")
    parser.add_argument("--seqs", type=int, default=4)
    parser.add_argument("--kv-blocks", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=192)
    parser.add_argument("--spec-num-tokens", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024, help="测试上下文上限；tiny 检查点可用 512")
    parser.add_argument("--prefix-repeats", type=int, default=40, help="共享前缀重复次数；短窗口检查点可用 20")
    parser.add_argument("--min-hit-blocks", type=int, default=2, help="热请求必须命中的最少块数")
    parser.add_argument("--host-only", action="store_true", help="只执行小状态主机契约")
    parser.add_argument("--preempt", action="store_true", help="额外用 4 块 KV 触发双请求真实抢占")
    parser.add_argument("--reuse-off", action="store_true", help="复用已有关闭缓存的基线，只重跑开启用例")
    parser.add_argument("--http", action="store_true", help="临时启动 server 验证 HTTP 缓存与 SSE")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--cache", choices=["on", "off"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.host_only:
        host_check()
        return
    if not args.model:
        parser.error("实机验收需要本地模型目录")
    if args.http:
        http_check(args)
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        result = worker(args)
        (output / f"{args.case}-{args.cache}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    host_check()
    for case in args.cases.split(","):
        results = []
        for cache in ("off", "on"):
            command = [sys.executable, "-m", "scripts.check_qwen35_prefix", args.model,
                       "--worker", "--case", case, "--cache", cache, "--draft", args.draft,
                       "--backend", args.backend, "--output", str(output),
                       "--seqs", str(args.seqs), "--kv-blocks", str(args.kv_blocks),
                       "--block-size", str(args.block_size),
                       "--chunk-size", str(args.chunk_size), "--max-tokens", str(args.max_tokens),
                       "--spec-num-tokens", str(args.spec_num_tokens),
                       "--max-model-len", str(args.max_model_len),
                       "--prefix-repeats", str(args.prefix_repeats),
                       "--min-hit-blocks", str(args.min_hit_blocks)]
            if args.preempt:
                command.append("--preempt")
            env = {**os.environ, "FLASHENGINE_CUDA_ATTENTION":
                   "flashinfer" if case.startswith("fi_") else "flash_attn"}
            print(f"运行 {case} cache={cache}", flush=True)
            baseline = output / f"{case}-off.json"
            if not (cache == "off" and args.reuse_off and baseline.exists()):
                with (output / f"{case}-{cache}.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            results.append(json.loads((output / f"{case}-{cache}.json").read_text(encoding="utf-8")))
        for name in ("rows", "batch"):
            exact = [a["tokens"] == b["tokens"] for a, b in zip(results[0][name], results[1][name])]
            assert all(exact), f"{case}/{name} 冷热贪心分歧：{exact}"
            print(f"{case}/{name} 严格一致 {sum(exact)}/{len(exact)}", flush=True)
        cold, warm = results[1]["rows"][0], results[1]["rows"][2]
        print(f"{case} TTFT: 冷 {cold['ttft_ms']:.1f}ms → 热 {warm['ttft_ms']:.1f}ms，"
              f"命中 {warm['hit_blocks']} 块", flush=True)


if __name__ == "__main__":
    main()
