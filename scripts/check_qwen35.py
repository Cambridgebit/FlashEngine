"""Qwen3.5 dense/MoE 验收：HF 文本参考、普通/投机 eager/Graph、状态生命周期。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager

QUESTIONS = ["What is 2+2? Answer with a number.", "写一个Python函数返回两个数之和。",
             "Explain why the sky is blue in two sentences.", "List three differences between Python and Rust."]


def prompts(tokenizer):
    questions = [*QUESTIONS, "Reference: write concise answers. " * 35 + QUESTIONS[1]]
    return [tokenizer.encode(tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)) for question in questions]


@contextmanager
def controlled_proposals(args):
    """仅验收用：保留真实草稿前向，随后注入已知提案，覆盖每种接受长度。"""
    if not args.controlled_proposals or not (args.case.startswith("spec") or args.case == "fi_spec"):
        yield {}
        return
    import torch
    from unittest.mock import patch
    from flashengine.engine.model_runner import ModelRunner
    from transformers import AutoTokenizer

    # 必须先跑同参数 base_eager；不修改目标 logits、接受器或状态提交代码。
    reference = json.loads((Path(args.output) / "base_eager.json").read_text(encoding="utf-8"))
    inputs = prompts(AutoTokenizer.from_pretrained(args.model, local_files_only=True))
    expected = {tuple(prompt): row["tokens"] for prompt, row in zip(inputs, reference["rows"])}
    fill, draft, readback = (ModelRunner._fill_spec_meta, ModelRunner._df_draft_block_graph,
                             ModelRunner._df_readback_commit)
    histogram = {}
    iteration = 0
    pending = {}

    def fill_meta(runner, seqs):
        nonlocal iteration
        size = fill(runner, seqs)
        pending.clear()
        if not hasattr(runner, "_check_proposals"):
            runner._check_proposals = torch.zeros_like(runner._sg["drafted_ids"])
        proposals = torch.zeros_like(runner._check_proposals, device="cpu")
        k = runner.config.spec_num_tokens
        for row, seq in enumerate(seqs):
            tokens = expected.get(tuple(seq.token_ids[:seq.num_prompt_tokens]))
            if tokens is None:
                continue  # 构造期预热，不计入验收。
            start = seq.num_completion_tokens
            accepted = (iteration + row) % (k + 1)
            for col in range(k):
                token = tokens[start + col] if start + col < len(tokens) else 0
                proposals[col, row] = (token if col < accepted else
                                       (token + 1) % runner.config.model_config.vocab_size)
            if start + k < len(tokens):
                pending[seq.seq_id] = accepted + 1
        runner._check_proposals.copy_(proposals)
        iteration += 1
        return size

    def draft_block(runner, size, use_cuda_graph):
        draft(runner, size, use_cuda_graph)
        # 捕获期绑定固定 buffer；replay 前 fill_meta 更新其中的提案。
        if not hasattr(runner, "_check_proposals"):
            runner._check_proposals = torch.zeros_like(runner._sg["drafted_ids"])
        runner._sg["drafted_ids"][:, :size].copy_(runner._check_proposals[:, :size])

    def read_commit(runner, seqs):
        result = readback(runner, seqs)
        for seq, tokens in zip(seqs, result):
            if seq.seq_id in pending:
                want = pending[seq.seq_id]
                assert len(tokens) == want, ("受控提案接受长度错误", len(tokens), want)
                histogram[want] = histogram.get(want, 0) + 1
        return result

    with (patch.object(ModelRunner, "_fill_spec_meta", fill_meta),
          patch.object(ModelRunner, "_df_draft_block_graph", draft_block),
          patch.object(ModelRunner, "_df_readback_commit", read_commit)):
        yield histogram


def hf_worker(args):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from flashengine.utils.loader import iter_safetensors

    outer_config = AutoConfig.from_pretrained(args.model)
    config = getattr(outer_config, "text_config", outer_config)
    config._attn_implementation = "eager"
    config._experts_implementation = "eager"
    previous = torch.get_default_dtype()
    torch.set_default_dtype(getattr(torch, args.reference_dtype))
    try:
        model = AutoModelForCausalLM.from_config(config, dtype=getattr(torch, args.reference_dtype)).eval()
    finally:
        torch.set_default_dtype(previous)
    for layer in model.model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.A_log.data = layer.linear_attn.A_log.data.float()
            layer.linear_attn.dt_bias.data = layer.linear_attn.dt_bias.data.float()
    loaded = set()
    expert_shards = {}
    for name, tensor in iter_safetensors(args.model):
        if name.startswith(("model.visual.", "visual.", "mtp.", "model.mtp.")):
            continue
        if name.startswith("model.language_model."):
            key = "model." + name[len("model.language_model."):]
        elif name.startswith("language_model."):
            key = name[len("language_model."):]
        else:
            key = name
        # 独立输出头必须装载；只有绑词嵌入时允许由 tie_weights 补齐。
        if key == "lm_head.weight" and config.tie_word_embeddings:
            continue
        if ".mlp.experts." in key:
            prefix, suffix = key.split(".experts.", 1)
            parts = suffix.split(".")
            # 旧检查点逐专家保存，新版 Transformers 将全部专家融合为三维参数。
            if parts[0].isdigit():
                index, proj = int(parts[0]), parts[1]
                field = "down_proj" if proj == "down_proj" else "gate_up_proj"
                target_key = prefix + ".experts." + field
                parameter = model.get_parameter(target_key)
                size = config.moe_intermediate_size
                if proj == "down_proj":
                    parameter.data[index].copy_(tensor)
                else:
                    begin = 0 if proj == "gate_proj" else size
                    parameter.data[index, begin:begin + size].copy_(tensor)
                expert_shards.setdefault(target_key, set()).add((index, proj))
                continue
            key = key.removesuffix(".weight")
        parameter = model.get_parameter(key)
        parameter.data.copy_(tensor)
        loaded.add(key)
    for key, shards in expert_shards.items():
        projections = ("down_proj",) if key.endswith(".down_proj") else ("gate_proj", "up_proj")
        expected = {(i, proj) for i in range(config.num_experts) for proj in projections}
        if shards != expected:
            raise ValueError(f"HF 参考专家分片不完整: {key}: {expected - shards}")
        loaded.add(key)
    model.tie_weights()
    missing = set(dict(model.named_parameters())) - loaded
    if config.tie_word_embeddings:
        missing.discard("lm_head.weight")
    if missing:
        raise ValueError(f"HF 文本参考缺权重: {sorted(missing)}")
    model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = []
    with torch.inference_mode():
        for prompt in prompts(tokenizer):
            ids = torch.tensor([prompt], dtype=torch.int64, device="cuda")
            output = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_tokens,
                                    do_sample=False, use_cache=True, return_dict_in_generate=True,
                                    output_logits=True, eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.eos_token_id)
            tokens = output.sequences[0, len(prompt):].tolist()
            trace = []
            for logits in output.logits:
                values, indices = logits[0].topk(8)
                trace.append({"ids": indices.tolist(), "logits": values.float().tolist()})
            rows.append({"tokens": tokens, "text": tokenizer.decode(tokens), "trace": trace})
    return {"case": "hf", "rows": rows, "model_type": config.model_type,
            "dtype": args.reference_dtype}


def engine_worker(args):
    from flashengine import LLM, SamplingParams
    rows = []
    batch_rows = []
    spec = args.case.startswith("spec") or args.case == "fi_spec"
    eager = args.case.endswith("eager") or args.case == "fi_spec"
    with controlled_proposals(args) as histogram, LLM(args.model, backend=args.backend, spec_model=args.draft if spec else "", enforce_eager=eager,
             max_num_seqs=args.seqs, max_model_len=512, max_num_batched_tokens=128,
             num_kvcache_blocks=args.kv_blocks, gpu_memory_utilization=0.9,
             spec_num_tokens=args.spec_num_tokens) as llm:
        probe = None
        if args.trace_logits:
            if not eager or spec:
                raise ValueError("logits 诊断只支持普通 eager 路径，不计入性能")
            from scripts.cuda_logits_probe import LogitsProbe
            probe = LogitsProbe(llm)
        input_ids = prompts(llm.tokenizer)
        if args.kv_blocks == 1:
            input_ids = input_ids[:4]
        sample = SamplingParams(temperature=0, max_tokens=args.max_tokens)
        for index, prompt in enumerate(input_ids):
            started = time.perf_counter()
            output = llm.generate([prompt], sample, use_tqdm=False)[0]
            rows.append({"tokens": output["token_ids"], "text": output["text"],
                         "wall_s": time.perf_counter() - started})
            if probe is not None:
                rows[-1]["trace"] = probe.drain()[0]
            assert llm.runtime_snapshot()["recurrent"]["used_slots"] == 0
        repeated = llm.generate([input_ids[0]], sample, use_tqdm=False)[0]
        assert repeated["token_ids"] == rows[0]["tokens"], "槽回收后重复请求输出漂移"
        if args.seqs > 1:
            batch_ids = input_ids[1:3] if args.kv_blocks == 1 else input_ids[:min(args.seqs, 3)]
            batch = llm.generate(batch_ids, sample, use_tqdm=False)
            assert all(row["token_ids"] for row in batch)
            batch_rows = [{"tokens": row["token_ids"], "text": row["text"]} for row in batch]
            assert llm.runtime_snapshot()["recurrent"]["used_slots"] == 0
        if args.kv_blocks == 1 and args.seqs > 1:
            assert llm.runtime_snapshot()["scheduler"]["preemptions"] > 0
        if args.boundary:
            token = llm.tokenizer.encode("hello")[0]
            edge = llm.generate([[token] * 500], SamplingParams(temperature=0, max_tokens=64, ignore_eos=True), use_tqdm=False)[0]
            assert len(edge["token_ids"]) == 12, edge
            assert llm.runtime_snapshot()["recurrent"]["used_slots"] == 0
        snapshot = llm.runtime_snapshot()
        if probe is not None:
            probe.close()
        if spec:
            assert snapshot["spec"]["rounds"] > 0, snapshot
            acceptance = snapshot["spec"]["committed"] / snapshot["spec"]["rounds"]
            assert acceptance >= args.min_acceptance_length, snapshot
        if args.case.endswith("graph") or args.case == "fi_graph":
            assert snapshot["graph"]["capture_success"] > 0, snapshot
            assert snapshot["graph"]["replay"] > 0, snapshot
            assert snapshot["graph"].get("fallback", snapshot["graph"].get("eager_rounds", 0)) == snapshot["graph"].get("context_limited_rounds", 0), snapshot
        if args.controlled_proposals and spec:
            assert set(histogram) == set(range(1, llm.config.spec_num_tokens + 2)), histogram
    return {"case": args.case, "rows": rows, "batch_rows": batch_rows, "runtime": snapshot,
            "controlled_acceptance_histogram": histogram}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Qwen3.5 本地模型目录")
    parser.add_argument("--draft", default="", help="匹配的 DFlash 本地目录；省略时只验收普通推理")
    parser.add_argument("--backend", default="cuda",
                        choices=["auto", "cuda", "corex", "ascend310p"],
                        help="推理后端（板上传 ascend310p；fi_* 用例仅 CUDA 支持）")
    parser.add_argument("--output", default="bench_results/qwen35-check", help="测试工件目录")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--cases", default="", help="逗号分隔的用例；默认根据是否提供草稿选择")
    parser.add_argument("--reference-dtype", default="bfloat16", choices=["bfloat16", "float16"],
                        help="HF 参考精度；310P 对照使用 float16")
    parser.add_argument("--trace-logits", action="store_true", help="普通 eager 记录 logits，仅用于数值诊断")
    parser.add_argument("--controlled-proposals", action="store_true",
                        help="测试专用：按 base_eager 工件注入提案，检查全拒绝/部分/全接受；不是性能数据")
    parser.add_argument("--seqs", type=int, default=1, help="引擎容量；可用 4 验证 3→4 桶 padding")
    parser.add_argument("--kv-blocks", type=int, default=8, help="测试用 KV 块数；1 可触发抢占")
    parser.add_argument("--spec-num-tokens", type=int, default=3, help="默认由草稿派生；小显存并发回归可用 2")
    parser.add_argument("--min-acceptance-length", type=float, default=1.5,
                        help="最低 E；随机草稿功能验收可显式设为 1，不代表性能验收")
    parser.add_argument("--boundary", action="store_true", help="验证 prompt+output 在上下文上限硬停")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--compare-only", action="store_true", help="只读取已有工件，不重新加载模型")
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.min_acceptance_length < 1:
        parser.error("--min-acceptance-length 不得小于 1")
    if not args.cases:
        args.cases = ("hf,base_eager,base_graph,spec_eager,spec_graph,fi_graph,fi_spec"
                      if args.draft else "hf,base_eager,base_graph")
    if any(case.startswith("spec") or case == "fi_spec" for case in args.cases.split(",")) and not args.draft:
        parser.error("投机验收用例需要 --draft")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        os.environ["FLASHENGINE_CUDA_ATTENTION"] = "flashinfer" if args.case.startswith("fi_") else "flash_attn"
        result = hf_worker(args) if args.case == "hf" else engine_worker(args)
        (output / f"{args.case}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    results = {}
    for case in args.cases.split(","):
        env = {**os.environ, "FLASHENGINE_CUDA_ATTENTION": "flashinfer" if case.startswith("fi_") else "flash_attn"}
        command = [sys.executable, "-m", "scripts.check_qwen35", args.model, "--draft", args.draft,
                   "--worker", "--case", case, "--output", str(output), "--max-tokens", str(args.max_tokens),
                   "--seqs", str(args.seqs), "--backend", args.backend,
                   "--reference-dtype", args.reference_dtype]
        command.extend(["--kv-blocks", str(args.kv_blocks), "--spec-num-tokens", str(args.spec_num_tokens)])
        command.extend(["--min-acceptance-length", str(args.min_acceptance_length)])
        if args.boundary:
            command.append("--boundary")
        if args.trace_logits:
            command.append("--trace-logits")
        if args.controlled_proposals:
            command.append("--controlled-proposals")
        if not args.compare_only:
            print(f"运行 {case}", flush=True)
            with (output / f"{case}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[case] = json.loads((output / f"{case}.json").read_text(encoding="utf-8"))
    base = results.get("base_eager", results.get("hf", next(iter(results.values()), None)))
    comparison = {}
    if base:
        for case, result in results.items():
            assert result["rows"] and len(base["rows"]) == len(result["rows"]), f"{case} 请求数不一致"
            exact = [a["tokens"] == b["tokens"] for a, b in zip(base["rows"], result["rows"])]
            comparison[case] = exact
            print(f"{case}: 严格一致 {sum(exact)}/{len(exact)}", flush=True)
            if base.get("batch_rows") and result.get("batch_rows"):
                batch_exact = [a["tokens"] == b["tokens"] for a, b in zip(base["batch_rows"], result["batch_rows"])]
                comparison[case + "_batch"] = batch_exact
    (output / "summary.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    if any(not all(exact) for exact in comparison.values()):
        raise SystemExit("存在贪心 token 分歧，请查看工件并核对 logits，不能静默豁免。")


if __name__ == "__main__":
    main()
