"""Qwen3.5 CUDA 验收：HF 文本参考、普通/投机 eager/Graph、状态生命周期。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

QUESTIONS = ["What is 2+2? Answer with a number.", "写一个Python函数返回两个数之和。",
             "Explain why the sky is blue in two sentences.", "List three differences between Python and Rust."]


def prompts(tokenizer):
    questions = [*QUESTIONS, "Reference: write concise answers. " * 35 + QUESTIONS[1]]
    return [tokenizer.encode(tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)) for question in questions]


def hf_worker(args):
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from flashengine.utils.loader import iter_safetensors

    config = AutoConfig.from_pretrained(args.model).text_config
    config._attn_implementation = "eager"
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = Qwen3_5ForCausalLM(config).eval()
    torch.set_default_dtype(previous)
    for layer in model.model.layers:
        if hasattr(layer, "linear_attn"):
            layer.linear_attn.A_log.data = layer.linear_attn.A_log.data.float()
            layer.linear_attn.dt_bias.data = layer.linear_attn.dt_bias.data.float()
    loaded = set()
    for name, tensor in iter_safetensors(args.model):
        if not name.startswith("model.language_model."):
            continue
        key = "model." + name[len("model.language_model."):]
        parameter = model.get_parameter(key)
        parameter.data.copy_(tensor)
        loaded.add(key)
    model.tie_weights()
    missing = set(dict(model.named_parameters())) - loaded - {"lm_head.weight"}
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
    return {"case": "hf", "rows": rows}


def engine_worker(args):
    from flashengine import LLM, SamplingParams
    rows = []
    batch_rows = []
    spec = args.case.startswith("spec") or args.case == "fi_spec"
    eager = args.case.endswith("eager") or args.case == "fi_spec"
    with LLM(args.model, backend="cuda", spec_model=args.draft if spec else "", enforce_eager=eager,
             max_num_seqs=args.seqs, max_model_len=512, max_num_batched_tokens=128,
             num_kvcache_blocks=args.kv_blocks, gpu_memory_utilization=0.9,
             spec_num_tokens=args.spec_num_tokens) as llm:
        input_ids = prompts(llm.tokenizer)
        if args.kv_blocks == 1:
            input_ids = input_ids[:4]
        sample = SamplingParams(temperature=0, max_tokens=args.max_tokens)
        for index, prompt in enumerate(input_ids):
            started = time.perf_counter()
            output = llm.generate([prompt], sample, use_tqdm=False)[0]
            rows.append({"tokens": output["token_ids"], "text": output["text"],
                         "wall_s": time.perf_counter() - started})
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
        if spec:
            assert snapshot["spec"]["committed"] / snapshot["spec"]["rounds"] > 1.5, snapshot
        if args.case.endswith("graph") or args.case == "fi_graph":
            assert snapshot["graph"]["capture_success"] > 0, snapshot
            assert snapshot["graph"]["replay"] > 0, snapshot
            assert snapshot["graph"].get("fallback", snapshot["graph"].get("eager_rounds", 0)) == snapshot["graph"].get("context_limited_rounds", 0), snapshot
    return {"case": args.case, "rows": rows, "batch_rows": batch_rows, "runtime": snapshot}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Qwen3.5 本地模型目录")
    parser.add_argument("--draft", required=True, help="匹配的 DFlash 本地目录")
    parser.add_argument("--output", default="bench_results/qwen35-check", help="测试工件目录")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--cases", default="hf,base_eager,base_graph,spec_eager,spec_graph,fi_graph,fi_spec")
    parser.add_argument("--seqs", type=int, default=1, help="引擎容量；可用 4 验证 3→4 桶 padding")
    parser.add_argument("--kv-blocks", type=int, default=8, help="测试用 KV 块数；1 可触发抢占")
    parser.add_argument("--spec-num-tokens", type=int, default=3, help="默认由草稿派生；小显存并发回归可用 2")
    parser.add_argument("--boundary", action="store_true", help="验证 prompt+output 在上下文上限硬停")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--compare-only", action="store_true", help="只读取已有工件，不重新加载模型")
    parser.add_argument("--case", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        result = hf_worker(args) if args.case == "hf" else engine_worker(args)
        (output / f"{args.case}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    results = {}
    for case in args.cases.split(","):
        env = {**os.environ, "FLASHENGINE_CUDA_ATTENTION": "flashinfer" if case.startswith("fi_") else "flash_attn"}
        command = [sys.executable, "-m", "scripts.check_qwen35", args.model, "--draft", args.draft,
                   "--worker", "--case", case, "--output", str(output), "--max-tokens", str(args.max_tokens),
                   "--seqs", str(args.seqs)]
        command.extend(["--kv-blocks", str(args.kv_blocks), "--spec-num-tokens", str(args.spec_num_tokens)])
        if args.boundary:
            command.append("--boundary")
        if not args.compare_only:
            print(f"运行 {case}", flush=True)
            with (output / f"{case}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[case] = json.loads((output / f"{case}.json").read_text(encoding="utf-8"))
    base = results.get("base_eager", results.get("hf", next(iter(results.values()), None)))
    comparison = {}
    if base:
        for case, result in results.items():
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
