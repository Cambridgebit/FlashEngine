"""沿同一 Python token 前缀诊断首次分歧，避免自由生成分叉污染后续 logits。"""
import argparse
import os
from pathlib import Path

from flashengine.deployment import CONTRACT_VERSION, read_json, sha256, source_state, write_json
from scripts.native_bench_common import validate_workload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("bundle")
    p.add_argument("workload")
    p.add_argument("reference", help="Python benchmark worker 的结果")
    p.add_argument("output")
    p.add_argument("--input-id", required=True)
    p.add_argument("--attention", choices=("flash_attn", "flashinfer"), default="flash_attn", help="仅诊断时切换 Python attention；不能冒充原基线")
    args = p.parse_args()
    if Path(args.output).exists():
        raise ValueError("拒绝覆盖 teacher oracle")
    workload, reference = read_json(args.workload), read_json(args.reference)
    validate_workload(workload)
    bundle = Path(args.bundle)
    if workload["bundle_sha256"] != sha256(bundle / "bundle.json") or reference["workload_sha256"] != sha256(args.workload):
        raise ValueError("teacher 输入来源不匹配")
    if read_json(bundle / "engine/config.json")["draft"]:
        raise ValueError("teacher 诊断暂仅支持普通 decode")
    for name, item in read_json(bundle / "bundle.json")["files"].items():
        if name.startswith("weights/target/") and sha256(Path(args.model) / Path(name).name) != item["sha256"]:
            raise ValueError("Python 权重与 bundle 不匹配")
    request = next(row for row in workload["requests"] if row["id"] == args.input_id)
    row = next(row for row in reference["rows"] if row["input_id"] == args.input_id and row["repeat"] == 0)
    expected = row["token_ids"]
    os.environ["FLASHENGINE_BACKEND"] = "cuda"
    os.environ["FLASHENGINE_CUDA_ATTENTION"] = args.attention
    from flashengine import LLM, SamplingParams
    from flashengine.engine.sequence import Sequence
    import torch
    cfg = workload["config"]
    with LLM(args.model, **cfg, backend="cuda", enforce_eager=True, enable_prefix_cache=False) as llm:
        pending, samples, actual = [], [], []
        original = llm.model_runner.model.compute_logits
        def logits(hidden):
            result = original(hidden)
            values, ids = result[-1].float().topk(8)
            pending.append(dict(ids=ids.tolist(), values=values.tolist()))
            return result
        llm.model_runner.model.compute_logits = logits
        seq = Sequence(request["prompt_token_ids"], SamplingParams(temperature=0, max_tokens=len(expected), ignore_eos=True))
        llm.scheduler.add(seq)
        while not llm.is_finished():
            before = seq.num_completion_tokens
            pending.clear()
            llm.step()
            if seq.num_completion_tokens > before:
                actual.append(seq.last_token)
                samples.append(pending[-1])
                seq.token_ids[-1] = seq.last_token = expected[before]
        environment = dict(dtype=str(llm.model_runner.dtype), torch=torch.__version__, device=torch.cuda.get_device_name(), attention=args.attention)
    case = dict(prompt_token_ids=request["prompt_token_ids"], output_token_ids=expected, finish_reason="length", sampling=dict(temperature=0, max_tokens=len(expected), ignore_eos=True), samples=samples, python_teacher_argmax=actual)
    write_json(args.output, dict(contract_version=CONTRACT_VERSION, oracle=dict(**source_state(), bundle_sha256=sha256(Path(args.bundle) / "bundle.json"), environment=environment), teacher_source=dict(workload_sha256=sha256(args.workload), reference_sha256=sha256(args.reference), input_id=args.input_id), python_self_consistent=actual==expected, cases=[case]))
    print(f"teacher oracle 已导出，Python 自洽={actual==expected}")


if __name__ == "__main__":
    main()
