"""导出固定 token 输入的 Python 引擎 golden；投机轮使用独立诊断钩子。"""
import argparse
import importlib.metadata
import platform
import os
from pathlib import Path

from flashengine.deployment import CONTRACT_VERSION, normalize_model, read_json, sha256, source_state, write_json
from scripts.native_bench_common import checked_greedy_round


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="本地目标模型")
    parser.add_argument("bundle", help="对应部署 bundle")
    parser.add_argument("output", help="新 golden JSON 文件")
    parser.add_argument("--draft", help="本地 DFlash/EAGLE3 草稿")
    parser.add_argument("--eager", action="store_true", help="生成 eager 参考")
    parser.add_argument("--backend", choices=("cuda", "corex", "ascend310p"), default="cuda", help="参考引擎后端")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), help="显式固定参考计算类型；310P 只支持 float16")
    parser.add_argument("--logits", action="store_true", help="导出普通/投机 top-8 logits，限 eager")
    parser.add_argument("--max-tokens", type=int, default=32, help="输出 token 上限")
    parser.add_argument("--max-model-len", type=int, default=512, help="参考上下文上限")
    parser.add_argument("--max-num-seqs", type=int, default=1, help="参考批上限")
    parser.add_argument("--gpu-mem-util", type=float, default=.70, help="设备内存预算")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("拒绝覆盖 golden")
    os.environ["FLASHENGINE_BACKEND"] = args.backend
    from flashengine import LLM, SamplingParams
    import torch
    bundle = Path(args.bundle)
    config = read_json(bundle / "engine/config.json")
    # 0.1 契约的 dense bundle 没有显式 model_type，按原约定补齐再比较。
    config["target"].setdefault("model_type", "qwen3")
    source_config = normalize_model(read_json(Path(args.model) / "config.json"))
    if any(config["target"].get(key) != value for key, value in source_config.items()):
        raise ValueError("oracle 目标配置与 bundle 不一致")
    if bool(args.draft) != bool(config["draft"]):
        raise ValueError("oracle 草稿配置与 bundle 不一致")
    for kind, directory in (("target", args.model), ("draft", args.draft)):
        if not directory:
            continue
        for name, entry in read_json(bundle / "bundle.json")["files"].items():
            if name.startswith(f"weights/{kind}/") and sha256(Path(directory) / Path(name).name) != entry["sha256"]:
                raise ValueError(f"oracle 权重与 bundle 不一致：{name}")
    kwargs = dict(backend=args.backend, enforce_eager=args.eager, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, gpu_memory_utilization=args.gpu_mem_util, enable_prefix_cache=False)
    if args.dtype:
        kwargs["dtype"] = args.dtype
    if args.draft:
        k = config["draft"]["num_tokens"]
        kwargs.update(spec_model=args.draft, spec_num_tokens=k)
        if config["draft"].get("spec_method")=="eagle3":kwargs["spec_aux_layers"]=tuple(config["draft"]["aux_layers"])
        if k in (3, 4):
            raise ValueError("Python 历史 CLI 将 k=3/4 视为默认值，请改用其他明确 k 导出参考")
    cases = []
    with LLM(args.model, **kwargs) as llm:
        rounds = []
        samples = []
        if args.logits and not args.eager:
            raise ValueError("Python logits 诊断只支持 eager，禁止向捕获区域加入 host 读回")
        if args.logits and not args.draft:
            original_logits = llm.model_runner.model.compute_logits

            def trace_logits(hidden):
                result = original_logits(hidden)
                values, ids = result[-1].float().topk(8)
                samples.append(dict(ids=ids.tolist(), values=values.tolist()))
                return result

            llm.model_runner.model.compute_logits = trace_logits
        if args.draft:
            runner = llm.model_runner
            if args.logits:
                original_sample = runner._sg_sample

                def trace_sample(logits, temps, topps, topks, use_cuda_graph):
                    values, ids = logits.float().topk(8)
                    stage = "draft" if logits.shape[0] == config["draft"]["num_tokens"] else "verify"
                    samples.append(dict(stage=stage, ids=ids.tolist(), values=values.tolist()))
                    return original_sample(logits, temps, topps, topks, use_cuda_graph)

                runner._sg_sample = trace_sample
            method = "_readback_spec_commit" if config["draft"].get("spec_method") == "eagle3" else "_df_readback_commit"
            original = getattr(runner, method)

            def trace_round(seqs):
                n, k = len(seqs), config["draft"]["num_tokens"]
                g = runner._sg
                drafts = (g["d_pad"][:n, :k] if config["draft"].get("spec_method")=="eagle3"
                          else g["drafted_ids"][:k, :n].t()).tolist()
                verified = g["v_token_ids"][:n * (k + 1)].view(n, k + 1).tolist()
                committed = original(seqs)
                for seq, d, v, c in zip(seqs, drafts, verified, committed):
                    rounds.append(checked_greedy_round(d,v,c,len(seq)-1))
                return committed

            setattr(runner, method, trace_round)
        prompts = ["The capital of France is", "用一句话介绍北京。", "def fibonacci(n):\n"]
        for text in prompts:
            ids = llm.tokenizer.encode(text, add_special_tokens=False)
            rounds.clear()
            samples.clear()
            out = llm.generate([ids], SamplingParams(temperature=0, max_tokens=args.max_tokens), use_tqdm=False)[0]
            tokens = out["token_ids"]
            cases.append(dict(prompt_token_ids=ids, output_token_ids=tokens, sampling=dict(temperature=0, max_tokens=args.max_tokens), finish_reason="stop" if tokens and tokens[-1] == llm.config.eos else "length", rounds=list(rounds), samples=list(samples)))
        device_api = getattr(torch, llm.model_runner.backend.device_name)
        environment = dict(python=platform.python_version(), torch=torch.__version__, transformers=importlib.metadata.version("transformers"), device=device_api.get_device_name(), backend=args.backend, dtype=str(llm.model_runner.dtype), runtime=llm.runtime_snapshot())
    write_json(args.output, dict(schema_version=1, contract_version=CONTRACT_VERSION, oracle=dict(**source_state(), bundle_sha256=sha256(bundle / "bundle.json"), environment=environment), cases=cases))
    print(f"已导出 {len(cases)} 个 oracle 样例：{args.output}")


if __name__ == "__main__":
    main()
