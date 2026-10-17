"""冻结同一模型、配置和 token 请求，供 Python/C++ 严格验证及基准共用。"""
import argparse
from pathlib import Path

from flashengine.deployment import read_json, sha256, source_state, write_json
from scripts.bench_utils import load_prompts
from scripts.native_bench_common import validate_workload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", help="本地 tokenizer/目标模型目录")
    p.add_argument("bundle", help="相应原生部署 bundle")
    p.add_argument("output", help="新的 workload JSON")
    p.add_argument("--questions", default="bench_data/question.jsonl")
    p.add_argument("--prompts-json", help="可选：字符串数组，优先于 questions")
    p.add_argument("--category", default="coding")
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--max-num-batched-tokens", type=int, default=512)
    p.add_argument("--max-num-partial-prefills", type=int, default=2)
    p.add_argument("--kv-block-size", type=int, default=256)
    p.add_argument("--kv-blocks", type=int, default=32)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--respect-eos", action="store_true", help="正常 EOS 停止；默认忽略 EOS 固定生成长度")
    p.add_argument("--raw-prompts", action="store_true", help="不套 chat template；默认 thinking-off")
    p.add_argument("--prompt-lengths", help="仅 raw 测试：循环已编码输入到指定 token 长度列表")
    p.add_argument("--synthetic", action="store_true", help="明确标记未训练夹具，不能用于真实性能声明")
    args = p.parse_args()
    lengths = [int(x) for x in args.prompt_lengths.split(",")] if args.prompt_lengths else None
    if lengths and (not args.raw_prompts or min(lengths) < 1):
        raise ValueError("指定 token 长度仅适用于 raw 引擎测试")
    if Path(args.output).exists():
        raise ValueError("拒绝覆盖 workload")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    texts = read_json(args.prompts_json) if args.prompts_json else load_prompts(args.questions, args.category, args.num_prompts, args.seed)
    config = dict(dtype=args.dtype, max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens, max_num_partial_prefills=args.max_num_partial_prefills, kvcache_block_size=args.kv_block_size, num_kvcache_blocks=args.kv_blocks)
    rows = []
    for index, text in enumerate(texts):
        rendered = text if args.raw_prompts else tok.apply_chat_template([dict(role="user", content=text)], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok.encode(rendered, add_special_tokens=False)
        if lengths:
            if len(lengths) != len(texts) or not ids:
                raise ValueError("prompt-lengths 数量不符或输入为空")
            ids = (ids * ((lengths[index] + len(ids) - 1) // len(ids)))[:lengths[index]]
        rows.append(dict(id=str(index), prompt_token_ids=ids, max_tokens=args.max_tokens, ignore_eos=not args.respect_eos))
    bundle = Path(args.bundle)
    fixture = Path(args.model).parent / "fixture.json"
    synthetic = args.synthetic or (fixture.is_file() and read_json(fixture).get("synthetic", False))
    workload = dict(schema_version=1, bundle_sha256=sha256(bundle / "bundle.json"), vocab_size=read_json(bundle / "engine/config.json")["target"]["vocab_size"], config=config, requests=rows, synthetic_model=bool(synthetic), thinking="raw" if args.raw_prompts else "off", source=dict(path=args.prompts_json or args.questions, sha256=sha256(args.prompts_json or args.questions)), oracle_source=source_state())
    validate_workload(workload)
    write_json(args.output, workload)
    print(f"已冻结 {len(rows)} 个输入，prompt 长度={[len(r['prompt_token_ids']) for r in rows]}，synthetic={synthetic}")


if __name__ == "__main__":
    main()
