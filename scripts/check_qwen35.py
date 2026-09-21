"""Qwen3.5/DFlash 验收校验（阶段1/2/3 验收门的可复跑落成）。

四个子命令（两两配对使用，两次调用须分开进程——WSL 下上一进程显存回收有
数秒延迟，同进程先后建双引擎会互相挤爆 KV 预算；引擎预算报错自带诊断数字）：

  ref    transformers 5.15 原生贪心参考输出 → json（阶段1 门：位等锚点）
  base   本引擎不开投机跑一遍 → json
  spec   本引擎开 DFlash 投机跑一遍 → json（阶段2 门：spec vs base 位等）
  check  对比两份 json：逐 prompt 报告首个分歧 token 位置（位等门）

判定口径（与 AGENTS.md「Qwen3.5 + DFlash」节一致）：
- 串行 bs=1 temp0 下期望全等或仅 bf16 平票翻转（deep-position 翻转、
  top-1/top-2 logit 打平或差 1 ULP）；并发/图 padding 场景允许形状漂移翻转
  （GEMM/attention 批形状差 → 归约序 ULP），翻转需逐条记录位置与深度；
- spec vs base 的翻转属「投机 vs 非投机逐 token 一致只在无翻转负载上成立」
  的既有口径（EAGLE3 同款），验收看 E 是否健康而非强求逐位。

用法（必须从仓库根目录以 -m 方式运行）：
    python3 -m scripts.check_qwen35 ref  <Qwen3.5-2B 目录> --out /tmp/ref.json
    python3 -m scripts.check_qwen35 base <Qwen3.5-2B 目录> --out /tmp/base.json
    python3 -m scripts.check_qwen35 check /tmp/ref.json /tmp/base.json
    python3 -m scripts.check_qwen35 spec <Qwen3.5-2B 目录> --draft <DFlash 目录> \
        --out /tmp/spec.json --graph
    python3 -m scripts.check_qwen35 check /tmp/base.json /tmp/spec.json --label spec-vs-base
"""
import argparse
import json

# 校验 prompt 集（中英混合 raw 续写，短长交错；与阶段1/2/3 验收门同款）。
PROMPTS = [
    "The capital of France is",
    "人工智能的未来发展方向包括",
    "def quicksort(arr):\n    if len(arr) <= 1:\n",
    "1 + 1 = 2, 2 + 2 = 4, 3 + 3 =",
    "Once upon a time, a young programmer discovered",
    "量子计算与经典计算的根本区别在于",
    "The quadratic formula is x =",
    "中国的四大发明是火药、造纸术、",
    "Machine learning models learn from",
    "生活就像一盒巧克力，",
    "In a shocking finding, scientists discovered a herd of unicorns living in a remote valley",
    "快速排序的时间复杂度分析：",
]


def _engine_kwargs(args):
    return dict(max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
                gpu_memory_utilization=args.gpu_mem_util, enforce_eager=not args.graph)


def cmd_ref(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # device_map 路径需要 accelerate（未装），改 load 后 .to("cuda")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).to("cuda").eval()
    ref = []
    for p in PROMPTS:
        ids = tok.encode(p)
        inp = torch.tensor([ids], device="cuda")
        with torch.no_grad():
            out = model.generate(inp, do_sample=False, max_new_tokens=args.max_tokens,
                                 pad_token_id=tok.eos_token_id)
        ref.append(out[0].tolist())
        print(f"[ref] prompt={len(ids)} tok -> +{len(out[0]) - len(ids)}")
    with open(args.out, "w") as f:
        json.dump({"prompts": PROMPTS, "prompt_lens": [len(tok.encode(p)) for p in PROMPTS],
                   "full_ids": ref, "kind": args.kind}, f)
    print(f"REF_DONE -> {args.out}")


def _engine_run(args, spec: bool):
    from nanovllm import LLM, SamplingParams

    kw = _engine_kwargs(args)
    if spec:
        llm = LLM(args.model, spec_model=args.draft, **kw)
    else:
        llm = LLM(args.model, **kw)
    outs = llm.generate(PROMPTS, [SamplingParams(temperature=0, max_tokens=args.max_tokens)] * len(PROMPTS),
                        use_tqdm=False)
    llm.exit()
    return outs


def cmd_engine(args, spec: bool):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    outs = _engine_run(args, spec)
    with open(args.out, "w") as f:
        json.dump({"prompts": PROMPTS,
                   "prompt_lens": [len(tok.encode(p)) for p in PROMPTS],
                   "gen_ids": [o["token_ids"] for o in outs],
                   "kind": args.kind}, f)
    print(f"{'SPEC' if spec else 'BASE'}_DONE -> {args.out}")


def cmd_check(args):
    with open(args.a) as f:
        a = json.load(f)
    with open(args.b) as f:
        b = json.load(f)
    assert a["prompts"] == b["prompts"], "Prompt sets differ between the two runs."
    # 统一成生成段：ref 存全序列（prompt+gen），引擎侧存生成段
    def gen_side(d, i):
        if "full_ids" in d:
            return d["full_ids"][i][d["prompt_lens"][i]:]
        return d["gen_ids"][i]

    n_ok = 0
    for i, p in enumerate(a["prompts"]):
        ga, gb = gen_side(a, i), gen_side(b, i)
        if ga == gb:
            n_ok += 1
            print(f"[OK   {i}] {len(ga)} tok")
            continue
        first = next((j for j, (x, y) in enumerate(zip(ga, gb)) if x != y), min(len(ga), len(gb)))
        depth = f"{first / max(len(ga), 1):.0%}" if len(ga) == len(gb) else "length-mismatch"
        print(f"[DIFF {i}] first@{first}/{len(ga)} depth={depth} lens={len(ga)}/{len(gb)}")
    print(f"{args.label}: {n_ok}/{len(a['prompts'])} token-exact matches")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_model=True):
        if need_model:
            p.add_argument("model", help="Local Qwen3.5 model directory")
        p.add_argument("--out", default=".omo/check_out.json", help="Output JSON path")
        p.add_argument("--max-tokens", type=int, default=64)
        p.add_argument("--max-model-len", type=int, default=2048)
        p.add_argument("--max-num-seqs", type=int, default=2)
        p.add_argument("--gpu-mem-util", type=float, default=0.95)
        p.add_argument("--graph", action="store_true", help="Enable graph mode; eager is the default")
        p.add_argument("--kind", default="", help="JSON label; defaults to the subcommand name")

    common(sub.add_parser("ref", help="Save transformers reference outputs to JSON"))
    common(sub.add_parser("base", help="Save base-engine outputs to JSON"))
    p_spec = sub.add_parser("spec", help="Save DFlash speculative outputs to JSON")
    common(p_spec)
    p_spec.add_argument("--draft", required=True, help="DFlash draft model directory")
    p_ck = sub.add_parser("check", help="Compare two JSON result files")
    p_ck.add_argument("a", help="JSON A, such as ref or base")
    p_ck.add_argument("b", help="JSON B, such as base or spec")
    p_ck.add_argument("--label", default="PARITY")

    args = ap.parse_args()
    if args.cmd in ("ref", "base", "spec") and not args.kind:
        args.kind = args.cmd
    {"ref": cmd_ref, "base": lambda a: cmd_engine(a, False),
     "spec": lambda a: cmd_engine(a, True), "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
