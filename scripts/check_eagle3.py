# -*- coding: utf-8 -*-
"""EAGLE3 投机解码验收脚本（spec_bench 串行基线，QA 回归用）。

工作负载：
- spec_bench 指定类别的 N 条 prompts（取 turns[0]），temperature=0，串行逐条生成
- 5 条预热（histogram 在预热后清零），逐 prompt 记录 completion token-id
- 引擎配置：spec_num_tokens=3, max_model_len=4096, max_num_seqs=8,
  kvcache_block_size=256（与回归基线 out_*.json 同口径）
- 指标：E（acceptance length）、逐位置接受率、提交直方图、TPS
  （total_completion_tokens / 预热后串行总墙钟，含 prefill）

必须以模块方式从仓库根目录运行：`python -m scripts.check_eagle3 ...`
（不能 `python scripts/check_eagle3.py`，见 AGENTS.md 命令一节）。

--save-out 给出时写出 JSON（schema 与回归 compare 流程一致：
per_prompt_completion_token_ids / total_completion_tokens 等），
可与基线 out_*.json 逐 token 对比（compare_outputs.py）。
"""
import argparse
import json
import os
import time
import traceback

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
import nanovllm.engine.model_runner as mr

# 按轮记录提交 token 数（c=1..k+1）构建直方图（monkey-patch spec_round）。
_orig_spec_round = mr.ModelRunner.spec_round
hist = {}


def _patched_spec_round(self, seqs):
    r = _orig_spec_round(self, seqs)
    for x in r:
        hist[len(x)] = hist.get(len(x), 0) + 1
    return r


mr.ModelRunner.spec_round = _patched_spec_round


def load_prompts(questions, category, num):
    qs = []
    with open(questions) as f:
        for line in f:
            d = json.loads(line)
            if d['category'] == category:
                qs.append(d['turns'][0])
    return qs[:num]


def build_prompt_ids(llm, q):
    # 与回归基线口径一致：enable_thinking=False
    return llm.tokenizer.apply_chat_template(
        [{'role': 'user', 'content': q}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False)['input_ids']


def main():
    ap = argparse.ArgumentParser(
        description='EAGLE3 投机解码验收（spec_bench 串行，temp 0，k=3）')
    ap.add_argument('target', help='目标模型目录（本地路径，不能用 hub id）')
    ap.add_argument('--draft', required=True, help='EAGLE3 draft 模型目录')
    ap.add_argument('--questions', default='',
                    help='spec_bench question.jsonl 路径（缺省读环境变量 '
                         'NANOVLLM_SPEC_BENCH_QUESTIONS）')
    ap.add_argument('--category', default='coding', help='spec_bench 类别过滤（默认 coding）')
    ap.add_argument('--num-prompts', type=int, default=100, help='prompt 数（默认 100）')
    ap.add_argument('--max-tokens', type=int, default=512, help='每条最大生成 token 数（默认 512）')
    ap.add_argument('--tag', default='run', help='指标打印 tag（默认 run）')
    ap.add_argument('--save-out', default='',
                    help='JSON 输出路径（缺省只打印指标；给出时写同 schema JSON，'
                         '可与基线 out_*.json 逐 token 对比）')
    args = ap.parse_args()

    questions = args.questions or os.environ.get('NANOVLLM_SPEC_BENCH_QUESTIONS', '')
    if not questions:
        ap.error('缺少 spec_bench 数据：请传 --questions 或设置 NANOVLLM_SPEC_BENCH_QUESTIONS')

    qs = load_prompts(questions, args.category, args.num_prompts)
    print(f'{args.category} prompts: {len(qs)}')

    engine_kwargs = dict(
        tensor_parallel_size=1, spec_model=args.draft, spec_num_tokens=3,
        max_model_len=4096, max_num_seqs=8, kvcache_block_size=256)
    llm = LLM(args.target, **engine_kwargs)
    try:
        sp = SamplingParams(temperature=0, max_tokens=args.max_tokens)

        # 预热 5 条（histogram 预热后清零）
        for q in qs[:5]:
            seq = Sequence(build_prompt_ids(llm, q), sp)
            llm.scheduler.add(seq)
            while not llm.is_finished():
                llm.step()
        hist.clear()

        # 串行正式跑：计时从预热后开始（prefill 计入，引擎加载不计入）
        completions = []
        t0 = time.perf_counter()
        for q in qs:
            seq = Sequence(build_prompt_ids(llm, q), sp)
            llm.scheduler.add(seq)
            while not llm.is_finished():
                llm.step()
            completions.append(list(seq.completion_token_ids))
        wall = time.perf_counter() - t0

        rounds = sum(hist.values())
        total_tokens = sum(len(c) for c in completions)
        commits = sum(k * v for k, v in hist.items())
        if commits != total_tokens:
            print(f'警告: histogram 提交数 {commits} != completion 总数 {total_tokens}')
        e = commits / rounds if rounds else 0.0
        # per-position：p_j = frac(该轮提交数 c >= j+2)，j=0..2
        per_pos = [sum(v for k, v in hist.items() if k >= j + 2) / rounds
                   for j in range(3)] if rounds else [0.0, 0.0, 0.0]
        tps = total_tokens / wall if wall > 0 else 0.0

        print(f'\n==== metrics (tag={args.tag}) ====')
        print(f'prompts={len(qs)}  total_completion_tokens={total_tokens}  '
              f'decode_rounds={rounds}')
        print(f'E (acceptance length) = {e:.3f}')
        print(f'per-position accept = {per_pos[0]:.1%} / {per_pos[1]:.1%} / {per_pos[2]:.1%}')
        print(f'commit hist = {dict(sorted(hist.items()))}   (c=1,2,3,4)')
        print(f'TPS = {tps:.1f} tok/s (serial, incl prefill, excl load/warmup)')
        print(f'wall_seconds = {wall:.1f}')

        if args.save_out:
            result = {
                'tag': args.tag,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': {
                    'base_model': args.target, 'draft_model': args.draft,
                    'dataset': questions, 'category': args.category,
                    'num_prompts': len(qs), 'max_tokens': args.max_tokens,
                    'temperature': 0, 'warmup_prompts': 5,
                    'workload': 'serial (one seq at a time), hist cleared after warmup',
                    'engine_kwargs': {'model': args.target, **engine_kwargs},
                },
                'per_prompt_completion_token_ids': completions,
                'total_completion_tokens': total_tokens,
                'total_decode_rounds': rounds,
                'commit_count_histogram': {str(k): hist[k] for k in sorted(hist)},
                'E': round(e, 4),
                'per_position_accept': [round(p, 4) for p in per_pos],
                'TPS': round(tps, 2),
                'wall_seconds': round(wall, 2),
            }
            with open(args.save_out, 'w') as f:
                json.dump(result, f)
            print(f'out = {args.save_out}')
    finally:
        llm.exit()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
