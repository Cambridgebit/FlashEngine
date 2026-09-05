#!/usr/bin/env bash
# CoreX（Iluvatar）实机验证清单：五步串行，全部统一走 CoreX 后端，日志逐部落盘。
#
# 用法:
#   bash scripts/verify_corex.sh <target模型目录> <EAGLE3 draft目录> [spec_bench jsonl] [输出目录]
#
# 注意：
# ① scripts/check_eagle3.py 的引擎 kwargs 硬编码 max_model_len=4096 / max_num_seqs=8 /
#    kvcache_block_size=256，小显存 CoreX 卡可能 OOM（可改脚本，或等 --max-model-len 参数落地）；
# ② 运行环境需已安装 ixformer（CoreX 后端依赖，pyproject 未声明，仅 Iluvatar 硬件可用）；
# ③ 步骤判读：第 4 步启动时若打印「spec CUDA Graph 捕获失败，降级为 eager」= CoreX 上不可捕获，
#    该步与第 3 步等价；若无告警则 spec 图生效，对比两步的 E / TPS 即为图收益。

set -o pipefail

# ---- 参数解析 ----
# $1=target 模型目录（必填）  $2=EAGLE3 draft 目录（必填）
# $3=spec_bench 数据集 jsonl 路径（默认 ~/vllm-lab/benchdata/spec_bench/question.jsonl）
# $4=输出目录（默认 ~/vllm-lab/benchdata/corex_verify）
if [ "$#" -lt 2 ]; then
  echo "用法: $0 <target模型目录> <EAGLE3 draft目录> [spec_bench jsonl] [输出目录]" >&2
  exit 1
fi
TARGET="$1"
DRAFT="$2"
QUESTIONS="${3:-$HOME/vllm-lab/benchdata/spec_bench/question.jsonl}"
OUT="${4:-$HOME/vllm-lab/benchdata/corex_verify}"

for d in "$TARGET" "$DRAFT"; do
  if [ ! -d "$d" ]; then
    echo "模型目录不存在: $d" >&2
    exit 1
  fi
done
mkdir -p "$OUT"

# 绝对化后再切目录，避免相对路径失效
TARGET="$(realpath "$TARGET")"
DRAFT="$(realpath "$DRAFT")"
QUESTIONS="$(realpath "$QUESTIONS")"
OUT="$(realpath -m "$OUT")"

# 全部步骤统一走 CoreX 后端
export NANOVLLM_BACKEND=corex
# scripts/check_eagle3.py 需要该 env（或 --questions 参数）提供 spec_bench 数据集
export NANOVLLM_SPEC_BENCH_QUESTIONS="$QUESTIONS"

# 允许用 PY= 覆盖解释器（如 CoreX 机器上的特定 venv）
PY=${PY:-python3}
# 按脚本所在位置自适应仓库根；引擎必须以模块方式从仓库根运行
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==== CoreX 实机验证五步清单 ===="
echo "target=$TARGET draft=$DRAFT questions=$QUESTIONS out=$OUT PY=$PY"

# 步骤 1/5：decode CUDA Graph 回归 —— V1.3 已实证的路径 + 本次新增的图内采样内容
echo "---- 步骤 1/5: decode CUDA Graph 回归 ----"
$PY -m scripts.check_prefix_cache "$TARGET" --backend corex --graph 2>&1 | tee "$OUT/prefix_graph.log"

# 步骤 2/5：图内采样对照 —— NANOVLLM_GRAPH_SAMPLE=0 关掉 exponential_ 图内采样再跑一遍，隔离其捕获风险
echo "---- 步骤 2/5: 图内采样对照（NANOVLLM_GRAPH_SAMPLE=0）----"
NANOVLLM_GRAPH_SAMPLE=0 $PY -m scripts.check_prefix_cache "$TARGET" --backend corex --graph 2>&1 | tee "$OUT/prefix_graph_nosample.log"

# check_eagle3.py 无 --backend 参数，CoreX 后端由上方 export 的 NANOVLLM_BACKEND 指定
# 步骤 3/5：spec eager 轮 —— 显式禁用 spec 图，与 CUDA 基线 E=2.095 对照
echo "---- 步骤 3/5: spec eager 轮（NANOVLLM_SPEC_GRAPH=0）----"
NANOVLLM_SPEC_GRAPH=0 $PY -m scripts.check_eagle3 "$TARGET" --draft "$DRAFT" --tag corex_eager_spec 2>&1 | tee "$OUT/eager_spec.log"

# 步骤 4/5：spec 图捕获尝试 —— 默认放行捕获（supports_spec_graph=True）；
# 若启动打印「spec CUDA Graph 捕获失败，降级为 eager」= 不可捕获，与第 3 步等价；
# 若无告警则图生效，对比两步 E / TPS。
echo "---- 步骤 4/5: spec 图捕获尝试 ----"
$PY -m scripts.check_eagle3 "$TARGET" --draft "$DRAFT" --tag corex_spec_graph 2>&1 | tee "$OUT/spec_graph.log"

# 步骤 5/5：TPS sweep —— decode 吞吐随 batch 扫描（1/2/4/8）
echo "---- 步骤 5/5: TPS sweep ----"
$PY -m scripts.benchmark graph "$TARGET" --batch 1 2 4 8 --backend corex 2>&1 | tee "$OUT/bench_sweep.log"

echo "==== 五步验证完成，日志与结果输出目录: $OUT ===="
