#!/usr/bin/env bash
# Ascend 310P（Atlas 300I Duo / 200I Pro）实机验证清单：环境预检 + 四步串行，
# 全部统一走 ascend310p 后端，日志逐部落盘。
#
# 用法:
#   bash scripts/verify_ascend310p.sh <target模型目录> <EAGLE3 draft目录> [spec_bench jsonl] [输出目录]
#
# 注意：
# ① 310P 仅支持 fp16（ATB 算子不支持 bf16）——引擎已按后端强制 dtype，bf16 权重
#    加载时隐式转换，无需手工转模；
# ② 勿依赖 max-model-len 自动探测：310P 的 prefill 注意力按 max_model_len 全量构建
#    causal mask，长度过大有 OOM 风险（vllm-ascend 同款已知问题），建议显式指定；
# ③ 310P 目标形态为 TP=1（多卡受硬件 event-id 资源限制，且 spec 本就仅 TP=1）；
# ④ spec 全路径在 310P 被引擎层拒绝（supports_spec=False：verify 恒传 block_tables
#    → attention 层走整池 4D 分支，TND 算子不消费；paged-TND 适配后再开）——步骤 3
#    验证的就是这个守卫：期待报错信息命中守卫文案（init 处单点 RuntimeError），
#    深栈形状错误 / 无报错意外成功均 FAIL；
# ⑤ 步骤判读：步骤 2 启动时若打印「捕获失败，降级为 eager」= 图不可捕获，与 eager
#    步等价；若无告警则图生效，对比 E / TPS 即为图收益；
# ⑥ 已知限制（勿当回归报）：prefix 命中的 prefill 会把 k/v 换成整池 4D 切片，TND
#    不消费——步骤 1 的 warm 轮（prefix-hit）预期以形状报错终止，判据只看 R1 cold
#    轮的 PASS（R1 = TND 路径正确性）；
# ⑦ 运行环境需 Ascend 硬件 + torch_npu（vllm-ascend 钉 torch-npu==2.10.0.post4 /
#    CANN 9.1.0）；步骤 0 会对四个载荷算子做存在性探测，缺失即明确报错。

set -o pipefail

# ---- 参数解析 ----
if [ "$#" -lt 2 ]; then
  echo "用法: $0 <target模型目录> <EAGLE3 draft目录> [spec_bench jsonl] [输出目录]" >&2
  exit 1
fi
TARGET="$1"
DRAFT="$2"
QUESTIONS="${3:-$HOME/vllm-lab/benchdata/spec_bench/question.jsonl}"
OUT="${4:-$HOME/vllm-lab/benchdata/ascend310p_verify}"

for d in "$TARGET" "$DRAFT"; do
  if [ ! -d "$d" ]; then
    echo "模型目录不存在: $d" >&2
    exit 1
  fi
done
mkdir -p "$OUT"

TARGET="$(realpath "$TARGET")"
DRAFT="$(realpath "$DRAFT")"
QUESTIONS="$(realpath -m "$QUESTIONS")"
OUT="$(realpath -m "$OUT")"

# 全部步骤统一走 Ascend310P 后端
export NANOVLLM_BACKEND=ascend310p
export NANOVLLM_SPEC_BENCH_QUESTIONS="$QUESTIONS"
# 310P 起步保守：图内采样关闭（NPU RNG 图安全性未验证；实验可显式置 1）
export NANOVLLM_GRAPH_SAMPLE=${NANOVLLM_GRAPH_SAMPLE:-0}

PY=${PY:-python3}
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==== Ascend 310P 实机验证清单（环境预检 + 四步）===="
echo "target=$TARGET draft=$DRAFT questions=$QUESTIONS out=$OUT PY=$PY"

# 步骤 0：环境预检 —— npu 可见性 / torch_npu 版本 / 四个载荷算子存在性 / pinned 分配
echo "---- 步骤 0: 环境预检 ----"
$PY - <<'EOF'
import os

import torch
print('torch:', torch.__version__)
try:
    import torch_npu
except ImportError as e:
    print('torch_npu 导入失败（需 Ascend 环境 + torch-npu==2.10.0.post4 / CANN 9.1.0）:', e)
    raise SystemExit(1)
print('torch_npu:', getattr(torch_npu, '__version__', 'unknown'))
print('npu available:', torch.npu.is_available(), '| device_count:', torch.npu.device_count())
print('ASCEND_RT_VISIBLE_DEVICES:', os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '<unset>'))
missing = [op for op in ('_npu_reshape_and_cache',
                         'npu_fused_infer_attention_score_v2',
                         'npu_add_rms_norm',
                         'npu_swiglu') if not hasattr(torch_npu, op)]
if missing:
    print('载荷算子缺失:', missing, '——请升级 torch_npu/CANN 后重试')
    raise SystemExit(1)
print('载荷算子存在性: OK')
try:
    torch.empty(1024, dtype=torch.float16, pin_memory=True)
    print('pinned 分配: OK')
except Exception as e:
    print('pinned 分配失败（MetaBlob 热路径依赖 pinned allocator）:', e)
    raise SystemExit(1)
EOF
if [ $? -ne 0 ]; then
  echo "环境预检失败，终止" >&2
  exit 1
fi

# 步骤 1/4：eager 数值回归 —— 判据只看 R1 cold 轮（TND 路径正确性）；warm 轮为
# prefix-hit prefill，触达已知限制 ⑥（310P 预期以 TND 形状报错终止，非回归）。
# 结构口径：310P 为 fp16，与 CUDA bf16 基线不逐 token 位等，看 PASS 判据不看具体 token。
echo "---- 步骤 1/4: eager 数值回归（R1 cold 判据；warm 轮预期命中已知限制 ⑥）----"
$PY -m scripts.check_prefix_cache "$TARGET" --backend ascend310p 2>&1 | tee "$OUT/prefix_eager.log"

# 步骤 2/4：decode 图捕获尝试（NPUGraph）—— 成功则 graph==eager 跨路一致 + TTFT 对照；
# 失败则打印「decode 图捕获失败，降级为 eager」= 与步骤 1 等价
echo "---- 步骤 2/4: decode 图捕获尝试（NPUGraph）----"
$PY -m scripts.check_prefix_cache "$TARGET" --backend ascend310p --graph 2>&1 | tee "$OUT/prefix_graph.log"

# 步骤 3/4：spec 拒绝守卫验证 —— 期待引擎层「清晰报错 + 干净退出」（见注意 ④）；
# 出现裸 traceback（未走守卫）或意外成功均 FAIL
echo "---- 步骤 3/4: spec 拒绝守卫验证 ----"
NANOVLLM_SPEC_GRAPH=0 $PY -m scripts.check_eagle3 "$TARGET" --draft "$DRAFT" --tag ascend310p_spec_guard > "$OUT/spec_guard.log" 2>&1
RC=$?
if [ $RC -ne 0 ] && grep -q "暂不支持 spec" "$OUT/spec_guard.log"; then
  echo "spec 拒绝守卫: PASS（引擎层清晰拒绝，exit=$RC）"
  grep -m1 "暂不支持 spec" "$OUT/spec_guard.log"
else
  echo "spec 拒绝守卫: FAIL——期待守卫报错（exit=$RC）；日志尾部：" >&2
  tail -15 "$OUT/spec_guard.log" >&2
fi

# 步骤 4/4：TPS sweep —— decode 吞吐随 batch 扫描（1/2/4/8）
echo "---- 步骤 4/4: TPS sweep ----"
$PY -m scripts.benchmark graph "$TARGET" --batch 1 2 4 8 --backend ascend310p 2>&1 | tee "$OUT/bench_sweep.log"

echo "==== 验证完成，日志与结果输出目录: $OUT ===="
