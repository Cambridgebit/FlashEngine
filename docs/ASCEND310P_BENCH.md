# 310P 板卡测试配置速查（2026-10-16 整理）

本文件汇总 310P 实机的常规测试配置，用于自测复现（历史一次性脚本已于 2026-10-16 清理，
日志保留在板卡 `/data2/*.log`）。方法与指标口径见 [`AGENTS.md`](../AGENTS.md) 与
[`docs/QWEN35.md`](QWEN35.md)；支持清单见 [`docs/SUPPORT_MATRIX.md`](SUPPORT_MATRIX.md)。

## 运行环境（板卡）

- 操作链：`ssh ascend` → 容器 `vllm018` 内运行；仓库在 `/data2/FlashEngine`，模型在 `/data`。
- 两个保留的运行器（封装了 CANN 环境/工作目录/离线开关）：
  - **t5 venv 用例**（Qwen3.5，需 transformers≥5.16）：`bash /data2/fe_py_t5.sh <参数...>`
    （等价手写：`source /usr/local/Ascend/ascend-toolkit/set_env.sh` +
    `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FLASHENGINE_VLLM_GEMM=0` +
    `/data2/fe-t5-venv/bin/python3 -B -u ...`，cwd=`/data2/FlashEngine`）
  - **系统 python 用例**（Qwen3/30B）：`bash /data2/fe_py.sh <参数...>`
    （python3 = `/usr/local/python3.11.10/bin/python3`，无需离线开关与 GEMM=0）
- venv 重建（如需）：`bash /data2/fe_install_t5.sh`（离线 wheelhouse）。

## 性能基准（`scripts.benchmark`，稳态解码 TPS + E）

通用口径：单序列、温度 0、`--no-think`、无前缀缓存、3 题（seed=0 洗牌）、64 token/题、`--warmup 1`。
指标 = Steady-state decode throughput（排除 TTFT/prefill）+ E（接受长度）。

### Qwen3-8B + DFlash

```bash
BASE8B="--backend ascend310p --no-think --gpu-mem-util 0.5 --max-model-len 2048 \
  --max-num-seqs 1 --warmup 1 --num-prompts 3 --max-tokens 64 \
  --questions bench_data/question.jsonl --no-prefix-cache"
cd /data2/FlashEngine
bash /data2/fe_py_t5.sh -m scripts.benchmark /data/Qwen3-8B $BASE8B
bash /data2/fe_py_t5.sh -m scripts.benchmark /data/Qwen3-8B $BASE8B --spec-model /data/Qwen3-8B-Dflash
```

2026-10-16 实测：无投机 **10.7**；+DFlash（k15 默认）**39.0**，E=4.878。
k7 反而慢（33.4，E=4.229）——8B 保持默认 k15（行边际便宜，长 k 划算）。

### Qwen3.5-4B + DFlash

同 `$BASE8B`：

```bash
bash /data2/fe_py_t5.sh -m scripts.benchmark /data/Qwen3.5-4B $BASE8B
bash /data2/fe_py_t5.sh -m scripts.benchmark /data/Qwen3.5-4B $BASE8B \
  --spec-model /data/Qwen3.5-4B-Dflash --spec-num-tokens 7
```

2026-10-16 实测：无投机 **15.9**；+DFlash k7 **32.8~32.9**，E=4.283
（k 曲线：k5 31.3 / k7 32.8 / k10 31.3 / k15 28.6）。

注意：**`--spec-num-tokens 3/4` 视为「未显式指定」**（实际跑默认 15）。
4B 草稿是滑窗草稿（5×sliding + 1×full），在 310P 走稠密掩码草稿路径
（分页草稿算子只支持全局非因果），比 2026-10-15 的分页路径（36.8）慢约 11%。

### Qwen3-30B-A3B + DFlash

```bash
BASE30B="--backend ascend310p --no-think --gpu-mem-util 0.85 --max-model-len 1024 \
  --max-num-seqs 1 --warmup 1 --num-prompts 3 --max-tokens 64 \
  --questions bench_data/question.jsonl --no-prefix-cache"
bash /data2/fe_py.sh -m scripts.benchmark /data/Qwen3-30B-A3B $BASE30B
bash /data2/fe_py.sh -m scripts.benchmark /data/Qwen3-30B-A3B $BASE30B \
  --spec-model /data/Qwen3-30B-A3B-Dflash --spec-num-tokens 7
```

2026-10-16 实测：无投机 **22.2**；+DFlash k7 **26.3~26.6**，E=3.328
（k2：18.2~26.5 波动，E=2.244）。装载约 5 分钟（61GB fp16）。

判读：**出现 ~2× 离群先复测再记录**（游离慢态，同命令同代码可复现正常值；
E 逐位不变可作对照）；持续慢态建议重启板卡后再测。

### W8A8 量化模型（msmodelslim 导出，2026-10-16 首测）

```bash
# 串行（同 BASE30B 口径，系统 python）
bash /data2/fe_py.sh -m scripts.benchmark /data/Qwen3-8B-W8A8 $BASE30B
bash /data2/fe_py.sh -m scripts.benchmark /data/Qwen3-Coder-30B-A3B-Instruct-w8a8 $BASE30B
# 批量聚合（含 prefill；工具脚本 /data2/fe_batch_bench.py 已保留）
bash /data2/fe_py.sh /data2/fe_batch_bench.py --model /data/Qwen3-8B-W8A8 \
  --gpu-mem-util 0.5 --max-model-len 2048
```

实测：**Qwen3-8B-W8A8 串行 15.5**（fp16 10.7 → **1.45×**）、批量 bs1/4/8 = 14.8/58.6/**100.7** agg；
**Coder-30B-w8a8 串行 30.3**（fp16 19.0 → **1.59×**）。

⚠️ **已知问题（未修）**：W8A8 **MoE** 在 token 数大时崩——`op[QuantBatchMatmulV3], Input x1
format should be ND, actual format is FRACTAL_NZ` → aicore 507015（异步不可恢复）；
引擎预热的多序列 prefill（如 8×1024）必触发，`max_num_seqs=1` 串行可用——
Coder 的批量档因此缺失。排查与尝试过的规避见 AGENTS.md「310P W8A8 实机首测」。

## 正确性验收（check_qwen35 四用例，严格逐 token 一致）

```bash
bash /data2/fe_py_t5.sh -m scripts.check_qwen35 /data/Qwen3.5-4B \
  --draft /data/Qwen3.5-4B-Dflash --backend ascend310p \
  --cases base_eager,base_graph,spec_eager,spec_graph \
  --max-tokens 32 --output /data2/qwen35_check_out
```

判别：`base_eager` 为基准，其余三行均须「严格一致 5/5」。

## 常量备忘

- 草稿默认 k（配置派生）：8B=15、4B=15（block16）、30B=7（block8，speculators `speculative_tokens=7`）。
- 模型目录：`/data/Qwen3-8B(-Dflash)`、`/data/Qwen3.5-4B(-Dflash)`、`/data/Qwen3-30B-A3B(-Dflash)`；
  W8A8：`/data/Qwen3-8B-W8A8`、`/data/Qwen3-Coder-30B-A3B-Instruct(-w8a8)`、`/data/Qwen3-Coder-30B-A3B-DFlash`。
- 服务端一键入口：`bash /data2/run_qwen35_server.sh`（Qwen3.5-4B+DFlash，`:8000`；
  `FE_PORT/FE_MAX_SEQS/FE_MML` 可覆盖）。
- 板卡保留脚本：`run_qwen35_server.sh`、`fe_py.sh`、`fe_py_t5.sh`、`fe_install_t5.sh`、`fe_batch_bench.py`（批量聚合对照工具）。
