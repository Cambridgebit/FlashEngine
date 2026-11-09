# 310P 综合基准报告（2026-10-16）

13 组模型/草稿组合的实机基准。协议与复现命令见 [`ASCEND310P_BENCH.md`](ASCEND310P_BENCH.md)；
数据源日志：板卡 `/data2/mega_bench.log`、`/data2/mega_bench_30b_rerun.log`。

## 口径

- `scripts.benchmark`（与 vllm bench-serve 对齐）：3 题（seed=0 洗牌）、64 token/题、warmup 1、
  贪心（temp=0）、think-off、无前缀缓存、单序列（`--max-num-seqs 1`）
- 指标：**稳态解码 TPS**（排除 TTFT/prefill）、**E**（投机接受长度）、**TTFT**（进程内，
  mean/median，不含排队）、**TPOT**（首 token 后每 token 时延）、agg = Output token throughput（含 prefill）
- 环境：310P / CANN 9.0.T3 / torch_npu 2.10.0.post4；代码 `f7cd8f9`。
  Qwen3.5 系 = t5 venv（transformers 5.16 + `FLASHENGINE_VLLM_GEMM=0`）；Qwen3/Coder 系 = 系统 python
- 密度协议：8B/4B/9B/W8A8-8B = `mml2048 / util0.5`；30B/Coder/W8A8-30B = `mml1024 / util0.85`

## 结果总表

| # | 模型 | 模式 | 稳态 TPS | 加速比 | E | TTFT med (ms) | TTFT mean (ms) | TPOT med (ms) | agg TPS |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Qwen3.5-4B | decode | 15.8 | 1× | — | 784 | 783 | 63.1 | 13.4 |
| 2 | Qwen3.5-4B + DFlash(k7) | spec | **32.8** | **2.08×** | 4.283 | 776 | 773 | 31.8 | 23.7 |
| 3 | Qwen3.5-9B | decode | 9.5 | 1× | — | 770 | 769 | 105.3 | 8.7 |
| 4 | Qwen3.5-9B + DFlash(k15) | spec | **22.5** | **2.37×** | **5.100** | 935 | 936 | 46.7 | 17.1 |
| 5 | Qwen3-8B | decode | 10.9 | 1× | — | 175 | 173 | 91.7 | 10.8 |
| 6 | Qwen3-8B + DFlash(k15) | spec | **37.6** | **3.45×** | 4.878 | 183 | 183 | 29.2 | 34.3 |
| 7 | Qwen3-8B + DSpark(k7) | spec | **40.9** | **3.75×** | 5.289 | 180 | 179 | 25.1 | 37.2 |
| 8 | Qwen3-8B-W8A8 | decode | 15.4 | 1.41×† | — | 233 | 233 | 64.8 | 14.8 |
| 9 | Qwen3-30B-A3B | decode | 22.2‡ | 1× | — | — | — | — | — |
| 10 | Qwen3-30B-A3B + DFlash(k7) | spec | **26.4** | **1.19×‡** | 3.328 | 331 | 324 | 37.9 | 23.6 |
| 11 | Coder-30B fp16 | decode | 22.4 | 1× | — | 290 | 291 | 44.7 | 20.6 |
| 12 | Coder-30B + Coder-DFlash(k15) | spec | **38.6** | **1.72×** | **6.839** | 307 | 310 | 23.8 | 32.9 |
| 13 | Coder-30B-w8a8 | decode | 30.4 | 1.36×† | — | 336 | 336 | 32.9 | 26.6 |

† 相对同模型 fp16 的量化增益（8B：15.4/10.9；Coder：30.4/22.4）。
‡ 第 9 组本次撞上游离慢态（6.7 / 16.0，TTFT 亦膨胀），表中取当日上午同代码复现值 22.2；
加速比按 26.4/22.2 计。判读规则见 [ASCEND310P_BENCH.md](ASCEND310P_BENCH.md)。

## 分模型结论

**Qwen3-8B 系（单序列最快组合）**
- DSpark(k7) **40.9** > DFlash(k15) 37.6 —— 8B 上 DSpark 全线更好（E 5.289 vs 4.878）；
  两者 TTFT 与无投机几乎相同（~180ms vs 175ms）
- 8B 的行边际便宜（0.9ms/行），甜点 k 在 15（k7 反而掉到 33.4）
- 量化版（W8A8）**15.4**（1.41×）：本报告里 TTFT 唯一明显抬升的稠密模型（233 vs 175ms，量化算子开销），
  decode 收益显著；w8a8 暂不支持投机（E≈1.0，见已知问题）

**Qwen3.5 系（GDN 线性注意力）**
- **2026-10-30 复测（含草稿注意力窄列宽优化）见下方专节**：4B 最优 k=7 为 **34.6 tok/s**（2.18×），
  9B 最优 k=10 为 **24.9 tok/s**（2.62×）；下表为 10-16 基线，保留作对照。
- 4B：15.8 → +DFlash(k7) **32.8**（2.08×，E=4.283）；注：10-15 分页草稿路径时代为 36.8，
  滑窗草稿语义修正后走稠密掩码路径（-11%）
- 9B：9.5 → +DFlash(k15) **22.5**（2.37×，**E=5.100** 为 Qwen3.5 系最高）
- **两者的 TTFT 都在 770-935ms**（远高于纯注意力模型）——瓶颈是 310P 上 GDN 的
  prefill 路径（`chunk_gated_delta_rule` 纯 torch fp32 分块推进）；9B 带草稿再 +160ms。
  这是后续可优化的大头（TTFT 与 8B 差 ~600ms）

**30B / Coder 系（MoE）**
- 30B-A3B：无投机 22.2（慢态抽签注意）→ +DFlash(k7) 26.4（1.19×，E=3.328）
- Coder-30B：22.4 → +Coder-DFlash(k15) **38.6**（1.72×，**E=6.839 全场最高**）
  —— 全 full-attention 草稿 + k15 的完美组合；TTFT 仅 +17ms（307 vs 290）
- Coder-w8a8：**30.4**（1.36×）—— MoE 的量化增益（int8 专家）与稠密模型量级一致

**横向对比**
- 投机加速比与草稿类型强相关：Coder-DFlash 6.8 / 8B-DSpark 5.3 / 9B-DFlash 5.1 / 8B-DFlash 4.9 / 4B-DFlash 4.3 / 30B-DFlash 3.3
- 单序列吞吐排序：8B+DSpark 40.9 > Coder+DFlash 38.6 > 8B+DFlash 37.6 > 4B+DFlash 32.8 > Coder-w8a8 30.4 > 30B+DFlash 26.4 > 9B+DFlash 22.5 > Coder fp16 22.4 无投机

## 已知问题与判读提醒

1. **30B 游离慢态**：同命令同代码的进程级抽签（本次午后 6.7/16.0，上午 22.2/26.4 正常）。
   ≥2× 离群先复测；持续慢态建议重启板卡。
2. **W8A8 模型暂不支持投机**：8B-W8A8 / Coder-w8a8 配 DFlash 时 E≈1.0（零接受），
   已在 AGENTS.md 记录待查；本表 w8a8 行均为无投机。
3. **W8A8 MoE 大 prefill 崩**：`npu_quant_grouped_matmul_dequant(pertoken)` 在 310P 上
   token 数大时触发 `QuantBatchMatmulV3 x1 NZ` aicore 异常——w8a8 MoE 目前仅
   `--max-num-seqs 1` 串行可用（本表口径满足）。
4. **Qwen3.5 的 TTFT（770-935ms）** 是 GDN prefill 的算子路径成本，不是配置问题。
5. 常态复测建议用当日上午的 30B 值（22.2/26.4）作对照；本表其余组当天均为稳定样本。

## 合并后回归验证（2026-10-28，板卡 192.168.16.118 / 容器 vllm018）

把 FP16 统一版（`nano_flashengine_fp16_1_0`）功能并入本仓后，按上表同口径复测 Qwen3-8B，
确认既有能力未退化。代码 = 本地合并工作区（HEAD `ea3543f` + 本轮新增模块）；
环境与上表一致（310P / CANN 9.0.T3 / torch_npu 2.9.0.post2-dev / t5 venv）。
日志：板卡 `/data2/logs/{bench_base_8b,bench_spec_8b,bench_eager_8b,switch_matrix2,batch_prefix,server_test,service_mgr}.log`。

| 项 | 本轮实测 | 上表/锚点 | 判定 |
|---|---:|---:|---|
| Qwen3-8B decode（图） | 10.7–11.0 | 10.9 | 一致 |
| Qwen3-8B + DFlash(k15) | 38.3 / **E=4.878** | 37.6 / E=4.878 | E 逐位相同，TPS +1.9% |
| Qwen3-8B decode（`--eager`） | 7.4 | — | 图相对 eager **1.46×** |
| 批量聚合 bs=1/4/8（agg） | 10.7 / 41.3 / 80.8 | — | 线性扩展正常 |
| Qwen3-8B-W8A8 decode | 15.4 | 15.4 / 15.5 | 一致（头号特性 W8A8 长跑修复未回退） |

**新增开关矩阵**（Qwen3-8B decode，图路径，`LITEQ_DECODE_LINEAR_BACKEND` × `LITEQ_GREEDY_BACKEND`）：
四档 `functional|selective` × `npu|cpu` 分别 **11.0 / 10.8 / 10.6 / 10.5** tok/s，
`captures_ok=1, fallback=0, replay_rate=100%` —— 即 selective LINEAR 与 CPU 端 greedy 归约
都能正常进图，且图数值守卫（`FLASHENGINE_GRAPH_NUMERIC_GUARD` 默认开）未把任何桶判失效。
差异 ≤4.5%，属开关自身开销与噪声，无功能退化。

**其余链路**（同一代码、同一容器）：

- 精度守卫：`require_unquantized_checkpoint` 对 `/data/Qwen3-8B-W8A8` 与
  `/data/Qwen3-Coder-30B-A3B-Instruct-w8a8` 正确拒绝（识别依据 `quant_model_description.json`），
  对 `/data/Qwen3-8B`、`/data/Qwen3-8B-Dflash` 放行。
- HTTP：`/healthy`、`/ready`（`{status,pid,service,version}`）、`/v1/runtime`（含新增
  `precision` / `linear_validation` 段）、`/v1/stats`、`/metrics`、`/v1/chat/completions` 全 200；
  旧别名 `/runtime` 正确 404。投机服务端 `graph_mode=spec_graph`、`spec.method=dflash`、
  `spec.num_tokens=15`，单请求 `path=spec_graph` E=2.32 / 19.1 tok/s（80 token 短答，含 prefill）。
- 服务托管：`start` → `FRAMEWORK_SERVICE_READY` → `status running=true ready=true` →
  `stop` → `FRAMEWORK_SERVICE_STOPPED` → `status running=false`，全链路 0 退出。
  注意 `stop` 只管理 `start` 自己登记的进程（精确核对 argv/cwd/start_ticks），
  手工用别的包装脚本起 `serve` 后 `stop` 报 `FRAMEWORK_SERVICE_NOT_RUNNING` 属预期，
  需自行终止。
- Prefix cache：`--num-repeats 2` 的 ON/OFF 均正常（TTFT cold 177/181 ms）；
  本题集 prompt 仅 ~48 token < block_size 128，不命中属预期。

**判读提醒**：`configs/*.json` 里的 `model` 是占位路径 `/models/qwen3-8b`，板卡实际模型在
`/data`，**必须用 `--model /data/Qwen3-8B` 覆盖**（`config` 子命令只回显不校验路径）。

## Qwen3.5 系列全量复测（2026-10-30，含草稿注意力窄列宽优化）

本轮把「草稿注意力按实际上下文缩窄列宽」合入 `flashengine/engine/model_runner.py`
（默认开，`FLASHENGINE_DFLASH_NARROW_CTX=0` 关闭），并按上表同口径重测 Qwen3.5-4B / 9B 的
普通解码与 DFlash 各档草稿长度。板卡 `/data2/FlashEngine_integrated` 的
`model_runner.py` 与本地合入文件 SHA-256 一致（`dc7ae2de…`）。

口径：310P / `vllm018` / t5 venv（transformers 5.16、`FLASHENGINE_VLLM_GEMM=0`）、
`scripts.benchmark` 3 题（seed=0 洗牌）、64 token/题、warmup 1、贪心、think-off、
无前缀缓存、`--max-model-len 2048 --max-num-seqs 1 --gpu-mem-util 0.5`。
指标同前：稳态解码 TPS（排除 TTFT/prefill）、E、TTFT、TPOT、agg = 含 prefill 的输出吞吐。

### Qwen3.5-4B（普通 + DFlash 全 k 扫描）

| 模式 | 稳态 TPS | 加速比 | E | TTFT med (ms) | TTFT mean (ms) | TPOT med (ms) | agg TPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 普通 decode | 15.9 | 1× | — | 759 | 758 | 63.0 | 13.5 |
| DFlash k=2 | 25.5 | 1.60× | 2.595 | 785 | 784 | 39.7 | 19.6 |
| DFlash k=5 | 33.1 | 2.08× | 3.863 | 769 | 766 | 28.4 | 23.9 |
| **DFlash k=7** | **34.6** | **2.18×** | 4.283 | 790 | 787 | 30.2 | 24.5 |
| DFlash k=10 | 32.9 | 2.07× | 4.545 | 778 | 776 | 33.1 | 23.7 |
| DFlash k=15 | 29.5 | 1.86× | 4.767 | 781 | 784 | 37.8 | 21.9 |

### Qwen3.5-9B（普通 + DFlash 全 k 扫描）

| 模式 | 稳态 TPS | 加速比 | E | TTFT med (ms) | TTFT mean (ms) | TPOT med (ms) | agg TPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| 普通 decode | 9.5 | 1× | — | 773 | 774 | 105.6 | 8.6 |
| DFlash k=2 | 17.0 | 1.79× | 2.694 | 792 | 794 | 61.3 | 14.2 |
| DFlash k=5 | 23.8 | 2.51× | 4.021 | 775 | 774 | 44.7 | 18.7 |
| DFlash k=7 | 23.7 | 2.49× | 4.370 | 796 | 798 | 44.1 | 18.5 |
| **DFlash k=10** | **24.9** | **2.62×** | 5.000 | 832 | 830 | 41.1 | 19.0 |
| DFlash k=15 | 23.1 | 2.43× | 5.100 | 854 | 853 | 45.6 | 17.8 |

### 结论与判读

- **甜点 k 因模型而异**：4B 仍是 k=7 最快（34.6，2.18×），与历史一致；**9B 在本负载下
  k=10 最快（24.9，2.62×）**，k=15 只有 23.1。E 随 k 单调上升（9B k15 的 E=5.100 最高），
  但每轮验证成本上升更快，E 最高不等于吞吐最高。业务上线应按自身负载扫 k。
- 全部 12 组均为 `captures_ok=2`、`replay_rate=100%`、`eager_rounds=0`、`sampling_fallbacks=0`：
  `captures_ok=2` 是窄列宽优化多捕获的一张 256 列图，短序列走它、长序列回落整宽图。
- **与 2026-10-16 表对比**：4B +DFlash(k7) 32.8 → **34.6**（+5.5%）；9B +DFlash(k15)
  22.5 → **23.1**（+2.7%）。差异与 `docs/QWEN35_OPT.md` 里三组配对 A/B 的中位 +5.5% 一致，
  9B 绝对增量小于 4B，因为其每轮时间更多被权重读取占据。
- TTFT 仍在 759–854 ms：本轮改动只影响 DFlash 轮内的草稿注意力，**不改善 GDN 预填充**，
  TTFT 由 `chunk_gated_delta_rule` 参考路径主导（见 `docs/QWEN35_OPT.md`）。
- **为什么 Qwen3.5 的 DFlash 加速比低于 Qwen3 系**（3.45× vs 2.2–2.6×）：主因是 GDN 多行验证
  必须逐行推进递归状态、状态读写随行数线性增长；310P 缺融合行核与窗口草稿注意力进一步放大。
  分解与公开证据见 [`QWEN35_OPT.md`](QWEN35_OPT.md#为什么-qwen35-的-dflash-加速比低于-qwen3架构-vs-310p)。
- 边界：单序列、无前缀缓存、64 token 输出；窄列宽只对 ≤256 token 的序列生效（更长序列回落，
  无收益也不受损）；同会话内重复测量有约 ±1% 波动，跨会话数字不宜直接比较。
  板卡上**没有 Qwen3.5 MoE 检查点**，因此本表不含 Qwen3.5 MoE；Qwen3 系（8B/30B/Coder）
  与量化结果见上表，本轮未重测。

复现（板卡容器内）：

```bash
cd /data2/FlashEngine_integrated
bash /data2/fe_py_t5.sh -m scripts.benchmark /data/Qwen3.5-4B \
  --backend ascend310p --no-think --gpu-mem-util 0.5 --max-model-len 2048 \
  --max-num-seqs 1 --warmup 1 --num-prompts 3 --max-tokens 64 \
  --questions bench_data/question.jsonl --no-prefix-cache \
  --spec-model /data/Qwen3.5-4B-Dflash --spec-num-tokens 7
```

9B 把模型/草稿换成 `/data/Qwen3.5-9B`、`/data/Qwen3.5-9B-DFlash`，`--spec-num-tokens` 取 2/5/7/10/15。
原始日志：板卡 `/data2/q35opt/full_matrix/{4b,9b}_{base,k2,k5,k7,k10,k15}.log`。

**同步方式与完整性**：板卡仓库（`/data2/FlashEngine`）的 `main` 与本机 `main` 是两条独立提交
谱系（板卡 HEAD `ce94022`「Qwen3.5 及 DFlash 前缀缓存」，无本地 `ea3543f` 对象），但两条谱系
的**工作区文件内容逐字节等价**——15 个被覆盖文件 `diff -w` 全为 0 行，差异纯粹是 CRLF/LF
换行（板卡文件由 Windows `docker cp` 带入，带 CRLF）。本次同步把 CRLF 归一为 LF，
板卡 `git status` 因此显示 66 项改动（原 51 项 + 新增模块/文档），**无内容丢失**。
同步前完整备份：板卡 `/data2/fe_board_backup_0119_0410.tar.gz`。
