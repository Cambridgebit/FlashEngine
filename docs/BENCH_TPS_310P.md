# 数据集 TPS · Atlas 300I Duo (310P)

> **状态**（2026-11-24）：在仓库自带 `bench_data/` 上跑六配置 × 中英数据集 + Qwen3-1.7B-ASR 的单并发实测。口径 = 每数据集前 10 条 `category=coding`、`max_tokens 256`、贪心、`--no-think`、单并发串行；聚合 TPS = Σ输出 token / 主跑墙钟（不含 1 条预热）。**这是 dataset TPS，不是 4K 长文档场景的延迟画像**（后者见 [BENCH_BS1_310P.md](BENCH_BS1_310P.md)）。

## 环境

- 板卡 `192.168.16.118` 容器 `vllm018`，`/data2/fe-structured`，venv `/data2/fe-t5-venv/bin/python3`。
- 服务：`--backend ascend310p --max-model-len 40960 --max-num-seqs 8 --gpu-mem-util 0.70 --block-size 16 --no-think`，`FLASHENGINE_VLLM_GEMM=1`。
- 数据集：`bench_data/question.jsonl`（英文 Spec-Bench coding）与 `bench_data/Zh_question.jsonl`（中文 coding），各取前 10 条 `turns[0]`，prompt 长度 100–200 token 量级（所以 TTFT 只有 0.15–0.8 s，别与长文档场景混比）。
- 客户端：`scratch/q35opt/bench_jsonl.py`（板上 `/data/hostmem/bench_jsonl.py`），流式 + `include_usage`；原始日志 `/data/hostmem/bench_tps.log`、服务端 `/data/lora/tps_<cfg>.log`。

## 结果

| 配置 | 执行路径 | 聚合 TPS（英） | 聚合 TPS（中） | TTFT 中位（英） | TTFT 中位（中） | E（英） | E（中） |
|---|---|---|---|---|---|---|---|
| Qwen3.5-9B | 普通图（decode_graph） | 9.19 | 9.16 | 592 | 743 | — | — |
| Qwen3.5-9B + DFlash (k=7) | spec_graph | 26.97 | 17.04 | 584 | 732 | 4.77 | 2.95 |
| Qwen3.5-9B + LoRA (4 槽) | decode_graph | 8.14 | 8.12 | 631 | 779 | — | — |
| Qwen3.5-9B + LoRA + DFlash (k=7) | spec_graph | 24.46 | 15.46 | 619 | 753 | 4.77 | 2.95 |
| Qwen3-8B | decode_graph | 10.55 | 10.67 | 155 | 148 | — | — |
| Qwen3-8B + DFlash (k=15) | spec_graph | 37.13 | 18.96 | 160 | 159 | 5.48 | 2.79 |

- 每配置 20 个计时请求（英 10 + 中 10），输出 token：英 2,505、中 2,560（`max_tokens 256` 下多数请求跑到上限）。
- DFlash 相对基座：9B 英 **2.94×** / 中 **1.86×**；LoRA 组合 **3.00× / 1.90×**；8B 英 **3.52×** / 中 **1.78×**。英文收益更高是因为接受长度更高（英 E≈4.8–5.5，中 E≈2.8–3.0），与上一轮 4K 长文档压测的方向一致。

### Qwen3-1.7B-ASR（离线 `scripts/asr_benchmark`，同一 310P）

| 项 | 值 |
|---|---|
| 样本 | `bench_data/asr/asr_en.wav`（15.05 s）、`asr_zh.wav`（4.20 s） |
| TTFT | mean 196.2 ms（p95 202.8 ms） |
| RTF | 聚合 0.0929（每请求均值 9.6%） |
| TPS | 聚合 **32.44**（每请求均值 29.37） |
| CER | 0.0%（两条都有参考，逐条 0.0%） |
| 生成 | 英 48 token / 中 10 token，共 58 |
| 关键参数 | `--max-model-len 2048 --max-num-seqs 1 --block-size 16 --gpu-mem-util 0.70`，fp16，图模式（`Graph capture: 0.929 s`，`Engine ready in 27.189 s`） |

## 注意事项

- **LoRA 槽是同一份权重且 `lora_alpha=0`**：`/data/lora/adapters/a0..a3` 全部符号链接到 `/data/lora/Qwen3.5-9B-lora-642/adapter_model.safetensors`，`r=8`、`alpha=0` → 缩放系数 0，适配器对输出**零贡献**。所以 LoRA 两行只证明「多槽加载 + 路由 + 运行时开销」：相对基座 −11%（9.19 → 8.14），并且 LoRA+DFlash 的 E 序列与纯 DFlash 逐项相同（同一 token 流）。要评估真实 LoRA 效果需换 `alpha>0` 的适配器权重。
- **DFlash 与 LoRA+DFlash 的 TPS 差**（26.97 vs 24.46 英）就是 LoRA 运行时开销，不是接受率差异（E 相同）。
- **ASR 必须 `source /usr/local/Ascend/ascend-toolkit/set_env.sh` 或在非登录 shell 里跑**：`docker exec … bash -lc` 会重置 `LD_LIBRARY_PATH`，实测 `RuntimeError: … aclInit, error code is 507008`；同样命令去掉 `-l` 就正常。
- 中文比英文慢的主因是接受长度低（E 2.8–3.0 vs 4.8–5.5），不是 prefill；两语言 prompt 都在 100–200 token，prefill 差异可忽略。
- 与 `scripts/benchmark` 的对表：本表用 HTTP 客户端，比进程内 benchmark 多一次协议税（实测 ~1–4%），跨表比较请留余量。
