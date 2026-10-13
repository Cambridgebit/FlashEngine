# 八组合并发压测与趋势图

两个项目均提供同一份 `scripts/bench_matrix.py`，从各自仓库根目录运行。目标模型、三个草稿均通过参数指定本地目录；使用对应硬件上已经安装好的 FlashEngine 运行环境。

默认矩阵为 **base / EAGLE3 / DFlash / DSpark × prefix-cache 关闭 / 开启 × batch 1 / 2 / 4 / 8**，每后端 32 个数据点。base 表示无投机解码。完整八组合使用支持三种草稿的 Qwen3 目标模型，各草稿必须匹配同一个目标。0924 中 Qwen3.5 等不支持某组合的模型会记录失败，不能代替完整 Qwen3 矩阵。

## 完整运行

```bash
python3 -m scripts.bench_matrix \
  --model /models/Qwen3-8B \
  --eagle3-model /models/Qwen3-8B-EAGLE3 \
  --dflash-model /models/Qwen3-8B-DFlash \
  --dspark-model /models/dspark_qwen3_8b_block7 \
  --backends cuda \
  --batch-sizes 1,2,4,8 \
  --questions bench_data/question.jsonl --category coding \
  --num-prompts 16 --repeats 3 --warmup 2 \
  --shared-prefix-tokens 512 --max-tokens 128 \
  --max-model-len 2048 --gpu-mem-util 0.85 \
  --output bench_results/cuda
```

CoreX 机器将后端改为 `--backends corex`，310P 机器改为 `--backends ascend310p`；也可用 `auto` 自动选择本机后端。只有在本机具备相应硬件及环境时才能运行指定后端；通常在三种硬件上分别运行，然后合并结果。每个组合在独立子进程中启动引擎，按顺序测四个 batch，默认组合间停 5 秒释放设备资源与冷却。各档 `max_num_seqs` 固定为最大 batch（默认 8），避免同时改变引擎容量造成比较混淆。若 GPU 空闲→升频或温度降频影响结果，应增加预热/冷却并重复测量。

可先仅验证参数与计划，不加载模型：

```bash
python3 -m scripts.bench_matrix \
  --model /models/Qwen3-8B \
  --eagle3-model /models/Qwen3-8B-EAGLE3 \
  --dflash-model /models/Qwen3-8B-DFlash \
  --dspark-model /models/dspark_qwen3_8b_block7 \
  --backends cuda --dry-run --output bench_results/plan
```

只测无投机解码时用 `--methods base`，仅测部分方法用 `--methods base,eagle3` 并传入所选草稿。缺少默认选中的草稿参数会明确报错，不静默跳过。`--eager` 关闭图；默认尝试图，报告记录实际 graph 模式、捕获成功/失败与主跑 replay/fallback 差值。`--spec-num-tokens 3` 默认控制 EAGLE3 的 k；DFlash/DSpark 的 3/4 按两仓现有规则自动派生，实际 k 记入 JSON。

## 指标与缓存负载

- 所有请求同批入引擎队列，用 `step()` 驱动到该批结束；下一批再入队。没有 HTTP 服务依赖，也不包含 HTTP/SSE 传输或 detokenization 成本。不能与 `bench_concurrent` 的 HTTP 数字混为同口径。
- **TTFT**：该批开始到每个请求第一次提交输出 token 的时间，包含引擎排队及 prefill。图中默认 p50，JSON/CSV 同时保留 p95。
- **TPOT**：每个请求 `(末次 token 提交时间 - 首次 token 提交时间) / (输出 token 数 - 1)`，再汇总 p50/p95。请求先结束时用它自己的末次提交时间；投机解码一次提交多个 token 仍按实际 token 数计分母。输出不足两个 token 时无定义，存为 null，图中断线。
- **TPS**：所有主跑批次的输出 token 总数 / 主跑批次墙钟总和，是聚合输出吞吐，包含 prefill。预先 tokenization、模型初始化、预热、缓存 seed、日志、出图与冷却均不计入。
- 默认 greedy、thinking 关闭、忽略 EOS，固定输出 128 token，便于比较相同 batch 下的趋势。`--thinking` 使用模板默认 thinking；`--respect-eos` 允许 EOS 提前结束，此时实际批大小可能下降。计数包含引擎提交的全部 token，包括显式允许的 EOS。
- 每个主跑批次先运行一次相同 system 前缀、不同 user 文本的 seed 请求，缓存开/关两边都做，seed 不计指标。共享 system 内容通过目标 tokenizer 扩充至 `--shared-prefix-tokens` 的最小长度；用实际完整模板 token 流计算共同满块数。每组前缀开头含不同组号，避免跨 batch/重复的缓存残留，也避免把整条问题的重复命中当成共享 system 缓存收益。
- 记录主跑 `prefix_hit_requests`、`prefix_hit_blocks`、`shared_blocks_min`。CUDA 的 KV block 通常为 256，310P 默认 128；默认 512 token 共享前缀能覆盖整块。前缀过短直接报错；开启缓存却没有命中会警告。后端自动关闭缓存时该开启组合标记失败。
- 记录实际调度的 `prefill_batch_peak` 与 `decode_batch_peak`，以及 KV 抢占次数。decode 峰值不足目标 batch 或发生抢占时，点标为空心，附原因，不能据此宣称完成了目标 batch 性能验收。
- E = 投机轮提交 token 数 / 轮请求数（含 bonus）。统计来自 scheduler 截断前的投机返回值，最后一轮可能略高估；它用于解释低接受负载下的趋势。

主跑每档每次重复的请求数向上补齐为 batch 整数倍，题集不足时循环取题；JSON 保存实际请求数与逐请求原始计时。八组使用同一题集、相同 seed 和相同共享前缀文本。

`prompt + max_tokens + 投机末轮余量` 必须落在实际模型窗口内；超长时记录失败，需增大 `--max-model-len` 或缩短共享前缀/输出。不切渲染后的模板 token 流。显存不足时调低模型/窗口或仅跑较小 batch，例如 `--batch-sizes 1,2,4`；固定 `--max-num-seqs` 仍必须覆盖所选最大 batch。

## 结果与跨后端/项目合并

输出目录包括：

- `report.html`：离线可打开的三指标趋势图、每点状态表与运行元数据。
- `trend_*.svg`：可单独打开/导入文档的矢量趋势图，无 CDN、无需 matplotlib。
- `results.json`：所有数据点、逐请求 token 数/TTFT/TPOT、实际引擎配置、图统计、错误。
- `summary.csv`：供 Excel/Pandas 使用的汇总数据（UTF-8 BOM）。
- `plan.json`、独立任务 JSON、子进程日志与每组合的中间结果：故障诊断及中断恢复的数据来源。

每完成一个组合即更新报告；组合失败仍继续其余组合，最终有失败时退出码为 1。已有输出目录可重复使用，主报告替换为本次结果，带唯一 run_id 的任务/日志/中间数据保留；历史报告需使用不同 `--output` 目录。

合并在任何有 Python 3.10+ 的机器上运行即可，不需要 torch、GPU 或模型：

```bash
python3 -m scripts.bench_matrix \
  --merge /results/V2_cuda/results.json \
          /results/V2_corex/results.json \
          /results/V2_310p/results.json \
          /results/0924_cuda/results.json \
  --output bench_results/comparison
```

报告按后端展示八组合曲线；有多个后端/项目/运行时，还按每个功能组合画跨后端/项目曲线。同一次运行不会因为重复导入文件而重复计数；独立重复运行分别保留曲线，不偷偷平均。

不同工作负载哈希不连线。哈希来自模型目录名、config.json 哈希、实际选中的题集与测量/引擎参数，不包含项目名和绝对路径，便于跨服务器搬运。对表时保持相同模型目录名和 config、题集、参数；JSON 另外保留实际 block size、k、dtype、图路径、KV 块数、后端算子信息、Python/平台与 `FLASHENGINE_*` 环境变量。模型权重版本、硬件型号/驱动和温度仍需在测试记录中注明，模型 config 相同不代表权重完全相同。
