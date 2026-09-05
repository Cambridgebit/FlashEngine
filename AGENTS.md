# AGENTS.md

Nano-vLLM：从零实现的极简 vLLM（研究/学习用代码库，非生产项目）。原版约 1200 行 Python；加入 EAGLE3 投机解码后约 2400 行。**无测试套件、无 CI、无 lint/typecheck 配置**；唯一验证手段是在有 GPU 的机器上对本地模型目录跑 `scripts/` 下的脚本。

## 命令

脚本需要**本地模型目录**（不能用 HuggingFace hub id）和 CUDA 兼容 GPU（NVIDIA，或 Iluvatar + `corex` 后端）。本机（RTX 4070 Laptop 8GB / WSL2）实际使用 `~/vllm-env/bin/python3`（Python 3.12, torch 2.11, transformers 5.15）：

```bash
~/vllm-env/bin/python3 -m scripts.benchmark graph   /path/to/Qwen3-1.7B [--batch 1 2 4 8] [--backend corex]
~/vllm-env/bin/python3 -m scripts.benchmark prefix  /path/to/Qwen3-1.7B [--backend corex]
~/vllm-env/bin/python3 -m scripts.check_prefix_cache /path/to/Qwen3-1.7B [--graph] [--backend corex]
~/vllm-env/bin/python3 -m scripts.check_eagle3 /path/to/Qwen3-1.7B --draft /path/to/Qwen3-1.7B-EAGLE3 [--num-prompts 10]
~/vllm-env/bin/python3 scripts/openai_server.py /path/to/Qwen3-1.7B [--port 8000] [--eager]
```

- 必须以模块方式从仓库根目录运行（`-m scripts.X`），不能 `python scripts/X.py`（本机 shell 有 ROS 的 PYTHONPATH，`scripts/__init__.py` 用于避免包遮蔽）。
- 脚本只跑 GPU 且对数值敏感：计时前先预热引擎，结果逐 token 对比；`check_prefix_cache` 失败时退出码非零。
- `--eager` / `enforce_eager=True` 关闭 CUDA Graph 捕获（小显存卡推荐）；CUDA Graph 需按 batch 形状逐桶捕获，遇到未捕获形状首轮会失败。
- `tqdm` 被引擎导入但 **pyproject.toml 未声明**，全新安装后需手动装。

## 测试环境与已知的坑（WSL2 + 4070 Laptop 8GB）

- 桌面进程占用 ~0.5-1.5GB 且波动，`num_kvcache_blocks` 计算结果在 33-75 块间漂移 → 并发数和共享前缀长度必须压低（建议 ≤4 并发、≤1200 token 前缀、max_model_len 2048、util 0.95）。
- WSL 下显存回收有数秒延迟：`allocate_kv_cache` 已做预算不足时的等待重试。
- 对照基线：vLLM 0.26 需 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 才能在 WSL2 跑通；spec_bench 数据在 `~/vllm-lab/benchdata/spec_bench/question.jsonl`，模型在 `~/vllm-lab/models/`（Qwen3-1.7B、Qwen3-1.7B-EAGLE3=腾讯 AngelSlim 官方等）。
- EAGLE3 draft checkpoint 的 config.json **没有** aux 层字段 / fc_norm / norm_before_fc；aux 层用引擎默认 (2, n//2, n-3)。
- spec_bench 验收脚本已晋升为 `scripts/check_eagle3.py`（E/逐位置接受率/直方图/TPS；`--save-out`/`--tag` 输出与回归 compare 流程同 schema）；测试工件在 `~/vllm-lab/benchdata/nv_regression/`。
- 数值确定性：bf16 下目标模型自身就随 batch 形状漂移（逐个 vs 批量跑同一批 prompt 有 3/10 输出分歧）——"投机 vs 非投机逐 token 一致"只在无分歧翻转的负载上成立。

## 后端与硬件

- `nanovllm/backends/` 抽象 attention/kvcache 算子。选择方式：`backend=` 参数、`NANOVLLM_BACKEND` 环境变量、或自动探测（`torch.corex` 标记 → corex，否则 cuda）。
- `corex` 面向 Iluvatar（天数智芯）GPU，导入外部 `ixformer` 包（`ixformer.inference.functions`、`ixformer.contrib.vllm_flash_attn`），**pyproject 未声明**，只能在 Iluvatar 硬件上运行。
- 两后端差异：KV 布局（`get_kv_cache_shape`，CUDA = `[2, layers, blocks, block_size, kv_heads, head_dim]`；CoreX = HND `[2, layers, blocks, kv_heads, block_size, head_dim]`）与 **slot_mapping dtype**（`graph_slot_mapping_dtype`，CUDA int32 / CoreX int64）。`cu_seqlens`/`context_lens`/`block_tables` 两后端统一 int32。改 kvcache/attention 代码时不要假设单一布局。
- **spec 路径的硬件依赖已全部参数化**：槽位类 buffer（chain_slots/v_slots/eager 路径 slots）统一走 `backend.graph_slot_mapping_dtype()`；meta blob 统一存 int64，图内 `copy_` 自动转型。
- **spec CUDA Graph 捕获失败会优雅降级**：`capture_spec_cudagraph` 捕获异常时打印警告、释放图池、回落到同构 eager spec 轮（为 CoreX 上 ixformer kernel 可能不可捕获而设计）。
- **桶表覆盖 max_bs**：`_graph_buckets` 在标准模式（[1,2,4,8]+16 步进）未覆盖 max_num_seqs 时追加尾桶并一次性告警（非 2 幂 max_num_seqs 不再静默降级 eager）；graph padding 复位时 cu_k 尾段置零长窗口（`cu_k[n+1:] = cu_k[n]`，cu 单调不降）。
- backend 能力声明：`dist_name`（TP 初始化）与 `supports_spec_graph`（spec 图捕获放行，失败自动降级 eager spec）；decode 图捕获同样有 try/except 降级（不再崩溃）。CoreX 实机验证：`bash scripts/verify_corex.sh <target> <draft>` 五步清单。

## 架构

- 公开 API：`LLM`（继承 `LLMEngine`，`nanovllm/llm.py`）+ `SamplingParams`，对齐 vLLM 的 `LLM.generate`。支持的模型：**Qwen3**（`nanovllm/models/qwen3.py`）+ EAGLE3 draft（`nanovllm/models/eagle3.py`，官方 cnets.py 的链式推理等价实现）。
- 引擎主循环（`nanovllm/engine/llm_engine.py`）：spec 关闭时 `schedule → run → postprocess`；spec 开启时 `step_spec()`：prefill 走 `run_prefill_spec()`（target 前向捕获 aux 隐层 + draft 同 chunk prefill），decode 每步一整轮 `spec_round()` → `commit_spec()` 多 token 提交。
- `model_runner.py` 是最大最复杂的文件：`run_prefill_spec` / `spec_round`（图 or eager 分发）→ 统一 `_spec_round_eager`（采样行引导 → `_sblob` 元数据 → 未捕获 `_sg_round` → `_readback_spec_commit`，冷/热共用引导；`spec_carry` 读回后恢复 `(采样行隐层, draft1)` 供冷→图交接）/ `_sg_*` 方法族（scatter/draft_chain/verify/acceptance/refill/round）/ `_fill_spec_meta`/`_load_spec_carry`/`_push_spec_meta` / `capture_spec_cudagraph`（每桶一张单图整轮）/ `_spec_round_graph`；decode/spec 双 pinned 元数据 blob 统一为声明式 `MetaBlob`。

### EAGLE3 投机解码（shifted draft 流口径，对齐 vLLM padded 分支）

- **流约定**：rope 位置 p 的 draft 行 embed `tok@p+1`、条件 `fc(aux@p)`（look-ahead aux，目标已消费该行提案 token 后的隐层）、输出预测 `tok@p+2`。draft prefill 与 decode 虚构窗口都用此口径（embed 右移一位、末槽补 next_token）。
- **单次 replay 顺序**：scatter（pinned meta 分发）→ 链步 1..k-1（seed=上轮采样行隐层，写在流位置 `l-2+i`，读窗 `ctxlen=l+i-1` = 提交前缀+自己的流槽，任何接受数下不读陈旧槽）→ verify（target 一次前向 [末位提交 token, d1..dk]，行 j 预测位置 l+j，图内 lm_head+采样）→ 图内链式接受（最长匹配前缀 m，提交 token + cnt 打包在 `commit_buf` 最后一列）→ refill（虚构窗口重放：k+1 行 = verify 行的位置/槽位/aux，行 m 覆盖为 bonus 并**采样出下一轮 draft1**）。
- **采样行**：refill 行 m（vLLM kernel 口径 `index_to_sample = 段尾 - 被拒数`）；其隐层+采样经 carry 交给下一轮链步。
- **跨轮状态按 seq_id 存字典**（batch 组成每轮会变，位置索引只在单次 replay 内自洽）：`spec_carry[seq_id]`（采样行隐层+draft1）、`spec_hidden[seq_id]`（aux 行 m，eager 复算/fallback 用）、`Sequence.spec_cold`（冷启动标记，实例属性）。
- **冷启动**：prefill 后首轮无 carry → 整批走 eager（采样行由 `fc(spec_hidden)+emb(last_token)` 在 rope l-2 复算，与 carry 同源），之后进图。
- **已验证的口径事实**（勿回退）：embeds 必须过 `input_layernorm`（去掉会崩到 E≈1.15）；链步 seed 用 raw midlayer 输出（vLLM 同）；checkpoint 无 fc_norm/norm_before_fc（纯 fc）；aux=(2,14,25) input-to-layer；eager 轮链步 1 的 seed = fc(spec_hidden)（引导行输入，非其输出）——与图轮稳态（refill 行 m 输出）不同源，均为已验证现状；冷→图交接 spec_carry = (采样行隐层, draft1)，refill 产物不跨 eager→图传递。
- 链步 i 的 embed token 恒为 drafted_ids[i-1]（步 1 = draft1 = tok@l，绝不回退为 last_token；右移口径的流续写前提）。C1 曾在此断裂（out_c1-c3 接受统计漂移而输出位等），C4 统一后由 drafted_ids 缓冲天然保证。
- 被拒 draft 的 KV 槽位不清理，靠后续写入覆盖（读窗恒在 `context_len` 内，残留无害）。

### Prefix caching
- `engine/block_manager.py`：block 级 hash 复用，仅覆盖 prefill 阶段（spec 提交的 token 不做 hash，避免被拒提案污染）；`reset_prefix_stats()`/`get_prefix_stats()` 供脚本统计。

## 性能基线（RTX 4070 Laptop 8GB / WSL2，spec_bench coding，k=3，串行，temp 0）

| 指标 | vLLM 0.26（graphs，bench 全程口径） | 本引擎 |
|---|---|---|
| Acceptance Length（=1+accepted/rounds） | 2.22-2.25 | 2.095 |
| 逐位置接受率 | 62.8-64.4 / 38-39 / 21.3% | 60.4 / 32.5 / 16.2% |
| 吞吐（含 prefill） | 129.8 tok/s | ~106-112 tok/s |

- 消融（4 并发、1100 token 共享前缀、96 token 生成）：基线 126 tok/s → +graph 155（+23%）→ +spec 179（+42%，spec 必须有图，eager spec 无收益）→ 全开 187（+49%）；prefix cache 只降 TTFT（后续请求 -46%），不影响 TPS（还略负 hash 开销）。
- 重构后（C1-C6，2026-09）最终 E=2.095、TPS 112-114 与重构前一致；输出位级不变贯穿全程，但 C1-C3 阶段逐轮接受统计有漂移（链步 1 embed 断裂，评审发现），C4 起恢复与基线逐项一致。
- 与 vLLM 的剩余 ~14% 差距：~7% 是每轮 host 耗时（vLLM 有异步调度重叠，我们没有），~6% 是两边生成文本数值漂移导致的接受率统计差（结构已完全对齐）。
- 短问答负载（接受率低）的参考值：spec k=2 ≈ 81 tok/s，base graph ≈ 67 tok/s。

## CoreX（Iluvatar）移植状态

- 代码层已就绪：spec 路径硬件依赖已参数化到 backend；图捕获失败自动降级 eager。
- **待实机验证**（本机无法测）：ixformer `flash_attn_varlen_func` 在虚构窗口/verify 路径的正确性；pinned memory + non_blocking H2D 行为。spec 图捕获现在会默认尝试（`supports_spec_graph=True`），成败由 `scripts/verify_corex.sh` 第 4 步实测（失败自动降级 eager spec，收益打折）。
- CoreX 上收益预期需重估：verify 摊薄权重读取的结构性收益不变，但 ixformer kernel host 开销若偏大，chain/refill 占比上升，净收益可能低于 CUDA。
- Ascend 预留：Tier-2 设备适配层接口位已就绪（backend.device 代理 / graph API 工厂 / pin 策略 / dist_name），接入新硬件时实现；多硬件方案全文见 .omo/plans/multi-hardware.md。

## 约定

- 所有注释、docstring、脚本 help 均用**中文**——新增注释请沿用。
- `nanovllm/config.py` 是 slotted dataclass；引擎 kwargs 会按其字段过滤，加配置项必须先加字段。
- `compile=True` 开启 `torch.compile`（主要加速 eager prefill；CoreX 上仅 ~8%，因 Qwen3 attention 在 CoreX 保持 eager）。Sampler 上禁用 compile（torch 2.11 Inductor scatter bug）。
- 本地 `main` 分支通常领先 `origin/main` 且带未提交工作——动代码前先 `git status`。
