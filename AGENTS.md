# AGENTS.md

Nano-vLLM：从零实现的极简 vLLM（研究/学习用代码库，非生产项目）。原版约 1200 行 Python；加入 EAGLE3 投机解码后约 2400 行。**无测试套件、无 CI、无 lint/typecheck 配置**；唯一验证手段是在有 GPU 的机器上对本地模型目录跑 `scripts/` 下的脚本。

## 命令

脚本需要**本地模型目录**（不能用 HuggingFace hub id）和 CUDA 兼容 GPU（NVIDIA，或 Iluvatar + `corex` 后端）。本机（RTX 4070 Laptop 8GB / WSL2）实际使用 `~/vllm-env/bin/python3`（Python 3.12, torch 2.11, transformers 5.15）：

```bash
~/vllm-env/bin/python3 -m scripts.benchmark /path/to/Qwen3-1.7B [--spec-model <draft 目录>] [--no-think] [--backend corex]
~/vllm-env/bin/python3 -m scripts.openai_server /path/to/Qwen3-1.7B [--port 8000] [--eager] [--spec-model <draft 目录>] [--spec-num-tokens k] [--no-think] [--reserve-output N]
# Qwen3.5 + DFlash（分支 qwen35-dflash；spec-num-tokens 缺省自动 = draft block_size-1 = 15）
~/vllm-env/bin/python3 -m scripts.benchmark /path/to/Qwen3.5-2B --spec-model /path/to/Qwen3.5-2B-Dflash
~/vllm-env/bin/python3 -m scripts.check_qwen35 base /path/to/Qwen3.5-2B --out /tmp/b.json
~/vllm-env/bin/python3 -m scripts.check_qwen35 spec /path/to/Qwen3.5-2B --draft /path/to/Qwen3.5-2B-Dflash --out /tmp/s.json
~/vllm-env/bin/python3 -m scripts.check_qwen35 check /tmp/b.json /tmp/s.json --label spec-vs-base
```

- **Qwen3.5 校验脚本两两配对须分进程跑**（WSL 上一进程显存回收延迟，同进程先后建双引擎会互挤 KV 预算；预算报错自带 used/peak/current 诊断数字）。`check` 子命令逐 prompt 报告首个分歧 token 位置与深度，判定口径见「Qwen3.5 + DFlash」节。

- **2026-09 精简**：benchmark 仅保留 question.jsonl 串行基准（vllm bench serve 对齐口径不变：seed shuffle / chat template / warmup 首条 / 单墙钟）；`benchmark graph|prefix` 子命令与 `check_eagle3` / `check_prefix_cache` / `verify_corex.sh` / `verify_ascend310p.sh` 已删除——文中历史锚点（E=2.095、CoreX 五步清单等）出自已移除的脚本，口径描述仍有效。
- **benchmark 内置 spec 接受统计 + `--no-think`（2026-09-08）**：`--spec-model` 时汇总打印 Acceptance length (E) / token 接受率 / 逐位置接受率——脚本侧 monkey-patch `spec_round`（每轮返回列表长度 = 接受数 m + 1 bonus，图/eager、冷/热四路统一从 `_readback_spec_commit` 出），预热后安装、仅计主跑；EOS/max_tokens 截断发生在 scheduler 层、图内 cnt 不感知，收尾轮轻微高估（CoreX 实测 2560 token 多计 5 个 ≈0.2%，可忽略）。check_eagle3 删除后 E 锚点复现职责由 benchmark 接管。`--no-think` = 对 chat template 显式传 `enable_thinking=False`（默认不传 kwargs，与 vllm bench 逐字节一致），表头打印 `thinking=` 状态便于对表。
- **openai_server 口径与新增开关（2026-09）**：① `--no-think` 关 thinking（默认 thinking-on；benchmark TPS/E 锚点为 thinking-off，对比两边须一致），每请求可用 `chat_template_kwargs.enable_thinking` 覆盖。② **上下文淘汰**：chat 请求 prompt 超 `max_model_len` 时保 system + 最近轮次、从最旧历史逐条丢（不再砍尾巴吞掉最新提问），为输出预留 `--reserve-output`（默认 512）token；输出 `max_tokens` 再按剩余上下文 clamp（旧实现允许 prompt+输出越界，隐患）。③ **block_size 硬约束**：CUDA 走 flash-attn paged 内核（`backends/cuda.py` 的 `flash_attn_varlen_func`/`flash_attn_with_kvcache`），`kvcache_block_size` 必须为 256 的倍数（默认 256；调小即 `RuntimeError: Paged KV cache block size must be divisible by 256`）→ **前缀缓存仅对 ≥256 token 共享前缀生效**，短对话 <256 不命中属预期（可加一段 ≥256 token 固定 system prompt 触发第一个块命中）。④ **启动预热**：`LLMEngine.__init__` 跑一次 max_tokens=2 的 dummy generate，消除 server 首请求 ~0.5s 的 decode/spec kernel 冷启动（实测同进程首轮 0.604s → 0.067s）。⑤ **服务端日志系统**：固定 5 字符标签 + `HH:MM:SS.mmm` 时间戳（TTY 上色）——`[HTTP ]` 传输层、`[REQ  ]` 首次被调度时由 `scheduler.schedule` 钩子打印（含引擎调度等待 `wait=`，rid 贯穿 HTTP/REQ/DONE）、`[DONE ]`（in/out/E/TTFT/TPS/ctx/cache/finish；TTFT=实际模型处理不含排队，ctx=已用/上限）、`[INFO ]` 周期聚合（winTPS/active/cache_hit/avgE）、`[WARN ]`/`[ERROR]`。⑥ **流式延迟**：driver 每步产出新 token 即 `st.event.set()` 唤醒 HTTP 线程（原仅 finish 时 set → `_run_llm` 只能 50ms 轮询、delta 被批处理）——同请求实测客户端 chunk 间隔 19.5ms（修复）vs 50.3ms（禁用）；同时 fanout 改用 O(1) 的 `num_tokens` 判增量、只切新增尾部（原每步全量拷贝 `completion_token_ids`）。
- **前缀缓存 A/B 与 Agent 基准数据集（2026-09）**：benchmark 已有 `--num-repeats R`（同 prompt 连跑 R 次、rep>0 全命中）+ `--no-prefix-cache`（冷/热 TTFT A/B）。新增 Agent 工作流风格数据集 `~/projects/agent_prefix_bench/`（`gen.py` 生成器 + `question.jsonl`，schema 同 spec_bench）：每条 = 共享 system+tools 前缀 + 组内共享会话上下文 + 各异任务；24 条、prompt≈1446 tok、同组共享≈1415 tok → 5 个满块（1280 tok）可命中，用 `--category agent --seed -1`（seed -1 保文件序、组内相邻才复用；默认 seed 0 会 shuffle 打散）。实测（4070 spec k=3 / no-think / mml2048 / ms8 / max_tokens64 / warmup0）：cache ON 24 请求命中 23、TTFT med **38.9ms** / TPS **94.1** vs cache OFF TTFT med **121.2ms** / TPS **83.6**（TTFT −68%、TPS +12.6%；E 两边 ≈2.0 不变，干净对照）。
- **TPS 对表警告（2026-09）**：同命令同机冷态 vs 热态可差 ~1.6×（实测 no-spec 42.6→69.8、spec 60.1→106.8，E 不变）——`nvidia-smi` 的 `SW Thermal Slowdown` 累计达千秒级、持续负载 SM 锁 ~1050MHz（上限 3105）。TPS 落差先排除热降频再怀疑代码；server 与 benchmark 稳态 TPS/TTFT 实测对齐（~96%），差异仅来自 server 首请求的 GPU 空闲→升频爬升。
- 必须以模块方式从仓库根目录运行（`-m scripts.X`），不能 `python scripts/X.py`（本机 shell 有 ROS 的 PYTHONPATH，`scripts/__init__.py` 用于避免包遮蔽）。
- 脚本只跑 GPU 且对数值敏感：计时前先预热引擎。
- `--eager` / `enforce_eager=True` 关闭 CUDA Graph 捕获（小显存卡推荐）；CUDA Graph 需按 batch 形状逐桶捕获，遇到未捕获形状首轮会失败。
- `tqdm` 被引擎导入但 **pyproject.toml 未声明**，全新安装后需手动装。
- 空板卡（无桌面占用显存）跑任何脚本建议传 `--gpu-mem-util 0.85`：KV 池按 util 贴地分配，图捕获/大桶采样峰值需余量（32GB CoreX 卡实测贴地后仅剩 ~112MB，图内采样 OOM）。

## 测试环境与已知的坑（WSL2 + 4070 Laptop 8GB）

- 桌面进程占用 ~0.5-1.5GB 且波动，`num_kvcache_blocks` 计算结果在 33-75 块间漂移 → 并发数和共享前缀长度必须压低（建议 ≤4 并发、≤1200 token 前缀、max_model_len 2048、util 0.95）。
- WSL 下显存回收有数秒延迟：`allocate_kv_cache` 已做预算不足时的等待重试。
- 对照基线：vLLM 0.26 需 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 才能在 WSL2 跑通；spec_bench 数据在 `~/vllm-lab/benchdata/spec_bench/question.jsonl`，模型在 `~/vllm-lab/models/`（Qwen3-1.7B、Qwen3-1.7B-EAGLE3=腾讯 AngelSlim 官方等）。
- EAGLE3 draft checkpoint 的 config.json **没有** aux 层字段 / fc_norm / norm_before_fc；aux 层用引擎默认 (2, n//2, n-3)。
- spec_bench 验收脚本曾晋升为 `scripts/check_eagle3.py`（E/逐位置接受率/直方图/TPS；`--save-out`/`--tag` 输出与回归 compare 流程同 schema），2026-09 已随精简删除（E/接受率统计职责移入 benchmark `--spec-model`，见「命令」节）；测试工件在 `~/vllm-lab/benchdata/nv_regression/`。
- 数值确定性：bf16 下目标模型自身就随 batch 形状漂移（逐个 vs 批量跑同一批 prompt 有 3/10 输出分歧）——"投机 vs 非投机逐 token 一致"只在无分歧翻转的负载上成立。
- Cherry 式流式对比以 stream_bench 为准；spec 开关对低 E 负载可能负收益，对比时须同时报告 E。

## 后端与硬件

- `nanovllm/backends/` 抽象 attention/kvcache 算子。选择方式：`backend=` 参数、`NANOVLLM_BACKEND` 环境变量、或自动探测（`torch.corex` 标记 → corex，否则 cuda）。
- `corex` 面向 Iluvatar（天数智芯）GPU，导入外部 `ixformer` 包（`ixformer.inference.functions`、`ixformer.contrib.vllm_flash_attn`），**pyproject 未声明**，只能在 Iluvatar 硬件上运行。
- 两后端差异：KV 布局（`get_kv_cache_shape`，CUDA = `[2, layers, blocks, block_size, kv_heads, head_dim]`；CoreX = HND `[2, layers, blocks, kv_heads, block_size, head_dim]`）与 **slot_mapping dtype**（`graph_slot_mapping_dtype`，CUDA int32 / CoreX int64）。`cu_seqlens`/`context_lens`/`block_tables` 两后端统一 int32（CoreX 硬约束：ixformer paged_attention 硬性检查 context_lens 为 Int&&contiguous&&cuda——图捕获静态 buffer 不得与 slot 共块）。改 kvcache/attention 代码时不要假设单一布局。
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

### OpenAI server 流式基线（stream_bench：3 条中文散文提示词，temp0.7，串行 bs=1；2026-09-06 采样瘦身会话，ms=8 统一默认后）

| 服务 | TPS 中位 | 平均 TTFT |
|---|---|---|
| vLLM 0.26（无 spec，ms=8） | 66.7 | 34ms |
| vLLM 0.26（eagle3 spec k=3，ms=8） | 64.4 | 34ms |
| nanovllm-initial（变体 server） | 70.1 | 53ms |
| 本引擎 spec-off（ms=8 统一默认） | 61.7 | 54ms |
| 本引擎 spec-on(k=3) | 59.7 | 70ms |

- **ms=8 统一默认**（openai_server 对 nospec/spec 都默认 max_num_seqs=8）：宽桶/宽 buffer 拖慢单步回放，实测 nospec 512→8 = 59.0→66.8 tok/s（+13%）；需要更高并发的用户显式传 `--max-num-seqs`。
- **sampler 瘦身**（top_p≥1 走 top-k 掩码+Gumbel-argmax，图安全无分支）：temp0 位等门 ×2 PASS（spec_bench 10 条逐 token 与基线完全一致）、E=2.095 带守恒、离线 decode 绝对值 60.1→64.1-66.6（+7-10%）、spec TPS 新高 108.7-109.6。top_p<1 保留 legacy 原路（fast_top_p=False 回退）。
- 该负载 E 仅 ~1.15-1.19（spec_bench coding 为 2.095；vLLM 自报 SpecDecoding metrics acceptance length 1.14-1.17）：低接受负载下 spec 两边都负收益（本引擎 61.7→59.7，vLLM 66.7→64.4——跨引擎互证）。**低接受负载勿开 --spec-model**；高接受负载（coding/temp0）收益 +64% 以上（离线 108.7 vs 66.2，见下节）。
- decode 残差现状：采样机器已修复（绝对值 +7-10%）；**GEMM 同 kernel 异常已复检排除**——采样修复后 back-to-back 重剖析（16 稳态步），128x2 GEMM ours 167.6 vs initial 167.7 ms（±0.002%）、128x1 ours 反而快 2.1%：pre-fix 的 +21-23% 是会话时钟漂移的测量伪影，非引擎固有。在线配对 A/B（v3，背靠背）：ours-nospec 68.1（TTFT 52ms）vs initial 70.2（53ms）= **0.97**，decode 基本速度与 initial 对齐（≥0.95 门通过）；残差 ~3% 与离线口径一致（66.2 vs 68.2 = 0.97）。
- server 路径税实测仅 ~1ms/轮（~4%）：temp0/GIL 切片/增量 detok/子进程隔离四个正交干预全部无效——瓶颈是负载 E 不是 host 路径。子进程化补丁（+167 LOC，串行零收益未采用）存于 .omo/drafts/subprocess-server.patch，未来测高并发 SSE 可捡回。
- 数据来源：~/vllm-lab/benchdata/stream_cmp/*-v2.json 与 offline_cmp/*-v2.json（2026-09-06 采样瘦身会话）；离线探针 /tmp/probe_offline*.py（会话级，未入库）。

### 离线横向对比（spec_bench coding 前 10 条，串行 bs=1，temp0，k=3，max_num_seqs=8；2026-09-06 采样瘦身会话）

| 配置 | AGG TPS（含 prefill，保守） | E |
|---|---|---|
| ours-decode | 66.2 | — |
| ours-spec(k=3) | 108.7 | 2.095 |
| initial-decode | 68.2 | — |
| vllm-decode | 69.5 | — |
| vllm-spec(eagle3) | 125.6 | — |

- 口径：ours 用 check_eagle3 同款 kwargs（max_model_len 4096 / max_num_seqs 8 / kvcache_block_size 256，E 直方图 monkey-patch spec_round）；initial 同构（其 SamplingParams 不支持 temp0，用 1e-9 近似 greedy）；vLLM 用 `LLM(max_model_len=2048, max_num_seqs=8, gpu_memory_utilization=0.85)`（graphs 默认 FULL_AND_PIECEWISE），vllm-spec 加 `speculative_config={'method':'eagle3','model':<draft>,'num_speculative_tokens':3}`（离线 LLM 传 dict；serve CLI 传 JSON 字符串）。
- spec 结构收益两引擎一致：ours 108.7/66.2 = 1.64×，vllm 125.6/69.5 = 1.81×；E=2.095 与历史锚点完全一致。ours decode 距 initial 0.97（66.2 vs 68.2，采样修复后从 0.92 收窄）、距 vllm decode 0.95（66.2 vs 69.5）——残差 ~3-5% 与流式配对口径一致（GEMM 异常已复检排除，见流式节），不是 spec 路径特有的。

### vllm bench serve 在线口径 vs benchmark jsonl（2026-09 对齐后可直接同表）

- **vllm bench spec_bench 真实口径（读 vllm-env 0.26 bench 源码核对，旧记载有误已修正）**：prompt **不是 raw 文本**——`CustomDataset.sample`（SpecBench 继承）对 prompt **套 chat template**（`tokenize=False` 字符串、不传 kwargs → Qwen3 默认 enable_thinking）后发 completions；数据集加载即 `random.seed(--seed)`+shuffle 取前 N（`--seed` 默认 0）；输出上限 = `--spec-bench-output-len` 默认 **256**；预热 = 就绪检查 1 次 + `--num-warmups 5` 次同一首条 prompt（均不计指标）。
- **jsonl 已同构复刻（默认即对齐）**：① prompt 构造——模板字符串再 encode（V1.8 的 `--raw-prompt` 已随对齐删除）；② 采样——`--seed` 默认 0 + shuffle（`-1`=文件序「前 N 条」历史口径），同 seed 即同 prompt 集；③ `--max-tokens` 默认 256；④ `--warmup` 默认 5、统一打 qs[0]；⑤ 吞吐改**单墙钟**（Σtokens / 主跑总墙钟，含逐条 tokenize/打印）= vllm Output token throughput 同口径。**同 seed → 同 prompt 集 → E 相同 → spec TPS 可直接同表**（原「spec TPS 强依赖 E 不可直接比」结论作废）。
- **残余口径差**：① TTFT——vllm 为 client 端（HTTP 往返，高几 ms），jsonl 为进程内，只比趋势；② vllm 侧 client 协议税（逐 chunk SSE 解析在其计时窗内，实测 ~1-4%）使 vllm TPS 天然偏低；③ EOS 计数差 1 token/条（<1%）。其余前提不变：vLLM server 显式开 spec（bench 只是 client）、确认同硬件。
- **thinking 口径对 E 的影响（2026-09-08 CoreX 实测，勿误判回归）**：Qwen3 模板不传 kwargs = enable_thinking=True，输出先吐 `<think>` 推理文本，draft 可预测性显著低于直答。同 seed 同题集单变量对照（k=3, max_tokens 256, temp0）：thinking-on **E=1.748 / 85.3 tok/s** → `--no-think` **E=2.125 / 101.3 tok/s**（E +22%）。**历史 E 锚点（4070 的 2.095、CoreX 的 2.050）全部是 thinking-off 口径**——对齐 vllm bench（thinking-on）跑出 E≈1.7-1.8 属正常，勿与锚点直接比；对比时认准 benchmark 表头 `thinking=` 状态，两边口径必须一致。
- **历史口径回溯**：旧 jsonl 数字（如 CoreX 三配置 8.1/76.5/104.1）= 文件序前 N + token 化模板（enable_thinking=False）+ 512 + warm 3 + 分墙钟——`--seed -1 --max-tokens 512 --warmup 3 --no-think` 重现（2026-09-08 CoreX 实测 E=2.050、TPS=103.7，与 check_eagle3 锚点逐位一致：benchmark 内置统计与旧脚本同构互证，check_eagle3 删除后的锚点复现即此命令；4070 的 E=2.095 同款命令待复测）。

## CoreX（Iluvatar）移植状态——实机验证通过（2026-09，32GB 板卡 / corex-4.4.0 / Python 3.10 / transformers 4.57.6）

- **五步清单全过，spec 单图整轮一次捕获成功**（无需降级）：decode 图 wall 551ms vs eager 3980ms（7.2×，逐 token 位等）；spec 图 E=2.050 / TPS **103.7**（≈4070 的 108.7 同量级）；spec eager 仅 11.3 tok/s——与 CUDA 侧「spec 必须有图」结论互证。
- **图是 CoreX 刚需**：eager decode 仅 7.8 tok/s（kernel launch 税被 CoreX torch 放大），图后 bs=1 72.9 / bs=8 415.8（TPS sweep **9.3×**，远超 4070 的 +23%）——图收益主要来自消除 launch 税；eager 模式在 CoreX 上近乎不可用级别地慢。
- **离线三配置基准**（spec_bench coding 前 10 条，串行 bs=1，temp0，k=3，`benchmark jsonl --gpu-mem-util 0.85`；2026-09 实机）：EAGER **8.1** → Graph **76.5**（9.4×）→ Spec+Graph **104.1**（+36%，vs eager **12.9×**）；TTFT 三组 130-136ms 持平（prefill 主导，预热热命中偏差可忽略，median≈avg）。③ 与 verify 步骤 4 的 check_eagle3 103.7 交叉一致 ±0.4%（两口径互证）。与 4070 同口径对照：decode 76.5 vs 66.2（**+15%**）、spec 104.1 vs 108.7（-4%）——spec 收益比 1.36× 低于 4070 的 1.64×（图已抬高基线，launch 税红利被吃掉）。
- **板卡高温**：连续高负载触发 XID 8 温度告警（实测 96°C）——bench 间留冷却间隔；告警后数据点可能受降频影响，采集时标注温度条件。
- **实机修出的三个坑（均已入仓）**：
  1. `context_lens` 必须独立 int32 buffer——ixformer `paged_attention` 硬性检查 Int&&contiguous&&cuda；图捕获静态 buffer 曾与 slot 共块跟随 `graph_slot_mapping_dtype()` 变 int64 而炸（CUDA 上 slot 恰好同为 int32 掩盖，见 `capture_cudagraph`）。
  2. 空板卡 KV 池按 util 贴地吃满（32GB 卡 KV 后仅剩 ~112MB），图捕获/大桶图内采样（bs=512 桶 argsort ~600MB）必 OOM——verify 五步统一 `--gpu-mem-util 0.85`；`check_prefix_cache` 补 `--max-num-seqs`（默认 8，对齐 openai_server 结论；脚本默认值未动不伤 4070 基线）。
  3. 旧版 transformers `apply_chat_template(tokenize=True)` 返回 list 而非 dict（4.57.6 实测仍如此）——4 个脚本共 7 处调用点已加 `isinstance(out, (list, tuple))` 兼容。
- **判读口径**：`benchmark graph` 的 correct 列在 CoreX 仅作参考——temp0 greedy 门下图内 lm_head 与 eager 的 GEMV kernel 选择不同，ULP 级 logit 差可致 argmax 翻转（实测 bs=1/2 FAIL、bs=4/8 PASS；同机 check_prefix_cache 位等门 PASS）。主判据 = 同配置图 vs 图、eager vs eager（两项全 PASS）。
- `check_eagle3` 的「histogram 提交数 != completion 总数」警告 = EOS 截断正常效应（spec 轮多 token 被 commit_spec 提前截断），非 bug。
- 收益预期修正（vs 上版文档）：spec 图收益远高于保守预期；ixformer varlen（虚构窗口/verify/refill）实测正确；此前「ixformer host 开销偏大致净收益低于 CUDA」的预测不成立（图内无 launch 税后反超）。
- Ascend 310P：已实现并注册为第三后端（见下节；auto-detect 分支在 backends/__init__.py）；多硬件方案全文见 .omo/plans/multi-hardware.md。

## Ascend 310P（Atlas 300I Duo / 200I Pro）适配状态

- 主仓后端 `backends/ascend.py` 已按 nano-vllm-ascend 实机路径收口（Qwen3-0.6B / CANN 9.1 / torch-npu 2.10.0.post4）：NZ KV + `_npu_flash_attention` prefill + `_npu_paged_attention` decode + NPUGraph 走现有 `capture_cudagraph`（不迁独立 NPUGraphDecodeRunner / TorchAir）。
- 生产算子（勿回退到 FIA v2）：KV = 每层 `empty_with_format(FRACTAL_NZ)`，shape `(blocks, hidden/16, block_size, 16)`；写入 `_npu_reshape_and_cache(..., slot_indices=)`；prefill 默认 `auto`（运行时同时具备 `_npu_flash_attention_v3` + `_npu_paged_attention_splitfuse_v2` 时走 v3 + 2048 压缩 NZ causal mask，否则回落 `_npu_flash_attention`；`NANOVLLM_310P_FLASH_OP=legacy|v3|atb_v2` 可强制）；decode `_npu_paged_attention`（静态 out）；RoPE 优先 `_npu_rotary_embedding`、缺失时回落纯 torch。运行时：`allow_internal_format=True`、`jit_compile=False`、`NPU_FUZZY_COMPILE_BLACKLIST=ReduceNansum`。
- 能力矩阵：decode 图 = NPUGraph 尝试+降级（lm_head + greedy argmax 入图；非 greedy 仍图外采样）；eager spec = on (EAGLE3/dense DFlash, multi-query PA); spec-graph try+smoke (host PA meta + greedy argmax + float accept)；DFlash draft 在 SplitFuse 可用时走独立分页 FRACTAL_NZ KV + 非因果 SplitFuse，复用 target block table/slot，并纳入同一 KV 内存预算；融合 = `npu_rms_norm`/`npu_add_rms_norm`/`npu_swiglu`/`_npu_rotary_embedding`；`graph_sample` 默认 False（指随机采样）。图 padding = 末块 dummy（block=N-1、ctx=1），BlockManager 只分配前 N-1 块。
- 硬件硬规矩：仅 fp16；`kvcache_block_size` 仅 {64,128}（config 把 benchmark 默认 256 覆盖为 128）；`block_size * head_dim <= 128*128`；请显式压低 max_model_len；TP=1。
- Prefix-hit/chunked prefill：运行时有 `_npu_paged_attention_splitfuse_v2` 时走固定 2048 ND 压缩 mask（mask_type=5），否则走 legacy SplitFuse + 动态 NZ mask；两者都缺失时自动关闭 prefix cache。spec 图默认尝试（PA meta host 预展开；图内 greedy argmax；float/static 验收避 AICPU Add；冒烟失败降级）。Sampler 仍支持 temperature=0 greedy。
- 实机验证（`scripts/benchmark.py`，不迁 bench_ttft_tps.py）：

```bash
python3 -m scripts.benchmark /data/Qwen3-0.6B \
  --backend ascend310p \
  --gpu-mem-util 0.5 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --warmup 1 \
  --num-prompts 3 \
  --max-tokens 64 \
  --no-prefix-cache
```

  `--eager` 为 eager decode；默认 enforce_eager=False 走 NPUGraph（对应旧仓 --mode aclgraph）。
  EAGLE3: --spec-model DRAFT --no-think. dense DFlash: paged target verify + paged draft SplitFuse，整轮 NPUGraph 捕获失败自动降级；DFlash 自动关闭 prefix cache；hybrid still cuda-only.
- 环境钉：torch-npu==2.10.0.post4 / CANN 9.1.0 / 310p 专用镜像。

## Qwen3.5 + DFlash 适配（hybrid + 块草稿投机，分支 qwen35-dflash）

- **架构事实（勿回退）**：Qwen3.5 = 24 层 hybrid（18×Gated DeltaNet 线性注意力 + 6×门控全注意力，`full_attention_interval=4`），text-only（视觉塔不加载，等价 vllm `--language-model-only`），text 形状在嵌套 `text_config`。GDN 纯 torch：分块 prefill（逐行移植 transformers `torch_chunk_gated_delta_rule` 语义）+ 向量化单步 decode；门控 `g=-exp(A_log)·softplus(a+dt_bias)`、`beta=sigmoid(b)`、q/k 逐头 L2、RMSNormGated(norm_before_gate)。rope = NeoX 半分式 partial 64/256（`mrope_interleaved` 只影响多模态 T/H/W 行混合，text-only 退化为标准 rotary）；**Gemma 式 RMSNorm（weight+1）仅 Qwen3.5 主体**，DFlash draft 是标准 RMSNorm。门控注意力 q_proj 输出翻倍（q+gate 逐头），attn 输出 ×sigmoid(gate)——decode 路径 flash_attn 返回 `[n,1,H,D]`，**必须先归形 `[n,H,D]` 再过 gate**（阶段1 踩过广播膨胀坑）。权重：多模态壳 `model.language_model.` 前缀剥离 + `mtp.*`/`model.visual.*` 跳过（`Qwen3_5ForCausalLM.load_weights` 自带，引擎经 `hasattr(model,'load_weights')` 优先分发）。
- **DFlash 口径事实（勿回退）**：draft 5 层标准 qwen3 注意力（**非因果**），checkpoint 无 embed/lm_head（共享 target）。每轮 = 上下文追加 → 块草稿 → verify → 最长前缀接受+bonus（commit_buf 契约与 EAGLE 一致）。aux 层 = draft config `target_layer_ids` [1,6,11,16,21]，语义 = **0-based 层输出**（qwen3 的 EAGLE3 是层输入口径，两者不同）。**上下文追加 = verify 的 v_aux 行 0..m**（vllm `_prepare_dflash_inputs_kernel` 口径：行 0 = 上轮 bonus 的 aux，只在本轮 verify 行 0 能算出，漏掉会在 L-1 位留永久空洞；行 m+1 是被拒提案的错位特征——任务书初版「行 1..m+1」为 off-by-one，勿回退）。draft 上下文 KV 走稠密缓冲（[ms, mml, 5层, 2, 4头, 128] bf16，不进 block manager）；k = block_size-1 = 15 自动派生。
- **GDN 状态 × spec（关键正确性）**：状态池 conv[ms×(k+2)+1 槽, 6144, 3] + ssm[同, 16, 128, 128] fp32（ms=4 ≈1.3GB）；verify 走 'row' 模式逐行落槽（行 j 终态 → 槽 base+j，base=批内序×(k+1)），行 0 读 carry 槽（上轮 base+m，读后写同槽安全）；非 spec decode/prefill 走 'final' 模式（行为与阶段1 逐字节一致）。图捕获的 row 模式用 host 常量段界（`gdn_row_bases`）绕开捕获期 tolist 同步。
- **引擎门控（config hybrid 分支）**：强制关 prefix cache（GDN 状态不可 block hash 复用）；`max_num_batched_tokens` 封顶到 max_model_len（禁 chunked prefill + 把 warmup 峰值压到 1×mml——8GB 卡按 8×2048 批预留会把 KV 池挤到 0 块）；仅 cuda 后端 + TP=1。四格配对矩阵：qwen3+eagle3（原路径）/ qwen3_5+dflash / **qwen3+dflash（已实现，待远端实测）** 三格放行；eagle3+hybrid 干净拒绝（唯一未实现格：_sg 路径的 GDN verify 分叉未接，硬跑会静默错乱）。
- **Qwen3+dflash（dense）口径**：aux 换算在 config 层完成——qwen3.py 钩子是层输入口径，层 l 输出 ≡ 层 l+1 输入，故 `spec_aux_layers = target_layer_ids + 1`（Qwen3-4B 的 [1,9,17,25,33] → (2,10,18,26,34)），qwen3.py 零改动，与 vllm qwen3 dense 同构；`_df_*` 路径（eager+图）GDN 触点已 `is_hybrid` 条件化（dense 无 GDN 状态分叉，df_carry 段留零不消费）。CUDA/CoreX 保留稠密 draft KV；**310P 使用每层独立分页 FRACTAL_NZ draft KV**，复用 target block table/slot，context/query KV 均由 `_npu_reshape_and_cache` 写入并用非因果 SplitFuse 读取；分页 draft KV 已纳入每块显存预算，prefix cache 自动关闭。**spec 图整轮捕获**失败时统一降级 eager spec。**dense 机械冒烟（2026-09-09，4070 + Qwen3-1.7B + 合成 mini dflash draft）全绿**：config 派生→setup→warmup→eager 轮→spec 图捕获（rows=16）→回放→读回提交全链走通，graph-vs-eager 4/4、base-vs-spec 4/4 位等；冒烟修出两个 dense 专属坑：① scheduler admission 状态槽门控；② warmup dummy 状态回收。CUDA/CoreX 的已知限制仍是 prefix 命中不重算 aux；310P 已禁用。**验证注意**：spec-vs-base 位等门对 aux 口径错误不敏感，**E 数字才是 aux 正确性的最终信号**（该配对应显著 >2；E≈1-1.2 先查 aux 换算）。
- **位等判定口径（三阶段实测沉淀）**：串行 bs=1 temp0 期望全等或 bf16 平票翻转（分歧点 top-1/top-2 打平或差 1 ULP、top-8 其余逐位一致）；**并发/图 padding 的深位置翻转 = 批形状漂移**（eager bs=n vs 图 bs=桶 的 GEMM/attention 归约序差，AGENTS「数值确定性」条同款）——base-vs-base 独立重跑 5/5 确证 base 自身确定、翻转源自 spec/图路径的形状差，判豁免不判 bug。验收锚点数据：阶段1 transformers 位等 8/10（2 平票）；阶段2 spec-vs-base 9/12+3 豁免、E=2.502（raw prompt）/5.179（benchmark coding，与论文 AL 4.78@block16 量级互证）；阶段3 decode 图串行 12/12、spec 图串行 12/12+E 守恒（ΔE=0.000，收尾截断口径差 ≤0.003）、双路径捕获降级干净。
- **显存现实（8GB 卡）**：spec 引擎 init 后 ~5.7GB（权重 4.0 + GDN 池 ~0.65 + draft ctx 0.04 + 杂项），KV 池仅剩百 MB 级——spec 图 ms=4 需 `mml=512 / util 0.99`，ms=8 不现实；无 spec 的 decode 图 ms=4/mml1024 可全桶捕获。
- **EAGLE `_spec_round_graph` padding 复位 bug（2026-09 已修）**：原 `vs[:, n:]` 轴序错位（`v_slots` 视图 (M,rows)，seq 维在轴 0，应为 `vs[n:, :]`——原写法清的是**所有序列的列**）+ 复位条件 `bucket < _spec_graph_active` 漏「首次扩桶 / 同桶内含 padding」两个窗口——并发（n<桶）下真实序列的 verify 行槽被清 -1 → 跳写 → NaN 零流 / 全 "！"；串行（n==桶）不受影响，故长期未被发现。已改为 `vs[n:, :]` + 条件 `n < bucket`（对齐 DFlash 版），并顺带在 no-spec `prepare_decode` 的 padding 复位里补 `ids/pos` 清零。验证：并发 spec（n=3→桶 4，正是触发窗口）输出正常、serial spec E=2.035 逐位不变。**遗留：建议补 n<bucket 并发位等门**（本仓无自动化测试）。

## 约定

- 所有注释、docstring、脚本 help 均用**中文**——新增注释请沿用。
- `nanovllm/config.py` 是 slotted dataclass；引擎 kwargs 会按其字段过滤，加配置项必须先加字段。
- `compile=True` 开启 `torch.compile`（主要加速 eager prefill；CoreX 上仅 ~8%，因 Qwen3 attention 在 CoreX 保持 eager）。Sampler 上禁用 compile（torch 2.11 Inductor scatter bug）。
- 本地 `main` 分支通常领先 `origin/main` 且带未提交工作——动代码前先 `git status`。
