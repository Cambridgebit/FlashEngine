# 能力状态（唯一状态源）

> **状态**（2026-11-24）：本文件是 FlashEngine **实测状态**的唯一来源。
> 其它文档里的「已验收 / 待验收 / 未验证」若与本文件冲突，以本文件为准；
> 专题文档负责记录过程、命令与失败细节，不负责给出总体结论。

三件事必须分清，混在一起就会得出错误结论：

| 层 | 含义 | 单一来源 |
|---|---|---|
| **声明支持** | 组合被引擎接受、启动不报 `NotImplementedError` | [`support_matrix.py`](../flashengine/support_matrix.py)（[说明](SUPPORT_MATRIX.md)） |
| **实测状态** | 在真机上跑出过什么、效果如何、证据有多强 | 本文件 |
| **实现细节** | 为什么会这样、怎么修的、踩过哪些坑 | 各专题文档 |

**声明支持 ≠ 实测有效。** 典型反例：`qwen3 + ascend310p + w8a8 + dflash` 在声明矩阵里放行，
实现也跑得通，但实测接受长度 E≈1（草稿几乎不被接受，比不开投机更慢），见第 3 节。
`support_matrix` 只回答「能不能启动」，不回答「值不值得开」。

---

## 0. 证据分级

| 级别 | 定义 | 可否对外宣称 |
|---|---|---|
| **A** | 多负载/多轮真机实测，指标与输出质量均有记录，当前推荐配置在生产服务里跑 | 可以，按记录口径 |
| **B** | 真机跑通，但只有单场景/单模型/少量样本；趋势可信、绝对数字不可外推 | 可以说「实测可用」，附条件 |
| **C** | 仅冒烟、合成权重、tiny 夹具、或纯主机单测 | 只能说「功能通」，不能说「有效/可部署」 |
| **D** | 仅声明 + 代码路径，**没有**对应硬件的真机运行 | 只能说「已实现」，不能说「已验证」 |
| **X** | 实测不成立 / 已知会崩 / 明显负收益 | 必须明确劝阻 |

「合成夹具通过」不证明训练权重的接受率或加速比；「tiny 模型跑通」不证明真实模型能装下显存。

## 1. 目标模型 × 后端（普通推理）

| 模型 | CUDA | CoreX | Ascend 310P |
|---|---|---|---|
| Qwen3 dense | **A**：4070 8GB，graph/spec 多轮基准（[EAGLE3](EAGLE3.md)） | **A**：32GB 板卡，graph 76.5 tok/s、eager 8.1（[CoreX 移植](EAGLE3.md)） | **A**：8B graph 10.7 tok/s、30B-A3B 22.2（[310P 基准](ASCEND310P_BENCHMARK.md)） |
| Qwen3 MoE（30B-A3B） | **C**：tiny 夹具与 HF 逐 token 对照 | **D**：纯 torch 专家组，无真机 | **B**：真实 30B-A3B 跑通（含 W8A8 21.2 tok/s，[长跑](ASCEND310P_W8A8_LONGRUN.md)） |
| Qwen3.5 dense 文本 | **B**：4B/8B 真机验收（[Qwen3.5](QWEN35.md)） | **D**：共享 CUDA 路径，无真机 | **A**：4B/8B/9B 多轮基准 + 生产服务（[Qwen3.5](QWEN35.md)、[优化台账](QWEN35_OPT.md)） |
| Qwen3.5 MoE 文本 | **C**：64.6MB 公开 tiny 夹具（[Qwen3.5 MoE](QWEN35_MOE.md)） | **D** | **D**：仅有实现与静态对齐 |
| Qwen3-ASR | **A**：CER 0/0 + server 端点（[ASR](QWEN3_ASR.md)） | **B**：两条样本冒烟，CER 0/0，[ASR §4.3](QWEN3_ASR.md) | **B+**：eager/图两模式 CER 0/0（[ASR §4.2](QWEN3_ASR.md)） |

> ASR 的 CER 0/0 **只对 zh 4.2s / en 15.05s 两条样例成立**，逐字级（NFKC + 去标点）比对；
> 它不是「任意音频 CER 为 0」的承诺。310P 给 B+ 而非 A，是因为只跑了这两条 × 两种执行模式，
> 没有多负载、长音频、噪声/口音样本。

## 2. 投机解码

| 方法 | CUDA | CoreX | 310P |
|---|---|---|---|
| EAGLE3（dense） | **A**：E=2.095、108.7 tok/s（spec_bench coding，k=3，thinking-off） | **A**：E=2.050、103.7 tok/s；eager 仅 11.3（投机必须有图） | **D**：eager spec 路径就绪，无 310P E 记录 |
| DFlash（dense/MoE） | **B**：真实草稿跑通（[DFlash 性能](DFLASH_PERFORMANCE.md)） | **D**：稠密草稿路径共享 CUDA，无真机 | **A**：4B/8B/9B/30B 多组，见下表 |
| DSpark（dense） | **B**：4070 上结构化请求走 eager spec 轮，E=2.143 | **D** | **B**：0.6B/8B 单场景跑通（E=1.26 / 5.30），paged 形态未做独立验收 |
| 量化 × 投机 | **X**（W8A8 仅 310P） | **X** | **X**：8B-W8A8 + DFlash **E≈1、零接受**，见第 3 节 |

310P DFlash 实测（服务端 c1、贪心、无 thinking、忽略 EOS 的 Spec-Bench 口径，[310P 基准](ASCEND310P_BENCHMARK.md)、[Qwen3.5 优化](QWEN35_OPT.md)）：

| 目标 + 草稿 | k | 聚合 tok/s | E | 备注 |
|---|---|---|---|---|
| Qwen3-8B + DFlash | 7 | 36.21 | 4.754 | 普通 10.75，约 3.4×；**同一配置另一次会话 37.6 / 38.3 tok/s（E 仍 4.878）**，跨会话热态差异约 ±5%，引用时写区间而非单点 |
| Qwen3.5-9B + DFlash | 7 | 22.73 | 5.452 | 普通 9.12，约 2.5×；中文/业务负载 E≈2.76 |
| Qwen3.5-9B 融合 LoRA + DFlash | 7 | 13.68 | 3.03 | 业务长提示；k 扫描 k7 最优（k1/5/9/11/13/15 均更差） |
| Qwen3-4B + DFlash | 7 | 32.8 | 4.283 | 该次运行是滑窗草稿**改走稠密掩码**时期；现在滑窗草稿已改走分页 + 逐层窗口掩码（`prepare_dflash_paged_window`），尚未重测同口径 4B，故 36.8/32.8 都只作历史点 |
| Qwen3-30B-A3B + DFlash | 7 | 26.6 | 3.328 | MoE + 滑窗草稿 |

**E 与负载强相关**：同一模型中文/业务负载的 E 只有英文 coding 的一半左右，
拿不同数据集的 tok/s 直接对比没有意义；对比时必须同时报 E 与数据集。

## 3. 量化

| 组合 | 状态 | 说明 |
|---|---|---|
| `qwen3*` + `ascend310p` + `w8a8` | **B** | msmodelslim 真 int8 算子（注意力静态 W8A8、MoE 专家 W8A8_DYNAMIC）；30B-A3B 21.2 tok/s |
| `qwen3_moe` + `ascend310p` + `w8a8` | **B** | 1000 条引擎级长跑 0 bad、15001 次图回放 0 回退；**仅 seqs=1、无投机**（[长跑](ASCEND310P_W8A8_LONGRUN.md)） |
| `qwen3` + `ascend310p` + `w8a8` + 投机 | **X** | 8B-W8A8 / Coder-w8a8 + DFlash 实测 **E≈1**：草稿提案几乎全被拒，比不开投机更慢。**不要开** |
| W8A8 MoE + 大 prefill | **X** | 长 prompt 触发算子崩溃（[310P 基准](ASCEND310P_BENCHMARK.md:67)），当时只在短 prompt 串行下稳定 |
| `cuda` / `corex` + `quark_w8a8` | **B** | AMD Quark int8 检查点，**加载期反量化**成计算 dtype，运行时等同 bf16——不是 int8 激活计算；tiny MoE 与 HF 32/32 逐 token 全等 |
| Qwen3.5 / ASR + 任意量化 | **X** | 未接入（`support_matrix` 直接拒绝） |

## 4. 前缀缓存

| 后端 | 状态 | 说明 |
|---|---|---|
| CUDA | **A** | block 级 xxhash；256 token 对齐，短于 256 的共享前缀不命中属预期（[运行时](ENGINE_RUNTIME.md)） |
| 310P（普通 Qwen3） | **B** | 需要 SplitFuse；缺算子时自动关闭前缀缓存并告警 |
| 310P（Qwen3.5 + GDN） | **B** | 完整块边界存 conv/SSM 状态快照；已改设备侧 D2D：同一 34K 前缀冷 2358ms → 热 836ms、输出逐字相同；另有 2.239→0.813s 的一次记录（不同会话）。**都是单条前缀、无同负载旧码 A/B**，只能说明「命中后 TTFT 大幅下降」这个量级（[前缀缓存](QWEN35_PREFIX_CACHE.md)） |
| 310P（DFlash 草稿） | **B** | 分页草稿用 `keep_last_block_unhashed`；稠密草稿用主机快照 + 校验器，快照有界 LRU |
| 310P（Qwen3.5 会话租约） | **B** | `--block-size 16` + 会话租约/深度加权驱逐 + `--recurrent-snapshot-stride 8`：小池对抗场景 A 复访 0%（TTFT 3557ms）→ 44%（2247ms）。**收益只在池子被挤占时出现**，生产 16,680 块下为 0（[优化台账](QWEN35_OPT.md)） |
| 多 LoRA | **B** | KV 哈希与 GDN 快照哈希都含 `lora_id`（2026-11-24 修）：同一适配器跨请求命中、不同适配器互不串用。**运行时多 LoRA 只在 310P 真机成立**；CUDA/CoreX 的 LoRA 批次强制 eager（`engine/model_runner.py:1219-1223`），且与工具/约束请求的组合未测 |

## 5. 结构化输出与工具调用

| 项 | 状态 | 说明 |
|---|---|---|
| CPU 自检（三协议语法机） | **A** | `scripts/check_structured.py` 全过 |
| CUDA 端到端 | **A** | tool/json × 流式/非流式全过；约束请求走 eager spec 轮，E=2.143 |
| 310P 端到端 | **A** | 融合 LoRA + DFlash k7 上四档全过，协议探测 `hermes`；spec 与无 spec 逐 token 一致 |
| 带工具请求流式 | **B+**（2026-11-24 新增） | 只在 **`hermes` 协议 + thinking 关闭 + 无自定义 stop + 无路径约束** 时生效：正文按 token 流式、工具标记零泄漏，结束时补 `tool_calls` 分片；其余组合仍整段缓冲。pi 端到端只用手测（首段 2.534s / 末段 14.379s、191 字符），**「76 个正文分片」没有仓库存档**；可复现的对外数字用 `scripts.toolcall_bench.py`（已采 `first_delta_s`） |
| 约束请求性能 | **B** | 约束请求必须走 eager spec 轮（比整轮 spec 图慢，310P 每 token 190–217ms）；无约束请求不受影响 |

## 6. 服务端（OpenAI 兼容）

| 项 | 状态 | 说明 |
|---|---|---|
| `/v1/chat/completions` 流式 | **A** | 普通请求逐 token；带工具请求 2026-11-24 起走增量协议流式（pi 实测 76 个正文分片） |
| 34K 长系统提示（pi agent） | **B** | pi 自带提示约 33.9K token，**必须** `--max-model-len 40960`；冷 prefill ~88.5s，命中前缀后 0.5–1s（[结构化解码 §9](STRUCTURED_DECODING.md)） |
| `/v1/audio/transcriptions` | **A** | CUDA + 310P 全过；非 ASR 模型返回 400 |
| `--raw-log`（原始输入输出） | **B** | 逐请求原子追加 JSONL：`event=input` 含 `input_raw`（请求体 JSON 原文）、`event=output` 含输出文本 + `ttft/tpot/e2e` 元数据，两侧用 `request_id` 配对；文件权限 0600 由 `_open_raw_log` 显式 `chmod`（2026-11-24 补，之前只靠调用方 umask）。**不是** HTTP/SSE 原始字节，重启前发生的请求无法补录；日志含完整对话与工具返回，按敏感数据管理 |
| `/metrics`、`/v1/stats`、内置控制台 | **B** | 计数与面板已验证；`/metrics` 无鉴权，需在网络边界限制 |
| 4 并发尾延迟 | **X（未解决）** | HTTP 四并发时偶发单条 TPOT 700ms+，2K 与 32K 都出现，未定位 |

## 7. 长上下文与显存

| 配置 | 状态 | 说明 |
|---|---|---|
| 310P 40K 窗口服务（Qwen3.5-9B + DFlash k7） | **A** | `--max-model-len 40960 --max-num-seqs 8 --gpu-mem-util 0.70`，KV 4247×64=271,808 token，引擎就绪约 65s |
| 310P 100K 窗口 | **C** | 能启动（KV 1999×64=127,936 token，ready 190.9s）并跑通 1.9K 提示的前缀复用，但 **52,910 token 输入把进程 OOM kill**（`Constraint=CONSTRAINT_MEMORY_POLICY`）。**不要宣称支持 52K 以上输入** |
| 310P 显存预算 | **B** | 共享主机 87GiB，服务占用时只剩 6.3GiB 可用且无 swap；`gpu-mem-util 0.7` 是实测安全线，`0.95` 只能在没有其它服务的独占机器上试 |
| 310P prefill 吞吐（Qwen3.5-9B + DFlash k7，40K 窗口） | **A** | `FLASHENGINE_VLLM_GEMM=1`（vllm-ascend `unquantized_gemm`）比默认 0 快 9–12%：2,144 tok 5.1s、4,433 tok 9.1–9.5s、8,853 tok 16.6–17.0s、17,054 tok 32.6s。层分解：GDN 占 88%（8.25s / 8,800 字符≈5.1K token；`layer_time.py` 的 `PROMPT_TOKENS` 是字符数）、全注意力 1.14s、MLP 1.73s；裸 matmul 峰值 21–27 TFLOPS 而端到端仅 8.6 TFLOPS，说明瓶颈是 GDN 的 fp32 分块递推而非 GEMM |
| 310P GDN 递推优化（两遍扫描 + fp16 操作数） | **A** | 同进程交错 2×2（5,123 token，各 3 次）：row+fp32 8.562s、row+fp16 8.383s（−2.1%）、scan+fp32 8.679s（+1.4%）、**scan+fp16 8.257s（−3.6%）**；服务级冷请求 −2.6%（5,132 tok）/ −0.5%（2,170 tok）；875 字文档输出与 fp32 逐字相同。**代码默认 fp32，生产脚本开 fp16**；chunk 64 在 fp16 下仍最优（128→10.9s、256→26.8s）（[优化台账](QWEN35_OPT.md:367)） |
| 310P 数据集 TPS（bench_data） | **B** | 六配置 × 中英 coding 前 10 题（max_tokens 256、贪心、单并发）：9B 9.19/9.16、9B+DFlash 26.97/17.04、9B+LoRA 8.14/8.12、9B+LoRA+DFlash 24.46/15.46、8B 10.55/10.67、8B+DFlash 37.13/18.96；ASR-1.7B TPS 32.44 / TTFT 196ms / RTF 0.0929 / CER 0（[数据集 TPS](BENCH_TPS_310P.md)） |
| 310P BS=1 六配置压测 | **B** | 9B/9B+DFlash/9B+LoRA/9B+LoRA+DFlash/8B/8B+DFlash × 中英 × 长上下文/解码/多轮/工具：解码 9.2 → 14.1–16.6 tok/s（DFlash）、LoRA −10%（+DFlash 收回大半）；每场景 1 次请求、含首请求图捕获，不是稳态吞吐（[压测台账](BENCH_BS1_310P.md)） |
| 310P 小页 KV（block 16） | **B** | 1,229 token 提示：冷 prefill 64→16 为 2,549→2,172ms（−12%），解码 TPOT 114→118ms（+3%）；6 轮工具轨迹两档都在噪声内，无端到端收益可宣称 |
| 310P 长文档前缀复用（D1） | **A** | 875 字文档同一前缀：冷 2,297ms → 热 404/367ms（约 6×），换随机 nonce 强制冷回到 2,192ms，证明提速来自前缀命中；3,720 字随机长文 4,639 → 489ms。34K pi 冷 88.5s → 命中后 0.8s |
| 310P 主机内存约束（D4） | **A** | NPU 运行时以 26,175 个 2MiB 大页常驻 51.1GiB（`Private_Hugetlb`、371 段匿名 `MAP_HUGETLB`），主机 87GiB 只剩 ~10GiB；长 prefill 本身不涨主机内存（2.1K–17K token 全程 delta ≤112KiB），52,910 token 的 OOM 是 `gpu-mem-util 0.95 + mml 102400` 的配置级超预算，不是逐 token 泄漏 |

## 8. 当前推荐配置

**端侧单机（Atlas 300I Duo，本仓主战场）**

```bash
FLASHENGINE_VLLM_GEMM=1 \
MODEL=/data/Qwen3.5-9B  SPEC=/data/Qwen3.5-9B-DFlash \
  --max-model-len 40960 --max-num-seqs 8 --gpu-mem-util 0.70 --spec-num-tokens 7 \
  --block-size 16 --recurrent-snapshot-stride 8 --recurrent-prefix-cache-mb 1024
```

- `FLASHENGINE_VLLM_GEMM=1` 让 prefill 走 vllm-ascend 的 `unquantized_gemm`，实测比默认 0 快 9–12%；
  代价是启动时后台导入 vllm_ascend（约 6s，与权重加载重叠，就绪时间基本不变）。

- k=7 是实测甜点（[优化台账](QWEN35_OPT.md) k 扫描），不要用默认 k=15。
- 需要工具调用时保留 `--tools` 默认协议探测（Qwen3.5 模板探成 `hermes`），不要额外开 `--no-path-constraints`，除非模型自己会产出合法 XML。
- 量化版本只在「无投机」场景推荐开启。

**桌面 CUDA（4070 8GB）**：`--gpu-mem-util 0.85 --max-model-len 2048 --max-num-seqs 8`，
EAGLE3 k=3；DFlash/DSpark 在 CUDA 上只是跑通，没有与 vLLM 的正式对照。

## 9. 已知不成立 / 不要宣称

1. **W8A8 + 投机能加速** —— 310P 上 E≈1，负收益。
2. **100K / 200K 上下文可用** —— 52K 输入即 OOM。
3. **CoreX 上跑过 Qwen3.5 / MoE / DFlash / DSpark** —— 只有 Qwen3 dense + EAGLE3 与 ASR 冒烟。
4. **310P 上跑过 EAGLE3** —— 路径就绪但无 E 记录。
5. **DFlash 在 CUDA/CoreX 有正式加速比** —— 只有功能与单场景数据。
6. **「声明矩阵放行」等于「该组合值得用」** —— 见第 3 节反例。
7. **长跑结论能外推到高并发/投机** —— 长跑只有 seqs=1、无投机。
8. **tiny/合成草稿通过 = 训练草稿有效** —— 合成夹具只证明机制正确。
9. **310P 的 HTTP 并发吞吐** —— 现有 310P 表格几乎都是 `max_num_seqs=1`、3 题 × 64–128 token；
   并发、批量 prefill、长输出都没测过，不能拿聚合 tok/s 对标生产吞吐。
10. **前缀缓存的 TPS 收益** —— 命中只降 TTFT（甚至略增 hash 开销）；把它写成「提升吞吐」是错的。
11. **小时级稳定性 / 故障注入** —— 最长的一次是 1000 条引擎级长跑 + 15001 次图回放，不等于过夜压测。
12. **GDN 换成「幂零级数平方解」能提速** —— 2026-11-24 试过 6 级平方 `∏(I+L^{2^j})` 替换 63 步逐行前向：
   prefill −8%、fp16/bf16 单测逐位相等，但真实 400+ token 请求退化成只吐一个 `<think>` 就 EOS，
   已回退。机理是 fp32 条件数放大，不要只看单测结果就上线。
13. **把长上下文问题归给设备显存** —— 52.9K OOM 杀的是主机进程（`shmem-rss` 22.7GB），
   设备侧在 util 0.70 时还有 6.7GB 空闲；调 `util`/`mml`/KV 块数前先看主机 RAM。
14. **小页（block 16）在所有负载上都更快** —— 只有冷 prefill −12%，解码约 +3%，6 轮工具轨迹差异在噪声内；
   租约那组 −37% TTFT 是小池对抗淘汰场景，不是生产 QPS 收益。

## 10. 与 vLLM-Ascend 的同卡对照（未完成）

板端 `192.168.16.118` 的主机驱动为 `7.6.T7.0.B052`（CANN 8.5.0），而带 DFlash 支持的
`vllm-ascend v0.23.0-310p` 镜像内是 CANN 9.1.0：设备初始化阶段报
`Package_Error_Verify_OPP(E30006)` / `rtSetDevice 507033`，无法启动。
可用的 `vllm018` 容器（CANN 9.0.T3、vllm 0.18）没有 DFlash 实现，且板端无外网无法补装。

**结论：同卡 vLLM 对照数据尚不存在**，任何「比 vLLM 快/慢 N%」的说法都缺依据。
要解锁需要：升级主机 NPU 驱动到 CANN 9.1 兼容版本（板上有 24.1.1.3 包，需重启），
或给板端提供能安装 vllm 0.23 的网络。

## 11. 证据装置清单

对外引用数字时，至少要知道它出自哪个装置——本仓只保留这些入口（`scripts/` 共 12 个 py）：

| 装置 | 用途 |
|---|---|
| `scripts.benchmark` | 离线串行吞吐 / E（vllm bench serve 同构口径） |
| `scripts.bench_concurrent` | HTTP 并发压测（纯 stdlib，任何机器都能对着 server 跑） |
| `scripts.openai_server` | OpenAI 兼容服务 + 内置控制台 + `/metrics` + `--raw-log` |
| `scripts.asr_benchmark` / `scripts.toolcall_bench` | ASR 转写指标 / 工具调用四档 |
| `scripts.merge_lora` / `scripts.check_structured` / `scripts.check_docs` | LoRA 合并 / 语法机自检 / 本文档校验 |
| `tests/`（unittest） | 主机侧单测：chunk GDN 位等、recurrent LoRA 前缀、hermes 流式、LoRA 合并 |

已从 main 移除、但历史文档仍在引用的入口：`check_qwen35*`、`check_cuda_attention`、
`check_python_memory`、`probe_dflash_perf`、`export_bundle`、`export_oracle`、
`sync_cpp_contracts`、`check_native_replay`、`check_native_server`。这些命令**不可执行**，
对应专题文档已标注「历史入口」。
