# 多 LoRA 适配（1 基座 × 8 适配器，310P 实测）

本文件记录 FlashEngine 的多 LoRA 实现与实机验收结果。实现思路对齐 vLLM 的
punica/SGMV，但更薄：**权重栈 + forward hook + 逐 token 路由**，不修改任何模型
forward 签名，不影响非 LoRA 路径。

## 1. 结构与关键决定

| 模块 | 作用 |
|---|---|
| `flashengine/lora/config.py` | `LoRAConfig(max_loras, max_lora_rank, modules)`、`LoRARequest` |
| `flashengine/lora/loader.py` | 读 PEFT 适配器；HF 张量名 → 引擎模块树（剥离 `base_model.model.` / `model.language_model.`）；`in_proj_qkv` 单张量按 `(key,key,value)` 拆进 `in_proj_qkvz` 前 3 分片（A 共用、B 按行切） |
| `flashengine/lora/layers.py` | `LoRABinding`：**秩拼接**布局 `AT[in, max_loras*r]`、`BT[max_loras*r, out]`（缩放折进 BT）；增量 = `(x @ AT * M) @ BT`，`M` 为 `[T, max_loras*r]` 的行路由掩码。主机镜像 `_AT_cpu/_BT_cpu` + `flush()` 整块下发 |
| `flashengine/lora/manager.py` | 注册适配器到槽位（1..max_loras）、解析融合模块分片（qkv / gate_up / in_proj_qkvz / in_proj_ba）、挂 `forward_hook`；`build_mask/mask_into/fill_mask` 建掩码（常驻 pinned 主机缓冲 + 常驻设备缓冲 + 同批次组成去重） |
| `flashengine/lora/context.py` | 本步路由上下文：只存一份主机算好的 fp16 掩码（`set_active_mask`） |

接线：`Config.enable_lora/max_loras/max_lora_rank/lora_modules`、
`Sequence.lora_id`、`add_request(lora_request=)`、`generate(lora_requests=)`。

**三条不做会出错的硬规矩**：

1. **前缀缓存必须按 lora 分键**：`BlockManager.compute_hash` 把 `lora_id` 混进哈希
   （链首块也混）。同一段文本在基座与不同适配器下隐层完全不同，KV 不可互用。
2. **融合分片要按槽位切**：Qwen3.5 的 `in_proj_qkvz` 是 4 段
   `MergedColumnParallelLinear`，适配器侧只有 `in_proj_qkv`（覆盖前 3 段）+ `in_proj_z`。
   加载器把 B 按 `output_sizes` 切、A 复制——这正是 vLLM 0.18 在 Qwen3.5 上做不到的
   那一块（它按 2 个名字打包，`set_lora` 按 4 段索引直接越界）。
3. **图内必须走形状静态路径**：`torch.nonzero`/动态取行在图捕获里是非法算子
   （310P 上会 `aclnnNonzero` 崩流），因此 LoRA 启用后 `LoRAManager.set_static(True)`，
   路由一律用「主机预构建掩码 + 定长 GEMM」。
4. **掩码不能在设备上用整数比较生成**（2026-11 实测踩过）：图内的
   `idx.unsqueeze(1) == slot_col.unsqueeze(0)`（int32 下标 vs int64 常量广播）在
   NPUGraph 回放时与捕获期语义不一致，每轮 LoRA 增量错、投机**接受长度塌到 E≈1.04**，
   而 eager 同代码完全正常（E=5.05）。掩码必须由 `LoRAManager.build_mask` 在**主机**
   拼成 fp16 buffer、每步一次 `copy_`，图内只剩一次 `mul`。
5. **捕获期必须静音 trace**：hook 里任何 `.tolist()`/`.item()` 都是主机同步，图捕获中
   执行会 `aclrtMemcpy 107030` + `Not allow to synchronize captured-stream` →
   捕获整桶失败（退化为 eager，TTFT 直接翻倍）。`set_trace_enabled(False)` 由
   `set_static(True)` 调用，调试开关 `FLASHENGINE_LORA_TRACE=1` 只用于图外排查。
6. **图外也一律走定长掩码，不要「按槽窄切」**（2026-11-21 删除）：曾为整批同一适配器
   加过 `delta_narrow`（`AT[:, lo:hi].contiguous()`），310P 上每次前向都要为 248 个
   binding 重新生成 496 个 Transpose 算子，实测引发 **6.4–6.8 s 一次性停顿**，并在
   并发探针里直接打成 `aicore timeout ... 507014` + `fault kernel_name=Transpose_...`
   把整机挂死（需断电重启）。省下的 FLOPs 在 310P 上本来就不值钱（见 §5）。

## 2. 310P 实机验收（Atlas 300I Duo / CANN 9.0.T3 / 容器 vllm018 / Qwen3.5-9B）

前置：容器内跑引擎**不能设 `PYTHONPATH=/data2/FlashEngine`**（会触发 torch_npu 的
`ACL_ALLOW_HF32` compileopt 失败，见 §4），必须 `cd /data2/FlashEngine` 后直接跑脚本。

### 2.1 eager 多 LoRA（8 适配器）

判据与结果（`t_lora_fe.py`）：

| 判据 | 含义 | 结果 |
|---|---|---|
| A | 适配器 a0（alpha=0，增量恒 0）输出与基座**逐位一致** | ✅ True |
| B | a3（alpha=32，训练值）输出与基座不同 | ✅ True |
| C | 8 适配器同批 vs 各自单跑 | 7/8 逐位一致（a6 因 fp16 批形状漂移翻转，见 §5） |

启动日志：`LoRA: {'a0': 1 … 'a7': 8} bindings=248 modules=152 hooks=152 stack=330.2MiB`。

### 2.2 NPUGraph 解码图 + LoRA

| 判据 | 结果 |
|---|---|
| 捕获 | 4 个桶全部成功、`numeric_failures=0` |
| 回放 | 380 次，`fallback=0` |
| a0 == base（**图路径**下逐位一致） | ✅ True |
| 8 适配器一批 | 35.0 s（同负载 eager 249.1 s，约 7×） |

### 2.3 DFlash 投机 + LoRA + 投机整轮图

`mode=spec_graph`，`capture_success=8 / replay=72 / eager_rounds=0`：

| 判据 | 结果 |
|---|---|
| method / k | `dflash` / 15 |
| 接受统计 | rounds=188, accepted=530, **E=3.819** |
| a0 == base 逐位 | ✅ True |
| a3 != base | ✅ True |

实现要点：投机整轮 verify 的行数是 `bs*(k+1)`，每 seq 连续 `k+1` 行。
`_prepare_spec_lora(bs)` 在捕获期为每桶建一份静态路由张量并绑定上下文，
`_fill_spec_lora` 在回放前按 `row // (k+1)` 填值；`_fit_rows` 在 hook 内把
下标对齐到本次前向行数（图外兜底）。

### 2.4 性能（参考配置口径：3 题 × 64 token，mml2048／ms1）

**统一口径**（与 `docs/QWEN35_OPT.md` 的基线表一致）：Qwen3.5-9B +
`/data/Qwen3.5-9B-DFlash`（k=15），`bench_data/question.jsonl` coding 前 3 题
（seed 0 shuffle，合计 189 输出 token，每条题都用 `enable_thinking=False`），
**max_tokens 64、mml2048、max-num-seqs 1、gpu-mem-util 0.5、关前缀缓存、
warmup 1**，`ignore_eos` 关闭（与 benchmark 默认一致）。
LoRA 侧注册 3 个适配器（a0/a1/a2，`max-loras 3 / max-lora-rank 8`）、按题轮转。
脚本：`scripts/benchmark`（`/data2/run_ref6.sh`），日志 `/data/lora/ref6.log`。

| 模式 | 基座 TPS | LoRA TPS | ΔTPS | 基座 decode | LoRA decode | Δdecode | TTFT 基座→LoRA | TPOT 基座→LoRA | E 基座→LoRA |
|---|---|---|---|---|---|---|---|---|---|
| eager | 3.98 | 3.19 | −19.8% | 4.1 | 3.3 | −19.5% | 793→830 ms | 242.5→304.8 ms | – |
| graph | 8.59 | 7.84 | −8.7% | 9.4 | 8.6 | −8.5% | 780→856 ms | 105.9→116.0 ms | – |
| **spec（DFlash）** | **19.42** | **18.31** | **−5.7%** | **25.4** | **23.9** | **−5.9%** | 811→855 ms | 39.4→41.9 ms | **5.100→5.025** |

- `TPS` = 总输出 token / 总墙钟（含 prefill）；`decode` = 排除首 token 的稳态解码。
- **参考配置下 LoRA 的真实税只有 5.7%**（spec 图路径），图路径的 496 个额外小算子
  被回放摊薄；差距主要在 eager（每步都在图外付 launch）与 prefill。

### 2.5 并发档（8 题 × 64 token，mml2048／ms8）

同一套输出长度与上下文，只把并发放开（`t_matrix.py <base|lora> <cfg> 8`，
`MATRIX_MML=2048 MATRIX_MS=8 MATRIX_PROMPTS=8 MATRIX_MAXTOK=64`，util 0.60），
日志 `/data/lora/conc6.log`。

| 模式 | 基座 TPS | LoRA TPS | ΔTPS | 基座 decode | LoRA decode | Δdecode | TTFT 基座→LoRA | TPOT 基座→LoRA | E 基座→LoRA |
|---|---|---|---|---|---|---|---|---|---|
| eager | 23.45 | 20.25 | −13.6% | 30.87 | 25.24 | −18.2% | 5508→5308 ms | 259.2→317.0 ms | – |
| graph | 35.02 | 32.86 | −6.2% | 52.39 | 48.39 | −7.6% | 4999→5163 ms | 152.7→165.3 ms | – |
| spec（DFlash） | 31.16 | 27.07 | −13.1% | **53.66** | **56.10** | **+4.5%** | 7039→9933 ms | 149.1→142.6 ms | 4.981→5.515 |

- 8 并发下 **稳态解码 LoRA 反而更快**（+4.5%）：8 个不同适配器改写出的输出更规律，
  草稿命中率上升（E 5.515 vs 4.981）。
- spec 的聚合 TPS 看起来差 13.1%，是因为批次 prefill 被 LoRA 拖长（TTFT 7.0→9.9 s，
  8 条题一起等），而输出只有 64 token —— **比较并发能力要看 decode 列**。

### 2.6 为什么「参考配置」的数字高，我自己的压测量出来低

同一台板、同一份代码，spec 基座可以同时测出 **25.4 / 19.4 / 17.0 / 15.5** 四个数，
差别全在口径。用实测值可以逐项分解：

| 口径 | E | decode | 端到端 TPS |
|---|---|---|---|
| 参考配置（3 题 × 64 token，steady） | 5.100 | 25.4 | 19.42 |
| 同上改成 1 题（64 token，steady） | 3.556 | 18.8 | — |
| 我最初的压测（1 题 × **128** token，mml1024/ms8，steady） | 3.256 | 17.02 | 15.50 |

`25.4 × (3.256 / 5.100) × (15.50 / 17.02) ≈ 14.7 ≈ 15.5`，逐项含义：

1. **TPS ≈ E / 每轮耗时，而每轮耗时几乎恒定（~190 ms）**。所以 spec 的 TPS 几乎
   正比于接受长度 E，E 是**内容属性**：3 题聚合 5.100、单题只有 3.556、同一题生成到
   128 token 又降到 3.256（越往后越不可预测）。**把 E=5.1 的数字与 E=3.3 的数字放一起
   比，等于在比两道不同的题**。
2. **端到端 vs 稳态（−9%）**：TPS 的分子含 prefill。64 token 的输出只有 ~2.5 s 解码，
   0.8 s 的 TTFT 就占掉 24%；输出越短，prefill 摊得越薄。
3. **自定义压测的两个额外开销**：`mml1024/ms8` 下多桶图捕获、以及**首版脚本没有 warmup**
   （第一轮 request 里混进了图捕获，TTFT 被记成 4.7–8.2 s），都会进一步压低 TPS。
   现在的 `t_matrix.py` 已加 warmup（与 benchmark `--warmup 1` 同口径）。

**结论**：文档里贴 spec 数字必须同时给 E、输出长度、是否含 prefill；
本文件此后统一用 2.4 / 2.5 两套口径（64 token 输出、mml2048）。

**开销归因（实测）**：310P 上小 GEMM 的成本与 rank 无关、基本是常数项
（`[59,4096]×[4096,64]` 0.117 ms、`[59,4096]×[4096,8]` 同样是 0.117 ms），
248 个 binding × 2 GEMM = **496 个额外算子/前向**：图内回放约 12 µs/算子（每轮 +6 ms，
被 E 的收益盖过）；图外约 0.09–0.12 ms/算子（prefill 每次 +60~80 ms、eager 每步
+70~80 ms）。**再往下压只能减算子数**：把同一 Linear 的多个分片合成一次 GEMM
（248→152 个模块）或做「单适配器合并进基座权重」的 solo 模式；窄 rank 快路径已删除
（只减 FLOPs，反倒在 310P 上引入 Transpose 编译停顿）。

### 2.7 服务端并发（HTTP `/v1/chat/completions`，独立口径）

压测脚本 `bench_lora_http.py <每线程请求数> <max_tokens> <线程数>`：每线程固定一个
适配器（`MODEL_PREFIX=Qwen3.5-9B` 时全部打基座，用于对照），64 token/请求，
prompt 72 token，服务端 `mml1024 / ms8 / util 0.60 / DFlash k=15`。

| 并发 | 基座 tok/s | 基座 p50 | LoRA tok/s | LoRA p50 | Δ吞吐 |
|---|---|---|---|---|---|
| 1 | 11.22 | 5.9 s | 10.40 | 6.2 s | −7.3% |
| 4 | 16.67 | 15.4 s | 16.64 | **13.5 s** | −0.2% |
| 8 | 23.32（1024 tok） | 22.0 s | 19.15（906 tok） | 21.7 s | −17.9%（token 少 11.5%） |

**修前修后（同一 4 并发、8 请求负载）**：墙钟 **204.1 s → 30.8 s**、p50 **175.2 s →
13.5 s**、聚合 **2.51 → 16.64 tok/s**；8 并发从「0/16 成功（server 退出）」变成
「16/16 成功」。修复后 8 并发下 `[DONE]` 里**不再出现 `preempt=`**（之前每条请求
`preempt=43`），每请求 prefill 步数从 43 步降到 1 步。

**三个根因（按影响排序，均已修）**：

1. **KV 池被 util 压没 → 大面积抢占**（最像「并发卡死」的那条）：310P 上
   `--gpu-mem-util` 是相对整卡 87.5 GiB 算的，而 Qwen3.5-9B + DFlash + GDN recurrent
   常驻就要 ~43 GiB（权重 20.4 + recurrent 6.7 + draft 2.4 + 图/碎片）。util 0.50 只剩
   0.7 GiB，实测**只分到 15 个块（960 tokens）**，8 并发 ×136 tokens 必然抢占，
   每条请求 `preempt=43`、`Steps: 91 (prefill=43, decode=48)`、TPOT 700+ ms/tok。
   **LoRA 场景请用 util 0.60**（实测 4496 块 = 287k tokens），并核对启动摘要的
   `KV cache:` 行——引擎现在会在池子过小时打 `[WARN] KV 池只有 N tokens ...`。
2. **投机状态回写按索引长度重复编译**：见 §6.1（每条请求 +6.7 s 长停顿）。
3. **宿主 OOM**：util 0.70 时设备内存以 shmem 计入宿主，87 GiB 被吃满后
   OOM killer 静默杀进程（`main process disappeared!`），见 §6.2。



### 2.8 业务效果评测（mission_report_sft_eval，2026-11-21）

**先说 alpha 是什么**：LoRA 训练出来的增量是低秩乘积 `B@A`，真正加到基座线性层上的
是 `(alpha/r) · B@A`——`r` 是秩（本适配器 8），`alpha` 是缩放常数（训练时 32，等于
缩放 4.0）。所以「同一份权重、不同 alpha」不是不同模型，而是把同一份增量按不同倍数
放大：alpha=0 等价于不加（应与基座逐位相同），alpha=32 是训练原值，alpha=192 是
训练值的 6 倍。板上 `/data/lora/adapters/a0…a7` 就是把训练好的那份权重导出 8 份、
只改 `alpha=(0,8,16,32,64,96,128,192)`，用来把「强度」当变量扫；**a3 = 训练原值，
是唯一该上线的那个**。这个扫法只验证强度，不验证「不同业务的不同适配器」。

**数据集**：`D:\AVIC\mission_report_sft_eval.jsonl`，302 条，全部是同一个函数
`mission_report`（任务简报）。每条 = `system`（约 1.4 KB：角色、四段输出格式
【任务】【进度】【态势】【建议】、建议动作词表）+ `user`（一句口语化提问 +
`[态势数据]` 字段清单：任务类型/限时/已执行时间/距离/弹药/燃油/bingo/joker/威胁等级/
数据链…）+ `golden`（标准四段简报，或 `【无法处理】` 拒答）。

`evalmeta.pert_type` = 这条样本被做了什么手脚，**它决定该不该拒答**：

| pert_type | 含义（真实样本） | 全库条数 | 期望行为 |
|---|---|---|---|
| `none` | 数据完整正常：`剩余目标：9个 / 剩余目标分布：左右分散分布` | 210 | 出正常四段简报 |
| `degraded` | 部分字段降级为「未知」：`护航机地速：未知`、`剩余燃油：未知` | 61 | **多数仍出简报**（59 条）但要标低置信度；只有叠加矛盾时才拒答（2 条） |
| `missing_arg` | 关键字段缺失/无效：`目标状态：未知`、`打击弹药预计到达时间：无` | 11 | 拒答并点出缺哪些参数 |
| `illegal_val` | 数值非法或物理不可能：`本机距目标距离：-20km` | 10 | 拒答并点名非法字段 |
| `conflict` | 字段自相矛盾：`剩余目标：2个` 但 `剩余目标分布：无剩余目标` | 9 | 拒答并指出矛盾对 |
| `out_of_scope` | 任务类型不在支持列表：`任务类型为：后勤补给` | 1 | 拒答并说明支持哪些类型 |

**指标口径**（脚本 `/data/lora/eval/lora_eval.py`，原始结果
`/data/lora/eval/full48.jsonl`）：

- `sim`：输出与 golden 的字符级相似度（`difflib`，先剥掉 think 段）。
- `refuse_acc`：**只在 `should_refuse=True` 的样本上算**，看是否以 `【无法处理】` 开头。
- 四段结构合规 / `【建议】` 动作词表合法 / `【进度】` 数值一致：**只在非拒答样本上算**
  ——拒答样本的正确输出只有一行 `【无法处理】`，把它们算进分母会把合格率压低
  （本文表格已按此修正，早先口径 0.53 的分母含拒答样本，属统计错误）。
- 抽样：分层 43 条（none 16 / degraded 8 / missing_arg 6 / illegal_val 6 / conflict 6 /
  out_of_scope 1，其中应拒答 20 条；`out_of_scope` 全库仅 1 条）。

**主结果（43 条 × 2 模型，0 报错）**

| 指标 | 基座 Qwen3.5-9B | LoRA a3（alpha=32） |
|---|---|---|
| `sim` 均值（全部 43 条） | 0.2563 | **0.8991** |
| `sim` 均值（非拒答 23 条） | 0.3087 | **0.9798** |
| 拒答正确率（应拒答 20 条） | **0.050** | **1.000** |
| 四段结构合规（非拒答 23 条） | 1.000 | 1.000 |
| `【建议】` 动作词表合法（非拒答） | 0.652 | **1.000** |
| `【进度】` 数值与 golden 一致（非拒答） | 0.174 | **1.000** |
| 平均延迟 / 平均输出 token | 51.65 s / 140.3 | 44.33 s / 81.1 |

43 条里 **23 条与 golden 近乎逐字一致**（`sim>0.995`）。

**分层（`sim` 均值）**

| 分层 | 条数 | 基座 | LoRA a3 |
|---|---|---|---|
| none（正常） | 16 | 0.314 | **0.993** |
| degraded（不拒答的 7 条） | 7 | 0.297 | **0.950** |
| degraded（要拒答的 1 条） | 1 | 0.198 | 0.526 |
| illegal_val | 6 | 0.166 | 0.886 |
| missing_arg | 6 | 0.191 | 0.784 |
| conflict | 6 | 0.143 | **0.763** |
| out_of_scope | 1 | 0.721 | 1.000 |

**强度曲线（同一批 12 条，a0…a7 只差 alpha）**

| 适配器 | alpha | 等效缩放 | `sim` | 拒答正确 | 动作合法 | 数值一致 | 输出 token |
|---|---|---|---|---|---|---|---|
| a0 | 0 | 0（= 基座） | 0.3059 | 0.917 | 0.833 | 0.182 | 143.8 |
| a1 | 8 | 1.0 | 0.4695 | 0.917 | 1.000 | 0.545 | 98.0 |
| **a3** | **32** | **4.0（训练值）** | **0.9278** | **1.000** | 0.917 | **1.000** | 114.8 |
| a7 | 192 | 24.0（6× 训练值） | 0.0496 | 0.583 | 0.000 | 0.000 | 224.3 |

a0 与基座几乎完全一致（0.3059 vs 0.3073）→ 零增量路径与逐 token 路由都是对的；
alpha=8 只有一半强度（0.47）；alpha=192 直接崩坏，输出退化成
`【无法处理】该本世纪末款制所域上表文明底额，可买用。`——**加乘不是越大越好，
必须用训练时的 alpha=32**。

**两个典型对比**

`mission_report_0001`（正常样本）：

- golden：`【进度】10%（目标进度：已清除1／共10）｜时间进度89%（40.2／45min），任务进度落后｜燃油进度：燃油不足` … `【建议】继续按计划执行；分兵两路同步清剿`
- 基座（sim 0.299）：`时间进度 89.3%，任务完成度 10.0%`、`剩余燃油 32% 等于 bingo 油量`、`【建议】立即返航` —— 数值口径错、把「bingo 油量」误判成「燃油不足」的反面结论
- a3（sim 0.957）：四段与 golden 逐字一致，只有第二条建议不同（`转向搜索剩余空域` vs `分兵两路同步清剿`）

`mission_report_0031`（应拒答：目标已摧毁，但本机距目标 210 km > 射程 200 km）：

- golden：`【无法处理】参数矛盾：目标已摧毁，但本机距目标210km超出武器射程200km；参数缺失：护航机地速、剩余燃油为未知，相关结论置信度：低。`
- 基座（sim 0.198）：**完全不拒答**，编了一份「任务完成度 100%、建议立即返航」的完整报告
- a3（sim 0.526）：拒答了，但只点出「参数缺失」，漏掉「参数矛盾」这一支

**结论**：LoRA 的业务效果是数量级差异（非拒答 sim 0.31→0.98、拒答 5%→100%、
进度数值 17%→100%），而性能侧只付 −5.7%（参考配置）/ −0.7%（8 并发 spec）；
短板是 `conflict`（0.763）与 `missing_arg`（0.784）两类——全库这两类合计只有 20 条，
属训练数据覆盖不足，不是融合或路由问题。

## 3. 已知限制

- `max_loras` 是硬上限，注册数 = 同时在线数；超出直接报错，不静默回落基座。
- 只支持「注意力/MLP 线性层」类适配器；MoE 专家、embedding、视觉塔未接。
- 图捕获目前按 310P 的 NPUGraph 路径实现；CUDA/CoreX 的 LoRA 批次仍强制走 eager。
- 效果评测目前只覆盖 `mission_report` 一个函数（§2.8）；`conflict` / `missing_arg` 两类
  最弱（0.76 / 0.78），全库这两类合计仅 20 条，属训练数据覆盖不足。
- DFlash + LoRA 时草稿模型不带 LoRA（验证仍精确，只有接受率受影响）。参考配置
  3 题 E 5.025 vs 5.100（−1.5%）；8 并发 64 token 档 E 5.515 vs 4.981（+10.7%，
  适配器让输出更规律）。

## 4. 板子环境的两个坑（已带代码兜底）

1. **`PYTHONPATH` 触发 NPU 初始化失败**：先 `import transformers` 再首次
   `torch.npu.set_device()` 会让 `_npu_init` 的 `aclSetCompileopt(ACL_ALLOW_HF32)`
   返回 500001，并毒化驱动（此后同机所有进程都失败，只能重启板卡）。
   规避：`flashengine/npu_bootstrap.py` 在 `import flashengine` 时先把设备建好，
   且**不要设 `PYTHONPATH`**（用 cwd 或 `sys.path.insert`）。
2. **锁页分配同理**：`pin_memory=True` 会提前触发同类 ACL 调用，可用
   `FLASHENGINE_PIN_MEMORY=0` 关闭；`MetaBlob` 另有异常回落。

## 5. 判读口径（勿误判为 bug）

310P fp16 下，**同一 prompt 在 bs=1 与 bs=8 的贪心输出本就会偶发翻转**
（GEMM 归约顺序不同，ULP 级差异经 argmax 放大）。因此「批量 vs 单跑逐位相等」
只能作为参考信号：eager 的 A/C 与图路径的 a0 位等已经足够证明路由与权重正确；
个别行翻转不代表串味。

对比 LoRA 与基座的吞吐时**必须固定输出 token 数**（`SamplingParams(ignore_eos=True)`）：
适配器会改写输出、部分序列提前 EOS，不固定 token 数时「总 token 少 12%」会被误读成加速或减速。

## 6. 310P 基础设施三处修正（2026-11-21，收益不限于 LoRA）

做多 LoRA 并发压测时暴露的三个**与 LoRA 无关**的 310P 缺陷，全部已修：

1. **投机状态回写必须逐槽切片拷，不能用 `index_select`/`index_copy_`**
   （`flashengine/engine/recurrent_state.py:167 commit_verify`）。
   Ascend 按**索引张量长度**各自编译 aclnn 算子：`index_select(1, src)` 首次遇到
   n=1/2/4 分别要 **12.4 / 9.5 / 13.5 s**，逐槽 `cache[:, dst].copy_(cache[:, src])`
   首次 42 ms、稳态 11 ms。症状是「并发每多一条就 6.7 s 长停顿」，
   修后同负载墙钟 18.09 s → **8.55 s**、bs=1 每步 359 → **192 ms**，
   参考配置稳态解码 23.2 → **25.4 tok/s**（E 恒为 5.100，正确性不变）。
2. **`gpu_mem_util` 在 310P 有下限，且设备内存算宿主 shmem**：
   Qwen3.5-9B 常驻（权重 20.4 + recurrent 6.7 + draft 2.4）≈ **30.1 GB**，
   util < 0.34 会直接 `MemoryBudgetError`（引擎预算=0）；而 util 0.70（≈62 GB KV）
   叠加 RAGFlow 5.5 GB 会把 87 GB 宿主 memory+shm 吃满，**OOM killer 静默杀进程**
   ——症状是 `TBE Subprocess[...] main process disappeared!` + `30 leaked semaphore
   objects`，`npu-smi` 仍 Health OK，极易误判成竞态/驱动崩溃。
   dmesg 证据：`Killed process ... shmem-rss:21767984kB`。**310P 多 LoRA 建议 0.50**。
3. **GDN recurrent 前缀快照改为设备侧常驻**：新能力位
   `BackendBase.prefix_snapshots_on_device`（CUDA 仍走异步锁页 D2H，310P 置 True）。
   原来 310P 上每个块边界都做**阻塞** D2H（单序列状态 ≈49 MiB），8 条互不相同的
   prompt 实测 `saved=3 / 147.4 MiB` 却 `prefix_hits=0`——纯税：TTFT mean +119 ms、
   p99 +1064 ms、TPS −1.4%（解码 28.3 vs 28.4 几乎不变）。改设备侧后拷贝变成同流
   D2D，只剩显存占用；启动摘要会打印 `device-resident snapshots`。


## 7. 离线融合：把适配器烘进基座权重（`scripts/merge_lora.py`，2026-11-23）

**什么时候用**：只要 1 个适配器、且愿意为它付一份模型权重时，把 LoRA 永久相加进
基座比运行时 `--lora-module` 更省：少一套 330 MB 适配器张量、少每步的增量 GEMM 与
定长掩码、服务启动参数也不必带适配器名。代价是**每个适配器一份 ~19 GB 模型目录**，
换适配器 = 换服务；要 8 个适配器热切仍走引擎的运行时 LoRA（§1–§2）。

**用法**（纯 CPU、不占 NPU；容器内用 `/data2/fe-t5-venv/bin/python3`，
宿主 `python3` 没有 safetensors）：

```bash
# 最小用法：只给基座 + 适配器（--out 省略时产出 <适配器目录>-merged）
python -m scripts.merge_lora --base /data/Qwen3.5-9B \
    --adapter /data/lora/Qwen3.5-9B-lora-642 --out /data2/merged/a3

# 310P 部署要 fp16 时显式指定；校验强度可用 all / off / 正整数
python -m scripts.merge_lora --base /data/Qwen3.5-9B \
    --adapter /data/lora/Qwen3.5-9B-lora-642 --out /data2/merged/a3 \
    --dtype float16 --verify-tensors all

# 干跑（只报映射与体积，不落盘）/ 批量 / 只审适配器（不需要基座）
python -m scripts.merge_lora --base /data/Qwen3.5-9B --adapter <adapter> --check
python -m scripts.merge_lora --base /data/Qwen3.5-9B \
    --adapters-dir /data/lora/adapters --all --out /data2/merged --dtype float16
python -m scripts.merge_lora --adapter /data/lora/qwen35-35b-a3b-lora --audit-adapter
```

**训练参数全部自动读，不需要手工填**（只需选基座、适配器、输出目录）：

| 参数 | 来源 | 说明 |
|---|---|---|
| `rank` / `alpha` / `use_rslora` | `adapter_config.json` | 另读 `rank_pattern` / `alpha_pattern` 逐模块覆盖 |
| `scaling` | 上式推导 | `alpha/r`，rsLoRA 为 `alpha/sqrt(r)` |
| dtype | 基座 `config.json` | 顶层 `dtype`/`torch_dtype` → `text_config` → `llm_config`/`language_config`，全缺省则 bfloat16 |
| 输出目录 | `--out`，可省略 | 省略 = `<适配器目录>-merged` |
| 分片大小 | 固定默认 | 4 GiB（`--shard-size-gb` 可改） |
| 校验 | 默认开启 | `auto` = 32 个张量且覆盖每种模块模式 |

**数学与命名**：`W += (alpha/r)·(B@A)`（`use_rslora=true` 时 `alpha/sqrt(r)`，
`rank_pattern`/`alpha_pattern` 逐模块覆盖）。适配器键是 PEFT 的
`base_model.model.model.language_model...`，剥掉 `base_model.model.` 后与基座
`model.safetensors.index.json` 的键同名；仅当基座换了多模态外壳前缀时按
`model.language_model.` / `language_model.` / `model.` 顺序回退匹配。
`in_proj_qkv` 在基座里是**单个** `[3*dim, in]` 张量、B 的行已覆盖 q/k/v 全序，
所以整张量相加即可——拆 `in_proj_qkvz`/`in_proj_ba` 只是引擎内部的融合细节
（`flashengine/lora/loader.py`），离线维度上不需要。

**安全阀**（任一不满足就报错退出，绝不产出「部分融合」的模型）：

- 适配器模块在基座里找不到、两个模块映射到同一张量、只有 `lora_A` 没有 `lora_B`
  → 退出码 2；
- `use_dora` / `bias != none` / `modules_to_save` / `target_parameters` / 出现
  `lora_magnitude_vector` 等非 `lora_A|lora_B` 张量 → 明确拒绝（不近似）；
- `visual.` / `mtp.` / `conv1d` / `lm_head` / `embed_tokens` 按引擎 loader 同口径跳过；
- 磁盘余量按基座体积 ×1.05 预检；输出目录非空需 `--overwrite`。

**产物自洽**：先写 `merge-tmp-*.safetensors`，全部写完后按最终分片数改名并生成
`model.safetensors.index.json`（`total_size` 为实际字节数）；`config.json` /
`generation_config.json` 的 `dtype`/`torch_dtype`（含 `text_config` 内）同步改写；
`--dtype` 作用于**全部**张量，避免「语言层 fp16 + 视觉塔 bf16」的混合产物；
**自动校验**（默认开启，`--verify-tensors auto`）：从产物里回读抽样张量，用
`base + s·B@A` 在 CPU 上重算并比对（atol：fp16 5e-3 / bf16 5e-2 / fp32 1e-5）；
取样先保证**每种模块模式至少一个**（`layers.N` 归一化后去重），再均匀补足到 32 个，
所以 9B 的 12 种模式、35B 的 13 种模式都会被验到。`all` 校验全部融合张量（9B 248 个，
多花一两分钟），`off` 关闭，正整数为自定义抽样数；校验失败即报错退出（退出码 2），
不会留下「看起来成功」的产物。

**测试**：`tests/test_merge_lora.py`（**17 项**，容器内
`python -m unittest tests.test_merge_lora -v`）覆盖数学等价、多分片与 index 自洽、
未融合张量透传、dtype 与 config 改写、`--check` 不落盘、未命中模块/DoRA/缺 B/
非空输出目录报错、pattern 缩放，以及本轮新增的「默认行为」：dtype 自动推断（含
`torch_dtype` 优先与兜底）、`--verify-tensors auto` 覆盖全部模块模式、省略 `--out`
时落到 `<适配器>-merged`、MoE 夹具。

**板上实跑（2026-11-23，容器 vllm018，Qwen3.5-9B + `Qwen3.5-9B-lora-642`）**：
248 个目标模块全部命中（`modules=248 targets=248 skipped=0 scaling=['4']`，
r=8/alpha=32），输出 5 分片、19,306,208,736 字节（≈18.0 GiB）、775 个张量与基座
一一对应；与既有手工融合产物 `/data/Qwen3.5-9B-lora642-fp16` 抽样 16 个张量
（含 `in_proj_qkv`/`in_proj_z`/`o_proj`/`down_proj`/`conv1d` 与视觉塔）
**最大绝对误差 0.0（bit 级一致）**，dtype 同为 fp16；脚本自身的 `--verify-tensors`
对基座重算的最大绝对误差 1.2e-04，属 fp16 舍入量级。即本脚本与历史手工融合等价，
可直接替换现有部署目录。

**MoE（Qwen3.5-35B-A3B）同样一条路径（2026-11-23）**：`--adapter` 指向
`v1-20260923-094720/checkpoint-2206`（已放到板上 `/data/lora/qwen35-35b-a3b-lora`），
`--audit-adapter` 实测结果：`r=32 alpha=64` → **scaling 2**（rsLoRA 关）、适配器张量为
bf16（加载即转 fp32）、**350 个模块 / 45.0M delta 参数 / 13 种模式**：

| 层（40 层） | 命中模块 | 数量 |
|---|---|---|
| 30 层 GDN | `linear_attn.in_proj_a` / `in_proj_b` / `in_proj_qkv` / `in_proj_z` / `out_proj` | 30×5 |
| 10 层全注意力 | `self_attn.q_proj` / `k_proj` / `v_proj` / `o_proj` | 10×4 |
| 40 层 MoE | `mlp.shared_expert.gate_proj` / `up_proj` / `down_proj` | 40×3 |
| 40 层 MoE 门控 | `mlp.shared_expert_gate`（输出 1 维标量门） | 40 |

**关键点：该适配器不含路由专家**（没有 `mlp.experts.<i>.*`），`shared_expert*` 在 HF
里就是普通 `nn.Linear`，所以 35B-A3B 与 9B 走完全相同的「同名键 + 整张量相加」路径，
不需要任何 MoE 专用分支。若将来适配器按 `mlp.experts.<i>.<proj>` 逐专家训练，Qwen 系
检查点本身就是逐专家存权重（`/data/Qwen3-30B-A3B` 的 index 可查），同名键照样能直接
融合；只有基座把专家堆叠成融合张量（`[num_experts, out, in]`）时，形状校验才会报错并
提示「需要逐专家导出的适配器」。

**新增 `--audit-adapter`（不需要基座、不需要输出目录）**：打印 rank/alpha/缩放、按层归一化
的模块模式与计数、delta 参数量、是否命中逐专家权重，并给出结论，用于在拿到基座前先确认
适配器可融合性：

```bash
python -m scripts.merge_lora --adapter /data/lora/qwen35-35b-a3b-lora --audit-adapter
# → modules=350 delta_params=45.0M scaling=2 … 结论: OK
```

同轮其它加固：`fan_in_fan_out=true` 明确拒绝（Conv1D 权重按 `[in, out]` 存储，delta 需
转置，静默相加会算错）；`--dtype` 对全部张量统一；形状不一致时的报错区分「模块未命中」
与「MoE 专家融合张量」两种情况。单测增至 **11 项**（新增 MoE 基座夹具：shared_expert 三项
+ `shared_expert_gate` + 逐专家 `experts.0/1`，覆盖 scaling、融合数学、未命中专家透传、
`--audit-adapter` 不写盘）。

**35B-A3B 目前只差基座**：板上有 `/data/Qwen3.5-9B`、`/data/Qwen3.5-4B`、
`/data/Qwen3-30B-A3B`（Qwen3 代次，无 shared expert），没有
`Qwen3.5-35B-A3B`；适配器里的 `base_model_name_or_path` 是训练机路径
`/mnt/si003010rsbi/default/models/Qwen3.5-35B-A3B`。要出产物需把该基座放到板上
（bf16 ≈70 GB；输出同样 ≈70 GB，按默认 4 GiB/分片约 18 片，`/data2` 现余 336 GB），
命令与 9B 完全一致，只换 `--base/--adapter/--out`。
