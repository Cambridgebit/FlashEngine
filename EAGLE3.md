# EAGLE3 × nano-vLLM 耦合实现说明

> 本文面向想读懂 / 改动本项目投机解码路径的开发者，说明 EAGLE3 draft 模型是如何与
> nano-vLLM 引擎（调度、KV 管理、CUDA Graph、多硬件后端）耦合的。
> 全部内容以源码为准：`nanovllm/models/eagle3.py`、`nanovllm/engine/model_runner.py`、
> `nanovllm/engine/llm_engine.py`、`nanovllm/engine/scheduler.py`、`nanovllm/models/qwen3.py`。
> 性能数据与历史口径见 [AGENTS.md](AGENTS.md)。

---

## 0. TL;DR

- **draft 是"半个模型"**：单层 decoder（midlayer）+ fc(3H→H) + 压缩词表头；embedding 与
  target **共享**，KV cache 与 target **同槽位布局、独立存储**。
- **target 是"验证器"**：只在 3 个固定层前捕获 aux 隐层（`Qwen3Model` 加一个元组字段），
  验证时 `[末位提交 token, d1..dk]` 一次前向，顺带产出供 draft 用的 aux。
- **编排全在 ModelRunner**：一轮投机 = 采样行引导 → draft 链式起草 → target 验证 →
  图内链式接受 → refill（虚构窗口重放 + 下一轮采样行）。模型层保持"哑"的，
  所有口径、状态、缓冲都在引擎层。
- **eager 与图同构**：eager 轮直接复用图轮 body（`_sg_round(use_cuda_graph=False)`），
  冷启动只在 eager 里多一步"采样行引导"；图捕获失败自动降级，功能不受影响。
- **整轮单图**：draft 链 + verify + 接受 + refill + 采样全部录进一张 CUDA Graph，
  每轮 2 次 H2D（pinned 元数据 blob）+ 1 次 D2H（commit_buf），实测"spec 必须有图"
  （CoreX：eager spec 11.3 tok/s → 图 103.7 tok/s）。

---

## 1. 设计原则

| 原则 | 落地方式 |
|---|---|
| 模型层最小侵入 | target 只加 `aux_hidden_state_layers` 元组；draft 是普通模块；不改算子层签名 |
| 编排集中在引擎层 | `ModelRunner` 独占投机流程；`LLMEngine.step_spec()` 只做步进分派与状态清理 |
| eager / 图同构 | 图轮 body（`_sg_*` 方法族）同时是 eager 轮的执行体，杜绝两套实现数值漂移 |
| 状态按 seq_id 键控 | batch 组成每轮会变，跨轮状态（carry/隐层）用 `dict[seq_id]`，不用位置索引 |
| 能力位裁决，只降速不崩溃 | 后端声明 `supports_spec` / `supports_spec_graph` / `graph_sample_default`；图捕获失败 try/except 降级 |
| 图安全优先 | 元数据走 pinned blob + 图内 scatter；采样用无动态 shape 的 Gumbel fast path |

约束：**spec 仅支持 TP=1**（draft 种子状态在 rank0，`config.py` 显式断言）。

---

## 2. 组件地图

```
config.py                 spec_model / spec_num_tokens / spec_aux_layers 字段 + 校验
engine/llm_engine.py      step_spec(): prefill 分派 / decode 轮编排 / 完成态清理
engine/scheduler.py       prepare_spec_round() 容量预分配；commit_spec() 多 token 提交 + EOS 截断
engine/block_manager.py   ensure_capacity() 投机轮预分配 block
engine/model_runner.py    ★ 核心：run_prefill_spec / spec_round / _sg_* / MetaBlob / 图捕获
models/qwen3.py           Qwen3Model.aux_hidden_state_layers：aux 隐层捕获点
models/eagle3.py          Eagle3DraftModel：fc + midlayer + 压缩词表头 + d2t/t2d
layers/sampler.py         forward_native()：图安全采样（top-k 掩码 + Gumbel-argmax）
backends/*                能力位、slot dtype、KV 布局、图 API 工厂
```

---

## 3. 配置与初始化

### 3.1 配置项（`config.py`）

| 字段 | 默认 | 说明 |
|---|---|---|
| `spec_model` | `""` | draft checkpoint 目录（官方 EAGLE-3 权重）；空 = 关闭投机 |
| `spec_num_tokens` | `4` | 每轮起草 token 数 k（文档示例/基准常用 k=3） |
| `spec_aux_layers` | `()` | aux 捕获层索引；未显式给则取 `(2, n//2, n-3)`（与官方 EAGLE / vLLM 的 Qwen3 捕获点一致，如 1.7B n=28 → (2,14,25)） |

校验（`__post_init__`）：`tensor_parallel_size == 1`、draft 目录存在、`k >= 1`、恰好 3 个 aux 层
（EAGLE3 的 fc 输入就是 3 层隐层拼接）。

### 3.2 初始化顺序（`ModelRunner.__init__`）

1. 加载 target（Qwen3ForCausalLM）；
2. spec 启用时：**先**打开 aux 捕获（`model.model.aux_hidden_state_layers = spec_aux_layers`，
   必须在 warmup 前，否则 prefill 前向不返回 aux），再从 draft 目录读 config、构建
   `Eagle3DraftModel` 并加载权重；
3. `warmup_model()`：spec 路径连 draft prefill 一起预热（dummy 序列，预热后清空 `spec_hidden`）；
4. `allocate_kv_cache()`：**draft KV 池的显存一并计入预算**（见 §4.3）；
5. 分配 spec 专用缓冲：`_sblob`（pinned 元数据）与 `_sg`（图静态 buffer）——
   在 KV 分配**之后**，保持 KV 预算口径与基线一致；
6. 图捕获：普通路径 `capture_cudagraph()`；spec 路径过三重门后 `capture_spec_cudagraph()`（§9）。

### 3.3 draft 权重加载（`eagle3.py: load_eagle3_draft`）

- 支持 `pytorch_model.bin` 与 safetensors；
- `q/k/v_proj → qkv_proj`、`gate/up_proj → gate_up_proj` 的分片重映射**复用** Qwen3 的
  `PACKED_MODULES_MAPPING`（draft midlayer 与 target 同命名约定）；
- `d2t` / `t2d` 是注册 buffer，直接 `copy_`；
- **checkpoint 中不含 embedding**——`Eagle3DraftModel.__init__` 直接收 target 的
  `embed_tokens` 模块引用，权重天然共享。

---

## 4. 两个模型怎么"握手"

### 4.1 aux 隐层捕获（target → draft 的信息通道）

`Qwen3Model` 增加一个字段：

```python
self.aux_hidden_state_layers: tuple[int, ...] = ()   # 空 = 普通路径
```

forward 中在**运行第 i 层之前**记下该层的输入隐状态（含残差、未过 final norm），
即"第 i-1 层的完整输出"；结束时返回 `(final_hidden, cat([low, mid, high], -1))`，
拼接顺序与 draft fc 权重的训练顺序一致。普通路径（元组为空）零开销。

EAGLE3 的 fc 以 `cat(aux_low, aux_mid, aux_high)`（3×hidden）为输入投影回 hidden——
这三个捕获点就是 draft 每步的"条件信号"来源。

### 4.2 draft 结构（`Eagle3DraftModel`）

```
embed_tokens   ← 与 target 共享（同一模块引用）
fc             Linear(3*hidden → hidden)：project_aux()
midlayer       单层 decoder（Eagle3MidLayer）：
                 - 注意力输入 = cat([input_layernorm(embeds), hidden_norm(x)])，2*hidden
                 - llama 风格：无 q/k norm；qkv 输入维度与 hidden 解耦
                 - 写自己的 KV cache（单层 → draft_kv_cache[.,0]）
norm + lm_head 输出投影到 draft_vocab_size（压缩词表，如 32k vs target 152k）
d2t            draft id → target id 的偏移映射：real_id = draft_idx + d2t[draft_idx]
```

两个前向入口：
- `forward_decode(positions, seed, token_ids) → (draft_hidden, draft_logits)`：链式起草步，
  `draft_hidden` 直接作为下一步 seed；
- `forward_prefill(positions, seed, token_ids) → hidden`：varlen 路径（context 由调用方
  经 `set_context` 设置），用于 draft prefill 与 refill 的虚构窗口重放。

### 4.3 KV cache：同槽位、双池

- draft 与 target **共用同一 `block_table` 与槽位映射**（`_slot_of(seq, p)`），
  但各有一块独立连续池：`self.kv_cache`（target，L 层）与 `self.draft_kv_cache`
  （draft，1 层 midlayer）。布局一致性由 `backend.get_kv_cache_shape` 保证
  （cuda/ascend = BSHD、corex = HND，同后端两模型必同布局）。
- 显存预算（`allocate_kv_cache`）在算 `block_bytes` 时把 draft 的层开销一并扣除，
  因此 `num_kvcache_blocks` 对两个池同时够用。
- 被 draft 写脏、但提案被拒绝的槽位**不清理**：注意力读窗恒在 `context_len` 内，
  残留值会被后续轮次写入覆盖（§7.5、§10.4）。

---

## 5. 核心口径：shifted draft 流

这是全路径统一的数据流约定（官方 EAGLE cnets 的 TTT 流式口径，与 vLLM padded 分支对齐）：

> **rope 位置 p 的 draft 行**：embed 输入为 `tok@p+1`（右移一位），条件 seed 为
> `fc(aux@p)` 或上一步 draft 隐层（look-ahead aux：target 已"看过"该行 embed 的那个
> token），输出预测 `tok@p+2`。

draft prefill 与 decode 的虚构窗口（refill）都用此口径（embed 右移一位、末槽补
next_token）。由此推导出各阶段的位置/槽位/读窗公式，见 §6、§7 的表格。

记号：`l = len(seq)`（本轮开始时已提交的 token 数），已提交 token 占据位置 `0..l-1`，
`seq.last_token = tok@l-1`；`k = spec_num_tokens`，`rows = k+1`。

---

## 6. Prefill 耦合：`run_prefill_spec`

引擎 `step_spec()` 的 prefill 分支（与普通 prefill 共用 `prepare_prefill` 的全部输入组装）：

1. **target 前向**：因 aux 捕获已打开，返回 `(hidden, aux)`；正常采样出首 token；
2. **draft 同 chunk prefill**：`seed = fc(aux)` 后过 midlayer——
   embed 取**右移一位**的输入：`cat([input_ids[off+1 : off+len], 首个采样 token])`，
   即 rope p 的行 embed `tok@p+1`、末行补 next_token（§5 口径）。
   context 与 target 完全一致（token 布局/槽位相同）→ prompt 每个位置的 draft KV 都写入；
3. **落状态**：本步完成 prefill 的序列记录
   `spec_hidden[seq_id] = aux 最后一行`（即预测首个采样 token 时的 aux@l-2，维度 [1, 3H]），
   并置 `seq.spec_cold = True`（§8）。

分块 prefill（chunked prefill）天然兼容：只有 chunk 补完 prompt 的那一步才记录 `spec_hidden`。
draft prefill 的输出**丢弃**——它的意义只是把 draft KV 流填到 `l-2`；
"采样行隐层"由冷启动轮复算得到（§8.2）。

---

## 7. Decode 耦合：一轮 `spec_round` 的五个阶段

引擎 decode 分支：`prepare_spec_round(k)`（§10.1）→ `spec_round(active)` →
`commit_spec()`（§10.2）→ 清理完成序列的 spec 状态。

`spec_round` 的分派逻辑：

```python
if spec_graphs is not None and seqs and not any(seq.spec_cold for seq in seqs):
    bucket = _bucket(spec_graph_bs, len(seqs))
    if bucket is not None: return _spec_round_graph(seqs, bucket)   # 图路径
return _spec_round_eager(seqs)                                      # eager 路径
```

有捕获图、桶覆盖、且**全部序列已有 carry**（无冷启动）才走图；否则整批走 eager。
两条路径执行**同一个 body**（`_sg_round`），仅引导方式不同：

```
┌──────────────────────────────────────────────────────────────────────┐
│ _sg_round(bs)  ——单图整轮（eager 时同函数逐算子执行）                    │
│                                                                      │
│  ① _sg_scatter     pinned 元数据 blob → 各静态 buffer（图内分发）       │
│  ② _sg_draft_chain 链步 1..k-1：draft 流式起草 + 逐步采样提案            │
│  ③ _sg_verify      target 一次前向 [tok@l-1, d1..dk]，图内采样 + 暂存 aux │
│  ④ _sg_acceptance  图内链式接受：最长匹配前缀 m，组装 commit_buf         │
│  ⑤ _sg_refill      虚构窗口重放：draft KV 回填 verify 槽位              │
│                     + 行 m 采样出下一轮 draft1 + carry_seed            │
└──────────────────────────────────────────────────────────────────────┘
  图路径：_fill_spec_meta → [padding 复位] → _load_spec_carry
          → _push_spec_meta(2×H2D) → graph.replay() → _readback_spec_commit(1×D2H)
  eager 路径：采样行引导（§8.2）→ 同一 _sg_round(use_cuda_graph=False) → 同一读回
```

### 7.1 链式起草（`_sg_draft_chain`，链步 i = 1..k-1）

| 项 | 值 | 说明 |
|---|---|---|
| rope 位置 | `l + i - 2` | §5 口径：行预测 `tok@l+i` |
| embed | `drafted_ids[i-1]` | 上一步的采样提案 = `tok@l+i-1 = tok@p+1`，**绝不回退 last_token**（C1 教训，§12） |
| 条件 seed | 上一步 midlayer 输出（步 1 = `carry_seed`） | raw 输出，未过额外 norm |
| KV 写槽 | `slot_of(l+i-2)` | draft 流延伸 |
| 读窗 | `context_lens = l+i-1` | 提交前缀 + 本行自己的流槽——**任何接受数下都不读陈旧槽** |
| 采样 | draft 词表（top_k 上限取 `draft_vocab_size`） | `map_to_target_ids` 后写入 `drafted_ids[i]` |

链步 i=1 的 seed 来自上一轮（carry 或冷启动引导），i≥2 在本轮内递推。

### 7.2 target 验证（`_sg_verify`）

verify 输入在 `_sg_round` 内组装：`v[:, 0] = chain_tokens`（tok@l-1）、
`v[:, 1:] = drafted_ids[0..k-1]`（d1..dk），共 rows = k+1 行。

| 项 | 值 |
|---|---|
| rope 位置 | 行 j（0..k）：`l-1+j` |
| KV 写槽 | `slot_of(l-1+j)`（target 池） |
| 读窗 | 每段 K 窗长 `l+k`（`v_cu_k`），varlen + block_table 右对齐 → 行 j 读 `[0, l+j)` |
| aux | `v_aux` 暂存 3 层捕获拼接（供 refill 的 look-ahead 条件） |
| 采样 | 图内 `forward_native`，行 j 的采样 `t_j` 即 target 对位置 `l+j` 的预测 |

注意 `v_cu_q = arange(bs+1) * rows` 是**捕获后恒定**的（每序列恰好 rows 行，无需每轮更新）；
`max_seqlen_k` 用 `max_model_len` 静态上界——varlen kernel 只用它做 grid sizing，
正确性由 `cu_seqlens` 决定，避免把随轮次变化的 python int 烘焙进图。

### 7.3 图内接受（`_sg_acceptance`）

全静态算子（无数据依赖 shape，图安全）：

```python
match = t[:, :k] == d          # target 逐行预测 vs 提案
m = match.all(-1) ? k : match.int().argmin(-1)     # 最长匹配前缀
commit_buf[:, :rows] = where(ar <= m, [d_0..d_{m-1}, t_m, -1...], -1)
commit_buf[:, rows]  = m + 1   # 提交 token 数 = 接受数 + 1 bonus
```

即提交内容 = **m 个被接受的提案 + 1 个 bonus**（target 在首个分歧行的预测），
cnt 打包在 `commit_buf` 最后一列。

### 7.4 refill：虚构窗口重放（`_sg_refill` + `_refill_ids`）

对齐 vLLM `eagle_prepare_inputs_padded_kernel` 的 shift 口径，用 verify 的
**同一套位置 / 槽位 / aux / cu** 重放 draft midlayer：

| 项 | 值 |
|---|---|
| rope 位置 / KV 写槽 | 行 j：`l-1+j` / `slot_of(l-1+j)`——与 verify 完全相同（写的是 draft 池） |
| 条件 | 行 j：`fc(v_aux[j])`（look-ahead aux） |
| embed | verify token 右移一位（`t_ext = [t_1..t_k, t_k]`），**行 m 覆盖为 bonus `t_m`** |
| 行 m 输出 `h_m` | `norm(h_m) @ lm_head` 采样出**下一轮 draft1**；`h_m` 写入 `carry_seed`（下一轮链种子） |

作用有二：① 把本轮已知的 token 流回填进 draft KV 槽位，使下一轮链步的读窗
（提交前缀 + 自己的流槽）连续可读；② 产出下一轮的采样行（carry）。
行 m（`index_to_sample = 段尾 - 被拒数`，vLLM kernel 口径）就是下一轮的"采样行"。

### 7.5 统一读回（`_readback_spec_commit`）

**每轮唯一一次大批量 D2H**：`commit_buf[:n].tolist()`。此后：

- `results[i] = row[:cnt]`（提交 token 列表）；
- `spec_carry[seq_id] = (carry_seed[i:i+1].clone(), drafted_ids[0][i].item())`
  ——refill 行 m 的隐层与采样提案，**留在 GPU 上跨轮传递**（不落 host）；
- `spec_hidden[seq_id] = v_aux 行 m`（eager 复算采样行 / 图不可用时的 fallback 用）；
- `seq.num_scheduled_tokens = 0`（调度状态复位）。

---

## 8. 跨轮状态与冷启动

### 8.1 三份状态

| 状态 | 位置 | 内容 | 生命周期 |
|---|---|---|---|
| `spec_carry` | `ModelRunner.spec_carry: dict[seq_id → (隐层[1,H], draft1)]` | 下一轮链步 1 的 seed 与首提案 | 每轮读回时更新 |
| `spec_hidden` | `ModelRunner.spec_hidden: dict[seq_id → aux 行[1,3H]]` | 采样行引导用的 aux（fc 的输入） | prefill 落、每轮读回刷 |
| `spec_cold` | `Sequence` 实例属性 | 冷启动标记 | prefill 置 True；首轮 eager 结束清 False |

按 seq_id 键控的原因：连续批处理下 batch 组成每轮变化，batch 位置索引不安全。
序列结束（EOS/max_tokens）由引擎 `_drop_finished_spec_state` → `drop_spec_state(seq_id)`
清理两本字典；**抢占**（preempt）后重新 prefill 会重置 `spec_cold` 并覆盖 `spec_hidden`，
carry 在下一轮 eager 中被重算覆盖——冷启动机制同时是抢占恢复路径。

### 8.2 冷启动：采样行引导（`_spec_round_eager` 开头）

prefill 后首轮无 carry（`spec_cold=True`）→ 整批走 eager，先补一步"采样行"：

```
h_seed = cat(spec_hidden[seq])          # aux@l-2（prefill 记录）
seed   = fc(h_seed)
row_m_hidden, logits = draft.forward_decode(
    positions = l-2,                    # rope p = l-2
    seed      = seed,                   # 条件 = fc(aux@p)
    token_ids = seq.last_token)         # embed tok@l-1 = tok@p+1   ← §5 口径
draft1  = map_to_target_ids(sample(logits))
```

这一步与 draft prefill 的最后一行**同位同内容**（同 rope、同 embed、同条件），
即把 prefill 丢弃的那个"采样行隐层"复算回来；它预测 `tok@l`，采样出 draft1。

之后把 `carry_seed ← seed`（注意：是 **fc(aux) 本身，即引导行的输入**，不是
`row_m_hidden`——与旧 eager 实现链步 1 逐位一致，已验证口径，勿"修正"）、
`drafted_ids[0] ← draft1`，再执行与图轮同构的 `_sg_round` body。

**冷→图交接**：eager 轮结束后 `spec_carry = (row_m_hidden, draft1)`（采样行隐层），
图轮稳态的链步 1 seed 则来自 refill 行 m 的输出——两者不同源，均为已验证现状
（§12）；读回写入的 refill 产物不跨 eager→图传递。之后 `spec_cold=False`，进图。

---

## 9. CUDA Graph：spec 单图整轮

### 9.1 为什么必须整轮单图

一轮 = k-1 个链步（各含 attention+MLP+采样）+ verify（一次 target 前向 + 图内采样）+
接受/提交 + refill（一次 draft varlen 前向 + 采样），全是小 kernel，
host launch 税会吞掉投机收益：CoreX 实测 **eager spec 11.3 tok/s vs 图 103.7 tok/s**。
每桶一张图覆盖整轮，replay 一次即完成全部五个阶段。

### 9.2 捕获（`capture_spec_cudagraph`）

- 桶表 `_graph_buckets(max_bs)`：`[1,2,4,8] + 16 步进`；未覆盖 `max_num_seqs` 时追加尾桶并告警；
- **从大到小**逆序捕获，共享 `spec_graph_pool`（内存池复用）；
- 每桶先 warmup（triton/flash kernel lazy init 必须在 capture 前完成；warmup 数据用
  零/-1——slot=-1 时 store kernel 跳过 KV 写、ctxlen=0 合法），再
  `backend.graph_ctx` 内捕获 `_sg_round(bs, use_cuda_graph=True)`；
- **try/except 优雅降级**：捕获异常（如 vendor kernel 含 host 同步/动态 workspace）时
  打印警告、释放图池、`spec_round` 自动回落到同构 eager 轮——只降速、不崩溃。

捕获门（三重）：`TP==1` 且 `backend.supports_spec_graph` 且
（`NANOVLLM_SPEC_GRAPH` env 优先，否则 `backend.graph_sample_default`——
spec 图无条件录 Gumbel 采样，RNG 图安全性未验证的后端不默认尝试）。

### 9.3 数据通路：MetaBlob + 图内 scatter

每轮的动态元数据（链步/verify 的位置、槽位、采样参数、block_table、cu_k）由 host 写进
`_sblob`（声明式 pinned blob，`MetaBlob` 类统一管理 int64/f32 两平面与命名视图）：

```
host:  _fill_spec_meta()  只写 [0:n] 行（+ 桶缩小时的 padding 复位）
H2D:   _push_spec_meta()  每个 dtype 平面一次整块 copy_（共 2 次，零逐段切片）
图内:  _sg_scatter()      设备孪生 → 各静态 buffer（int64 → 图 buffer dtype 转型在 copy_ 内）
```

配套事实：
- blob 统一存 int64；槽位类图 buffer 的 dtype 由 `backend.graph_slot_mapping_dtype()`
  决定（CUDA int32 / CoreX int64），`context_lens`/`cu`/`block_tables` 两后端恒 int32
  （CoreX 的 ixformer paged_attention 硬性检查 Int&&contiguous&&cuda）；
- `carry_seed` / `drafted_ids[0]` 由 host 直写设备 buffer（`_load_spec_carry`），
  与 blob 无关；
- padding：桶缩小时把尾段补 -1/0，且 `cu_k[n+1:] = cu_k[n]`——尾段 K 窗零长、
  cu 单调不降，kernel 跳过空段。

### 9.4 通信量预算

| 每轮 | 次数 | 内容 |
|---|---|---|
| H2D | 2 | `_sblob` 的 i64/f32 平面整块 |
| D2H | 1 | `commit_buf`（token + cnt 打包）；carry/aux 均留 GPU |

---

## 10. 调度、KV 与采样的配合

### 10.1 容量预分配（`Scheduler.prepare_spec_round`）

每个 running 序列保证 block_table 能容纳 `len + k + 1` 个 token
（draft 流最深写槽 = `l+k-3`，verify 覆盖到 `l+k-1`，再留 1 余量）；
空闲块不足则**抢占**该序列（丢弃 KV 退回 waiting，等重 prefill——之后走冷启动）。
`BlockManager.ensure_capacity` 预分配的块在拒绝回滚后**不回收**：被拒提案写脏的槽位
靠覆盖失效（§4.3）。

### 10.2 提交与截断（`Scheduler.commit_spec`）

逐 token `append_token`，逐 token 判 `_maybe_finish`（EOS / max_tokens）提前截断并结束
序列。注意：**投机产生的 token 不做 block hash**——前缀缓存只覆盖 prefill 阶段，
避免被拒提案污染哈希表。EOS 截断发生在调度层，图内 cnt 不感知 → 统计口径对 E 有
~0.2% 的轻微高估（可忽略）。

### 10.3 采样（`Sampler.forward_native`）

- draft 侧 top_k 上限取 `draft_vocab_size`（ctopks），verify 侧取 target vocab；
- 图内一律走 Gumbel fast path（top-k 掩码 + Gumbel-argmax，无动态 shape、无数据依赖
  分支；PyTorch CUDA RNG 是 graph-safe 的，replay 每次产生新随机数）；
  `fast_top_p=False` 可回退 legacy 全流水（host 属性分支，捕获时常量折叠）。

### 10.4 残留 KV 的无害性

被拒提案 / padding 写入的 KV 槽位不清理，安全性由两个不变式保证：
① 注意力读窗恒在 `context_len` / `cu_seqlens` 界内（链步读窗 = 提交前缀 + 自己的流槽）；
② 任何位置在被读取前必然先被本轮有效写入覆盖（decode/prefill 先写后读）。

---

## 11. 多后端适配（CUDA / CoreX / Ascend 310P）

spec 对后端的依赖全部参数化（能力位 + 工厂方法，无 `torch.cuda.*` 直呼）：

| 后端 | `supports_spec` | `supports_spec_graph` | slot dtype | KV 布局 | 说明 |
|---|---|---|---|---|---|
| CUDA | ✅ | ✅（实证） | int32 | BSHD | flash-attn varlen/with_kvcache + triton store（slot==-1 跳过写） |
| CoreX | ✅ | ✅（实机验证：单图整轮一次捕获成功，E=2.050 / 103.7 tok/s） | **int64**（`graph_slot_mapping_dtype` 覆写） | HND | ixformer；meta blob 仍存 int64，图内 copy_ 自动转型 |
| Ascend 310P | ❌ 引擎层干净拒绝 | 尝试+降级 | — | — | verify 恒传 block_tables → attention 走整池 4D 分支，TND 算子不消费；`supports_spec=False` 时出现 spec 配置直接报错 |

其余接口面：`new_graph()` / `graph_ctx()`（图 API 工厂，Ascend = NPUGraph）、
`NANOVLLM_SPEC_GRAPH` / `NANOVLLM_GRAPH_SAMPLE` 两个 env 覆盖后端默认。
spec 路径的槽位类 buffer（chain_slots/v_slots/eager slots）统一走
`graph_slot_mapping_dtype()`，两后端共用同一套 `_sg_*` 代码。

---

## 12. 已固化的口径事实（改动前必读，均经位等门验证）

1. **embeds 必须过 `input_layernorm`**——去掉会崩到 E≈1.15；
2. 链步 seed 用 **raw midlayer 输出**（与 vLLM 同）；
3. EAGLE3 checkpoint **无 fc_norm / norm_before_fc**（纯 fc），aux 层默认 (2, n//2, n-3)
   input-to-layer；
4. **eager 轮链步 1 的 seed = fc(spec_hidden)（引导行输入，非其输出）**——与图轮稳态
   （refill 行 m 输出）不同源，均为已验证现状，勿"统一"；
5. **冷→图交接 `spec_carry = (采样行隐层, draft1)`**，refill 产物不跨 eager→图传递；
6. 链步 i 的 embed token **恒为 `drafted_ids[i-1]`**（步 1 = draft1 = tok@l），绝不回退
   last_token——C1 曾在此断裂（接受统计漂移而输出位等），靠 `drafted_ids` 缓冲天然保证；
7. 被拒 draft 的 KV 槽位不清理，靠后续写入覆盖（读窗恒在界内，残留无害，§10.4）；
8. 投机 token 不做 block hash（prefix cache 仅 prefill）；
9. spec 图默认门同时要求 `graph_sample_default`；捕获失败降级 eager spec 轮是**特性**。

---

## 13. 验证钩子与性能锚点

- **基准内置 spec 统计**：`scripts/benchmark --spec-model` 汇总 Acceptance Length (E)、
  token 接受率、逐位置接受率（脚本侧 monkey-patch `spec_round`，图/eager、冷/热四路
  统一从 `_readback_spec_commit` 出数）；`--no-think` 控制 chat template 口径。
- **锚点**（spec_bench coding，k=3，temp0，thinking-off）：
  - RTX 4070 Laptop：E=2.095，spec 108.7 tok/s vs decode 66.2（1.64×）；
  - CoreX 32GB：E=2.050，spec 图 103.7 tok/s（spec eager 仅 11.3——"spec 必须有图"）。
- thinking-on 负载 E 会掉到 1.7~1.8（draft 可预测性低），属正常，勿与锚点直比（AGENTS.md）。

---

## 14. 一次投机轮全流程速查（单序列，k=3）

```
prefill  : target 前向(捕获 aux) → 首 token；draft 同 chunk 右移 prefill → draft KV@0..l-2
           落 spec_hidden=aux@l-2, spec_cold=True

第 1 轮(eager/冷):
  引导   : fc(aux@l-2) + emb(tok@l-1) @ rope l-2 → row_m_hidden, draft1(tok@l)
  链步1-2: seed=fc(aux@l-2)起步 → drafted_ids[1..2]（KV@l-1, l）
  verify : [tok@l-1, d1, d2, d3] @ l-1..l+2 → t0..t3；KV(target)@l-1..l+2
  接受   : m = 最长匹配(t_j vs d_j) → 提交 [d1..dm, t_m]（m+1 个）
  refill : fc(aux@l-1+j) + emb(t_{j+1}) @ l-1+j（行 m = bonus）→ draft KV@l-1..l+2
           行 m 输出 h_m → 采样 draft1' + carry_seed
  读回   : commit_buf(D2H×1)；spec_carry=(h_m, draft1')；spec_cold=False

第 2 轮(图): fill blob → 2×H2D → replay(scatter→链→verify→接受→refill) → 1×D2H
  链步 1 : seed=carry_seed(=h_m), embed=draft1' @ rope l'-1（l' = l+m+1，即重写
           上轮 refill 行 m+1 的流槽，流连续）
```

---

*文档基于 2026-09 代码状态撰写；改代码时以源码与 AGENTS.md 的口径记录为准。*
