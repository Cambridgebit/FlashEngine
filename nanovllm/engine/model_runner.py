import os
import pickle
import time
import traceback
from itertools import accumulate

import torch
import torch.nn.functional as F
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.backends import get_backend
from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.models.eagle3 import Eagle3DraftModel, load_eagle3_draft
from nanovllm.models.dflash import DFlashDraftModel, load_dflash_draft
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model, iter_safetensors


def _bucket(table, n):
    """桶搜索：table 中第一个 >= n 的元素，无则 None（host 侧纯 Python）。"""
    return next((x for x in table if x >= n), None)


def _graph_buckets(max_bs):
    """图桶表：标准模式 [1,2,4,8] + 16 步进；未覆盖 max_bs 时追加尾桶并告警一次
    （否则该并发将静默回退 eager）。"""
    buckets = [x for x in ([1, 2, 4, 8] + list(range(16, max_bs + 1, 16))) if x <= max_bs]
    if not buckets or buckets[-1] != max_bs:
        buckets.append(max_bs)
        print(
            f"[nano-vllm] Added graph bucket {max_bs} because max_num_seqs={max_bs} "
            "is not covered by the standard bucket set.",
            flush=True,
        )
    return buckets


def _refill_ids(t2d, m_idx, rows: int, k: int, ar_f=None, shift_idx=None):
    """refill shifted-ids; ar_f/shift_idx static to keep NPUGraph AICPU-free."""
    bonus = t2d.gather(1, m_idx[:, None])
    if shift_idx is None:
        t_ext = torch.cat([t2d[:, 1:], t2d[:, k:k + 1]], dim=1)
    else:
        t_ext = t2d.gather(1, shift_idx.expand(t2d.size(0), rows))
    if ar_f is None:
        ar = torch.arange(rows, device=t2d.device)
        return torch.where(ar[None, :] == m_idx[:, None], bonus, t_ext).reshape(-1)
    m_f = m_idx.to(torch.float32)[:, None]
    return torch.where(ar_f[None, :] == m_f, bonus, t_ext).reshape(-1)


class MetaBlob:
    """声明式 pinned 元数据 blob：按 (段名, 尺寸) 声明 int64/f32 两个平面，init 时
    一次性分配 host pinned 张量并缓存全部命名视图（host 侧写、设备孪生侧读），
    热路径零逐段切片/view 构造。段偏移 = 声明顺序累加，与旧手工偏移表
    （_pin_off / _meta_off / _metaf_off）逐一对应。"""

    def __init__(self, i64_segments, f32_segments):
        self._i64_off, self._i64_size = {}, {}
        pos = 0
        for name, size in i64_segments:
            self._i64_off[name], self._i64_size[name] = pos, size
            pos += size
        # host 平面恒为 pinned CPU（spec blob 构建时 default device 已是 cuda，
        # 须显式 device='cpu' 才能分配 pinned）。
        self.i64 = torch.zeros(pos, dtype=torch.int64, device='cpu', pin_memory=True)
        self._f32_off, self._f32_size = {}, {}
        pos = 0
        for name, size in f32_segments:
            self._f32_off[name], self._f32_size[name] = pos, size
            pos += size
        self.f32 = torch.zeros(pos, dtype=torch.float32, device='cpu', pin_memory=True)
        self._views = {}
        self._dev = {}

    def offset(self, name):
        return self._i64_off.get(name, self._f32_off.get(name))

    def _slice(self, plane, name):
        if plane == 'i64':
            return self._i64_off[name], self._i64_size[name]
        return self._f32_off[name], self._f32_size[name]

    def view_i64(self, name, shape=None):
        key = ('i64', name, shape)
        v = self._views.get(key)
        if v is None:
            off, size = self._slice('i64', name)
            v = self.i64[off:off + size]
            if shape is not None:
                v = v.view(*shape)
            self._views[key] = v
        return v

    def view_f32(self, name, shape=None):
        key = ('f32', name, shape)
        v = self._views.get(key)
        if v is None:
            off, size = self._slice('f32', name)
            v = self.f32[off:off + size]
            if shape is not None:
                v = v.view(*shape)
            self._views[key] = v
        return v

    def device_i64(self):
        """设备孪生（图内 scatter 的整块拷贝目标），惰性分配于当前默认设备。"""
        if self._dev.get('i64') is None:
            self._dev['i64'] = torch.zeros(self.i64.numel(), dtype=torch.int64)
        return self._dev['i64']

    def device_f32(self):
        if self._dev.get('f32') is None:
            self._dev['f32'] = torch.zeros(self.f32.numel(), dtype=torch.float32)
        return self._dev['f32']

    def device_view_i64(self, name, shape=None):
        key = ('di64', name, shape)
        v = self._views.get(key)
        if v is None:
            off, size = self._slice('i64', name)
            v = self.device_i64()[off:off + size]
            if shape is not None:
                v = v.view(*shape)
            self._views[key] = v
        return v

    def device_view_f32(self, name, shape=None):
        key = ('df32', name, shape)
        v = self._views.get(key)
        if v is None:
            off, size = self._slice('f32', name)
            v = self.device_f32()[off:off + size]
            if shape is not None:
                v = v.view(*shape)
            self._views[key] = v
        return v

    def h2d(self, non_blocking=True):
        """host → 设备孪生整块拷贝（每 dtype 一次，共 2 次 H2D）。"""
        self.device_i64().copy_(self.i64, non_blocking=non_blocking)
        self.device_f32().copy_(self.f32, non_blocking=non_blocking)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        startup_t0 = time.perf_counter()
        self.startup_stats: dict[str, float] = {}
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.backend = get_backend()
        self.dflash_paged = bool(
            config.spec_model
            and config.spec_method == "dflash"
            and getattr(self.backend, "supports_dflash_paged", False)
        )
        if self.dflash_paged and config.spec_num_tokens + 1 > self.block_size:
            raise ValueError(
                "Ascend 310P paged DFlash requires draft block_size <= "
                f"KV-cache block size ({config.spec_num_tokens + 1} > {self.block_size})."
            )
        if config.enable_prefix_cache and self.backend.name == "ascend310p":
            disable_reason = None
            if config.spec_model and config.spec_method == "dflash":
                # A prefix hit skips target prefill for the reused blocks. DFlash
                # also needs the auxiliary target states from those blocks to
                # populate its private projected KV cache, so accepting such a
                # hit would leave the draft context incomplete.
                disable_reason = "DFlash auxiliary states cannot be reconstructed on a prefix hit"
            elif not getattr(self.backend, "supports_splitfuse", False):
                disable_reason = "the runtime lacks paged-attention SplitFuse"
            if disable_reason is not None:
                print(
                    f"[nano-vllm] Ascend 310P prefix cache disabled: {disable_reason}.",
                    flush=True,
                )
                config.enable_prefix_cache = False
        # 设备语义串与 dtype 解析：后端强制 dtype（如 310P 仅 fp16）优先于 hf_config。
        # dtype 取 text_config：qwen3_5 外层多模态壳无顶层 dtype（实测 None），
        # text 侧才有；qwen3 的 text_config 即 hf_config，逐字节同值。
        self.device = self.backend.device_name
        self.dtype = self.backend.default_dtype or config.text_config.dtype
        # hybrid（Qwen3.5）总开关：GDN 状态池/槽生命周期/图与 spec 门控统一由此
        # 分支（qwen3 路径恒 False，所有 hybrid 分支不可达，行为零变化）。
        self.is_hybrid = config.is_hybrid
        # GDN 状态池字节数（hybrid 下由 _alloc_gdn_state_pool 填充；qwen3 恒 0）。
        self._gdn_state_bytes = 0
        # EAGLE3 投机解码（仅 TP=1，见 config 断言）。
        self.spec_enabled = bool(config.spec_model)
        if self.spec_enabled and self.is_hybrid and config.spec_method != "dflash":
            # 守卫（配对矩阵唯余此格拒绝）：hybrid（Qwen3.5）+ EAGLE3 未实现——
            # EAGLE3 链式草稿的 GDN verify 分叉未接入 _sg 路径（接入前硬跑会
            # 静默状态错乱，故干净拒绝）。四格矩阵现状：qwen3+eagle3（原路径）、
            # qwen3_5+dflash（hybrid 基建）、qwen3+dflash（dense：_df 路径 GDN
            # 触点已 is_hybrid 条件化）三者放行。
            raise RuntimeError(
                "Qwen3.5 hybrid targets support DFlash speculative decoding only "
                "(spec_method=dflash). EAGLE3 chained drafting and its three-layer "
                "auxiliary-state contract are not compatible with the GDN hybrid model."
            )
        if self.spec_enabled and not self.backend.supports_spec:
            # 能力位裁决（不做 name 字符串判断）。310P 已用多 query PA 承接
            # verify/refill；此守卫留给仍未适配 paged 形态的后端。
            raise RuntimeError(
                f"Backend {self.backend.name} does not support speculative decoding "
                "(supports_spec=False); remove spec_model."
            )
        # 每序列的 draft 种子隐层：fc(cat3 aux) 的输入，维度 [1, 3*hidden]。
        self.spec_hidden: dict[int, torch.Tensor] = {}
        # 每序列的 draft 流 carry：(row k 隐层, 首提案 token)。
        # 按 seq_id 存取（batch 顺序每轮可变，位置索引不安全）；冷启动轮由
        # eager 路径补齐。
        self.spec_carry: dict[int, tuple[torch.Tensor, int]] = {}
        # 端到端 graph 采样：lm_head + sampler 捕获进 decode 图，replay 后
        # 直接读 token id（省每步图外 1 次 GEMV launch + 5 个 full-vocab kernel）。
        # 开关优先级：env 显式设置 > 后端默认（graph_sample_default，Ascend 起步
        # 为 False——NPU RNG 图安全性未验证）；TP>1 时 lm_head 的 gather 不可
        # 捕获，world_size==1 守卫无条件保留（env=1 不得绕过）。
        _env_sample = os.environ.get("NANOVLLM_GRAPH_SAMPLE")
        self.graph_sample = (
            self.world_size == 1
            and (self.backend.graph_sample_default if _env_sample is None
                 else _env_sample != "0")
        )
        # graph static buffer 的 padding 状态跟踪（仅 bs，见 run_model）。
        self._graph_bs_active = 0
        self.graphs = None
        self.graph_pool = None
        # 310P decode 图走独立 NPUGraphDecodeRunner（CUDA/CoreX 仍 capture_cudagraph）。
        self.graph_runner = None
        # 每步 / 每轮复用的 pinned 暂存：避免热路径反复 cudaHostAlloc。
        # decode 元数据并入单个声明式 blob（MetaBlob），图内 scatter 分发，仅 2 次 H2D。
        self._max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 图/桶 batch 上限：单图捕获最大 bs，与 run_model 守卫共用。
        self._graph_max_bs = min(config.max_num_seqs, 512)
        M = self._graph_max_bs
        W = self._max_num_blocks + 1
        self._pin_M, self._pin_W = M, W
        # 段名/顺序/尺寸与旧 _pin_off 逐一对应（ids=0, pos=M, slots=2M, ctx=3M,
        # bt=4M, topks=4M+MW；f32: temperatures=0, top_ps=M）。
        self._dblob = MetaBlob(
            [('ids', M), ('pos', M), ('slots', M), ('ctx', M), ('bt', M * W), ('topks', M)],
            [('temperatures', M), ('top_ps', M)])

        dist.init_process_group(self.backend.dist_name, f"tcp://localhost:{config.hccl_port}", world_size=self.world_size, rank=rank)
        self.backend.set_device(rank)
        default_dtype = torch.get_default_dtype()
        if self.backend.default_dtype is not None and self.backend.default_dtype != config.text_config.dtype:
            print(
                f"[nano-vllm] Backend {self.backend.name} requires "
                f"dtype={self.backend.default_dtype}; model dtype={config.text_config.dtype}. "
                "Weights will be converted while loading.",
                flush=True,
            )
        torch.set_default_dtype(self.dtype)
        torch.set_default_device(self.backend.device_name)
        model_load_t0 = time.perf_counter()
        # 模型分发：qwen3_5（hybrid）→ Qwen3_5ForCausalLM（text 形状自取
        # hf_config.text_config）；qwen3 路径原样不动。
        if self.is_hybrid:
            self.model = Qwen3_5ForCausalLM(hf_config)
        else:
            self.model = Qwen3ForCausalLM(hf_config)
        # 模型自带 load_weights（Qwen3.5：多模态壳前缀剥离 + GDN in_proj/conv/A_log
        # 特例装配，见 qwen3_5.load_weights）优先；qwen3 走通用 loader，路径不变。
        if hasattr(self.model, "load_weights"):
            self.model.load_weights(iter_safetensors(config.model))
        else:
            load_model(self.model, config.model)
        if self.is_hybrid:
            # 契约核对：模型侧 is_hybrid 必须与 config 判定一致（防两处口径漂移）。
            assert getattr(self.model.model, "is_hybrid", False) is True, \
                "Model does not declare model.model.is_hybrid=True."
            # GDN 状态池在 warmup 之前分配：GDN forward（含预热轮）需要真实的
            # conv/ssm 缓存（Attention 的 k_cache 靠 numel()==0 跳过预热写，GDN
            # 无此占位语义）。其字节数此后已包含在 mem_get_info 的 used 里，
            # allocate_kv_cache 的预算循环自动计入（勿再显式扣除，避免双重计费）。
            self._alloc_gdn_state_pool()
        if self.spec_enabled:
            # aux 隐层捕获必须在 warmup 前打开（prefill 前向开始返回 (hidden, aux)）。
            self.model.model.aux_hidden_state_layers = tuple(config.spec_aux_layers)
            # draft config 由 config.py 统一加载（避免二次 IO），语义不变。
            self.draft_config = config.draft_config
            if getattr(self.draft_config, "draft_vocab_size", None) is None:
                # dflash 无独立 draft 词表（= target 词表，无 d2t）——_fill_spec_meta
                # 的 top_k 截断上界按 target 词表取值（vllm DFlashQwen3ForCausalLM
                # 构造时同款回填）。
                self.draft_config.draft_vocab_size = config.text_config.vocab_size
            if config.spec_method == "dflash":
                d = self.draft_config
                # DFlash draft：embed/lm_head 共享 target（构造传入）；rope 的
                # max_position 按 max_model_len 收紧（位置恒 < max_model_len，
                # 省 262144 全长 cos/sin 缓存的开销）。
                self.draft_model = DFlashDraftModel(
                    d, self.model.model.embed_tokens,
                    target_lm_head_weight=self.model.lm_head.weight,
                    max_position=config.max_model_len)
                load_dflash_draft(self.draft_model, config.spec_model)
                self.draft_model._build_fused_kv_buffers()
                # draft 上下文 KV 稠密缓冲：[slots, max_model_len, L, kvh, hd]×2
                # bf16（~84MB@ms8/mml2048；不进 block manager——上下文 KV 由 aux
                # 重投影而来，与分页 token KV 是两种内容）。零初始化：块草稿读取
                # 以 ctx_len 右界 + -inf mask 屏蔽界外残留，有限值保证无 NaN 传播。
                # 分配在 warmup 之前（预热 prefill 即写入），字节数随 mem_get_info
                # 自动计入 KV 池预算（勿再显式扣除）。
                hd = getattr(d, "head_dim", d.hidden_size // d.num_attention_heads)
                # 槽多配 1 个（_df_pad_ctx_slot = ms）：spec 图 padding 序列的
                # append 零写专用——padding 行若指向 0 号真实槽，会把运行序列的
                # prompt 上下文前 rows 位清零（verify bonus 仍正确、输出不受损，
                # 但该序列后续轮提案全废、E 受损），专用槽从分配上杜绝。
                if not self.dflash_paged:
                    ctx_shape = (config.max_num_seqs + 1, config.max_model_len,
                                 d.num_hidden_layers, d.num_key_value_heads, hd)
                    self.draft_model.ctx_k_cache = torch.zeros(ctx_shape, dtype=self.dtype)
                    self.draft_model.ctx_v_cache = torch.zeros(ctx_shape, dtype=self.dtype)
                self._df_pad_ctx_slot = config.max_num_seqs
                # 槽生命周期镜像 GDN 主槽（get_or_alloc / drop_seq_state 双路径清理）
                self.draft_ctx_slots: dict[int, int] = {}
                self._df_ctx_free = set(range(config.max_num_seqs))
                # DFlash 每轮状态（按 seq_id 记账，batch 组成每轮可变）：
                # spec_ctx_pending = 上轮 verify 已提交行的 aux（GPU 常驻——每轮
                # ≤16 行×10KB，省 D2H/H2D；CPU 常驻则每轮多两次拷贝，取舍取 GPU）；
                # gdn_spec_carry = 上轮 verify 行 m 的 GDN 槽号（绝对值）。
                self.spec_ctx_pending: dict[int, torch.Tensor | None] = {}
                self.gdn_spec_carry: dict[int, int] = {}
            else:
                self.draft_model = Eagle3DraftModel(self.draft_config, self.model.model.embed_tokens)
                load_eagle3_draft(self.draft_model, config.spec_model)
            # 本轮 verify 的 GDN 写基址（host int，读回更新 carry 用；仅 dflash 写）
            self._df_bases: list[int] = []
        weight_nz_count = self.backend.optimize_model_weights(self.model)
        if self.spec_enabled:
            weight_nz_count += self.backend.optimize_model_weights(self.draft_model)
            if self.config.spec_method == "dflash":
                weight_nz_count += self.backend.optimize_dflash_weights(
                    self.draft_model, self.model.lm_head
                )
        self.startup_stats["weight_layout_conversions"] = float(weight_nz_count)
        self.startup_stats["model_load_s"] = time.perf_counter() - model_load_t0
        compile_t0 = time.perf_counter()
        if config.compile:
            # torch.compile：CoreX 上 qkv/rope/attention 区域的 Inductor 融合会产出
            # 错误结果，Qwen3Attention.forward 已 @torch.compiler.disable 保持 eager，
            # 此处只编译 embed + RMSNorm + MLP（正确性已验证 PASS）。
            self.model = torch.compile(self.model)
        self.sampler = Sampler()
        warmup_t0 = time.perf_counter()
        self.warmup_model()
        self.startup_stats["model_warmup_s"] = time.perf_counter() - warmup_t0
        self.startup_stats["model_compile_s"] = (
            time.perf_counter() - compile_t0 if config.compile else 0.0
        )
        kv_t0 = time.perf_counter()
        self.allocate_kv_cache()
        self.startup_stats["kv_cache_allocation_s"] = time.perf_counter() - kv_t0
        if self.spec_enabled:
            # spec 解码缓冲在启用时即分配：eager 轮统一复用图轮 body（_sg_round），
            # 捕获被跳过/失败时缓冲仍需存在；须在 allocate_kv_cache 之后，保持
            # kv 预算计算不含本缓冲（与基线一致）。
            k = self.config.spec_num_tokens
            rows = k + 1
            M, W = self._pin_M, self._pin_W
            # spec 热路径 pinned 元数据 blob（段名/顺序/尺寸与旧 _meta_off/_metaf_off
            # 逐一对应；每轮 2 次 H2D，零 cudaHostAlloc）。dflash 专段追加在尾部
            # （eagle3 不声明，既有段偏移逐字节不变）：
            #   df_l1      = 各 seq 的 L-1（anchor 位置 = 块查询位置基址 = ctx_len）；
            #   df_c       = 上轮提交行数（pending 长度，append 位置回推 L-1-c）；
            #   df_ctxslot = draft 上下文稠密缓冲槽号；
            #   df_carry   = GDN verify 行 0 读槽（gdn_spec_carry 或主槽）。
            df_segments = ([('df_l1', M), ('df_c', M), ('df_ctxslot', M), ('df_carry', M),
                            ('df_ctx_pos', rows * M), ('df_ctx_phys_slots', rows * M),
                            ('df_total_lens', M)]
                           if self.config.spec_method == "dflash" else [])
            self._sblob = MetaBlob(
                [('chain_tokens', M), ('chain_pos', k * M), ('chain_ctxlen', k * M),
                 ('chain_slots', k * M), ('v_pos', rows * M), ('v_slots', rows * M),
                 ('v_cu_k', M + 1), ('ctopks', M), ('v_topks', rows * M), ('btables', M * W),
                 # 310P 图安全：verify/refill per-query context_lens + 展开 block_table
                 # （host 填，scatter copy_；禁止图内 index/Add）
                 ('v_ctx_lens', rows * M), ('v_btables_exp', rows * M * W)]
                + df_segments,
                [('ctemps', M), ('ctopps', M), ('v_temps', rows * M), ('v_topps', rows * M)])
            self._sg = self._alloc_spec_graph_vars(self._graph_max_bs, rows, k)
            if self.config.spec_method == "dflash":
                # dflash 专段静态缓冲（阶段3 图轮；eagle3 不分配，字节布局零变化）。
                self._sg.update(self._alloc_df_graph_vars(self._graph_max_bs, rows))
        self.spec_graphs = None
        self.spec_graph_pool = None
        self._spec_graph_active = 0
        self.spec_graph_stats = {
            "capture_success": 0,
            "capture_failure": 0,
            "replay": 0,
            "eager_rounds": 0,
            "sampling_fallbacks": 0,
            "bucket_hits": {},
        }
        self.graph_capture_attempted = False
        graph_t0 = time.perf_counter()
        if not self.enforce_eager:
            if not self.spec_enabled:
                if self.backend.name == "ascend310p":
                    # 310P 与 bench_ttft_tps --mode aclgraph 同一套图：每桶静态
                    # Tensor + copy_ + NPUGraph.replay；sampler 图外。失败按桶 eager。
                    from nanovllm.engine.npu_graph_runner import NPUGraphDecodeRunner
                    self.graph_capture_attempted = True
                    self.graph_runner = NPUGraphDecodeRunner(self)
                    self.graph_runner.capture_all()
                    if not any(
                        e.enabled and e.graph is not None
                        for e in self.graph_runner.entries.values()
                    ):
                        print(
                            "[nano-vllm] All 310P NPUGraph buckets failed to capture; "
                            "decode will use eager mode.",
                            flush=True,
                        )
                        self.graph_runner = None
                else:
                    # hybrid（Qwen3.5）与 qwen3 同构捕获 decode 图（阶段3 解除阶段1
                    # 门控）：GDN 单步（_forward_decode）本身无 host 同步/动态 shape，
                    # 状态槽经 context 的 device twin 静态段消费（见 _alloc_gdn_state_pool
                    # 与 prepare_decode 的图分支）；捕获失败仍走 try/except 优雅降级。
                    self.graph_capture_attempted = True
                    self.capture_cudagraph()
            else:
                if self.config.spec_method == "dflash":
                    # DFlash spec 单图整轮捕获（阶段3：上下文追加/块草稿/verify
                    # 全静态化，见 _df_round_body / capture_df_spec_cudagraph）。
                    # 图门与 EAGLE3 同款（supports_spec_graph + graph_sample_default，
                    # NANOVLLM_SPEC_GRAPH 显式覆盖）；失败自动降级 eager spec 轮。
                    # dense（Qwen3+dflash）与 hybrid 同门：图路径的 GDN 静态段已
                    # is_hybrid 条件化（dense 无 GDN 状态分叉，df_carry 段留零不
                    # 消费），CUDA/CoreX 均尝试捕获——CoreX 是 dense+dflash 图的
                    # 主目标形态（EAGLE 整轮图实机已证 ixformer verify 可捕获）。
                    _env_spec = os.environ.get("NANOVLLM_SPEC_GRAPH")
                    if (self.world_size == 1 and self.backend.supports_spec_graph
                            and (_env_spec != "0" if _env_spec is not None
                                 else self.backend.spec_graph_try_default)):
                        self.graph_capture_attempted = True
                        self.capture_df_spec_cudagraph()
                else:
                    # spec 解码路径图捕获（draft 链图 + verify 图，仅 TP=1）。spec 图
                    # 无条件录 Gumbel 采样（forward_native），默认门同时要求
                    # graph_sample_default（RNG 图安全性未验证的后端不默认尝试）；
                    # NANOVLLM_SPEC_GRAPH 显式设置优先后端默认，可实验性覆盖。
                    _env_spec = os.environ.get("NANOVLLM_SPEC_GRAPH")
                    if (self.world_size == 1 and self.backend.supports_spec_graph
                            and (_env_spec != "0" if _env_spec is not None
                                 else self.backend.spec_graph_try_default)):
                        self.graph_capture_attempted = True
                        self.capture_spec_cudagraph()
        self.startup_stats["graph_capture_s"] = time.perf_counter() - graph_t0
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
        self.startup_stats["model_runner_total_s"] = time.perf_counter() - startup_t0

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def graph_mode(self) -> str:
        if self.enforce_eager:
            return "eager"
        if self.spec_enabled:
            if self.spec_graphs:
                return "spec_graph"
            return "eager_fallback" if self.graph_capture_attempted else "eager"
        if self.graph_runner is not None or self.graphs is not None:
            return "decode_graph"
        return "eager_fallback" if self.graph_capture_attempted else "eager"

    def get_spec_graph_stats(self) -> dict:
        """Return a snapshot of speculative graph capture/runtime counters."""
        stats = dict(self.spec_graph_stats)
        stats["bucket_hits"] = dict(stats["bucket_hits"])
        return stats

    def reset_spec_graph_runtime_stats(self):
        """Keep capture results but start a fresh speculative runtime window."""
        self.spec_graph_stats["replay"] = 0
        self.spec_graph_stats["eager_rounds"] = 0
        self.spec_graph_stats["sampling_fallbacks"] = 0
        self.spec_graph_stats["bucket_hits"] = {}

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if self.graphs is not None:
            del self.graphs, self.graph_pool
        self.backend.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        self.backend.empty_cache()
        self.backend.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        if self.spec_enabled:
            # spec warmup：target prefill（含 aux 捕获）+ draft prefill 一并预热；
            # 产生的 h_seed 属于 dummy 序列，预热后清理。
            self.run_prefill_spec(seqs)
            self.spec_hidden.clear()
        else:
            self.run(seqs, True)
        self.backend.empty_cache()
        if self.is_hybrid:
            # 预热 dummy 序列占用的 GDN 状态槽全量回收：清映射、槽位全数归还
            # （池数据不擦除——槽重用后的首次消费恒为 has_initial=False 的
            # 零初始化 prefill，旧值永不被读，见 drop_seq_state）。
            self.gdn_slots.clear()
            self._gdn_free_slots = set(range(self.config.max_num_seqs))
        if self.config.spec_method == "dflash":
            # DFlash：预热 dummy 序列的 spec 状态同口径全量回收（pending aux /
            # GDN carry 记账 / draft 上下文槽）。hybrid 与 dense（qwen3+dflash）
            # 都需要——原实现嵌在 is_hybrid 块内，dense 首次机械冒烟实测 warmup
            # 假槽不回收 → 首个真实 prefill 槽耗尽 RuntimeError，故提出块外。
            self.spec_ctx_pending.clear()
            self.gdn_spec_carry.clear()
            self.draft_ctx_slots.clear()
            self._df_ctx_free = set(range(self.config.max_num_seqs))

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # 形状统一取 text_config（qwen3 的 text_config 即 hf_config，逐字节同值）。
        tc = config.text_config
        if self.is_hybrid:
            # hybrid：只有全注意力层进 KV 池（GDN 无 KV cache，状态在独立池，
            # 见 _alloc_gdn_state_pool）。层数显式取模型声明（契约字段
            # num_attention_layers），避免从 layer_types 二次推导产生口径漂移。
            num_attn_layers = self.model.model.num_attention_layers
        else:
            num_attn_layers = hf_config.num_hidden_layers
        num_kv_heads = tc.num_key_value_heads // self.world_size
        head_dim = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
        block_bytes = 2 * num_attn_layers * self.block_size * num_kv_heads * head_dim * self.dtype.itemsize
        if self.spec_enabled and (
                self.config.spec_method == "eagle3" or self.dflash_paged):
            # EAGLE3 and 310P DFlash paged KV share the target block-table/slot
            # layout, so their cache bytes must be included in the same budget.
            # 布局一致性由 backend.get_kv_cache_shape 保证（cuda/ascend=BSHD、corex=HND，
            # 两模型同后端必同布局）；draft 槽位写入与 target 走同一 _slot_of 映射。
            d = self.draft_config
            draft_kv_heads = d.num_key_value_heads // self.world_size
            draft_head_dim = getattr(d, "head_dim", d.hidden_size // d.num_attention_heads)
            draft_layers = (1 if self.config.spec_method == "eagle3"
                            else d.num_hidden_layers)
            block_bytes += (2 * draft_layers * self.block_size * draft_kv_heads
                            * draft_head_dim * self.dtype.itemsize)
        # WSL 下上一进程退出后驱动回收显存存在数秒延迟，`used` 可能虚高；
        # 预算不足时短暂等待重测，而非直接断言崩溃。
        # 状态池（hybrid）在 warmup 前已分配，其字节数包含在下面的 used 里、
        # 随本预算循环自动计入——勿再显式扣除（已占用的显存扣两次即双重计费）。
        requested = config.num_kvcache_blocks
        for _ in range(6):
            free, total = self.backend.mem_get_info()
            used = total - free
            peak = self.backend.memory_stats()["allocated_bytes.all.peak"]
            current = self.backend.memory_stats()["allocated_bytes.all.current"]
            computed = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
            if requested > 0:
                config.num_kvcache_blocks = requested
                break
            config.num_kvcache_blocks = computed
            if config.num_kvcache_blocks > 0:
                break
            time.sleep(5)
        else:
            raise RuntimeError(
                f"Insufficient device-memory budget for KV cache: "
                f"{block_bytes / 2**20:.1f} MiB is required per block"
                + (f"; GDN state pool uses {self._gdn_state_bytes / 2**20:.1f} MiB" if self.is_hybrid else "")
                + f"; gpu_memory_utilization={config.gpu_memory_utilization} leaves zero blocks "
                f"(used={used / 2**20:.0f} MiB, peak={peak / 2**20:.0f} MiB, "
                f"current={current / 2**20:.0f} MiB, "
                f"budget={total * config.gpu_memory_utilization / 2**20:.0f} MiB). "
                "Reduce max_model_len or block_size, or check other device-memory users."
            )
        layers = self.backend.allocate_kv_layers(
            num_layers=num_attn_layers,
            num_blocks=config.num_kvcache_blocks,
            block_size=self.block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        if layers is not None:
            attn_mods = [m for m in self.model.modules()
                         if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
            assert len(attn_mods) == len(layers) == num_attn_layers, (
                f"attn {len(attn_mods)} vs NZ {len(layers)}/{num_attn_layers}")
            for module, (k_cache, v_cache) in zip(attn_mods, layers):
                module.k_cache = k_cache
                module.v_cache = v_cache
            # EAGLE3 and paged DFlash draft attention also need private NZ
            # layers with the same block ids/slots as the target cache.
            if self.spec_enabled and (
                    self.config.spec_method == "eagle3" or self.dflash_paged):
                d = self.draft_config
                draft_attns = [m for m in self.draft_model.modules()
                                if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
                draft_kv_heads = d.num_key_value_heads // self.world_size
                draft_head_dim = getattr(
                    d, "head_dim", d.hidden_size // d.num_attention_heads)
                draft_layers = self.backend.allocate_kv_layers(
                    num_layers=len(draft_attns),
                    num_blocks=config.num_kvcache_blocks,
                    block_size=self.block_size,
                    num_kv_heads=draft_kv_heads,
                    head_dim=draft_head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                assert draft_layers is not None and len(draft_layers) == len(draft_attns)
                for module, (k_cache, v_cache) in zip(draft_attns, draft_layers):
                    module.k_cache = k_cache
                    module.v_cache = v_cache
            return
        shape = self.backend.get_kv_cache_shape(
            num_layers=num_attn_layers,
            num_blocks=config.num_kvcache_blocks,
            block_size=self.block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
        )
        self.kv_cache = torch.empty(shape)
        if self.is_hybrid:
            # KV wiring（hybrid）：显式按注意力序号映射（计划 1b 要求）。核实结论：
            # 模块树序 = decoder 层序 = attention_layer_ids 升序，故「按模块序递增」
            # 对 qwen3_5 本也算正确；但池切片索引与全局层号是两个概念（读写都经
            # module 上的池切片句柄，无需换算全局层号），这里按注意力序 0..5 显式
            # 枚举并断言模块数，防未来模型结构调整时静默错位。
            attn_mods = [m for m in self.model.modules()
                         if hasattr(m, "k_cache") and hasattr(m, "v_cache")]
            assert len(attn_mods) == num_attn_layers, (
                f"Full-attention module count ({len(attn_mods)}) does not match "
                f"KV-cache layer count ({num_attn_layers})."
            )
            for attn_ord, module in enumerate(attn_mods):
                module.k_cache = self.kv_cache[0, attn_ord]
                module.v_cache = self.kv_cache[1, attn_ord]
        else:
            layer_id = 0
            for module in self.model.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                    layer_id += 1
        if self.spec_enabled and self.config.spec_method == "eagle3":
            # EAGLE3 draft 的 paged KV cache（dflash 无此结构，见上方预算分支注释）
            d = self.draft_config
            draft_kv_heads = d.num_key_value_heads // self.world_size
            draft_head_dim = getattr(d, "head_dim", d.hidden_size // d.num_attention_heads)
            draft_shape = self.backend.get_kv_cache_shape(
                num_layers=d.num_hidden_layers,
                num_blocks=config.num_kvcache_blocks,
                block_size=self.block_size,
                num_kv_heads=draft_kv_heads,
                head_dim=draft_head_dim,
            )
            self.draft_kv_cache = torch.empty(draft_shape)
            attn = self.draft_model.midlayer.self_attn.attn
            attn.k_cache = self.draft_kv_cache[0, 0]
            attn.v_cache = self.draft_kv_cache[1, 0]

    def _df_drop_state(self, seq_id: int):
        """DFlash spec 状态清理（幂等；finish 的 drop_spec_state 与 preempt 的
        drop_seq_state 双路径共用）：pending aux、draft 上下文槽、GDN carry 记账。
        非 dflash 直通（qwen3+eagle3 / 非 spec hybrid 零开销）。"""
        if self.config.spec_method != "dflash":
            return
        self.spec_ctx_pending.pop(seq_id, None)
        self.gdn_spec_carry.pop(seq_id, None)
        slot = self.draft_ctx_slots.pop(seq_id, None)
        if slot is not None:
            self._df_ctx_free.add(slot)

    def drop_spec_state(self, seq_id: int):
        self.spec_hidden.pop(seq_id, None)
        self.spec_carry.pop(seq_id, None)
        self._df_drop_state(seq_id)

    def _alloc_gdn_state_pool(self):
        """hybrid（Qwen3.5）：GDN 线性注意力状态池分配 + 模块注入 + 槽生命周期初始化。

        在 warmup 之前调用（GDN forward 含预热轮都需要真实缓存）；KV 池预算
        （allocate_kv_cache 的 mem_get_info 重试循环）在本函数之后执行，状态池
        字节数经 used 自动计入，无需显式扣除。

        池形状（阶段1）：[num_gdn_layers, slots, *state_shape]，slots = max_num_seqs；
        阶段2 spec 槽扩展（slots × (1+spec_num_tokens)，verify 行分叉槽）只改本函数
        的 slots 来源与 fill/读侧语义。
        """
        spec = self.model.model.get_gdn_state_spec()
        tc = self.config.text_config
        n_gdn = tc.num_hidden_layers - self.model.model.num_attention_layers
        slots = self.config.max_num_seqs
        if self.config.spec_method == "dflash":
            # DFlash（阶段2）：池扩为「主槽区 ms + verify 行槽区 ms×(k+1)」。
            # verify 行槽静态布局：批内序号 i 的写基址 base_i = ms + i×(k+1)
            # （行 0..k 逐行落槽，见 qwen3_5._forward_rows / _df_verify）。
            # 主槽（prefill 终态 / 非 spec decode）保持 [0, ms) 的 get_or_alloc
            # 语义不变——必须与全部行槽区互斥：prefill 与 verify 是不同引擎步、
            # 批内序号在两步间可变，若 prefill 写进「本步批序号对应」的行槽区，
            # 会被其他序列的 verify 行覆盖（carry 读到脏状态）。偏差说明：计划
            # §3 写 slots = ms×(k+1)，实现取 ms×(k+2)，多出的 ms 槽即主槽区
            # （@ms8 ≈ +8MB ssm，正确性必需）。
            slots = self.config.max_num_seqs * (self.config.spec_num_tokens + 2)
        # 阶段3 图 padding 保留槽（池尾 +1）：decode 图按桶回放时 padding 行的 GDN
        # 读写需要一个「可安全弄脏」的槽——GDN 的 conv/ssm 读写是设备侧高级索引
        # （idx=-1 会回绕到末槽写真实状态，不能用 -1 哨兵），padding 行统一指向本
        # 保留槽：has_initial=False 零化 ssm 读、conv 读垃圾（输出丢弃）、写全落
        # 保留槽（不脏任何可分配槽——空闲集只覆盖主槽区 [0, ms)）。spec 图 verify
        # 的 padding 行写各自 base 行槽区（池静态布局互斥），同样不消费保留槽。
        # 代价 ≤1 槽 ssm（fp32 ~1MB），enforce_eager 下闲置无害。
        slots += 1
        self._gdn_pad_slot = slots - 1
        conv_shape = spec["conv_shape"]
        ssm_shape = spec["ssm_shape"]
        # conv 状态用模型 dtype、ssm 状态 fp32——形状与 dtype 均以 get_gdn_state_spec()
        # 声明为准（勿在引擎侧硬编码，模型侧按 text_config 计算口径唯一）。
        self.conv_pool = torch.empty((n_gdn, slots) + tuple(conv_shape), dtype=spec["conv_dtype"])
        self.ssm_pool = torch.empty((n_gdn, slots) + tuple(ssm_shape), dtype=spec["ssm_dtype"])
        self._gdn_state_bytes = (self.conv_pool.numel() * self.conv_pool.element_size()
                                 + self.ssm_pool.numel() * self.ssm_pool.element_size())
        # GDN 模块识别：模型侧契约字段（按 GDN 层序的权威模块引用，
        # qwen3_5 恒赋值）。
        gdn_mods = self.model.model.gdn_layers
        assert len(gdn_mods) == n_gdn, (
            f"Detected GDN module count ({len(gdn_mods)}) does not match "
            f"the configured layer count ({n_gdn}); the model must expose gdn_layers."
        )
        for gdn_ord, module in enumerate(gdn_mods):
            # 注入该层的池切片 [slots, *state_shape]：模块 forward 用
            # context.gdn_state_slots[row] 在 slot 维索引读写。
            module.conv_cache = self.conv_pool[gdn_ord]
            module.ssm_cache = self.ssm_pool[gdn_ord]
        # 槽生命周期（host 侧）：seq_id → slot 映射 + 空闲槽集合。分配在
        # prepare_prefill/prepare_decode 兜底（get_or_alloc，warmup 直构 Sequence
        # 绕过 scheduler 也由此覆盖）；释放在 finish（llm_engine）/ preempt
        # （scheduler 回调）——见 drop_seq_state。
        self.gdn_slots: dict[int, int] = {}
        self._gdn_free_slots: set[int] = set(range(slots))
        # 每步 GDN 元数据的 pinned host + device twin（图友好静态段：阶段3 图捕获
        # 直接消费 device twin 固定地址。长度取 max_num_seqs（≥ _graph_max_bs，
        # eager 路径覆盖全部并发；图桶在范围内）。host 槽位缓冲初始化为保留槽：
        # 图 padding 行在「首次收窄前」读到的恒是 pad 槽而非初始 0（0 是真实主槽，
        # padding 写会脏掉运行序列的状态——与 KV 池 slot_mapping 首扩桶 padding 行
        # 写槽 0 的隐患同源，这里从分配上杜绝）。
        N = self.config.max_num_seqs
        self._gdn_slots_pin = torch.full((N,), self._gdn_pad_slot, dtype=torch.int32,
                                         device='cpu', pin_memory=True)
        self._gdn_init_pin = torch.zeros(N, dtype=torch.bool, device='cpu', pin_memory=True)
        self._gdn_slots_dev = torch.full((N,), self._gdn_pad_slot, dtype=torch.int32)
        self._gdn_init_dev = torch.zeros(N, dtype=torch.bool)

    def _gdn_get_or_alloc_slot(self, seq: Sequence) -> int:
        """GDN 状态槽 get_or_alloc（prepare_prefill/prepare_decode 兜底调用）。

        新分配的槽内容无脏读风险：该序列的首次消费恒为 has_initial=False 的
        零初始化 prefill（重用槽的残留值不会被读，之后被终态覆盖）。
        """
        slot = self.gdn_slots.get(seq.seq_id)
        if slot is None:
            try:
                slot = self._gdn_free_slots.pop()
            except KeyError:
                # 调度门控（scheduler 的 hybrid admission）保证 live 序列数
                # ≤ max_num_seqs；此断言炸穿即门控失效。
                raise RuntimeError(
                    "GDN state slots are exhausted; live sequences exceeded "
                    f"max_num_seqs={self.config.max_num_seqs}."
                ) from None
            self.gdn_slots[seq.seq_id] = slot
        return slot

    def drop_seq_state(self, seq_id: int):
        """hybrid GDN 状态槽释放（finish / preempt 时经 llm_engine 调用，与
        drop_spec_state 同模式）。阶段1 不擦池数据：释放即断开映射，槽位重用时
        由 has_initial=False 的零初始化 prefill 覆盖写入，旧值永不被读。"""
        slot = self.gdn_slots.pop(seq_id, None)
        if slot is not None:
            self._gdn_free_slots.add(slot)
        # dflash：preempt 路径同步清 spec 状态（finish 路径由 drop_spec_state 清，
        # _df_drop_state 幂等，双路径重复调用安全）。
        self._df_drop_state(seq_id)

    def _graph_pad_block(self) -> int:
        return self.backend.graph_pad_block(self.config.num_kvcache_blocks)

    def _graph_pad_slot(self) -> int:
        b = self._graph_pad_block()
        return -1 if b < 0 else b * self.block_size

    def _graph_pad_ctx(self) -> int:
        return self.backend.graph_pad_context_len()

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        pad = self._graph_pad_block()
        block_tables = [seq.block_table + [pad] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        return block_tables

    def _slot_of(self, seq: Sequence, p: int) -> int:
        # 位置 p 的 KV 槽位（draft 与 target 共用同一 block_table/slot 布局）。
        return seq.block_table[p // self.block_size] * self.block_size + p % self.block_size

    def _fill_bt_row(self, buf, i: int, table: list[int]):
        # 填 block_table 第 i 行：有效段写 table，其余补 -1（host int64 视图）。
        pad = self._graph_pad_block()
        buf[i, :len(table)] = torch.tensor(table, dtype=torch.int64)
        buf[i, len(table):] = pad

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        # hybrid：GDN 状态槽元数据（pinned host → device twin 非阻塞 H2D，静态段，
        # 模仿 _dblob 写法）。整段 prefill（hybrid 禁 chunk）：每序列一个槽；
        # has_initial 恒 False——零初始化、终态写主槽，preempt 后重 prefill 同口径
        # 重建。host 缓冲的跨步覆写安全性同 _dblob：每步收尾必有 D2H 同步
        # （token_ids.tolist()），覆写时上一步 H2D 已完成。
        gdn_extra = {}
        if self.is_hybrid:
            n_seqs = len(seqs)
            for i, seq in enumerate(seqs):
                self._gdn_slots_pin[i] = self._gdn_get_or_alloc_slot(seq)
                self._gdn_init_pin[i] = False
            self._gdn_slots_dev[:n_seqs].copy_(self._gdn_slots_pin[:n_seqs], non_blocking=True)
            self._gdn_init_dev[:n_seqs].copy_(self._gdn_init_pin[:n_seqs], non_blocking=True)
            gdn_extra = dict(gdn_state_slots=self._gdn_slots_dev[:n_seqs],
                             gdn_has_initial=self._gdn_init_dev[:n_seqs])
        input_ids = torch.tensor(
            input_ids, dtype=torch.int64, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        positions = torch.tensor(
            positions, dtype=torch.int64, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        cu_seqlens_q = torch.tensor(
            cu_seqlens_q, dtype=torch.int32, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        cu_seqlens_k = torch.tensor(
            cu_seqlens_k, dtype=torch.int32, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        # 槽位 dtype 由后端决定（CUDA int32 / CoreX int64），cu_seqlens 两后端均为 int32。
        slot_mapping = torch.tensor(
            slot_mapping,
            dtype=self.backend.graph_slot_mapping_dtype(),
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, block_size=self.block_size, **gdn_extra)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        n = len(seqs)
        block_size = self.block_size
        blob = self._dblob
        ids_v = blob.view_i64('ids')
        pos_v = blob.view_i64('pos')
        ctx_v = blob.view_i64('ctx')
        slots_v = blob.view_i64('slots')
        bt = blob.view_i64('bt', (self._pin_M, self._pin_W))
        for i, seq in enumerate(seqs):
            ids_v[i] = seq.last_token
            pos_v[i] = len(seq) - 1
            ctx_v[i] = len(seq)
            slots_v[i] = seq.block_table[-1] * block_size + seq.last_block_num_tokens - 1
            self._fill_bt_row(bt, i, seq.block_table)
            if self.is_hybrid:
                # GDN 状态槽（host 侧先行填写）：decode 每行槽 + has_initial=True
                # ——单行消费主槽、写回主槽（计划 §3 语义）。
                self._gdn_slots_pin[i] = self._gdn_get_or_alloc_slot(seq)
                self._gdn_init_pin[i] = True
        # graph 直通：host 填 blob → run_model 2 次 H2D → 图内 scatter 分发。
        # gv buffer 是 capture 时（inference mode）创建的 inference tensor，
        # 相关写入须在 inference mode 内进行。hybrid（阶段3）与 qwen3 同构：
        # GDN 元数据消费 device twin 固定地址（capture 时经 set_context 烘焙进图，
        # 每步 H2D 刷新内容即可，无需图内 scatter——twin 本身就是静态段）。
        if self.graphs is not None:
            gv = self.graph_vars
            bucket = _bucket(self.graph_bs, n)
            if bucket is not None:
                with torch.inference_mode():
                    # padding 复位：n < bucket 即需（含首次扩桶——blob 初始为 0，
                    # padding 行 slot=0 会写脏 block 0/主槽 0；旧条件 bucket <
                    # active 漏掉首扩与同桶缩行两个窗口，属共享路径的隐患修复，
                    # qwen3 行为只增不减：padding 行从「写真实槽」变为「跳写」）。
                    if n < bucket:
                        pad_slot = self._graph_pad_slot()
                        pad_ctx = self._graph_pad_ctx()
                        pad_block = self._graph_pad_block()
                        slots_v[n:bucket].fill_(pad_slot)
                        ctx_v[n:bucket].fill_(pad_ctx)
                        ids_v[n:bucket].zero_()
                        pos_v[n:bucket].zero_()
                        bt[n:, :].fill_(pad_block)
                    self._graph_bs_active = bucket
                    if self.is_hybrid:
                        # GDN padding 行 → 保留槽 + has_initial=False（host 侧
                        # 复位，随下方 H2D 一起上卡）；真实行已在上面循环写好。
                        # 随后 twin 整段 H2D [:bucket]（图捕获消费 twin 固定地址）。
                        self._gdn_slots_pin[n:bucket] = self._gdn_pad_slot
                        self._gdn_init_pin[n:bucket] = False
                        self._gdn_slots_dev[:bucket].copy_(self._gdn_slots_pin[:bucket], non_blocking=True)
                        self._gdn_init_dev[:bucket].copy_(self._gdn_init_pin[:bucket], non_blocking=True)
                    set_context(False,
                                slot_mapping=gv['slot_mapping'][:n],
                                context_lens=gv['context_lens'][:n],
                                block_tables=gv['block_tables'][:n],
                                max_context_len=self.config.max_model_len,
                                use_cuda_graph=True,
                                block_size=self.block_size)
                return gv['input_ids'][:n], gv['positions'][:n]
        # 无图回退：直接 H2D 出 staging 张量（槽位 dtype 随后端；ctx/block_table 两后端均 int32）
        with torch.inference_mode():
            slot_dtype = self.backend.graph_slot_mapping_dtype()
            input_ids = ids_v[:n].to(self.device, non_blocking=True)
            positions = pos_v[:n].to(self.device, non_blocking=True)
            slot_mapping = slots_v[:n].to(self.device, non_blocking=True).to(slot_dtype)
            context_lens = ctx_v[:n].to(self.device, non_blocking=True).to(torch.int32)
            block_tables = bt[:n].to(self.device, non_blocking=True).to(torch.int32)
        # hybrid：GDN 状态槽元数据 H2D（pinned host → device twin 非阻塞拷贝；
        # 阶段3 图捕获直接消费 twin 固定地址）。跨步覆写安全性同 prepare_prefill。
        gdn_extra = {}
        if self.is_hybrid:
            self._gdn_slots_dev[:n].copy_(self._gdn_slots_pin[:n], non_blocking=True)
            self._gdn_init_dev[:n].copy_(self._gdn_init_pin[:n], non_blocking=True)
            gdn_extra = dict(gdn_state_slots=self._gdn_slots_dev[:n],
                             gdn_has_initial=self._gdn_init_dev[:n])
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, max_context_len=self.config.max_model_len, block_size=self.block_size, **gdn_extra)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence], is_prefill: bool = False):
        """返回 (temperatures, top_ps, top_ks) 张量；top_k 不生效时传词表大小（掩码恒假）。"""
        # 词表大小取 text_config（qwen3_5 外层多模态壳无 vocab_size；qwen3 同值）。
        vocab = self.config.text_config.vocab_size
        n = len(seqs)
        blob = self._dblob
        M = self._pin_M
        temps_v = blob.view_f32('temperatures')
        topps_v = blob.view_f32('top_ps')
        topks_v = blob.view_i64('topks')
        for i, seq in enumerate(seqs):
            temps_v[i] = seq.temperature
            topps_v[i] = seq.top_p
            topks_v[i] = seq.top_k if seq.top_k > 0 else vocab
        # graph 直通（decode）：run_model 的 scatter 分发，返回图内视图。
        if not is_prefill and self.graphs is not None:
            gv = self.graph_vars
            return gv['temperatures'][:n], gv['top_ps'][:n], gv['top_ks'][:n]
        with torch.inference_mode():
            off = blob.offset('topks')
            return (blob.f32[:n].to(self.device, non_blocking=True),
                    blob.f32[M:M + n].to(self.device, non_blocking=True),
                    blob.i64[off:off + n].to(self.device, non_blocking=True))

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, bool]:
        """端到端 graph 路径返回 (token_ids, True)，其余返回 (logits, False)。"""
        # FULL_DECODE_ONLY：prefill 永远 eager；只有 decode 走 CUDA Graph。
        if is_prefill or self.enforce_eager or input_ids.size(0) > self._graph_max_bs:
            return self.model.compute_logits(self.model(input_ids, positions)), False
        if self.graphs is None:    # 捕获失败降级：decode 全部走 eager 前向
            return self.model.compute_logits(self.model(input_ids, positions)), False

        bs = input_ids.size(0)
        gv = self.graph_vars

        # 真实 bs 向上取到最近的 graph bucket（bs=3 → graph 4）。
        graph_bs = _bucket(self.graph_bs, bs)
        if graph_bs is None:
            return self.model.compute_logits(self.model(input_ids, positions)), False
        # 防御：仅接受 prepare_decode 直通准备的输入（blob 分发依赖固定地址）。
        if input_ids.data_ptr() != gv["input_ids"].data_ptr():
            return self.model.compute_logits(self.model(input_ids, positions)), False

        # 2 次 H2D + replay；元数据分发（int64→int32 转换）在图内 scatter 完成。
        self._dblob.h2d()
        self.graphs[graph_bs].replay()
        if self.graph_sample:
            return gv["token_ids"][:bs], True
        return self.model.compute_logits(gv["outputs"][:bs]), False

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        # 310P ACLGraph：先试独立 runner（自备静态 buffer / set_context），
        # 勿先走 prepare_decode 的 CUDA Graph blob 路径。
        if not is_prefill and self.graph_runner is not None:
            graph_result = self.graph_runner.try_run(seqs)
            if graph_result is not None:
                result, sampled = graph_result
                if self.rank == 0:
                    if sampled:
                        token_ids = result.tolist()
                    else:
                        sample = self.prepare_sample(seqs, False)
                        token_ids = self.sampler(result, *sample).tolist()
                else:
                    token_ids = None
                reset_context()
                return token_ids
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        sample = self.prepare_sample(seqs, is_prefill) if self.rank == 0 else None
        result, sampled = self.run_model(input_ids, positions, is_prefill)
        if self.rank == 0:
            token_ids = (result.tolist() if sampled
                         else self.sampler(result, *sample).tolist())
        else:
            token_ids = None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def run_prefill_spec(self, seqs: list[Sequence]) -> list[int]:
        """spec 模式 prefill 步：target 前向（返回 aux 隐层）+ draft 侧状态初始化。

        EAGLE3：draft 同 chunk 前向（复用同一 context，为 prompt 每个位置写入
        draft KV），已完成 prefill 的序列记录 h_seed（aux 最后一行）。
        DFlash：走 _run_prefill_spec_dflash（aux 全行投影写上下文稠密缓冲）。
        """
        if self.config.spec_method == "dflash":
            return self._run_prefill_spec_dflash(seqs)
        assert self.rank == 0
        input_ids, positions = self.prepare_prefill(seqs)
        hidden, aux = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden)
        sample = self.prepare_sample(seqs)
        token_ids = self.sampler(logits, *sample).tolist()
        offsets = list(accumulate([0] + [seq.num_scheduled_tokens for seq in seqs]))
        # draft prefill：fc(3H -> H) 后逐位置过 midlayer；输出丢弃。
        # 口径与 decode 虚构窗口一致（vLLM set_inputs_first_pass）：rope p 的行
        # embed tok@p+1（末槽补本步首个采样 token），aux@p 前瞻一位。
        seed = self.draft_model.project_aux(aux)
        # 逐 chunk 构造：拓扑与 _refill_ids 的 (bs, rows) 矩阵不同，shifted-ids 口径一致。
        shifted = []
        for i, seq in enumerate(seqs):
            off0, off1 = offsets[i], offsets[i + 1]
            shifted.append(torch.cat([input_ids[off0 + 1:off1],
                                      input_ids.new_tensor(token_ids[i]).view(1)]))
        self.draft_model.forward_prefill(positions, seed, torch.cat(shifted))
        reset_context()
        for i, seq in enumerate(seqs):
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                # 该序列本步完成 prefill：h_seed 取其 chunk 最后一行 aux（预测首 token）。
                self.spec_hidden[seq.seq_id] = aux[offsets[i + 1] - 1:offsets[i + 1]].clone()
                seq.spec_cold = True
        return token_ids

    @torch.inference_mode()
    def spec_round(self, seqs: list[Sequence]) -> list[list[int]]:
        """一轮投机解码（入口）：有捕获图、桶足够且全部序列已有 carry 时走 CUDA
        Graph，否则 eager（冷启动轮在 eager 内补齐 carry）。

        DFlash：分发到 _df_round_graph（图）/ _df_round_eager（eager）；dflash 无
        链式 carry 依赖——prefill 后首轮（pending=None，append 全 padding 行零写、
        块草稿全量读 prefill 上下文）同样可走图，故不需要 EAGLE 的 spec_cold 门。
        commit_buf 契约与 EAGLE3 完全一致（committed = row[:row[rows]]，benchmark
        的 spec_round monkey-patch 统计依赖此入口与契约，勿改）。"""
        # Some backends (310P by default) deliberately capture only greedy
        # argmax in the speculative graph because RNG/sampling kernels are not
        # graph-safe. Never silently change a stochastic request into greedy:
        # use the graph only when all rows are greedy, otherwise run the
        # isomorphic eager body with the native sampler.
        sampling_compatible = (
            self.graph_sample
            or all(seq.temperature <= 1e-10 for seq in seqs)
        )
        if self.config.spec_method == "dflash":
            if self.spec_graphs is not None and seqs and sampling_compatible:
                bucket = _bucket(self.spec_graph_bs, len(seqs))
                if bucket is not None:
                    return self._df_round_graph(seqs, bucket)
            if self.spec_graphs is not None and seqs and not sampling_compatible:
                self.spec_graph_stats["sampling_fallbacks"] += 1
            return self._df_round_eager(seqs)
        if (self.spec_graphs is not None and seqs
                and sampling_compatible
                and not any(seq.spec_cold for seq in seqs)):
            bucket = _bucket(self.spec_graph_bs, len(seqs))
            if bucket is not None:
                return self._spec_round_graph(seqs, bucket)
        if self.spec_graphs is not None and seqs and not sampling_compatible:
            self.spec_graph_stats["sampling_fallbacks"] += 1
        return self._spec_round_eager(seqs)

    def _fill_spec_meta(self, seqs: list[Sequence]) -> int:
        """填充 spec pinned 元数据 blob 的 [0:n] 行（链步/verify 的位置槽位、采样
        参数、block_table、cu_k），返回 n；图内 scatter 分发（仅 2 次 H2D）。"""
        k = self.config.spec_num_tokens
        rows = k + 1
        n = len(seqs)
        lens = [len(seq) for seq in seqs]
        blob = self._sblob
        M, W = self._pin_M, self._pin_W
        draft_vocab = self.draft_config.draft_vocab_size
        # vocab 取 text_config：qwen3_5 外层多模态壳无顶层 vocab_size（实测）；
        # qwen3 的 text_config 即 hf_config，取值逐字节相同（行为零变化）。
        vocab = self.config.text_config.vocab_size
        chain_tokens = blob.view_i64('chain_tokens')
        ctopks = blob.view_i64('ctopks')
        ctemps = blob.view_f32('ctemps')
        ctopps = blob.view_f32('ctopps')
        cp = blob.view_i64('chain_pos', (k, M))
        cc = blob.view_i64('chain_ctxlen', (k, M))
        cs = blob.view_i64('chain_slots', (k, M))
        vp = blob.view_i64('v_pos', (M, rows))
        vs = blob.view_i64('v_slots', (M, rows))
        vt = blob.view_f32('v_temps', (M, rows))
        vpp = blob.view_f32('v_topps', (M, rows))
        cu_k = blob.view_i64('v_cu_k')
        v_topks = blob.view_i64('v_topks')
        bt = blob.view_i64('btables', (M, W))

        for i, seq in enumerate(seqs):
            chain_tokens[i] = seq.last_token
            ctopks[i] = seq.top_k if 0 < seq.top_k < draft_vocab else draft_vocab
            ctemps[i] = seq.temperature
            ctopps[i] = seq.top_p
        # 链步 1..k-1：采样行 rope 恒为 l-2（与 m 无关），流式续写 p = l-2+i，
        # 读窗 = 提交前缀 ctxlen = l+i-1（含本行流槽写）；行 0 已并入 refill
        # 采样行。
        for i in range(1, k):
            for j, (seq, l) in enumerate(zip(seqs, lens)):
                p = l + i - 2
                cp[i, j] = p
                cc[i, j] = l + i - 1
                cs[i, j] = self._slot_of(seq, p)
        cu_k[0] = 0
        acc = 0
        for i, (seq, l) in enumerate(zip(seqs, lens)):
            for j in range(rows):
                p = l - 1 + j
                vp[i, j] = p
                vs[i, j] = self._slot_of(seq, p)
                vt[i, j] = seq.temperature
                vpp[i, j] = seq.top_p
                v_topks[i * rows + j] = seq.top_k if seq.top_k > 0 else vocab
            acc += l + k
            cu_k[i + 1] = acc
        for i, seq in enumerate(seqs):
            self._fill_bt_row(bt, i, seq.block_table)
        # per-query PA 元数据（host）：context_lens[i,j] = l+j
        # (= kv_len - rows + j + 1，kv_len=l+k, rows=k+1)；block_table 按行展开。
        v_ctx = blob.view_i64('v_ctx_lens', (M, rows))
        v_bte = blob.view_i64('v_btables_exp', (M, rows, W))
        for i, l in enumerate(lens):
            for j in range(rows):
                v_ctx[i, j] = max(l + j, 1)
                v_bte[i, j, :] = bt[i]
        return n

    def _load_spec_carry(self, seqs: list[Sequence], n: int):
        """图轮专用：从 spec_carry 播种本轮链（carry_seed 隐层 + 首提案）。"""
        g = self._sg
        g['carry_seed'][:n] = torch.cat([self.spec_carry[seq.seq_id][0] for seq in seqs], dim=0)
        g['drafted_ids'][0][:n] = torch.tensor(
            [self.spec_carry[seq.seq_id][1] for seq in seqs], dtype=torch.int64)

    def _push_spec_meta(self):
        """pinned 元数据 blob 的 2 次 H2D（图内 scatter 分发）。"""
        self._sblob.h2d()

    def _readback_spec_commit(self, seqs: list[Sequence]) -> list[list[int]]:
        """统一读回（每轮唯一一次 D2H）：提取提交 token，并更新下一轮 carry
        （refill 行 k 隐层 + 其采样提案）、aux 行 m 与调度状态。commit_buf 的
        rows/rows+1 两列分别打包提交数和下一轮首提案，避免额外 ``.item()``
        触发第二次设备同步。"""
        g = self._sg
        rows = self.config.spec_num_tokens + 1
        n = len(seqs)
        rows_out = g['commit_buf'][:n].tolist()
        results = []
        for i, seq in enumerate(seqs):
            row = rows_out[i]
            results.append(row[:row[rows]])
            # 下一轮 carry：refill 行 k 隐层 + 其采样提案。
            self.spec_carry[seq.seq_id] = (g['carry_seed'][i:i + 1].clone(),
                                           row[rows + 1])
            # aux 行 m（供 eager 复算采样行 / 图轮 fallback）。
            base = i * rows + row[rows] - 1
            self.spec_hidden[seq.seq_id] = g['v_aux'][base:base + 1].clone()
            # 回滚未提交的 drafted token（KV 残留已被 refill / 后续写入覆盖）。
            seq.num_scheduled_tokens = 0
        return results

    @torch.inference_mode()
    def _spec_round_eager(self, seqs: list[Sequence]) -> list[list[int]]:
        """一轮投机解码（统一 eager 体：采样行引导 → 填 meta → 未捕获执行图轮
        body → 统一读回；冷启动轮由本函数引导补 carry，与图轮同构）。"""
        assert self.rank == 0
        n = len(seqs)
        if n == 0:
            return []
        self.spec_graph_stats["eager_rounds"] += 1
        lens = [len(seq) for seq in seqs]

        # ---- 采样行引导（冷/热一致，与旧 eager 实现逐位相同） ----
        draft_vocab = self.draft_config.draft_vocab_size
        temperatures = torch.tensor(
            [seq.temperature for seq in seqs],
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        top_ps = torch.tensor(
            [seq.top_p for seq in seqs],
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        # draft 采样在 draft 词表上，top_k 上限相应取 draft 词表大小。
        top_ks_draft = torch.tensor(
            [seq.top_k if 0 < seq.top_k < draft_vocab else draft_vocab for seq in seqs],
            dtype=torch.int64, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        h_seed = torch.cat([self.spec_hidden[seq.seq_id] for seq in seqs], dim=0)
        seed = self.draft_model.project_aux(h_seed)
        slot_dtype = self.backend.graph_slot_mapping_dtype()
        p = torch.tensor(
            [l - 2 for l in lens], dtype=torch.int64, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        ctx = torch.tensor(
            [l - 1 for l in lens], dtype=torch.int32, device="cpu", pin_memory=True
        ).to(self.device, non_blocking=True)
        # 1D slot_indices（Ascend ATB 硬性要求；[n,1] 会 ReshapeAndCache setup failed）
        slots = torch.tensor(
            [self._slot_of(seq, l - 2) for seq, l in zip(seqs, lens)],
            dtype=slot_dtype,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        tokens = torch.tensor(
            [seq.last_token for seq in seqs],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        set_context(False, slot_mapping=slots, context_lens=ctx,
                    block_tables=self.prepare_block_tables(seqs),
                    max_context_len=self.config.max_model_len,
                    block_size=self.block_size)
        row_m_hidden, draft_logits = self.draft_model.forward_decode(p, seed, tokens)
        row_m_hidden = row_m_hidden.clone()
        draft_ids = self.sampler.forward_native(draft_logits, temperatures, top_ps, top_ks_draft)
        draft1 = self.draft_model.map_to_target_ids(draft_ids)

        # ---- 播种本轮：carry_seed 取采样行的条件 seed（fc(aux)，与旧 eager 链步
        # 1 的输入逐位一致），drafted_ids[0] 为其首提案 ----
        g = self._sg
        g['carry_seed'][:n] = seed
        g['drafted_ids'][0][:n] = draft1

        # ---- 填 meta → H2D → 未捕获执行图轮 body → 统一读回 ----
        self._fill_spec_meta(seqs)
        self._push_spec_meta()
        self._sg_round(n, use_cuda_graph=False)
        results = self._readback_spec_commit(seqs)
        # 冷启动→图轮交接与旧实现逐位一致：图轮链步 1 的 seed/首提案取本轮采样行
        # （读回写入的 refill 产物仅用于图轮稳态续流）。
        for i, seq in enumerate(seqs):
            self.spec_carry[seq.seq_id] = (row_m_hidden[i:i + 1], draft1[i].item())
        reset_context()
        for seq in seqs:
            seq.spec_cold = False
        return results

    # ---- DFlash 块草稿投机（阶段2，eager；CUDA 图留阶段3）----
    # 轮次结构（z-lab dflash 语义，对齐 vllm DFlashProposer/DFlashSpeculator）：
    # ① 上下文追加（上轮 v_aux 已提交行 → fc+norm → k/v proj → rope → 稠密缓冲）
    # ② block 草稿（anchor+mask×k 单次非因果前向 → target 词表采样 [n, k]）
    # ③ verify（复用 EAGLE 骨架的 _sg 缓冲与验收口径；GDN 走 'row' 逐行落槽）
    # ④ 读回提交 + 更新 spec_ctx_pending / GDN carry（无链式 carry / refill）。

    def _df_ctx_get_or_alloc(self, seq: Sequence) -> int:
        """draft 上下文 KV 稠密缓冲的槽 get_or_alloc（prefill 时分配）。

        新槽内容无脏读风险：prefill 全量重写 0..P-1；块草稿读取恒以
        ctx_len = len-1 为右界 + -inf mask 屏蔽界外残留（缓冲零初始化，无 NaN）。
        """
        slot = self.draft_ctx_slots.get(seq.seq_id)
        if slot is None:
            try:
                slot = self._df_ctx_free.pop()
            except KeyError:
                raise RuntimeError(
                    "Draft-context slots are exhausted; live sequences exceeded "
                    f"max_num_seqs={self.config.max_num_seqs}."
                ) from None
            self.draft_ctx_slots[seq.seq_id] = slot
        return slot

    @torch.inference_mode()
    def _run_prefill_spec_dflash(self, seqs: list[Sequence]) -> list[int]:
        """DFlash prefill 步：target 前向（aux 5 层捕获）+ 首 token 采样 + prefill
        aux 全行（位置 0..P-1）投影写入 draft 上下文 KV 稠密缓冲。

        与 EAGLE3 prefill 的差异：无 draft 流式 prefill（dflash 无链式流——draft
        上下文 = aux 重投影，块查询行每轮由 embed 现算）；spec_ctx_pending 置 None
        （上下文已落盘，首轮块草稿的 ctx_len = P 恰为 anchor 前的全部位置）。
        """
        assert self.rank == 0
        input_ids, positions = self.prepare_prefill(seqs)
        hidden, aux = self.model(input_ids, positions)
        logits = self.model.compute_logits(hidden)
        token_ids = self.sampler(logits, *self.prepare_sample(seqs)).tolist()
        reset_context()
        offsets = list(accumulate([0] + [seq.num_scheduled_tokens for seq in seqs]))
        for i, seq in enumerate(seqs):
            if seq.num_cached_tokens + seq.num_scheduled_tokens != seq.num_tokens:
                continue  # hybrid 禁 chunked prefill（config 断言），防御分支
            # ``warmup_model`` runs before ``allocate_kv_cache`` so its dummy
            # sequences intentionally have no block table. CUDA/CoreX DFlash
            # can still write their already-allocated dense context cache, but
            # the 310P paged path has neither physical slots nor draft NZ cache
            # yet. Skip only that budget-warmup store; after allocation, graph
            # warmup and LLMEngine's runtime warmup execute the complete path.
            if (self.dflash_paged
                    and (not seq.block_table
                         or not self.draft_model.layers[0].self_attn.k_cache.numel())):
                self.gdn_spec_carry.pop(seq.seq_id, None)
                self.spec_ctx_pending[seq.seq_id] = None
                continue
            slot = self._df_ctx_get_or_alloc(seq)
            p0, p1 = offsets[i], offsets[i + 1]
            if self.dflash_paged:
                start = seq.num_cached_tokens
                context_slots = torch.tensor(
                    [self._slot_of(seq, p) for p in range(start, start + (p1 - p0))],
                    dtype=self.backend.graph_slot_mapping_dtype(),
                    device="cpu",
                    pin_memory=True,
                ).to(self.device, non_blocking=True)
            else:
                context_slots = torch.full(
                    (p1 - p0,), slot, dtype=torch.int64, device=self.device
                )
            # prefill aux 全行（0-based 层输出口径）→ 投影写缓冲（位置 0..P-1）
            self.draft_model.project_context_kv(
                aux[p0:p1], positions[p0:p1],
                context_slots)
            # 首 verify 的 GDN carry = prefill 终态主槽（gdn_slots），清 preempt 残留
            self.gdn_spec_carry.pop(seq.seq_id, None)
            self.spec_ctx_pending[seq.seq_id] = None
        return token_ids

    def _df_append_context(self, seqs: list[Sequence]):
        """① 上下文追加：spec_ctx_pending（上轮 v_aux 已提交行）→ 投影写稠密缓冲。

        行语义（对齐 vllm _prepare_dflash_inputs_kernel：上下文 = verify 前 m+1 个
        有效行 = [last_token, d_0..d_{m-1}]，位置 L-1..L+m-1）：上轮 verify 的
        v_aux 行 0..m 恰为这批新可得已提交 token 的 aux——行 0 = 本轮 last_token
        （上轮 bonus，其 aux 由上轮 verify 行 0 首次算出）、行 1..m = 被接受的提案。
        上下文覆盖不变式 = 已提交 token 位置 0..L-2 的真 aux（anchor 位置 L-1 不入
        上下文，由块查询行自身的 KV 表示）。注：任务书草案「行 1..m+1」有偏——
        行 m+1 是被拒提案的 aux（内容错位），行 0 不追加则 L-1 位永久空洞；以
        vllm 口径为准实现（偏差已记入报告，门B 以 E 数值佐证）。
        pending 行数 c 的位置可由长度反推：[L-1-c, L-2]（提交只增长度）。
        """
        rows = [(s, p) for s in seqs
                if (p := self.spec_ctx_pending.get(s.seq_id)) is not None]
        if not rows:
            return
        aux = torch.cat([p for _, p in rows], dim=0)              # [T, aux宽] GPU
        lens = [p.size(0) for _, p in rows]
        slot_list = [self.draft_ctx_slots[s.seq_id] for s, _ in rows]
        # 位置 = [L-1-c, L-1)（device int64）
        pos = torch.cat([
            torch.arange(len(s) - 1 - c, len(s) - 1, dtype=torch.int64, device=self.device)
            for (s, _), c in zip(rows, lens)])
        if self.dflash_paged:
            physical = []
            for (s, _), c in zip(rows, lens):
                start = len(s) - 1 - c
                physical.extend(self._slot_of(s, p) for p in range(start, start + c))
            slots = torch.tensor(
                physical,
                dtype=self.backend.graph_slot_mapping_dtype(),
                device="cpu",
                pin_memory=True,
            ).to(self.device, non_blocking=True)
        else:
            slots = torch.cat([
                torch.full((c,), slot, dtype=torch.int64, device=self.device)
                for slot, c in zip(slot_list, lens)])
        self.draft_model.project_context_kv(aux, pos, slots)
        for s, _ in rows:
            self.spec_ctx_pending[s.seq_id] = None

    def _df_draft_block(self, seqs: list[Sequence]) -> torch.Tensor:
        """② block 草稿：anchor+mask×k 单次非因果前向，逐 mask 行采样提案。

        返回 [n, k] target 词表 id（draft 词表 = target，无 d2t 映射）。采样参数
        按序列展开（temp0 → argmax 确定性提案；top_k 上界 = target 词表），
        logits 行序 (i, j) → i*k+j 与 repeat_interleave(k) 对齐。
        """
        k = self.config.spec_num_tokens
        rows = k + 1
        n = len(seqs)
        Ls = [len(seq) for seq in seqs]
        # 块查询位置 = [l'-1, ..., l'+k-1]（anchor = 最后已提交 token，l' = 序列长）
        positions = torch.cat([
            torch.arange(L - 1, L - 1 + rows, dtype=torch.int64, device=self.device)
            for L in Ls]).view(n, rows)
        last = torch.tensor(
            [seq.last_token for seq in seqs],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        # 上下文读窗：ctx_len = L-1（覆盖 0..L-2），右 padding 至批内最大（host int）
        ctx_lens = torch.tensor(
            [L - 1 for L in Ls],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        max_ctx = (self.config.max_model_len if self.dflash_paged
                   else max(L - 1 for L in Ls))
        ctx_slots = torch.tensor(
            [self.draft_ctx_slots[seq.seq_id] for seq in seqs],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        query_slots = block_tables = total_lens = None
        if self.dflash_paged:
            query_slots = torch.tensor(
                [self._slot_of(seq, p)
                 for seq, L in zip(seqs, Ls)
                 for p in range(L - 1, L - 1 + rows)],
                dtype=self.backend.graph_slot_mapping_dtype(),
                device="cpu",
                pin_memory=True,
            ).to(self.device, non_blocking=True)
            block_tables = self.prepare_block_tables(seqs)
            total_lens = torch.tensor(
                [L - 1 + rows for L in Ls],
                dtype=torch.int32,
                device="cpu",
                pin_memory=True,
            ).to(self.device, non_blocking=True)
        logits = self.draft_model.forward_block(
            last, positions, ctx_slots, ctx_lens, max_ctx, rows,
            query_slots=query_slots,
            block_tables=block_tables,
            total_lens=total_lens,
        )
        vocab = self.config.text_config.vocab_size
        temps = torch.tensor(
            [seq.temperature for seq in seqs],
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        topps = torch.tensor(
            [seq.top_p for seq in seqs],
            dtype=torch.float32,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        topks = torch.tensor(
            [seq.top_k if 0 < seq.top_k < vocab else vocab for seq in seqs],
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        ids = self.sampler.forward_native(
            logits, temps.repeat_interleave(k), topps.repeat_interleave(k),
            topks.repeat_interleave(k))
        return ids.view(n, k)

    def _df_verify(self, seqs: list[Sequence]):
        """③ target verify：[last_token, d1..dk] 一次前向，行 j 预测位置 L+j，
        图外采样并暂存 aux。

        与 _sg_verify 同构（同一批 _sg 静态缓冲），差异仅 GDN 上下文：'row' 模式
        + 写基址/ carry 槽——EAGLE3 路径零触碰红线，故独立成方法而非加参。
        GDN 元数据（host 逐 seq 解析，跨轮槽位按 seq_id 记账；仅 hybrid）：
        - 写基址 base_i = max_num_seqs + i×(k+1)（池静态布局，行 0..k 逐行落槽）；
        - 行 0 读槽 = gdn_spec_carry[seq]（后续轮 = 上轮 base+m）或 gdn_slots[seq]
          （首轮 = prefill 终态主槽）。
        dense（Qwen3+DFlash）无 GDN：上下文 kwargs 整组缺省，_df_bases 仍记录
        （读回侧 is_hybrid 门控，恒不消费）。
        """
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        n = len(seqs)
        ms = self.config.max_num_seqs
        bases = [ms + i * rows for i in range(n)]
        gdn_extra: dict = {}
        if self.is_hybrid:
            carries = [self.gdn_spec_carry.get(s.seq_id, self.gdn_slots[s.seq_id]) for s in seqs]
            base_dev = torch.tensor(
                bases, dtype=torch.int32, device="cpu", pin_memory=True
            ).to(self.device, non_blocking=True)
            carry_dev = torch.tensor(
                carries, dtype=torch.int32, device="cpu", pin_memory=True
            ).to(self.device, non_blocking=True)
            gdn_extra = dict(gdn_state_slots=base_dev, gdn_carry_slots=carry_dev, gdn_state_mode='row')
        # max_seqlen_k 用 max_model_len 上界（同 _sg_verify 的 varlen grid 口径）
        set_context(True,
                    cu_seqlens_q=g['v_cu_q'][:n + 1],
                    cu_seqlens_k=g['v_cu_k'][:n + 1],
                    max_seqlen_q=rows,
                    max_seqlen_k=self.config.max_model_len,
                    slot_mapping=g['v_slots'][:n * rows],
                    context_lens=g['v_ctx_lens'][:n * rows],
                    block_tables=g['v_btables_exp'][:n * rows],
                    block_size=self.block_size,
                    **gdn_extra)
        hidden, aux = self.model(g['v_input_ids'][:n * rows], g['v_positions'][:n * rows])
        logits = F.linear(
            hidden, getattr(self.model.lm_head, "weight_nz", self.model.lm_head.weight))
        g['v_token_ids'][:n * rows] = self.sampler.forward_native(
            logits, g['v_temps'][:n * rows], g['v_topps'][:n * rows], g['v_topks'][:n * rows])
        g['v_aux'][:n * rows].copy_(aux)
        self._df_bases = bases  # 供读回更新 carry（host int）

    def _df_readback_commit(self, seqs: list[Sequence]) -> list[list[int]]:
        """④ 统一读回（每轮唯一一次 D2H；commit_buf 契约与 EAGLE3 完全一致：
        committed = row[:row[rows]]，cnt 固定打包在第 rows 列）：
        提取提交 token + 更新 spec_ctx_pending（v_aux 行 0..m = 新可得已提交位置
        L-1..L+m-1，供下轮上下文追加）与 GDN carry（本轮行 m 的槽 base_i+m）。
        """
        g = self._sg
        rows = self.config.spec_num_tokens + 1
        n = len(seqs)
        rows_out = g['commit_buf'][:n].tolist()
        results = []
        for i, seq in enumerate(seqs):
            row = rows_out[i]
            c = row[rows]
            results.append(row[:c])
            # 下一轮上下文追加行：v_aux 行 0..c-1（GPU 常驻 clone）
            self.spec_ctx_pending[seq.seq_id] = g['v_aux'][i * rows:i * rows + c].clone()
            # GDN carry = 本轮行 m 槽（c-1 = m ≥ 0）；下轮行 0 读此槽（仅 hybrid）
            if self.is_hybrid:
                self.gdn_spec_carry[seq.seq_id] = self._df_bases[i] + (c - 1)
            # 回滚未提交的调度计数（同 _readback_spec_commit）
            seq.num_scheduled_tokens = 0
        return results

    @torch.inference_mode()
    def _df_round_eager(self, seqs: list[Sequence]) -> list[list[int]]:
        """DFlash 一轮投机（eager 全量体；CUDA 图留阶段3）。

        verify 段复用 _sg 静态缓冲与 _sg_acceptance（纯张量验收，零上下文依赖）；
        提案写入 g['drafted_ids'] 后验收口径与 EAGLE3 逐位一致。
        """
        assert self.rank == 0
        n = len(seqs)
        if n == 0:
            return []
        self.spec_graph_stats["eager_rounds"] += 1
        k = self.config.spec_num_tokens
        rows = k + 1
        # ---- ① 上下文追加 ----
        self._df_append_context(seqs)
        # ---- ② block 草稿 ----
        drafted = self._df_draft_block(seqs)   # [n, k] target 词表 id
        # ---- ③ verify：填 meta（复用 _fill_spec_meta，chain 段写而不读）→
        #      组装 v_input_ids = [last_token, d1..dk] → target 前向（GDN row）----
        self._fill_spec_meta(seqs)
        # 与 EAGLE3 eager 流程对齐的元数据分发：_fill_spec_meta 只写 pinned host
        # blob，_df_verify 消费的是 _sg 静态 buffer（v_positions/v_slots/v_cu_k/
        # v_temps/v_topps/v_topks/v_btables/chain_tokens）——须经 _push_spec_meta
        # （2 次 H2D 到设备孪生）+ _sg_scatter（孪生 → 静态 buffer）才生效。缺了
        # 这两步会读到初始化零值（实测 verify 行 0 被喂 token 0、KV 全写槽 0，
        # 整轮垃圾 → E≈1 且输出全错）。
        self._push_spec_meta()
        self._sg_scatter(n)
        g = self._sg
        v = g['v_input_ids'][:n * rows].view(n, rows)
        v[:, 0] = g['chain_tokens'][:n]        # = 各 seq.last_token
        v[:, 1:] = drafted                     # d1..dk（target 词表，无映射）
        g['drafted_ids'][:k, :n] = drafted.t()  # 验收缓冲（_sg_acceptance 消费）
        self._df_verify(seqs)
        self._sg_acceptance(n)
        # ---- ④ 读回提交 + 更新 pending / carry ----
        return self._df_readback_commit(seqs)

    # ---- DFlash spec CUDA Graph（阶段3）----
    # capture_spec_cudagraph（EAGLE3）的 dflash 等价物：EAGLE 骨架零触碰红线，
    # 故独立成方法族而非参数化复用（两 body 结构同构：scatter → 起草 → verify
    # 输入组装 → verify → 图内接受；差异 = dflash 无链式循环/refill，前置一段
    # 上下文追加）。每桶一张图覆盖整轮，每轮仅一次 D2H（commit_buf 读回）。
    #
    # 图内静态化三要点（与 eager _df_* 的对应关系）：
    # a) 上下文追加（_df_append_context_graph）：待追加行数 c = m+1 每 seq 变长
    #    → 按 rows=k+1 行 padding：无效行 aux 清零（_df_load_ctx_aux 装载时清）
    #    → fc/norm/k_norm/rope 对零向量恒产出零 → 写入 ctx 缓冲的是零特征。写位
    #    置 = (L-1-c)+j 连续延展到 [L-1, L-1-c+rows)：这些位置本round的块草稿读
    #    不到（ctx 读窗恒 0..L-2），且下轮 append 恰从 L-1 起以真 aux 逐位覆盖
    #    （不变式：ctx[0..L-2] 全真、ctx[L-1..] 无所谓——归纳可证，与「被拒提案
    #    残留无害」同一口径）；即使下轮 seq 结束也不再有任何读。故零写后无读，
    #    无需 mask。
    # b) 块草稿（_df_draft_block_graph）：ctx gather 固定 max_model_len 宽
    #    （eager 为批内最大 ctx_len）——gather 形状静态，越界 padding 由 -inf
    #    mask 屏蔽（forward_block 既有逻辑，掩码后零贡献精确不扰动求和）；块前
    #    向固定 [bs, k+1] 形状。padding 行消费 chain_tokens/df_l1 的陈旧值（恒
    #    为合法 token id / < max_model_len 的位置，产出垃圾提案被丢弃，不写任何
    #    KV/状态）。
    # c) verify（_df_verify_graph）：沿用 _sg 静态缓冲 + _sg_scatter；GDN 'row'
    #    模式在图内——写基址改为 host 常量列表注入（gdn_row_bases，按池静态布局
    #    只依赖批内序号，bake 进图即正确），行 0 读槽走 df_carry 静态 buffer
    #    （blob 每轮分发，动态值）；python 循环（逐 seq × 逐行）捕获期按常量
    #    展开（次数只依赖 bs 桶与 k+1）。padding 行写各自 base 行槽区（互斥），
    #    行 0 读 df_carry=0 号槽（只读不脏）。

    def _alloc_df_graph_vars(self, max_bs: int, rows: int) -> dict:
        """dflash 图轮专段静态缓冲（_sg 之外的补充段；eagle3 不分配）。"""
        # aux 宽度取 text_config（qwen3_5 外层壳无顶层 hidden_size）。
        aux_width = self.config.text_config.hidden_size * len(self.config.spec_aux_layers)
        slot_dtype = self.backend.graph_slot_mapping_dtype()
        return dict(
            # 上下文追加的 aux 暂存：host 每轮从 spec_ctx_pending（GPU clone）
            # 装载有效行、其余清零；图内 project_context_kv 消费固定 [bs*rows] 形状。
            df_ctx_aux=torch.zeros(max_bs * rows, aux_width, dtype=self.dtype),
            # GDN verify 写基址：桶静态常量 base_i = max_num_seqs + i*rows
            # （_alloc_gdn_state_pool 的行槽区布局），capture 期烘进图。
            df_bases=(self.config.max_num_seqs
                      + torch.arange(max_bs, dtype=torch.int32) * rows),
            # GDN verify 行 0 读槽：每轮经 _df_scatter 从 blob 孪生分发（动态）。
            df_carry=torch.zeros(max_bs, dtype=torch.int32),
            # 310P paged DFlash metadata. Keep these as static graph buffers so
            # int64->int32 conversion and non-contiguous last-column slicing do
            # not become operators inside the captured graph.
            df_ctx_positions=torch.zeros(max_bs * rows, dtype=torch.int64),
            df_ctx_slots=torch.full(
                (max_bs * rows,), self._graph_pad_slot(), dtype=slot_dtype),
            df_total_lens=torch.full(
                (max_bs,), self._graph_pad_ctx(), dtype=torch.int32),
        )

    def _fill_df_meta(self, seqs: list[Sequence]):
        """dflash 图轮专段填充（_fill_spec_meta 之后调用，host 写 pinned blob）：
        L-1 / pending 行数 / ctx 槽 / GDN 行 0 读槽。chain 段（chain_pos/ctxlen/
        slots）dflash 不消费、写而不读（与 eager 轮同口径）。"""
        blob = self._sblob
        l1 = blob.view_i64('df_l1')
        c = blob.view_i64('df_c')
        cs = blob.view_i64('df_ctxslot')
        cy = blob.view_i64('df_carry')
        rows = self.config.spec_num_tokens + 1
        M = self._pin_M
        ctx_pos = blob.view_i64('df_ctx_pos', (M, rows))
        ctx_phys = blob.view_i64('df_ctx_phys_slots', (M, rows))
        total_lens = blob.view_i64('df_total_lens')
        for i, seq in enumerate(seqs):
            l1_value = len(seq) - 1
            l1[i] = l1_value
            p = self.spec_ctx_pending.get(seq.seq_id)
            pending_count = 0 if p is None else p.size(0)
            c[i] = pending_count
            cs[i] = self.draft_ctx_slots[seq.seq_id]
            start = l1_value - pending_count
            total_lens[i] = l1_value + rows
            for j in range(rows):
                pos = start + j
                ctx_pos[i, j] = pos
                ctx_phys[i, j] = (
                    self._slot_of(seq, pos) if self.dflash_paged else int(cs[i])
                )
            # 行 0 读槽：与 eager _df_verify 同源（carry 优先，回落 prefill 主槽）。
            # dense（Qwen3+dflash）无 GDN：df_carry 段留零，_df_verify_graph 不消费。
            if self.is_hybrid:
                cy[i] = self.gdn_spec_carry.get(seq.seq_id, self.gdn_slots[seq.seq_id])

    def _df_load_ctx_aux(self, seqs: list[Sequence], bs: int):
        """上下文追加的 aux 暂存装载（图外 host 步，GPU→GPU）：spec_ctx_pending
        逐序列拷入静态段 [i*rows, i*rows+c)，段内其余行与 padding 序列清零。
        装载即消费——pop 掉 pending（本轮 replay 的 append 消费；下轮 pending 由
        _df_readback_commit 重写）。"""
        g = self._sg
        rows = self.config.spec_num_tokens + 1
        buf = g['df_ctx_aux']
        buf[:bs * rows].zero_()
        for i, seq in enumerate(seqs):
            p = self.spec_ctx_pending.pop(seq.seq_id, None)
            if p is not None:
                buf[i * rows:i * rows + p.size(0)].copy_(p)

    def _df_scatter(self, bs: int):
        """dflash 专段 blob → 静态 buffer 的图内分发（_sg_scatter 的 df 等价物；
        df_l1/df_c/df_ctxslot 为 int64 直通，图体直接消费 blob 孪生，无需中转）。"""
        g = self._sg
        blob = self._sblob
        rows = self.config.spec_num_tokens + 1
        M = self._pin_M
        g['df_carry'][:bs].copy_(blob.device_view_i64('df_carry')[:bs])
        g['df_ctx_positions'][:bs * rows].copy_(
            blob.device_view_i64('df_ctx_pos', (M, rows))[:bs].reshape(-1))
        g['df_ctx_slots'][:bs * rows].copy_(
            blob.device_view_i64('df_ctx_phys_slots', (M, rows))[:bs].reshape(-1))
        g['df_total_lens'][:bs].copy_(
            blob.device_view_i64('df_total_lens')[:bs])

    def _df_append_context_graph(self, bs: int):
        """① 上下文追加（图内静态化）：project_context_kv 全链（fc → hidden_norm
        → 融合 k/v GEMM → 逐层 k_norm → rope → 高级索引写稠密缓冲）无 host 同步、
        循环次数 = draft 层数常量，图安全。位置/槽位来自 blob 孪生（固定地址），
        padding 行 aux=0 → 写零特征（无害性论证见本节头注释 a）。"""
        g = self._sg
        rows = self.config.spec_num_tokens + 1
        aux = g['df_ctx_aux'][:bs * rows]
        pos = g['df_ctx_positions'][:bs * rows]
        slots = g['df_ctx_slots'][:bs * rows]
        self.draft_model.project_context_kv(aux, pos, slots)

    def _df_draft_block_graph(self, bs: int):
        """② block 草稿（图内静态化）：dflash.forward_block 原样复用——其内部
        （embed gather / ctx gather / arange mask / sdpa / 逐层循环）全部静态形状
        无 host 同步，只需调用方传静态 buffer：gather 宽度固定 max_model_len
        （eager 为批内 max，掩码屏蔽越界段，有效段数值一致）、ctx_lens = df_l1
        （= L-1）、块查询位置 = df_l1 + arange(rows)。采样参数复用 _sg_scatter
        产物 chain_temps/topps/topks（ctopks 的 draft_vocab 上界对 dflash = target
        词表，与 eager _df_draft_block 的 vocab 上界逐值相同）。"""
        g = self._sg
        blob = self._sblob
        k = self.config.spec_num_tokens
        rows = k + 1
        l1 = blob.device_view_i64('df_l1')[:bs]
        # v_positions is already host-expanded to [L-1, ..., L-1+k] by
        # _fill_spec_meta and copied into a static int64 graph buffer. Reusing
        # it avoids an NPU int64 arange+Add pair (AICPU/507011 on 310P graphs).
        block_pos = g['v_positions'][:bs * rows].view(bs, rows)
        logits = self.draft_model.forward_block(
            g['chain_tokens'][:bs], block_pos,
            blob.device_view_i64('df_ctxslot')[:bs], l1,
            self.config.max_model_len, rows,
            query_slots=(g['v_slots'][:bs * rows] if self.dflash_paged else None),
            block_tables=(g['v_btables'][:bs] if self.dflash_paged else None),
            total_lens=(g['df_total_lens'][:bs]
                        if self.dflash_paged else None))
        ids = self._sg_sample(
            logits,
            g['chain_temps'][:bs].repeat_interleave(k),
            g['chain_topps'][:bs].repeat_interleave(k),
            g['chain_topks'][:bs].repeat_interleave(k),
            True)
        g['drafted_ids'][:k, :bs] = ids.view(bs, k).t()

    def _df_verify_graph(self, bs: int):
        """③ target verify（图内静态化）：与 _df_verify 同构（同一批 _sg 静态
        buffer），差异仅在 GDN 元数据的图安全来源（仅 hybrid）——写基址 host
        常量列表（gdn_row_bases，capture 期烘进图）+ 行 0 读槽 df_carry 静态
        buffer（_df_scatter 每轮分发），取代 eager 版的两次 torch.tensor H2D
        （捕获期会把值烘焙成常量，carry 必须走 blob）。padding 行 base 指向各
        自行槽区。dense（Qwen3+dflash）无 GDN：上下文 kwargs 整组缺省。"""
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        gdn_extra: dict = {}
        if self.is_hybrid:
            ms = self.config.max_num_seqs
            bases = [ms + i * rows for i in range(bs)]
            gdn_extra = dict(gdn_state_mode='row',
                             gdn_carry_slots=g['df_carry'][:bs],
                             gdn_row_bases=bases,
                             gdn_row_seg=rows)
        # max_seqlen_k 用 max_model_len 上界（同 _sg_verify/_df_verify 的 varlen
        # grid 口径，避免 capture 烘焙随轮次变化的 python int）
        set_context(True,
                    cu_seqlens_q=g['v_cu_q'][:bs + 1],
                    cu_seqlens_k=g['v_cu_k'][:bs + 1],
                    max_seqlen_q=rows,
                    max_seqlen_k=self.config.max_model_len,
                    slot_mapping=g['v_slots'][:bs * rows],
                    context_lens=g['v_ctx_lens'][:bs * rows],
                    block_tables=g['v_btables_exp'][:bs * rows],
                    block_size=self.block_size,
                    **gdn_extra)
        hidden, aux = self.model(g['v_input_ids'][:bs * rows], g['v_positions'][:bs * rows])
        logits = F.linear(
            hidden, getattr(self.model.lm_head, "weight_nz", self.model.lm_head.weight))
        g['v_token_ids'][:bs * rows] = self._sg_sample(
            logits, g['v_temps'][:bs * rows], g['v_topps'][:bs * rows], g['v_topks'][:bs * rows],
            True)
        g['v_aux'][:bs * rows].copy_(aux)

    def _df_round_body(self, bs: int):
        """DFlash spec 单图整轮 body（capture 预热与图捕获共用，全静态算子）：
        scatter → 上下文追加 → block 草稿 → verify 输入组装 → verify → 图内接受。
        无 EAGLE 的链式循环/refill；提交组装复用 _sg_acceptance（纯张量验收，
        commit_buf 契约与 EAGLE3 一致）。"""
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        self._sg_scatter(bs)
        self._df_scatter(bs)
        self._df_append_context_graph(bs)
        self._df_draft_block_graph(bs)
        v = g['v_input_ids'][:bs * rows].view(bs, rows)
        v[:, 0] = g['chain_tokens'][:bs]
        v[:, 1:].copy_(g['drafted_ids'][:k, :bs].t())
        self._df_verify_graph(bs)
        self._sg_acceptance(bs)

    @torch.inference_mode()
    def capture_df_spec_cudagraph(self):
        """DFlash spec 单图整轮捕获（capture_spec_cudagraph 的 dflash 等价物，
        结构同构、body 换 _df_round_body；EAGLE 捕获函数逐字节未动）。"""
        max_bs = self._graph_max_bs
        self.spec_graph_bs = _graph_buckets(max_bs)
        g = self._sg    # 缓冲已在 __init__ 分配（spec 启用即存在，eager 轮共用）
        self.spec_graphs = {}
        try:
            M, W = self._pin_M, self._pin_W
            rows = self.config.spec_num_tokens + 1
            blob = self._sblob
            pad_block = self._graph_pad_block()
            pad_base = pad_block * self.block_size
            pad_slots = torch.arange(
                pad_base, pad_base + rows, dtype=torch.int64, device="cpu")
            pad_positions = torch.arange(rows, dtype=torch.int64, device="cpu")
            pad_context_lens = torch.arange(
                1, rows + 1, dtype=torch.int64, device="cpu")
            blob.view_i64('btables', (M, W)).fill_(pad_block)
            blob.view_i64('v_slots', (M, rows))[:] = pad_slots
            blob.view_i64('v_pos', (M, rows))[:] = pad_positions
            blob.view_i64('v_ctx_lens', (M, rows))[:] = pad_context_lens
            blob.view_i64('v_btables_exp', (M, rows, W)).fill_(pad_block)
            blob.view_i64('df_ctx_pos', (M, rows))[:] = pad_positions
            blob.view_i64('df_ctx_phys_slots', (M, rows))[:] = pad_slots
            blob.view_i64('df_total_lens').fill_(rows)
            blob.h2d()
            for bs in reversed(self.spec_graph_bs):
                # warmup：kernel lazy init 必须在 capture 前完成；warmup 数据用
                # 全零静态 buffer（token 0 / 槽 0 / 位置 0 合法，ctx 追加写零特征、
                # 块草稿/verify 产出垃圾输出但写入的是哑槽/被丢弃）。
                g['v_cu_k'][:bs + 1] = g['v_cu_q'][:bs + 1]
                for _ in range(self.backend.graph_warmup_iters()):
                    self._df_round_body(bs)
                    reset_context()
                    self.backend.synchronize()
                graph = self.backend.new_graph()
                with self.backend.graph_ctx(graph, pool=self.spec_graph_pool):
                    self._df_round_body(bs)
                if self.spec_graph_pool is None:
                    pool_fn = getattr(graph, 'pool', None)
                    self.spec_graph_pool = pool_fn() if callable(pool_fn) else None
                self.spec_graphs[bs] = graph
                reset_context()
                # 冒烟回放（与 EAGLE capture_spec 同款）：检出 AICPU/507011
                self._push_spec_meta()
                graph.replay()
                self.backend.synchronize()
                self.spec_graph_stats["capture_success"] += 1
        except Exception as e:
            # 与 EAGLE 同款优雅降级：spec_round 自动回落 _df_round_eager（阶段2
            # 已验收的 eager 全链路），功能不受影响。
            print(
                "[nano-vllm] DFlash speculative graph capture failed; falling back "
                f"to eager speculative rounds: {type(e).__name__}: {e}",
                flush=True,
            )
            self.spec_graph_stats["capture_failure"] += 1
            traceback.print_exc()
            try:
                reset_context()
                self.backend.synchronize()
            except Exception:
                pass
            self.spec_graphs = None
            self.spec_graph_pool = None
            try:
                self.backend.empty_cache()
            except Exception:
                pass

    @torch.inference_mode()
    def _df_round_graph(self, seqs: list[Sequence], bucket: int) -> list[list[int]]:
        """DFlash spec 图轮回放（_spec_round_graph 的 dflash 等价物）：fill meta
        → padding 复位 → 装载 append 暂存 → H2D → 单图 replay → 读回。"""
        n = self._fill_spec_meta(seqs)
        self._fill_df_meta(seqs)
        rows = self.config.spec_num_tokens + 1

        # padding 复位：n < bucket 即需。310P 的 ReshapeAndCache/SplitFuse 不使用
        # CUDA 的 -1 跳写约定，所以每个 dummy request 都指向末尾保留 block 的
        # 0..rows-1 合法槽，context_lens=1..rows、total_len=rows；输出全部丢弃。
        # 这样既不写脏真实序列，也不会向图中引入 slots<0 的动态修正算子。
        if n < bucket:
            M, W = self._pin_M, self._pin_W
            blob = self._sblob
            vs = blob.view_i64('v_slots', (M, rows))
            vp = blob.view_i64('v_pos', (M, rows))
            cu_k = blob.view_i64('v_cu_k')
            bt = blob.view_i64('btables', (M, W))
            # 注意轴序：v_slots 视图 (M, rows) 的 seq 维在轴 0——必须切
            # vs[n:, :]（padding 序列的全部行），不能切成 vs[:, n:]。
            pad_block = self._graph_pad_block()
            pad_base = pad_block * self.block_size
            pad_slots = torch.arange(
                pad_base, pad_base + rows, dtype=torch.int64, device="cpu")
            pad_positions = torch.arange(rows, dtype=torch.int64, device="cpu")
            pad_context_lens = torch.arange(
                1, rows + 1, dtype=torch.int64, device="cpu")
            vs[n:, :] = pad_slots
            vp[n:, :] = pad_positions
            # padding 段 K 窗零长（cu 单调不降、kernel 跳过空段）
            cu_k[n + 1:] = cu_k[n]
            bt[n:, :].fill_(pad_block)
            v_ctx = blob.view_i64('v_ctx_lens', (M, rows))
            v_bte = blob.view_i64('v_btables_exp', (M, rows, W))
            v_ctx[n:, :] = pad_context_lens
            v_bte[n:, :, :].fill_(pad_block)
            blob.view_i64('df_l1')[n:].zero_()
            blob.view_i64('df_c')[n:].zero_()
            # padding 序列 ctx 槽 → 专用 pad 槽（append 零写落 pad 槽，不碰任何
            # 真实序列的上下文；块草稿对 padding 行 gather pad 槽零特征，掩码屏蔽）
            blob.view_i64('df_ctxslot')[n:].fill_(self._df_pad_ctx_slot)
            blob.view_i64('df_carry')[n:].zero_()
            blob.view_i64('df_ctx_pos', (M, rows))[n:, :] = pad_positions
            blob.view_i64('df_ctx_phys_slots', (M, rows))[n:, :] = pad_slots
            blob.view_i64('df_total_lens')[n:].fill_(rows)
        self._spec_graph_active = bucket

        # ---- 装载 append 暂存 → H2D → 单图回放 → 读回 ----
        # 写基址 host 侧同步重算（图内不跑 python，_df_verify_graph 的 bases 只在
        # 捕获期存在；读回更新 GDN carry 需要与本桶一致的 host 值——桶静态布局，
        # 公式即真值）。
        self._df_bases = [self.config.max_num_seqs + i * rows for i in range(n)]
        self._df_load_ctx_aux(seqs, bucket)
        self._push_spec_meta()
        self.spec_graphs[bucket].replay()
        self.spec_graph_stats["replay"] += 1
        bucket_hits = self.spec_graph_stats["bucket_hits"]
        bucket_hits[bucket] = bucket_hits.get(bucket, 0) + 1
        return self._df_readback_commit(seqs)

    # ---- spec 解码 CUDA Graph ----
    # 每桶一张图覆盖整轮：draft 链式起草 → target verify → 图内接受/提交/种子
    # → refill（用 verify 的 target aux 回填 draft KV）。每轮仅一次 D2H。

    def _alloc_spec_graph_vars(self, max_bs: int, rows: int, k: int) -> dict:
        # 形状取 text_config（qwen3_5 外层壳无顶层 vocab_size/hidden_size；qwen3
        # 路径 text_config 即 hf_config，逐字节同值，行为零变化）。
        tc = self.config.text_config
        vocab = tc.vocab_size
        draft_vocab = self.draft_config.draft_vocab_size
        # bt 宽度含 +1 余量，与 decode 图捕获的 block_tables 一致（_pin_W）。
        max_num_blocks = self._pin_W
        # 槽位 dtype 由后端决定（CUDA int32 / CoreX int64）；ctx/cu_seqlens/
        # block_table 两后端均为 int32；meta blob 统一存 int64，图内 copy_ 自动转型。
        slot_dtype = self.backend.graph_slot_mapping_dtype()
        pad_block = self._graph_pad_block()
        pad_slot = self._graph_pad_slot()
        return dict(
            # draft 链静态 buffer；carry_seed 为上一轮 refill 采样行（row k）的
            # midlayer 输出（TTT 链种子），drafted_ids[0] 为其采样的首个提案。
            carry_seed=torch.zeros(max_bs, tc.hidden_size, dtype=self.dtype),
            chain_tokens=torch.zeros(max_bs, dtype=torch.int64),
            drafted_ids=torch.zeros(k, max_bs, dtype=torch.int64),
            chain_pos=torch.zeros(k, max_bs, dtype=torch.int64),
            chain_slots=torch.full((k, max_bs), pad_slot, dtype=slot_dtype),
            chain_ctxlen=torch.full((k, max_bs), self._graph_pad_ctx(), dtype=torch.int32),
            chain_btables=torch.full((max_bs, max_num_blocks), pad_block, dtype=torch.int32),
            chain_temps=torch.ones(max_bs, dtype=torch.float32),
            chain_topps=torch.ones(max_bs, dtype=torch.float32),
            chain_topks=torch.full((max_bs,), draft_vocab, dtype=torch.int64),
            # verify 静态 buffer
            v_input_ids=torch.zeros(max_bs * rows, dtype=torch.int64),
            v_positions=torch.zeros(max_bs * rows, dtype=torch.int64),
            # cu_seqlens_q 恒定：每序列恰好 k+1 行，capture 后无需更新
            v_cu_q=(torch.arange(max_bs + 1, dtype=torch.int32) * rows),
            v_cu_k=torch.zeros(max_bs + 1, dtype=torch.int32),
            v_slots=torch.full((max_bs * rows,), pad_slot, dtype=slot_dtype),
            v_btables=torch.full((max_bs, max_num_blocks), pad_block, dtype=torch.int32),
            # 310P verify/refill：展开后的 per-query PA 入参（与 decode 同构）
            v_ctx_lens=torch.full((max_bs * rows,), self._graph_pad_ctx(), dtype=torch.int32),
            v_btables_exp=torch.full((max_bs * rows, max_num_blocks), pad_block, dtype=torch.int32),
            v_temps=torch.ones(max_bs * rows, dtype=torch.float32),
            v_topps=torch.ones(max_bs * rows, dtype=torch.float32),
            v_topks=torch.full((max_bs * rows,), vocab, dtype=torch.int64),
            # aux 宽度按捕获层数参数化（原硬编码 *3 = EAGLE3 的 3 层；dflash 为
            # 5 层）。eagle3 下 len==3，宽度与旧硬编码逐位一致（行为零变化）。
            v_aux=torch.zeros(max_bs * rows, tc.hidden_size * len(self.config.spec_aux_layers), dtype=self.dtype),
            v_token_ids=torch.zeros(max_bs * rows, dtype=torch.int64),
            # 接受/提交输出：前 rows 列为 token，rows 为 cnt，rows+1 为
            # 下一轮 draft1。一次 D2H 同时读回提交结果与 carry token。
            commit_buf=torch.full((max_bs, rows + 2), -1, dtype=torch.int64),
            # 310P graph-safe static bufs (avoid AICPU int Add/arange in accept/refill)
            v_ar_f=torch.arange(rows, dtype=torch.float32),
            v_ar_clamp=torch.arange(rows, dtype=torch.int64).clamp(max=k),
            d_pad=torch.zeros(max_bs, rows, dtype=torch.int64),
            neg1_tok=torch.full((max_bs, rows), -1, dtype=torch.int64),
            accept_m=torch.zeros(max_bs, dtype=torch.int64),
            v_shift_idx=torch.cat([
                torch.arange(1, k + 1, dtype=torch.int64),
                torch.tensor([k], dtype=torch.int64)]),
        )

    def _sg_sample(self, logits, temps, topps, topks, use_cuda_graph: bool):
        if use_cuda_graph and not self.graph_sample:
            return logits.argmax(dim=-1)
        return self.sampler.forward_native(logits, temps, topps, topks)

    def _sg_scatter(self, bs: int):
        """图内把 meta blob 分发到各静态 buffer（int64→图 buffer dtype 转换在 copy 内）。"""
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        blob = self._sblob
        M, W = self._pin_M, self._pin_W
        g['chain_tokens'][:bs].copy_(blob.device_view_i64('chain_tokens')[:bs])
        cp = blob.device_view_i64('chain_pos', (k, M))
        cc = blob.device_view_i64('chain_ctxlen', (k, M))
        cs = blob.device_view_i64('chain_slots', (k, M))
        for i in range(k):
            g['chain_pos'][i][:bs].copy_(cp[i, :bs])
            g['chain_ctxlen'][i][:bs].copy_(cc[i, :bs])
            g['chain_slots'][i][:bs].copy_(cs[i, :bs])
        g['v_positions'][:bs * rows].copy_(blob.device_view_i64('v_pos')[:bs * rows])
        g['v_slots'][:bs * rows].copy_(blob.device_view_i64('v_slots')[:bs * rows])
        g['v_cu_k'][:bs + 1].copy_(blob.device_view_i64('v_cu_k')[:bs + 1])
        g['chain_topks'][:bs].copy_(blob.device_view_i64('ctopks')[:bs])
        g['v_topks'][:bs * rows].copy_(blob.device_view_i64('v_topks')[:bs * rows])
        bt = blob.device_view_i64('btables', (M, W))
        g['chain_btables'][:bs].copy_(bt[:bs])
        g['v_btables'][:bs].copy_(bt[:bs])
        g['v_ctx_lens'][:bs * rows].copy_(
            blob.device_view_i64('v_ctx_lens')[:bs * rows])
        g['v_btables_exp'][:bs * rows].copy_(
            blob.device_view_i64('v_btables_exp', (M * rows, W))[:bs * rows])
        g['chain_temps'][:bs].copy_(blob.device_view_f32('ctemps')[:bs])
        g['chain_topps'][:bs].copy_(blob.device_view_f32('ctopps')[:bs])
        g['v_temps'][:bs * rows].copy_(blob.device_view_f32('v_temps')[:bs * rows])
        g['v_topps'][:bs * rows].copy_(blob.device_view_f32('v_topps')[:bs * rows])

    def _sg_draft_chain(self, bs: int, use_cuda_graph: bool):
        """draft 链步 1..k-1：从上轮 refill 采样行（carry_seed）流式续写并逐步采样提案。"""
        g = self._sg
        k = self.config.spec_num_tokens
        # 链从上一轮 refill 采样行（carry_seed）续写：TTT 流式种子 + 上轮
        # 首提案；写在流位置（host 按 l-m+k+i-2 填充），读窗仍是提交前缀。
        set_context(False,
                    slot_mapping=g['chain_slots'][1][:bs],
                    context_lens=g['chain_ctxlen'][1][:bs],
                    block_tables=g['chain_btables'][:bs],
                    max_context_len=self.config.max_model_len,
                    use_cuda_graph=use_cuda_graph,
                    block_size=self.block_size)
        seed = g['carry_seed'][:bs]
        for i in range(1, k):
            if i > 1:
                set_context(False,
                            slot_mapping=g['chain_slots'][i][:bs],
                            context_lens=g['chain_ctxlen'][i][:bs],
                            block_tables=g['chain_btables'][:bs],
                            max_context_len=self.config.max_model_len,
                            use_cuda_graph=use_cuda_graph,
                            block_size=self.block_size)
            seed, logits = self.draft_model.forward_decode(
                g['chain_pos'][i][:bs], seed, g['drafted_ids'][i - 1][:bs])
            ids = self._sg_sample(
                logits, g['chain_temps'][:bs], g['chain_topps'][:bs], g['chain_topks'][:bs],
                use_cuda_graph)
            g['drafted_ids'][i][:bs] = self.draft_model.map_to_target_ids(ids)

    def _sg_verify(self, bs: int, use_cuda_graph: bool = True):
        """target verify：[chain_tokens, d1..dk] 一次前向，行 j 预测位置 L+j，图内采样并暂存 aux。"""
        g = self._sg
        rows = self.config.spec_num_tokens + 1
        # max_seqlen_k 用 max_model_len 上界（varlen kernel 只用于 grid sizing，
        # 正确性由 cu_seqlens 决定），避免 capture 烘焙随轮次变化的 python int。
        set_context(True,
                    cu_seqlens_q=g['v_cu_q'][:bs + 1],
                    cu_seqlens_k=g['v_cu_k'][:bs + 1],
                    max_seqlen_q=rows,
                    max_seqlen_k=self.config.max_model_len,
                    slot_mapping=g['v_slots'][:bs * rows],
                    context_lens=g['v_ctx_lens'][:bs * rows],
                    block_tables=g['v_btables_exp'][:bs * rows],
                    block_size=self.block_size)
        hidden, aux = self.model(g['v_input_ids'][:bs * rows], g['v_positions'][:bs * rows])
        logits = F.linear(hidden, self.model.lm_head.weight)
        g['v_token_ids'][:bs * rows] = self._sg_sample(
            logits, g['v_temps'][:bs * rows], g['v_topps'][:bs * rows], g['v_topks'][:bs * rows],
            use_cuda_graph)
        g['v_aux'][:bs * rows].copy_(aux)

    def _sg_acceptance(self, bs: int):
        """链式接受 + 提交组装（全部图内静态算子）。

        310P：避免 int Add / 动态 arange/full/cat（进 AICPU → NPUGraph 507011）。
        前缀长 m 用 float cumprod+sum；cnt=m+1 用 float Add；
        arange/clamp/neg1/d_pad 用 _alloc 静态 buffer。
        """
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        t = g['v_token_ids'][:bs * rows].view(bs, rows)
        g['d_pad'][:bs, :k].copy_(g['drafted_ids'][:k, :bs].transpose(0, 1))
        g['d_pad'][:bs, k].zero_()
        d = g['d_pad'][:bs, :k]
        match = t[:, :k] == d
        m_f = match.to(torch.float32).cumprod(dim=-1).sum(dim=-1)
        m = m_f.to(torch.int64)
        g['accept_m'][:bs] = m
        t_m = t.gather(1, m[:, None])
        ar_f = g['v_ar_f'][:rows]
        gathered = g['d_pad'][:bs].gather(1, g['v_ar_clamp'][:rows].expand(bs, rows))
        out = torch.where(ar_f.expand(bs, rows) < m_f[:, None], gathered, t_m.expand(bs, rows))
        g['commit_buf'][:bs, :rows] = torch.where(
            ar_f.expand(bs, rows) <= m_f[:, None], out, g['neg1_tok'][:bs])
        g['commit_buf'][:bs, rows] = (m_f + 1.0).to(torch.int64)

    def _sg_refill(self, bs: int, use_cuda_graph: bool = True):
        """虚构窗口重放（对齐 vLLM padded 分支）+ 下一轮 draft1 采样。

        行 j（= verify 行 j，位置 L-1+j）的 x = fc(v_aux[j])（look-ahead aux，
        目标消费过本行提案 token 后的隐层），embed 取 verify token[j+1]
        （vLLM 的 shift 口径：rope p 的行 embed tok@p+1，输出预测 tok@p+2），
        行 m 被覆盖为 bonus 并作为采样行（eagle_prepare_inputs_padded_kernel:
        index_to_sample = 段尾 - 被拒数）。KV 写入 verify 槽位，构成下一轮
        链步连续读取的 draft 流窗口；采样行隐层/提案经 host carry 交接。
        """
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        seed_r = self.draft_model.project_aux(g['v_aux'][:bs * rows])
        t = g['v_token_ids'][:bs * rows].view(bs, rows)
        m_idx = g['accept_m'][:bs]
        ids_r = _refill_ids(t, m_idx, rows, k, ar_f=g['v_ar_f'][:rows], shift_idx=g['v_shift_idx'][:rows])
        set_context(True,
                    cu_seqlens_q=g['v_cu_q'][:bs + 1],
                    cu_seqlens_k=g['v_cu_k'][:bs + 1],
                    max_seqlen_q=rows,
                    max_seqlen_k=self.config.max_model_len,
                    slot_mapping=g['v_slots'][:bs * rows],
                    context_lens=g['v_ctx_lens'][:bs * rows],
                    block_tables=g['v_btables_exp'][:bs * rows],
                    block_size=self.block_size)
        h = self.draft_model.forward_prefill(
            g['v_positions'][:bs * rows], seed_r, ids_r)
        h_m = h.view(bs, rows, -1).gather(
            1, m_idx[:, None, None].expand(bs, 1, h.size(-1))).squeeze(1)
        logits = F.linear(self.draft_model.norm(h_m), self.draft_model.lm_head.weight)
        ids = self._sg_sample(
            logits, g['chain_temps'][:bs], g['chain_topps'][:bs], g['chain_topks'][:bs],
            use_cuda_graph)
        g['drafted_ids'][0][:bs] = self.draft_model.map_to_target_ids(ids)
        g['carry_seed'][:bs] = h_m
        # 与提交 token/count 一起读回，避免热路径对 drafted_ids 再做一次 .item()。
        g['commit_buf'][:bs, rows + 1] = g['drafted_ids'][0][:bs]

    def _sg_round(self, bs: int, use_cuda_graph: bool):
        """spec 单图整轮：scatter → draft 链 → verify 输入组装 → verify → 图内接受 → refill。"""
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        self._sg_scatter(bs)
        self._sg_draft_chain(bs, use_cuda_graph)
        v = g['v_input_ids'][:bs * rows].view(bs, rows)
        v[:, 0] = g['chain_tokens'][:bs]
        v[:, 1:].copy_(g['drafted_ids'][:k, :bs].t())
        self._sg_verify(bs, use_cuda_graph)
        self._sg_acceptance(bs)
        self._sg_refill(bs, use_cuda_graph)

    @torch.inference_mode()
    def capture_spec_cudagraph(self):
        max_bs = self._graph_max_bs
        self.spec_graph_bs = _graph_buckets(max_bs)
        g = self._sg    # 缓冲已在 __init__ 分配（spec 启用即存在，eager 轮共用）
        self.spec_graphs = {}

        try:
            k = self.config.spec_num_tokens
            rows = k + 1
            M, W = self._pin_M, self._pin_W
            blob = self._sblob
            blob.view_i64('chain_ctxlen', (k, M)).fill_(self._graph_pad_ctx())
            blob.view_i64('chain_slots', (k, M)).fill_(self._graph_pad_slot())
            blob.view_i64('btables', (M, W)).fill_(self._graph_pad_block())
            blob.view_i64('v_slots', (M, rows)).fill_(self._graph_pad_slot())
            blob.view_i64('v_ctx_lens').fill_(1)
            blob.view_i64('v_btables_exp', (M, rows, W)).fill_(self._graph_pad_block())
            blob.h2d()
            for bs in reversed(self.spec_graph_bs):
                # warmup：kernel lazy init 必须在 capture 前完成（310P 默认 3 次）。
                g['v_cu_k'][:bs + 1] = g['v_cu_q'][:bs + 1]
                for _ in range(self.backend.graph_warmup_iters()):
                    self._sg_round(bs, use_cuda_graph=True)
                    reset_context()
                    self.backend.synchronize()
                graph = self.backend.new_graph()
                with self.backend.graph_ctx(graph, pool=self.spec_graph_pool):
                    self._sg_round(bs, use_cuda_graph=True)
                if self.spec_graph_pool is None:
                    pool_fn = getattr(graph, 'pool', None)
                    self.spec_graph_pool = pool_fn() if callable(pool_fn) else None
                self.spec_graphs[bs] = graph
                reset_context()
                # 冒烟回放：310P 上捕获期不报错、replay 才 AICPU/507011 的情况
                # 在此同步检出，整批降级 eager，避免热路径静默炸机。
                self._push_spec_meta()
                graph.replay()
                self.backend.synchronize()
                self.spec_graph_stats["capture_success"] += 1
        except Exception as e:
            # 后端不支持图捕获（如部分 vendor kernel 含 host 同步/动态 workspace）
            # 时优雅降级：spec_round 自动回落到同构 eager 轮，功能不受影响。
            print(
                "[nano-vllm] Speculative graph capture failed; falling back to eager "
                f"speculative rounds: {type(e).__name__}: {e}",
                flush=True,
            )
            self.spec_graph_stats["capture_failure"] += 1
            traceback.print_exc()
            try:
                reset_context()
                self.backend.synchronize()
            except Exception:
                pass
            self.spec_graphs = None
            self.spec_graph_pool = None
            try:
                self.backend.empty_cache()
            except Exception:
                pass

    @torch.inference_mode()
    def _spec_round_graph(self, seqs: list[Sequence], bucket: int) -> list[list[int]]:
        n = self._fill_spec_meta(seqs)

        # padding 复位（n < bucket 时 blob 段补 -1/0，scatter 覆盖图 buffer）。
        # 条件必须是 n < bucket（不再是 bucket < active）：后者漏「首次扩桶 /
        # 同桶内含 padding」两个窗口，padding 行会带陈旧槽位写脏真实序列的 KV
        # ——并发（n<桶）下 verify 行槽被清 -1 → 跳写 → NaN 零流 / 全 "！"。
        # DFlash 版同款修复见 _df_round_graph。轴序 vs[n:, :]（seq 维在轴 0，
        # 每序列占 rows 列）——旧 vs[:, n:] 清的是所有序列的列，属同类错位。
        if n < bucket:
            k = self.config.spec_num_tokens
            rows = k + 1
            M, W = self._pin_M, self._pin_W
            blob = self._sblob
            cp = blob.view_i64('chain_pos', (k, M))
            cc = blob.view_i64('chain_ctxlen', (k, M))
            cs = blob.view_i64('chain_slots', (k, M))
            vs = blob.view_i64('v_slots', (M, rows))
            cu_k = blob.view_i64('v_cu_k')
            bt = blob.view_i64('btables', (M, W))
            cp[:, n:].fill_(-1)
            cc[:, n:].fill_(self._graph_pad_ctx())
            cs[:, n:].fill_(self._graph_pad_slot())
            vs[n:, :].fill_(self._graph_pad_slot())
            # padding 段 K 窗零长（cu 单调不降、kernel 跳过空段，杜绝旧算式的非单调 cu/负窗长隐患）
            cu_k[n + 1:] = cu_k[n]
            bt[n:, :].fill_(self._graph_pad_block())
            v_ctx = blob.view_i64('v_ctx_lens', (M, rows))
            v_bte = blob.view_i64('v_btables_exp', (M, rows, W))
            v_ctx[n:, :].fill_(self._graph_pad_ctx())
            v_bte[n:, :, :].fill_(self._graph_pad_block())
        self._spec_graph_active = bucket

        # ---- H2D + 单图回放：scatter → draft 链 → verify → 图内接受 → refill ----
        self._load_spec_carry(seqs, n)
        self._push_spec_meta()
        self.spec_graphs[bucket].replay()
        self.spec_graph_stats["replay"] += 1
        bucket_hits = self.spec_graph_stats["bucket_hits"]
        bucket_hits[bucket] = bucket_hits.get(bucket, 0) + 1
        return self._readback_spec_commit(seqs)

    @torch.inference_mode()
    def capture_cudagraph(self):
        try:
            config = self.config
            # 形状取 text_config（qwen3_5 外层多模态壳无顶层 hidden_size/vocab_size，
            # 实测 None——原 hf_config 直取在 hybrid 下会 TypeError 进降级分支；
            # qwen3 的 text_config 即 hf_config，逐字节同值，行为零变化）。
            tc = config.text_config
            max_bs = self._graph_max_bs
            # + 1：runtime block_tables 宽度可能比 capture 时更宽
            # （nano-vLLM 公开 bug），static buffer 直接留出余量（_pin_W）。
            max_num_blocks = self._pin_W
            # 同 dtype 的输入合并为一块连续 buffer，prepare 阶段可单次 H2D 直写。
            io = torch.zeros(2 * max_bs, dtype=torch.int64)
            sc = torch.zeros(max_bs, dtype=self.backend.graph_slot_mapping_dtype())
            pad_block = self._graph_pad_block()
            pad_slot = self._graph_pad_slot()
            pad_ctx = self._graph_pad_ctx()
            sc.fill_(pad_slot)
            # context_lens 必须独立 int32：ixformer paged_attention 硬性检查
            # Int && contiguous && cuda，不能跟随 slot 的图 dtype（CoreX=int64）。
            # 旧两段式共 buffer 在 CUDA 上靠 slot 恰好同为 int32 掩盖。
            ctx = torch.zeros(max_bs, dtype=torch.int32)
            ctx.fill_(pad_ctx)
            input_ids, positions = io[:max_bs], io[max_bs:]
            slot_mapping, context_lens = sc, ctx
            block_tables = torch.full((max_bs, max_num_blocks), pad_block, dtype=torch.int32)
            outputs = torch.zeros(max_bs, tc.hidden_size, dtype=self.dtype)
            temperatures = torch.ones(max_bs, dtype=torch.float32)
            top_ps = torch.ones(max_bs, dtype=torch.float32)
            top_ks = torch.full((max_bs,), tc.vocab_size, dtype=torch.int64)
            token_ids = torch.zeros(max_bs, dtype=torch.int64)
            self.graph_bs = _graph_buckets(max_bs)
            self.graphs = {}
            self.graph_pool = None
            # decode 元数据 blob 的设备孪生：host 只做 2 次 H2D，分发在图内 scatter。
            ids_d = self._dblob.device_view_i64('ids')
            pos_d = self._dblob.device_view_i64('pos')
            slots_d = self._dblob.device_view_i64('slots')
            ctx_d = self._dblob.device_view_i64('ctx')
            bt_d = self._dblob.device_view_i64('bt', (max_bs, max_num_blocks))
            topks_d = self._dblob.device_view_i64('topks')
            temps_d = self._dblob.device_view_f32('temperatures')
            topps_d = self._dblob.device_view_f32('top_ps')
            self._dblob.view_i64('slots')[:max_bs] = pad_slot
            self._dblob.view_i64('ctx')[:max_bs] = pad_ctx
            self._dblob.view_i64('bt', (max_bs, max_num_blocks)).fill_(pad_block)
            self._dblob.h2d()

            def scatter(bs):
                """图内把 dmeta blob 分发到各静态 buffer（int64→int32 转换在 copy 内）。"""
                input_ids[:bs].copy_(ids_d[:bs])
                positions[:bs].copy_(pos_d[:bs])
                slot_mapping[:bs].copy_(slots_d[:bs])
                context_lens[:bs].copy_(ctx_d[:bs])
                block_tables[:bs].copy_(bt_d[:bs])
                top_ks[:bs].copy_(topks_d[:bs])
                temperatures[:bs].copy_(temps_d[:bs])
                top_ps[:bs].copy_(topps_d[:bs])

            for bs in reversed(self.graph_bs):
                graph = self.backend.new_graph()
                # Graph Context 必须静态：max_context_len 用 config.max_model_len，
                # 不要用 context_lens.max().item()（capture 后不可变 python 整数）。
                # hybrid：GDN 状态槽元数据走 device twin 固定地址（prepare_decode
                # 每步 H2D 刷新 twin 内容，replay 原样读取；capture 期 twin 为
                # pad 槽/False 的哑值，仅占位——replay 不运行 python，元数据不经
                # context 传递而是经 twin 内存）。qwen3 路径不注入（恒 None）。
                gdn_graph_extra = (
                    dict(gdn_state_slots=self._gdn_slots_dev[:bs],
                         gdn_has_initial=self._gdn_init_dev[:bs])
                    if self.is_hybrid else {})
                set_context(
                    False,
                    slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs],
                    max_context_len=config.max_model_len,
                    use_cuda_graph=True,
                    block_size=self.block_size,
                    **gdn_graph_extra,
                )
                # Warmup：IXFormer kernel lazy init / workspace / torch.compile
                # 必须在正式 capture 前完成。graph_sample 路径把 lm_head + sampler
                # 一并捕获（含 exponential_ 随机采样——PyTorch CUDA RNG 是
                # graph-safe 的，replay 每次产生新随机数）。
                scatter(bs)
                for _ in range(self.backend.graph_warmup_iters()):
                    hidden = self.model(input_ids[:bs], positions[:bs])    # warmup
                    if self.graph_sample:
                        token_ids[:bs] = self.sampler(
                            self.model.compute_logits(hidden), temperatures[:bs], top_ps[:bs], top_ks[:bs])
                    else:
                        outputs[:bs] = hidden
                    self.backend.synchronize()
                with self.backend.graph_ctx(graph, pool=self.graph_pool):
                    scatter(bs)
                    hidden = self.model(input_ids[:bs], positions[:bs])    # capture
                    if self.graph_sample:
                        token_ids[:bs] = self.sampler(
                            self.model.compute_logits(hidden), temperatures[:bs], top_ps[:bs], top_ks[:bs])
                    else:
                        outputs[:bs] = hidden
                if self.graph_pool is None:
                    pool_fn = getattr(graph, "pool", None)
                    self.graph_pool = pool_fn() if callable(pool_fn) else None
                self.graphs[bs] = graph
                self.backend.synchronize()
                reset_context()

            self.graph_vars = dict(
                block_tables=block_tables,
                outputs=outputs,
                top_ks=top_ks,
                token_ids=token_ids,
                input_ids=input_ids,
                positions=positions,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                temperatures=temperatures,
                top_ps=top_ps,
            )
        except Exception as e:
            # 图捕获失败（如 vendor kernel 含 host 同步/动态 workspace）时优雅
            # 降级：decode 全部走 eager 前向（run_model / prepare_* 的 graphs=None
            # 分支），与 spec 捕获的降级策略一致。
            print(
                "[nano-vllm] Decode graph capture failed; falling back to eager mode: "
                f"{type(e).__name__}: {e}",
                flush=True,
            )
            traceback.print_exc()
            reset_context()
            self.backend.synchronize()
            self.graphs = None
            self.graph_pool = None
            self.backend.empty_cache()
