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
        print(f'[nano-vllm] max_num_seqs={max_bs} 未被标准桶模式覆盖，已追加 {max_bs} 桶'
              f'（否则该并发将静默回退 eager）', flush=True)
    return buckets


def _refill_ids(t2d, m_idx, rows: int, k: int):
    """refill shifted-ids 口径：verify token 右移一位，行 m 覆盖为 bonus。

    对齐 vLLM eagle_prepare_inputs_padded_kernel 的 shift 口径。
    """
    bonus = t2d.gather(1, m_idx[:, None])
    ar = torch.arange(rows, device=t2d.device)
    t_ext = torch.cat([t2d[:, 1:], t2d[:, k:k + 1]], dim=1)
    return torch.where(ar[None, :] == m_idx[:, None], bonus, t_ext).reshape(-1)


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
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.backend = get_backend()
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
                'Qwen3.5（hybrid）目标仅支持 DFlash 投机解码（spec_method=dflash）：'
                'EAGLE3 draft 的链式起草与 3 层 aux 口径不适用于 GDN 混合架构，'
                '请改用含 dflash_config 的 DFlash draft checkpoint')
        if self.spec_enabled and not self.backend.supports_spec:
            # spec 全路径不可用的后端在此干净拒绝（能力位裁决，不做 name 字符串
            # 判断）：310P 上 verify 恒传 block_tables → attention 层走整池 4D
            # 分支，TND 算子不消费——否则首个 spec 轮即硬崩，违背「只降速不崩溃」。
            raise RuntimeError(
                f'{self.backend.name} 后端暂不支持 spec：verify 的 paged KV 路径 '
                f'TND 算子不消费（需 paged-TND 适配后再开，勿当回归报）——'
                f'请去掉 spec_model 配置')
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
            print(f'[nano-vllm] 后端 {self.backend.name} 强制 dtype={self.backend.default_dtype}'
                  f'（hf_config 为 {config.text_config.dtype}），权重将在加载时隐式转换', flush=True)
        torch.set_default_dtype(self.dtype)
        torch.set_default_device(self.backend.device_name)
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
                '模型未声明 is_hybrid（qwen3_5 契约：model.model.is_hybrid=True）'
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
        if config.compile:
            # torch.compile：CoreX 上 qkv/rope/attention 区域的 Inductor 融合会产出
            # 错误结果，Qwen3Attention.forward 已 @torch.compiler.disable 保持 eager，
            # 此处只编译 embed + RMSNorm + MLP（正确性已验证 PASS）。
            self.model = torch.compile(self.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
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
            df_segments = ([('df_l1', M), ('df_c', M), ('df_ctxslot', M), ('df_carry', M)]
                           if self.config.spec_method == "dflash" else [])
            self._sblob = MetaBlob(
                [('chain_tokens', M), ('chain_pos', k * M), ('chain_ctxlen', k * M),
                 ('chain_slots', k * M), ('v_pos', rows * M), ('v_slots', rows * M),
                 ('v_cu_k', M + 1), ('ctopks', M), ('v_topks', rows * M), ('btables', M * W)]
                + df_segments,
                [('ctemps', M), ('ctopps', M), ('v_temps', rows * M), ('v_topps', rows * M)])
            self._sg = self._alloc_spec_graph_vars(self._graph_max_bs, rows, k)
            if self.config.spec_method == "dflash":
                # dflash 专段静态缓冲（阶段3 图轮；eagle3 不分配，字节布局零变化）。
                self._sg.update(self._alloc_df_graph_vars(self._graph_max_bs, rows))
        self.spec_graphs = None
        self.spec_graph_pool = None
        self._spec_graph_active = 0
        if not self.enforce_eager:
            if not self.spec_enabled:
                # hybrid（Qwen3.5）与 qwen3 同构捕获 decode 图（阶段3 解除阶段1
                # 门控）：GDN 单步（_forward_decode）本身无 host 同步/动态 shape，
                # 状态槽经 context 的 device twin 静态段消费（见 _alloc_gdn_state_pool
                # 与 prepare_decode 的图分支）；捕获失败仍走 try/except 优雅降级。
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
                                 else self.backend.graph_sample_default)):
                        self.capture_df_spec_cudagraph()
                else:
                    # spec 解码路径图捕获（draft 链图 + verify 图，仅 TP=1）。spec 图
                    # 无条件录 Gumbel 采样（forward_native），默认门同时要求
                    # graph_sample_default（RNG 图安全性未验证的后端不默认尝试）；
                    # NANOVLLM_SPEC_GRAPH 显式设置优先后端默认，可实验性覆盖。
                    _env_spec = os.environ.get("NANOVLLM_SPEC_GRAPH")
                    if (self.world_size == 1 and self.backend.supports_spec_graph
                            and (_env_spec != "0" if _env_spec is not None
                                 else self.backend.graph_sample_default)):
                        self.capture_spec_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

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
        if self.spec_enabled and self.config.spec_method == "eagle3":
            # draft KV cache 与 target 共用同一 block_table/slot 布局，预算一并扣除。
            # 布局一致性由 backend.get_kv_cache_shape 保证（cuda/ascend=BSHD、corex=HND，
            # 两模型同后端必同布局）；draft 槽位写入与 target 走同一 _slot_of 映射。
            # dflash 不进此分支：上下文 KV 走独立稠密缓冲（ctx_k/v_cache，已在
            # __init__ 分配），块查询行 KV 仅本轮 transient，无 paged draft cache。
            d = self.draft_config
            draft_kv_heads = d.num_key_value_heads // self.world_size
            draft_head_dim = getattr(d, "head_dim", d.hidden_size // d.num_attention_heads)
            block_bytes += 2 * d.num_hidden_layers * self.block_size * draft_kv_heads * draft_head_dim * self.dtype.itemsize
        # WSL 下上一进程退出后驱动回收显存存在数秒延迟，`used` 可能虚高；
        # 预算不足时短暂等待重测，而非直接断言崩溃。
        # 状态池（hybrid）在 warmup 前已分配，其字节数包含在下面的 used 里、
        # 随本预算循环自动计入——勿再显式扣除（已占用的显存扣两次即双重计费）。
        for _ in range(6):
            free, total = self.backend.mem_get_info()
            used = total - free
            peak = self.backend.memory_stats()["allocated_bytes.all.peak"]
            current = self.backend.memory_stats()["allocated_bytes.all.current"]
            config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
            if config.num_kvcache_blocks > 0:
                break
            time.sleep(5)
        else:
            raise RuntimeError(
                f"kvcache 显存预算不足：需 {block_bytes / 2**20:.1f} MiB/block"
                + (f"（GDN 状态池已占 {self._gdn_state_bytes / 2**20:.1f} MiB）" if self.is_hybrid else "")
                + f"，按 gpu_memory_utilization={config.gpu_memory_utilization} 计算可用块数为 0"
                f"（used={used / 2**20:.0f} MiB / peak={peak / 2**20:.0f} MiB / current={current / 2**20:.0f} MiB /"
                f" 预算 {total * config.gpu_memory_utilization / 2**20:.0f} MiB——"
                "可调低 max_model_len / block_size，或检查其他进程的显存占用")
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
                f'全注意力模块数({len(attn_mods)})与 KV 池层数({num_attn_layers})不符'
                f'（qwen3_5 契约：全注意力复用现有 Attention 类，k_cache/v_cache 占位）')
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
            f'GDN 模块识别数({len(gdn_mods)})与层数({n_gdn})不符：模型侧需以 '
            f'gdn_layers 声明 GDN 模块（引擎按此注入状态池切片）')
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
                    f'GDN 状态槽耗尽：并发序列数超过 max_num_seqs={self.config.max_num_seqs}'
                    '（scheduler 的 hybrid admission 门控应保证不发生）') from None
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

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        return block_tables

    def _slot_of(self, seq: Sequence, p: int) -> int:
        # 位置 p 的 KV 槽位（draft 与 target 共用同一 block_table/slot 布局）。
        return seq.block_table[p // self.block_size] * self.block_size + p % self.block_size

    def _fill_bt_row(self, buf, i: int, table: list[int]):
        # 填 block_table 第 i 行：有效段写 table，其余补 -1（host int64 视图）。
        buf[i, :len(table)] = torch.tensor(table, dtype=torch.int64)
        buf[i, len(table):] = -1

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
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        # 槽位 dtype 由后端决定（CUDA int32 / CoreX int64），cu_seqlens 两后端均为 int32。
        slot_mapping = torch.tensor(slot_mapping, dtype=self.backend.graph_slot_mapping_dtype(), pin_memory=True).to(self.device, non_blocking=True)
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
                        slots_v[n:bucket].fill_(-1)
                        ctx_v[n:bucket].zero_()
                        # ids/pos 一并复位（旧值虽被 slot=-1/ctx=0 隔离，仍属
                        # 残留脏数据；清零后 padding 行恒为合法 token 0/位置 0）。
                        ids_v[n:bucket].zero_()
                        pos_v[n:bucket].zero_()
                        bt[n:, :].fill_(-1)
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
        if self.config.spec_method == "dflash":
            if self.spec_graphs is not None and seqs:
                bucket = _bucket(self.spec_graph_bs, len(seqs))
                if bucket is not None:
                    return self._df_round_graph(seqs, bucket)
            return self._df_round_eager(seqs)
        if (self.spec_graphs is not None and seqs
                and not any(seq.spec_cold for seq in seqs)):
            bucket = _bucket(self.spec_graph_bs, len(seqs))
            if bucket is not None:
                return self._spec_round_graph(seqs, bucket)
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
        """统一读回（每轮唯一一次 D2H，cnt 打包在 commit_buf 最后一列）：提取提交
        token，并更新下一轮 carry（refill 行 k 隐层 + 其采样提案）、aux 行 m 与
        调度状态。"""
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
                                           g['drafted_ids'][0][i].item())
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
        lens = [len(seq) for seq in seqs]

        # ---- 采样行引导（冷/热一致，与旧 eager 实现逐位相同） ----
        draft_vocab = self.draft_config.draft_vocab_size
        temperatures = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, pin_memory=True).to(self.device, non_blocking=True)
        top_ps = torch.tensor([seq.top_p for seq in seqs], dtype=torch.float32, pin_memory=True).to(self.device, non_blocking=True)
        # draft 采样在 draft 词表上，top_k 上限相应取 draft 词表大小。
        top_ks_draft = torch.tensor(
            [seq.top_k if 0 < seq.top_k < draft_vocab else draft_vocab for seq in seqs],
            dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        h_seed = torch.cat([self.spec_hidden[seq.seq_id] for seq in seqs], dim=0)
        seed = self.draft_model.project_aux(h_seed)
        slot_dtype = self.backend.graph_slot_mapping_dtype()
        p = torch.tensor([l - 2 for l in lens], dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        ctx = torch.tensor([l - 1 for l in lens], dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
        slots = torch.tensor([[self._slot_of(seq, l - 2)] for seq, l in zip(seqs, lens)],
                             dtype=slot_dtype, pin_memory=True).to(self.device, non_blocking=True)
        tokens = torch.tensor([seq.last_token for seq in seqs], dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
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
                    f'draft 上下文槽耗尽：并发序列数超过 max_num_seqs={self.config.max_num_seqs}'
                    '（scheduler admission 门控应保证不发生）') from None
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
            slot = self._df_ctx_get_or_alloc(seq)
            p0, p1 = offsets[i], offsets[i + 1]
            # prefill aux 全行（0-based 层输出口径）→ 投影写缓冲（位置 0..P-1）
            self.draft_model.project_context_kv(
                aux[p0:p1], positions[p0:p1],
                torch.full((p1 - p0,), slot, dtype=torch.int64, device=self.device))
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
        last = torch.tensor([seq.last_token for seq in seqs], dtype=torch.int64,
                            pin_memory=True).to(self.device, non_blocking=True)
        # 上下文读窗：ctx_len = L-1（覆盖 0..L-2），右 padding 至批内最大（host int）
        ctx_lens = torch.tensor([L - 1 for L in Ls], dtype=torch.int64,
                                pin_memory=True).to(self.device, non_blocking=True)
        max_ctx = max(L - 1 for L in Ls)
        ctx_slots = torch.tensor([self.draft_ctx_slots[seq.seq_id] for seq in seqs],
                                 dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
        logits = self.draft_model.forward_block(last, positions, ctx_slots, ctx_lens,
                                                max_ctx, rows)
        vocab = self.config.text_config.vocab_size
        temps = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32,
                             pin_memory=True).to(self.device, non_blocking=True)
        topps = torch.tensor([seq.top_p for seq in seqs], dtype=torch.float32,
                             pin_memory=True).to(self.device, non_blocking=True)
        topks = torch.tensor([seq.top_k if 0 < seq.top_k < vocab else vocab for seq in seqs],
                             dtype=torch.int64, pin_memory=True).to(self.device, non_blocking=True)
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
            base_dev = torch.tensor(bases, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
            carry_dev = torch.tensor(carries, dtype=torch.int32, pin_memory=True).to(self.device, non_blocking=True)
            gdn_extra = dict(gdn_state_slots=base_dev, gdn_carry_slots=carry_dev, gdn_state_mode='row')
        # max_seqlen_k 用 max_model_len 上界（同 _sg_verify 的 varlen grid 口径）
        set_context(True,
                    cu_seqlens_q=g['v_cu_q'][:n + 1],
                    cu_seqlens_k=g['v_cu_k'][:n + 1],
                    max_seqlen_q=rows,
                    max_seqlen_k=self.config.max_model_len,
                    slot_mapping=g['v_slots'][:n * rows],
                    block_tables=g['v_btables'][:n],
                    block_size=self.block_size,
                    **gdn_extra)
        hidden, aux = self.model(g['v_input_ids'][:n * rows], g['v_positions'][:n * rows])
        logits = F.linear(hidden, self.model.lm_head.weight)
        g['v_token_ids'][:n * rows] = self.sampler.forward_native(
            logits, g['v_temps'][:n * rows], g['v_topps'][:n * rows], g['v_topks'][:n * rows])
        g['v_aux'][:n * rows].copy_(aux)
        self._df_bases = bases  # 供读回更新 carry（host int）

    def _df_readback_commit(self, seqs: list[Sequence]) -> list[list[int]]:
        """④ 统一读回（每轮唯一一次 D2H；commit_buf 契约与 EAGLE3 完全一致：
        committed = row[:row[rows]]，cnt 打包在最后一列）：
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
        for i, seq in enumerate(seqs):
            l1[i] = len(seq) - 1
            p = self.spec_ctx_pending.get(seq.seq_id)
            c[i] = 0 if p is None else p.size(0)
            cs[i] = self.draft_ctx_slots[seq.seq_id]
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
        g['df_carry'][:bs].copy_(self._sblob.device_view_i64('df_carry')[:bs])

    def _df_append_context_graph(self, bs: int):
        """① 上下文追加（图内静态化）：project_context_kv 全链（fc → hidden_norm
        → 融合 k/v GEMM → 逐层 k_norm → rope → 高级索引写稠密缓冲）无 host 同步、
        循环次数 = draft 层数常量，图安全。位置/槽位来自 blob 孪生（固定地址），
        padding 行 aux=0 → 写零特征（无害性论证见本节头注释 a）。"""
        g = self._sg
        blob = self._sblob
        rows = self.config.spec_num_tokens + 1
        aux = g['df_ctx_aux'][:bs * rows]
        l1 = blob.device_view_i64('df_l1')[:bs]
        c = blob.device_view_i64('df_c')[:bs]
        ar = torch.arange(rows, device=aux.device)
        # 追加位置 = [L-1-c, L-1-c+rows)（连续展平，与 eager 的逐 seq arange 拼接
        # 同值域；padding 行落在 [L-1, ...) 无读区）
        pos = ((l1 - c)[:, None] + ar[None, :]).reshape(-1)
        slots = blob.device_view_i64('df_ctxslot')[:bs].repeat_interleave(rows)
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
        ar = torch.arange(rows, device=l1.device)
        block_pos = l1[:, None] + ar[None, :]          # [bs, rows] = L-1..L-1+k
        logits = self.draft_model.forward_block(
            g['chain_tokens'][:bs], block_pos,
            blob.device_view_i64('df_ctxslot')[:bs], l1,
            self.config.max_model_len, rows)
        ids = self.sampler.forward_native(
            logits,
            g['chain_temps'][:bs].repeat_interleave(k),
            g['chain_topps'][:bs].repeat_interleave(k),
            g['chain_topks'][:bs].repeat_interleave(k))
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
                    block_tables=g['v_btables'][:bs],
                    block_size=self.block_size,
                    **gdn_extra)
        hidden, aux = self.model(g['v_input_ids'][:bs * rows], g['v_positions'][:bs * rows])
        logits = F.linear(hidden, self.model.lm_head.weight)
        g['v_token_ids'][:bs * rows] = self.sampler.forward_native(
            logits, g['v_temps'][:bs * rows], g['v_topps'][:bs * rows], g['v_topks'][:bs * rows])
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
            for bs in reversed(self.spec_graph_bs):
                # warmup：kernel lazy init 必须在 capture 前完成；warmup 数据用
                # 全零静态 buffer（token 0 / 槽 0 / 位置 0 合法，ctx 追加写零特征、
                # 块草稿/verify 产出垃圾输出但写入的是哑槽/被丢弃）。
                g['v_cu_k'][:bs + 1] = g['v_cu_q'][:bs + 1]
                self._df_round_body(bs)
                reset_context()
                self.backend.synchronize()
                graph = self.backend.new_graph()
                with self.backend.graph_ctx(graph, pool=self.spec_graph_pool):
                    self._df_round_body(bs)
                if self.spec_graph_pool is None:
                    self.spec_graph_pool = graph.pool()
                self.spec_graphs[bs] = graph
                reset_context()
                self.backend.synchronize()
        except Exception as e:
            # 与 EAGLE 同款优雅降级：spec_round 自动回落 _df_round_eager（阶段2
            # 已验收的 eager 全链路），功能不受影响。
            print(f'[nano-vllm] DFlash spec CUDA Graph 捕获失败，降级为 eager spec 轮：'
                  f'{type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
            reset_context()
            self.backend.synchronize()
            self.spec_graphs = None
            self.spec_graph_pool = None
            self.backend.empty_cache()

    @torch.inference_mode()
    def _df_round_graph(self, seqs: list[Sequence], bucket: int) -> list[list[int]]:
        """DFlash spec 图轮回放（_spec_round_graph 的 dflash 等价物）：fill meta
        → padding 复位 → 装载 append 暂存 → H2D → 单图 replay → 读回。"""
        n = self._fill_spec_meta(seqs)
        self._fill_df_meta(seqs)
        rows = self.config.spec_num_tokens + 1

        # padding 复位：n < bucket 即需（含首次使用/扩桶——blob 初始为 0，padding
        # 行 v_slots=0 会把 verify KV 写进 0 号槽、cu_k 尾段 0 会造成非单调 cu/
        # 负窗长；EAGLE 版 _spec_round_graph 的「仅桶收缩复位」条件存在同源隐患，
        # 本 dflash 版从条件上杜绝，EAGLE 路径红线未动）。EAGLE 版同款段（v_slots/
        # cu_k/btables）语义一致；chain_pos/ctxlen/slots dflash 不消费，不复位；
        # v_pos padding 行保持陈旧值（恒 < max_model_len 的合法位置，垃圾输出被
        # 丢弃——与 EAGLE 版不复位 v_pos 同口径）。df 专段复位：l1/c=0 → append
        # 位置从 0 起、写零特征；ctxslot → 专用 pad 槽（零写不碰真实序列上下文）；
        # carry=0 只读（垃圾进垃圾出，输出丢弃）。padding 行产出全丢弃且不脏任何
        # 运行序列状态。
        if n < bucket:
            M, W = self._pin_M, self._pin_W
            blob = self._sblob
            vs = blob.view_i64('v_slots', (M, rows))
            cu_k = blob.view_i64('v_cu_k')
            bt = blob.view_i64('btables', (M, W))
            # 注意轴序：v_slots 视图 (M, rows) 的 seq 维在轴 0——复位必须切
            # vs[n:, :]（padding 序列的全部行）。切错轴（vs[:, n:]）会把真实
            # 序列 verify 行 n..k 的槽位清成 -1 → store_kvcache 跳写 → 这些行
            # 读到陈旧/捕获期垃圾 KV → 行 3+（首个读到未写位置的行）输出 NaN
            # （实测 layer11 起 NaN、vt[3:]=0、并发序列全零流）。
            vs[n:, :].fill_(-1)
            # padding 段 K 窗零长（cu 单调不降、kernel 跳过空段）
            cu_k[n + 1:] = cu_k[n]
            bt[n:, :].fill_(-1)
            blob.view_i64('df_l1')[n:].zero_()
            blob.view_i64('df_c')[n:].zero_()
            # padding 序列 ctx 槽 → 专用 pad 槽（append 零写落 pad 槽，不碰任何
            # 真实序列的上下文；块草稿对 padding 行 gather pad 槽零特征，掩码屏蔽）
            blob.view_i64('df_ctxslot')[n:].fill_(self._df_pad_ctx_slot)
            blob.view_i64('df_carry')[n:].zero_()
        self._spec_graph_active = bucket

        # ---- 装载 append 暂存 → H2D → 单图回放 → 读回 ----
        # 写基址 host 侧同步重算（图内不跑 python，_df_verify_graph 的 bases 只在
        # 捕获期存在；读回更新 GDN carry 需要与本桶一致的 host 值——桶静态布局，
        # 公式即真值）。
        self._df_bases = [self.config.max_num_seqs + i * rows for i in range(n)]
        self._df_load_ctx_aux(seqs, bucket)
        self._push_spec_meta()
        self.spec_graphs[bucket].replay()
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
        return dict(
            # draft 链静态 buffer；carry_seed 为上一轮 refill 采样行（row k）的
            # midlayer 输出（TTT 链种子），drafted_ids[0] 为其采样的首个提案。
            carry_seed=torch.zeros(max_bs, tc.hidden_size, dtype=self.dtype),
            chain_tokens=torch.zeros(max_bs, dtype=torch.int64),
            drafted_ids=torch.zeros(k, max_bs, dtype=torch.int64),
            chain_pos=torch.zeros(k, max_bs, dtype=torch.int64),
            chain_slots=torch.full((k, max_bs), -1, dtype=slot_dtype),
            chain_ctxlen=torch.zeros(k, max_bs, dtype=torch.int32),
            chain_btables=torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32),
            chain_temps=torch.ones(max_bs, dtype=torch.float32),
            chain_topps=torch.ones(max_bs, dtype=torch.float32),
            chain_topks=torch.full((max_bs,), draft_vocab, dtype=torch.int64),
            # verify 静态 buffer
            v_input_ids=torch.zeros(max_bs * rows, dtype=torch.int64),
            v_positions=torch.zeros(max_bs * rows, dtype=torch.int64),
            # cu_seqlens_q 恒定：每序列恰好 k+1 行，capture 后无需更新
            v_cu_q=(torch.arange(max_bs + 1, dtype=torch.int32) * rows),
            v_cu_k=torch.zeros(max_bs + 1, dtype=torch.int32),
            v_slots=torch.full((max_bs * rows,), -1, dtype=slot_dtype),
            v_btables=torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32),
            v_temps=torch.ones(max_bs * rows, dtype=torch.float32),
            v_topps=torch.ones(max_bs * rows, dtype=torch.float32),
            v_topks=torch.full((max_bs * rows,), vocab, dtype=torch.int64),
            # aux 宽度按捕获层数参数化（原硬编码 *3 = EAGLE3 的 3 层；dflash 为
            # 5 层）。eagle3 下 len==3，宽度与旧硬编码逐位一致（行为零变化）。
            v_aux=torch.zeros(max_bs * rows, tc.hidden_size * len(self.config.spec_aux_layers), dtype=self.dtype),
            v_token_ids=torch.zeros(max_bs * rows, dtype=torch.int64),
            # 接受/提交输出（cnt 打包在最后一列，每轮仅一次 D2H）
            commit_buf=torch.full((max_bs, rows + 1), -1, dtype=torch.int64),
        )

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
            ids = self.sampler.forward_native(
                logits, g['chain_temps'][:bs], g['chain_topps'][:bs], g['chain_topks'][:bs])
            g['drafted_ids'][i][:bs] = self.draft_model.map_to_target_ids(ids)

    def _sg_verify(self, bs: int):
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
                    block_tables=g['v_btables'][:bs],
                    block_size=self.block_size)
        hidden, aux = self.model(g['v_input_ids'][:bs * rows], g['v_positions'][:bs * rows])
        logits = F.linear(hidden, self.model.lm_head.weight)
        g['v_token_ids'][:bs * rows] = self.sampler.forward_native(
            logits, g['v_temps'][:bs * rows], g['v_topps'][:bs * rows], g['v_topks'][:bs * rows])
        g['v_aux'][:bs * rows].copy_(aux)

    def _sg_acceptance(self, bs: int):
        """链式接受 + 提交组装 + 下一轮 draft 种子（全部图内静态算子）。"""
        g = self._sg
        k = self.config.spec_num_tokens
        rows = k + 1
        t = g['v_token_ids'][:bs * rows].view(bs, rows)
        d = g['drafted_ids'][:k, :bs].t()
        match = t[:, :k] == d
        m = torch.where(match.all(-1),
                        torch.full((bs,), k, dtype=torch.int64, device=t.device),
                        match.int().argmin(-1))
        t_m = t.gather(1, m[:, None])
        ar = torch.arange(rows, device=t.device)
        d_pad = torch.cat([d, d.new_zeros(bs, 1)], dim=1)
        out = torch.where(ar < m[:, None], d_pad.gather(1, ar.clamp(max=k).expand(bs, rows)), t_m)
        # [0..m] 为提交 token，[rows] 列存 cnt=m+1；padding 行保持 -1
        g['commit_buf'][:bs, :rows] = torch.where(ar <= m[:, None], out, torch.full_like(out, -1))
        g['commit_buf'][:bs, rows] = m + 1

    def _sg_refill(self, bs: int):
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
        m_idx = g['commit_buf'][:bs, rows] - 1
        ids_r = _refill_ids(t, m_idx, rows, k)
        set_context(True,
                    cu_seqlens_q=g['v_cu_q'][:bs + 1],
                    cu_seqlens_k=g['v_cu_k'][:bs + 1],
                    max_seqlen_q=rows,
                    max_seqlen_k=self.config.max_model_len,
                    slot_mapping=g['v_slots'][:bs * rows],
                    block_tables=g['v_btables'][:bs],
                    block_size=self.block_size)
        h = self.draft_model.forward_prefill(
            g['v_positions'][:bs * rows], seed_r, ids_r)
        h_m = h.view(bs, rows, -1).gather(
            1, m_idx[:, None, None].expand(bs, 1, h.size(-1))).squeeze(1)
        logits = F.linear(self.draft_model.norm(h_m), self.draft_model.lm_head.weight)
        ids = self.sampler.forward_native(
            logits, g['chain_temps'][:bs], g['chain_topps'][:bs], g['chain_topks'][:bs])
        g['drafted_ids'][0][:bs] = self.draft_model.map_to_target_ids(ids)
        g['carry_seed'][:bs] = h_m

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
        self._sg_verify(bs)
        self._sg_acceptance(bs)
        self._sg_refill(bs)

    @torch.inference_mode()
    def capture_spec_cudagraph(self):
        max_bs = self._graph_max_bs
        self.spec_graph_bs = _graph_buckets(max_bs)
        g = self._sg    # 缓冲已在 __init__ 分配（spec 启用即存在，eager 轮共用）
        self.spec_graphs = {}

        try:
            for bs in reversed(self.spec_graph_bs):
                # warmup：triton/flash kernel lazy init 必须在 capture 前完成；
                # warmup 数据用零/-1（slot=-1 跳过 KV 写，ctxlen=0 合法）。
                g['v_cu_k'][:bs + 1] = g['v_cu_q'][:bs + 1]
                self._sg_round(bs, use_cuda_graph=True)
                reset_context()
                self.backend.synchronize()
                graph = self.backend.new_graph()
                with self.backend.graph_ctx(graph, pool=self.spec_graph_pool):
                    self._sg_round(bs, use_cuda_graph=True)
                if self.spec_graph_pool is None:
                    self.spec_graph_pool = graph.pool()
                self.spec_graphs[bs] = graph
                reset_context()
                self.backend.synchronize()
        except Exception as e:
            # 后端不支持图捕获（如部分 vendor kernel 含 host 同步/动态 workspace）
            # 时优雅降级：spec_round 自动回落到同构 eager 轮，功能不受影响。
            print(f'[nano-vllm] spec CUDA Graph 捕获失败，降级为 eager spec 轮：'
                  f'{type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
            reset_context()
            self.backend.synchronize()
            self.spec_graphs = None
            self.spec_graph_pool = None
            self.backend.empty_cache()

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
            cc[:, n:].zero_()
            cs[:, n:].fill_(-1)
            vs[n:, :].fill_(-1)
            # padding 段 K 窗零长（cu 单调不降、kernel 跳过空段，杜绝旧算式的非单调 cu/负窗长隐患）
            cu_k[n + 1:] = cu_k[n]
            bt[n:, :].fill_(-1)
        self._spec_graph_active = bucket

        # ---- H2D + 单图回放：scatter → draft 链 → verify → 图内接受 → refill ----
        self._load_spec_carry(seqs, n)
        self._push_spec_meta()
        self.spec_graphs[bucket].replay()
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
            sc.fill_(-1)
            # context_lens 必须独立 int32：ixformer paged_attention 硬性检查
            # Int && contiguous && cuda，不能跟随 slot 的图 dtype（CoreX=int64）。
            # 旧两段式共 buffer 在 CUDA 上靠 slot 恰好同为 int32 掩盖。
            ctx = torch.zeros(max_bs, dtype=torch.int32)
            input_ids, positions = io[:max_bs], io[max_bs:]
            slot_mapping, context_lens = sc, ctx
            block_tables = torch.full((max_bs, max_num_blocks), -1, dtype=torch.int32)
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
                    self.graph_pool = graph.pool()
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
            print(f'[nano-vllm] decode CUDA Graph 捕获失败，降级为 eager：'
                  f'{type(e).__name__}: {e}', flush=True)
            traceback.print_exc()
            reset_context()
            self.backend.synchronize()
            self.graphs = None
            self.graph_pool = None
            self.backend.empty_cache()
