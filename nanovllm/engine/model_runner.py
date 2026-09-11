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
        if self.spec_enabled and self.is_hybrid:
            # 阶段1 守卫：Qwen3.5 目标的投机解码（DFlash）在阶段2 提供，干净拒绝
            # （先于 draft 构建与 EAGLE3 检查；hybrid 的 spec_method 分发也是阶段2）。
            raise NotImplementedError(
                'Qwen3.5 目标的投机解码（DFlash）将在阶段2 提供，当前请去掉 spec_model 配置')
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
            from transformers import AutoConfig
            self.draft_config = AutoConfig.from_pretrained(config.spec_model)
            self.draft_model = Eagle3DraftModel(self.draft_config, self.model.model.embed_tokens)
            load_eagle3_draft(self.draft_model, config.spec_model)
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
            # 逐一对应；每轮 2 次 H2D，零 cudaHostAlloc）。
            self._sblob = MetaBlob(
                [('chain_tokens', M), ('chain_pos', k * M), ('chain_ctxlen', k * M),
                 ('chain_slots', k * M), ('v_pos', rows * M), ('v_slots', rows * M),
                 ('v_cu_k', M + 1), ('ctopks', M), ('v_topks', rows * M), ('btables', M * W)],
                [('ctemps', M), ('ctopps', M), ('v_temps', rows * M), ('v_topps', rows * M)])
            self._sg = self._alloc_spec_graph_vars(self._graph_max_bs, rows, k)
        self.spec_graphs = None
        self.spec_graph_pool = None
        self._spec_graph_active = 0
        if not self.enforce_eager:
            if not self.spec_enabled:
                if self.is_hybrid:
                    # 阶段1：hybrid decode 图不做——GDN 状态槽读写的图内静态分发
                    # 待阶段3 图适配；decode 全走 eager 前向，enforce_eager 语义不变。
                    print('[nano-vllm] Qwen3.5 阶段1 暂不走 decode 图（GDN 状态的图内分发待阶段3 适配）', flush=True)
                else:
                    self.capture_cudagraph()
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
        if self.spec_enabled:
            # draft KV cache 与 target 共用同一 block_table/slot 布局，预算一并扣除。
            # 布局一致性由 backend.get_kv_cache_shape 保证（cuda/ascend=BSHD、corex=HND，
            # 两模型同后端必同布局）；draft 槽位写入与 target 走同一 _slot_of 映射。
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
        if self.spec_enabled:
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

    def drop_spec_state(self, seq_id: int):
        self.spec_hidden.pop(seq_id, None)
        self.spec_carry.pop(seq_id, None)

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
        # 形状键名双兼容：模型侧实际返回 conv_shape/ssm_shape（阶段1b 契约文本
        # 写作 conv/ssm），以模型落地为准、两键名都认，防任一侧改动即炸。
        conv_shape = spec.get("conv_shape") or spec["conv"]
        ssm_shape = spec.get("ssm_shape") or spec["ssm"]
        # conv 状态用模型 dtype、ssm 状态 fp32——形状与 dtype 均以 get_gdn_state_spec()
        # 声明为准（勿在引擎侧硬编码，模型侧按 text_config 计算口径唯一）。
        self.conv_pool = torch.empty((n_gdn, slots) + tuple(conv_shape), dtype=spec["conv_dtype"])
        self.ssm_pool = torch.empty((n_gdn, slots) + tuple(ssm_shape), dtype=spec["ssm_dtype"])
        self._gdn_state_bytes = (self.conv_pool.numel() * self.conv_pool.element_size()
                                 + self.ssm_pool.numel() * self.ssm_pool.element_size())
        # GDN 模块识别：优先用模型侧声明的 gdn_layers（按 GDN 层序的权威模块
        # 引用，模型侧契约字段）；未暴露时退回与 k_cache/v_cache 同款鸭子类型
        # （契约：GDN 模块自带 conv_cache/ssm_cache 占位属性等待注入）。
        gdn_mods = getattr(self.model.model, "gdn_layers", None)
        if gdn_mods is None:
            gdn_mods = [m for m in self.model.modules()
                        if hasattr(m, "conv_cache") and hasattr(m, "ssm_cache")]
        assert len(gdn_mods) == n_gdn, (
            f'GDN 模块识别数({len(gdn_mods)})与层数({n_gdn})不符：模型侧需以 '
            f'gdn_layers 或 conv_cache/ssm_cache 占位属性声明 GDN 模块（引擎按此注入状态池切片）')
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
        # 直接消费 device twin 固定地址 + 图内 scatter）。长度取 max_num_seqs
        # （本 buffer 阶段1 不进图桶，无需对齐 _graph_max_bs；阶段3 再统一）。
        N = self.config.max_num_seqs
        self._gdn_slots_pin = torch.zeros(N, dtype=torch.int32, device='cpu', pin_memory=True)
        self._gdn_init_pin = torch.zeros(N, dtype=torch.bool, device='cpu', pin_memory=True)
        self._gdn_slots_dev = torch.zeros(N, dtype=torch.int32)
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
        # 相关写入须在 inference mode 内进行。
        # （防御性排除 hybrid：阶段1 不捕获图、graphs 恒 None，本分支不可达；
        # 显式排除防未来误开——图内尚未组装 GDN 元数据段。）
        if self.graphs is not None and not self.is_hybrid:
            gv = self.graph_vars
            bucket = _bucket(self.graph_bs, n)
            if bucket is not None:
                with torch.inference_mode():
                    if bucket < self._graph_bs_active:
                        slots_v[n:bucket].fill_(-1)
                        ctx_v[n:bucket].zero_()
                        bt[n:, :].fill_(-1)
                    self._graph_bs_active = bucket
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
        """spec 模式 prefill 步：target 前向（返回 aux 隐层）+ draft 同 chunk 前向。

        draft prefill 复用同一 context（token 布局/槽位完全一致），为 prompt 的
        每个位置写入 draft KV。已完成 prefill 的序列记录 h_seed（aux 最后一行，
        即预测首个采样 token 的隐层）。
        """
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
        Graph，否则 eager（冷启动轮在 eager 内补齐 carry）。"""
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
        vocab = self.config.hf_config.vocab_size
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

    # ---- spec 解码 CUDA Graph ----
    # 每桶一张图覆盖整轮：draft 链式起草 → target verify → 图内接受/提交/种子
    # → refill（用 verify 的 target aux 回填 draft KV）。每轮仅一次 D2H。

    def _alloc_spec_graph_vars(self, max_bs: int, rows: int, k: int) -> dict:
        hf_config = self.config.hf_config
        vocab = hf_config.vocab_size
        draft_vocab = self.draft_config.draft_vocab_size
        # bt 宽度含 +1 余量，与 decode 图捕获的 block_tables 一致（_pin_W）。
        max_num_blocks = self._pin_W
        # 槽位 dtype 由后端决定（CUDA int32 / CoreX int64）；ctx/cu_seqlens/
        # block_table 两后端均为 int32；meta blob 统一存 int64，图内 copy_ 自动转型。
        slot_dtype = self.backend.graph_slot_mapping_dtype()
        return dict(
            # draft 链静态 buffer；carry_seed 为上一轮 refill 采样行（row k）的
            # midlayer 输出（TTT 链种子），drafted_ids[0] 为其采样的首个提案。
            carry_seed=torch.zeros(max_bs, hf_config.hidden_size, dtype=self.dtype),
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
            v_aux=torch.zeros(max_bs * rows, hf_config.hidden_size * 3, dtype=self.dtype),
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

        # padding 复位（桶缩小时 blob 段补 -1/0，scatter 会覆盖图 buffer）
        if bucket < self._spec_graph_active:
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
            vs[:, n:].fill_(-1)
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
            hf_config = config.hf_config
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
            outputs = torch.zeros(max_bs, hf_config.hidden_size, dtype=self.dtype)
            temperatures = torch.ones(max_bs, dtype=torch.float32)
            top_ps = torch.ones(max_bs, dtype=torch.float32)
            top_ks = torch.full((max_bs,), hf_config.vocab_size, dtype=torch.int64)
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
                set_context(
                    False,
                    slot_mapping=slot_mapping[:bs],
                    context_lens=context_lens[:bs],
                    block_tables=block_tables[:bs],
                    max_context_len=config.max_model_len,
                    use_cuda_graph=True,
                    block_size=self.block_size,
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
