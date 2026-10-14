"""Qwen3.5 dense/MoE 文本模型：Gated DeltaNet + 门控全注意力。

GDN 数学来自历史 qwen35-dflash 分支，按 Transformers 的 chunk/recurrent
参考公式验证。全注意力、MLP、路由专家与共享专家走现有三后端接口；310P 的
短卷积采用静态展开。多模态检查点只加载 language_model，视觉塔与 MTP 不加载。

TargetModelAdapter 选择文本配置和层输出 aux；RecurrentStateRuntime 管理
conv/SSM 主槽、投机行槽与接受后回滚。只有 full_attention 层分配分页 KV。
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from flashengine.layers.attention import Attention
from flashengine.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from flashengine.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear, ReplicatedLinear
from flashengine.layers.layernorm import GemmaRMSNorm, RMSNormGated
from flashengine.layers.rotary_embedding import get_partial_rope
from flashengine.models.qwen3 import Qwen3MLP
from flashengine.models.qwen3_moe import Qwen3MoeSparseMoeBlock, Qwen3MoeForCausalLM
from flashengine.backends import get_backend
from flashengine.utils.context import get_context
from flashengine.utils.loader import default_weight_loader

# Qwen3.5 packed 权重重映射（repo 格式：检查点子串 → (参数子串, 分片号)）。
# qkv/gate_up 与 qwen3.py 相同；GDN 条目对应 vllm qwen3_5.py
# packed_modules_mapping 的 "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"] /
# "in_proj_ba": ["in_proj_b", "in_proj_a"]（检查点 [q,k,v,z] / [b,a] 非交织布局）。
# in_proj_qkv（单张量 [q|k|v] → 前三分片）在 load_weights 特判，不放此表。
PACKED_MODULES_MAPPING = {
    "q_proj": ("qkv_proj", "q"),
    "k_proj": ("qkv_proj", "k"),
    "v_proj": ("qkv_proj", "v"),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    "in_proj_z": ("in_proj_qkvz", 3),
    "in_proj_b": ("in_proj_ba", 0),
    "in_proj_a": ("in_proj_ba", 1),
}

# 检查点里跳过视觉塔和 MTP；lm_head 只在与 embedding 绑定时跳过。
_SKIP_PREFIXES = ("mtp.", "model.mtp.", "model.visual.", "visual.")
# 多模态壳内 text 权重前缀（index.json 实测：model.language_model.layers.N.* 等）。
_TEXT_PREFIX = "model.language_model."


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """逐头 L2 归一化（非 RMS）：x / ||x||₂（+eps）。

    逐行对表 transformers modeling_qwen3_5.py l2norm（L242-245，注释声明对齐
    FLA l2norm_fwd）。GDN 的 q/k 归一化是 L2 而非 RMSNorm。
    """
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _gdn_gating(a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor):
    """GDN 门控：g = -exp(A_log)·softplus(a + dt_bias)（逐 v-head 标量，fp32）；
    beta = sigmoid(b)（模型 dtype，递归内再升 fp32）。

    出处：transformers Qwen3_5GatedDeltaNet.forward L497-499（= vllm
    qwen_gdn_linear_attn.fused_gdn_gating kernel：g = -A_log.float().exp() *
    softplus(a.float() + dt_bias)，beta = b.sigmoid()）。fp32 计算防 fp16 下
    A → -inf（vllm 同款注释）。
    """
    g = -A_log.float().exp() * F.softplus(a.float() + dt_bias)
    beta = torch.sigmoid(b)
    return g, beta


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """分块 Gated Delta Rule 递归（prefill 金标准，纯 torch）。

    逐行移植 transformers modeling_qwen3_5.py torch_chunk_gated_delta_rule
    （L249-327，torch 原生 fallback 分支）。与原版的差异仅两点（已注释）：
    ① 固定 batch=1（本引擎 prefill 逐序列调用，n_seqs ≤ max_num_seqs 的 python 循环）；
    ② initial_state 接收 [H, K, V]（升维成 [1, H, K, V]；普通 prefill 传 None=零态，
    spec verify 行经 gdn_has_initial 传槽内状态）。

    输入：query/key/value [1, L, H, D]（模型 dtype）；g/beta [1, L, H]（g fp32）。
    输出：(core_attn_out [L, H, V] 模型 dtype, final_state [H, K, V] fp32)。
    chunk 大小 64 = transformers 默认参数（FLA_CHUNK_SIZE）。
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        # q/k 逐头 L2 归一化在输入 dtype 内进行，随后整批升 fp32（与原版同序）
        query = _l2norm(query, dim=-1)
        key = _l2norm(key, dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape 到 chunk：[1, H, NC, CS, D]
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # chunk 内衰减矩阵：decay_mask[i, j] = exp(g_i - g_j)（i ≥ j）
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    # 求解 (I - L) w = v_beta 的前向替换循环（原版逐行移植，含 clone 语义）
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value).unsqueeze(0)  # 差异②：[H, K, V] → [1, H, K, V]
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    # 逐 chunk 递推（状态跨 chunk 传递；chunk 内向量批量计算）
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    # 差异①：去掉 batch 维（恒为 1）
    return core_attn_out.squeeze(0), last_recurrent_state.squeeze(0)


class Qwen3_5Attention(nn.Module):
    """门控全注意力（full_attention 层）。

    出处：vllm qwen3_next.py Qwen3NextAttention（Qwen3.5 复用同结构）+ transformers
    Qwen3_5Attention。要点：
    - qkv 输出 2×q_size + 2×kv_size：检查点 q_proj 输出 2×q_size（q+gate 逐头配对，
      config attn_output_gate=true），经 QKVParallelLinear 以 total_heads×(1+gate) 装载
      （vllm qwen3_next.py L267-275）；
    - q/gate 逐头拆分：q_gate.view(H, 2×head_dim) 后 chunk（transformers
      Qwen3_5Attention.forward L667-670 同款，检查点行序 [head0_q|head0_gate|...]）；
    - q_norm/k_norm 为 Gemma 式 RMSNorm（weight+1，transformers Qwen3_5RMSNorm），
      作用于完整 head_dim、先 norm 后 rope；
    - attn 输出 × sigmoid(gate)（vllm qwen3_next.py L397-398）后过 o_proj。
    """

    def __init__(self, config) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        assert self.attn_output_gate, "Qwen3.5 全注意力层为门控注意力（attn_output_gate=true）"

        # q 输出翻倍含 gate：QKVParallelLinear 的「总头数」按 total×(1+gate) 传
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        # rope 参数：Qwen3.5 用新式 rope_parameters dict（rope_theta / partial_rotary_factor），
        # 兼容旧式顶层字段；mrope_section / mrope_interleaved 仅多模态有效（text-only
        # 三行位置相同，见 PartialRotaryEmbedding 注释）。Qwen3.5-2B：256×0.25 → 64 维旋转。
        rp = getattr(config, "rope_parameters", None) or {}
        rope_theta = rp.get("rope_theta", getattr(config, "rope_theta", 10000000.0))
        partial = rp.get("partial_rotary_factor", getattr(config, "partial_rotary_factor", 1.0))
        self.rotary_emb = get_partial_rope(
            self.head_dim,
            int(self.head_dim * partial),
            config.max_position_embeddings,
            rope_theta,
        )
        self.attn = Attention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads)
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    # attention 经后端算子执行，设备 Graph 不需要对整模型做 Dynamo 编译。
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        q_gate = q_gate.view(-1, self.num_heads, 2 * self.head_dim)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        # decode 路径 flash_attn_with_kvcache 返回 [n, 1, H, D]（prefill 为
        # [n, H, D]）——统一回 [n, H, D] 再过门控，否则 sigmoid(gate) 广播会把
        # 序列维错乘成 [n, H, H, D]（qwen3.py 用 view(-1, q_size) 侧身绕开，
        # 本层有逐头 gate 必须显式归形）。
        o = o.view(-1, self.num_heads, self.head_dim)
        # 门控：attn 输出 × sigmoid(gate)（逐头广播）
        o = o * torch.sigmoid(gate)
        return self.o_proj(o.flatten(1, -1))


class Qwen3_5GatedDeltaNet(nn.Module):
    """Gated DeltaNet 线性注意力（linear_attention 层，纯 torch 实现）。

    工程形态对齐 vllm qwen_gdn_linear_attn.py（Qwen3.5 非交织布局）：
    - in_proj_qkvz = MergedColumnParallelLinear(hidden → [q,k,v,z] 四分片，
      检查点 in_proj_qkv([q|k|v]) + in_proj_z 分片装入，见 load_weights）；
    - in_proj_ba = MergedColumnParallelLinear(hidden → [b,a])；
    - conv1d 深度可分离（groups=conv_dim，无 bias），检查点权重实测 [conv_dim, 1, K]
      （transformers nn.Conv1d 保存形态），按列 narrow 进本地 [dim_local, 1, K]（兼容 2D）；
    - conv 输出过 silu（activation = config.hidden_act，vllm gdn/base.py
      self.activation = config.hidden_act，causal_conv1d_fn/update 应用之）；
    - A_log / dt_bias 为 [num_v_heads] 的 fp32 参数（vllm A_log 显式 fp32；本任务
      契约两者均 fp32，计算时 .float() 升精度，见 _gdn_gating）；
    - 输出 norm = RMSNormGated（逐 v-head、norm 先 gate 后）→ out_proj。
    状态：conv_cache [slots, conv_dim, K-1]（模型 dtype，存 raw 输入窗口）+
    ssm_cache [slots, H, K, V]（fp32），均由引擎注入本层专属切片视图。
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.tp_rank = dist.get_rank()
        tp_size = dist.get_world_size()
        self.hidden_size = config.hidden_size
        self.total_num_k_heads = config.linear_num_key_heads
        self.total_num_v_heads = config.linear_num_value_heads
        assert self.total_num_k_heads % tp_size == 0
        assert self.total_num_v_heads % tp_size == 0
        self.num_k_heads = self.total_num_k_heads // tp_size
        self.num_v_heads = self.total_num_v_heads // tp_size
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.activation = config.hidden_act
        assert self.activation in ("silu", "swish"), f"不支持的 conv 激活 {self.activation}"
        self.rms_norm_eps = config.rms_norm_eps
        # GQA 展开（transformers L500-502；Qwen3.5-2B 为 16/16=1，即无展开）
        self.gqa_ratio = self.total_num_v_heads // self.total_num_k_heads

        # 投影（分片顺序 [q,k,v,z] / [b,a]，output_sizes 传完整尺寸、层内按 TP 切）
        self.in_proj_qkvz = MergedColumnParallelLinear(
            self.hidden_size,
            [
                self.total_num_k_heads * self.head_k_dim,
                self.total_num_k_heads * self.head_k_dim,
                self.total_num_v_heads * self.head_v_dim,
                self.total_num_v_heads * self.head_v_dim,
            ],
            bias=False,
        )
        self.in_proj_ba = MergedColumnParallelLinear(
            self.hidden_size,
            [self.total_num_v_heads, self.total_num_v_heads],
            bias=False,
        )
        # 深度可分离因果卷积：weight [conv_dim, 1, K]（本地分片），模块 forward 不用，
        # 前向经 F.conv1d(·, weight, groups=conv_dim)（与 transformers nn.Conv1d 同构）。
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
        )
        self.conv1d.weight.weight_loader = self._conv_weight_loader

        # 离散化参数（fp32，见类注释）；检查点 [total_v_heads]，TP 按头 narrow
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.empty(self.num_v_heads, dtype=torch.float32))
        self.dt_bias.weight_loader = self._head_param_weight_loader
        self.A_log.weight_loader = self._head_param_weight_loader

        # 输出：逐 v-head RMSNormGated（head_v_dim 作用域）+ RowParallel
        output_gate_type = getattr(config, "output_gate_type", "silu")
        if output_gate_type == "swish":
            output_gate_type = "silu"  # vllm L453-454 同款别名归一
        self.norm = RMSNormGated(self.head_v_dim, eps=self.rms_norm_eps, activation=output_gate_type)
        self.out_proj = RowParallelLinear(self.value_dim, self.hidden_size, bias=False)

        # 状态池切片（引擎注入；占位空张量与 Attention.k_cache 哨兵同款）
        self.backend = get_backend()
        self.conv_cache = torch.tensor([])
        self.ssm_cache = torch.tensor([])

    def _conv_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        # 检查点实测 [conv_dim_full, 1, K]（transformers nn.Conv1d 保存形态，3D）；
        # 兼容 2D [conv_dim_full, K]（vllm 检查点口径）——统一 narrow 后落成
        # 本地 [conv_dim_local, 1, K]（深度可分离，dim 1 恒为 1）。
        w = loaded_weight.unsqueeze(1) if loaded_weight.ndim == 2 else loaded_weight
        shard = param.data.size(0)
        param.data.copy_(w.narrow(0, self.tp_rank * shard, shard))

    def _head_param_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        # A_log / dt_bias：检查点 [total_v_heads] → 本地 [v_heads]（逐头 narrow）
        shard = param.data.size(0)
        param.data.copy_(loaded_weight.narrow(0, self.tp_rank * shard, shard))

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        # conv 输出激活 = config.hidden_act（silu ≡ swish）
        return F.silu(x)

    # prefill 逐序列执行；decode 与 verify 行段在设备图中保持静态形状。
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ctx = get_context()

        # ---- 投影（整批一次）----
        qkvz = self.in_proj_qkvz(hidden_states)
        ba = self.in_proj_ba(hidden_states)
        qkv_size = self.key_dim * 2 + self.value_dim
        # Qwen3.5 非交织布局：检查点/权重序即 [q,k,v,z]（vllm forward_cuda L856-863）
        mixed_qkv, z = qkvz.split([qkv_size, self.value_dim], dim=-1)
        z = z.view(-1, self.num_v_heads, self.head_v_dim)
        b, a = ba.chunk(2, dim=-1)  # [n, H]，检查点顺序 [b, a]

        if ctx.gdn_state_mode == "row":
            # DFlash verify 逐行落槽（阶段2）：slots=写基址、carry_slots=行 0 读槽
            # （has_initial 不消费，见 _forward_rows docstring）。图捕获静态模式
            # （阶段3）：gdn_row_bases 非 None 时写基址/段界由引擎以 host 常量注入
            # （省 tolist host 同步——捕获期禁止 D2H），此时不要求 gdn_state_slots。
            assert ctx.gdn_carry_slots is not None and (
                ctx.gdn_state_slots is not None or ctx.gdn_row_bases is not None), (
                "'row' 模式需注入 context.gdn_carry_slots（行 0 读槽）与 "
                "context.gdn_state_slots（eager 写基址）或 gdn_row_bases（图捕获 "
                "host 静态写基址）之一")
            core = self._forward_rows(mixed_qkv, b, a, ctx)
        else:
            # GDN 状态槽元数据由引擎注入 context（阶段 1b 契约字段）
            slots = ctx.gdn_state_slots
            has_initial = ctx.gdn_has_initial
            assert slots is not None and has_initial is not None, (
                "引擎未注入 GDN 状态槽元数据（context.gdn_state_slots / gdn_has_initial，阶段 1b 契约）")
            if ctx.is_prefill:
                core = self._forward_prefill(mixed_qkv, b, a, slots, has_initial, ctx)
            else:
                core = self._forward_decode(mixed_qkv, b, a, slots, has_initial)
        # 输出归一化：逐 (token, v-head) 的 RMSNormGated（norm(o) * silu(z)）→ out_proj
        core = self.norm(core, z)
        return self.out_proj(core.flatten(1, -1))

    def _forward_prefill(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        slots,
        has_initial,
        ctx,
    ) -> torch.Tensor:
        """prefill：按调度 chunk 推进各序列，保存末态供后续 chunk/decode 复用。"""
        # 复用引擎 host 元数据，不让 18 层分别将设备长度/槽号读回。
        if ctx.host_seq_lens_q is not None:
            cu = [0]
            for length in ctx.host_seq_lens_q.tolist():
                cu.append(cu[-1] + length)
        else:
            cu = ctx.cu_seqlens_q.tolist()
        slots_l = ctx.gdn_host_slots if ctx.gdn_host_slots is not None else slots.tolist()
        init_l = ctx.gdn_host_initial if ctx.gdn_host_initial is not None else has_initial.tolist()
        n_seqs = len(cu) - 1
        core = torch.empty(
            mixed_qkv.size(0), self.num_v_heads, self.head_v_dim,
            dtype=mixed_qkv.dtype, device=mixed_qkv.device,
        )
        conv_w = self.conv1d.weight  # [conv_dim, 1, K]
        state_len = self.conv_kernel_size - 1
        for i in range(n_seqs):
            s0, s1 = cu[i], cu[i + 1]
            seq_len = s1 - s0
            x_i = mixed_qkv[s0:s1].transpose(0, 1).unsqueeze(0)  # [1, conv_dim, L]
            slot = slots_l[i]
            # 因果卷积：零初始态 = 左补 state_len 个零后 valid 卷积
            #（等价 vllm causal_conv1d_fn 的 padding=K-1 取前 L；有初始态 = 左拼槽内
            # raw 窗口，spec verify 行经 gdn_has_initial 消费）。
            if init_l[i]:
                x_ext = torch.cat([self.conv_cache[slot].unsqueeze(0), x_i], dim=-1)
            else:
                x_ext = F.pad(x_i, (state_len, 0))
            conv_out = self.backend.gdn_causal_conv1d(x_ext, conv_w)[:, :, :seq_len]
            conv_out = self._act(conv_out)
            # conv 末态 = 未过卷积/激活的 raw 输入末 state_len 窗
            #（vllm causal_conv1d_fn / transformers causal_conv1d_update 均存 raw 窗口）
            self.conv_cache[slot] = x_ext[0, :, -state_len:].to(self.conv_cache.dtype)

            # 拆 q/k/v（conv 后）并按头展开
            qkv_i = conv_out.squeeze(0).transpose(0, 1)  # [L, conv_dim]
            q, k, v = qkv_i.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = q.view(seq_len, self.num_k_heads, self.head_k_dim)
            k = k.view(seq_len, self.num_k_heads, self.head_k_dim)
            v = v.view(seq_len, self.num_v_heads, self.head_v_dim)
            if self.gqa_ratio > 1:
                # k 头按 GQA 组展开到 v 头数（transformers L500-502）
                q = q.repeat_interleave(self.gqa_ratio, dim=1)
                k = k.repeat_interleave(self.gqa_ratio, dim=1)

            g, beta = _gdn_gating(a[s0:s1], b[s0:s1], self.A_log, self.dt_bias)
            if getattr(self.backend, "supports_fused_gdn_prefill", False):
                # 时间长度走运行时循环，避免每个 prompt 长度都编译一个递推内核。
                q = (_l2norm(q).float() * self.head_k_dim ** -0.5).contiguous()
                k = _l2norm(k).float().contiguous()
                core[s0:s1] = self.backend.gdn_recurrent(
                    q, k, v.contiguous(), g, beta, self.ssm_cache, slots[i:i + 1],
                    rows=seq_len, initial=has_initial[i:i + 1])
                continue
            init_state = self.ssm_cache[slot] if init_l[i] else None
            o_i, final_state = chunk_gated_delta_rule(
                q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                g.unsqueeze(0), beta.unsqueeze(0),
                initial_state=init_state,
            )
            core[s0:s1] = o_i.to(core.dtype)
            self.ssm_cache[slot] = final_state.to(self.ssm_cache.dtype)
        return core

    def _forward_decode(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        slots,
        has_initial,
    ) -> torch.Tensor:
        """decode：单行/序列，全批向量化（无 python 循环、无 host 同步，图捕获友好）。"""
        n = mixed_qkv.size(0)
        idx = slots.to(torch.long)  # 高级索引需 long（int32 契约 → 视图收窄转换）
        conv_w = self.conv1d.weight
        # ---- conv 单步：窗口 = [槽内 raw 末 K-1 输入, 新输入]，valid 卷积取末位
        #（= transformers causal_conv1d_update 的 seq_len=1 分支）----
        if getattr(self.backend, "supports_fused_gdn", False):
            conv_out = self.backend.gdn_conv(mixed_qkv, conv_w, self.conv_cache,
                                             slots, initial=has_initial)
        else:
            conv_state = self.conv_cache[idx]  # [n, conv_dim, K-1]
            conv_state = torch.where(has_initial[:, None, None], conv_state, torch.zeros_like(conv_state))
            window = torch.cat([conv_state, mixed_qkv.unsqueeze(-1)], dim=-1)  # [n, conv_dim, K]
            conv_out = self._act(self.backend.gdn_causal_conv1d(window, conv_w)[:, :, 0])
            self.conv_cache[idx] = window[:, :, 1:]

        q, k, v = conv_out.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.view(n, self.num_k_heads, self.head_k_dim)
        k = k.view(n, self.num_k_heads, self.head_k_dim)
        v = v.view(n, self.num_v_heads, self.head_v_dim)
        if self.gqa_ratio > 1:
            q = q.repeat_interleave(self.gqa_ratio, dim=1)
            k = k.repeat_interleave(self.gqa_ratio, dim=1)

        g, beta = _gdn_gating(a, b, self.A_log, self.dt_bias)  # [n, H]（g fp32）
        # q/k 逐头 L2 归一化（输入 dtype 内）→ fp32；q 另乘 1/sqrt(head_k)
        #（transformers torch_recurrent_gated_delta_rule L343-353 同序）
        q = _l2norm(q).float() * self.head_k_dim ** -0.5
        k = _l2norm(k).float()
        if getattr(self.backend, "supports_fused_gdn", False):
            return self.backend.gdn_recurrent(q, k, v, g, beta, self.ssm_cache,
                                              slots, initial=has_initial)
        v = v.float()
        # 状态读入 + 无初始态行清零（torch.where 无 host 同步，图安全）
        state = self.ssm_cache[idx]  # [n, H, K, V] fp32
        state = torch.where(has_initial.view(-1, 1, 1, 1), state, torch.zeros_like(state))
        # 单步递归（transformers torch_recurrent_gated_delta_rule 循环体 L364-375 逐行移植，
        # 全批向量化；q 已含 1/sqrt(K) 缩放）：
        #   S̃ = exp(g)·S；kv_mem = kᵀS̃；S = S̃ + k⊗(beta·(v − kv_mem))；o = Sᵀq
        state = state * g.exp().unsqueeze(-1).unsqueeze(-1)
        kv_mem = (state * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)
        out = (state * q.unsqueeze(-1)).sum(dim=-2)  # [n, H, V]
        self.ssm_cache[idx] = state
        return out.to(mixed_qkv.dtype)

    def _forward_rows(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        ctx,
    ) -> torch.Tensor:
        """DFlash verify 按固定段逐行推进，并保存每个 token 的终态。

        gdn_carry_slots 指请求主槽；gdn_row_bases 是独立验证行槽的静态基址。
        RecurrentStateRuntime 在验证后将接受行的终态复制回主槽。所有 carry 在
        写验证行之前先 gather，避免跨序列读写干扰。递归公式与单步 decode 相同，
        conv 状态保存未激活的历史窗口。

        段界与基址是 host 常量，Graph 内不调用 tolist；carry 槽号通过固定地址
        的设备缓冲传入。eager 与 Graph 使用相同的行递推实现。
        """
        if (getattr(self.backend, "supports_fused_gdn", False)
                and ctx.gdn_row_bases is not None):
            return self._forward_rows_fused(mixed_qkv, b, a, ctx)
        if ctx.gdn_row_bases is not None:
            seg = ctx.gdn_row_seg
            base_l = ctx.gdn_row_bases
            n_seqs = len(base_l)
            cu = [i * seg for i in range(n_seqs + 1)]
        else:
            cu = ctx.cu_seqlens_q.tolist()
            base_l = ctx.gdn_state_slots.tolist()
            n_seqs = len(cu) - 1
        core = torch.empty(
            mixed_qkv.size(0), self.num_v_heads, self.head_v_dim,
            dtype=mixed_qkv.dtype, device=mixed_qkv.device,
        )
        conv_w = self.conv1d.weight  # [conv_dim, 1, K]
        state_len = self.conv_kernel_size - 1
        # ---- 全批 carry 先 gather（关键次序①：写槽前完成全部读）----
        carry_idx = ctx.gdn_carry_slots.to(torch.long)  # int32 契约 → long 高级索引
        ssm_carry = self.ssm_cache[carry_idx]    # [n, H, K, V] fp32
        conv_carry = self.conv_cache[carry_idx]  # [n, conv_dim, K-1]
        for i in range(n_seqs):
            s0, s1 = cu[i], cu[i + 1]
            seg = s1 - s0
            # ---- conv：cat(carry raw 窗口, 段输入) 一次 valid 卷积 ----
            x_ext = torch.cat([conv_carry[i], mixed_qkv[s0:s1].transpose(0, 1)],
                              dim=-1).unsqueeze(0)  # [1, conv_dim, K-1+seg]
            conv_out = self.backend.gdn_causal_conv1d(x_ext, conv_w)
            conv_out = self._act(conv_out).squeeze(0).transpose(0, 1)  # [seg, conv_dim]
            # 每 token 的 conv 终态（raw 窗口右移，与 decode 同语义）批量写槽
            conv_states = torch.stack(
                [x_ext[0, :, j + 1:j + 1 + state_len] for j in range(seg)])
            self.conv_cache[base_l[i]:base_l[i] + seg] = conv_states.to(self.conv_cache.dtype)
            # ---- 拆 q/k/v 并按头展开（与 prefill 同构）----
            q, k, v = conv_out.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
            q = q.view(seg, self.num_k_heads, self.head_k_dim)
            k = k.view(seg, self.num_k_heads, self.head_k_dim)
            v = v.view(seg, self.num_v_heads, self.head_v_dim)
            if self.gqa_ratio > 1:
                q = q.repeat_interleave(self.gqa_ratio, dim=1)
                k = k.repeat_interleave(self.gqa_ratio, dim=1)
            g, beta = _gdn_gating(a[s0:s1], b[s0:s1], self.A_log, self.dt_bias)
            # ---- 逐行递归（单步公式 = _forward_decode 循环体，fp32）----
            q = _l2norm(q).float() * self.head_k_dim ** -0.5  # [seg, H, K]
            k = _l2norm(k).float()
            v = v.float()
            state = ssm_carry[i]  # [H, K, V] fp32（读槽①已 gather）
            states, outs = [], []
            for j in range(seg):
                state = state * g[j].exp().unsqueeze(-1).unsqueeze(-1)
                kv_mem = (state * k[j].unsqueeze(-1)).sum(dim=-2)
                delta = (v[j] - kv_mem) * beta[j].unsqueeze(-1)
                state = state + k[j].unsqueeze(-1) * delta.unsqueeze(-2)
                outs.append((state * q[j].unsqueeze(-1)).sum(dim=-2))
                states.append(state)
            # 行终态批量落槽（base_i..base_i+seg-1 连续 → 一次切片写）
            self.ssm_cache[base_l[i]:base_l[i] + seg] = torch.stack(states)
            core[s0:s1] = torch.stack(outs).to(core.dtype)
        return core

    def _forward_rows_fused(self, mixed_qkv, b, a, ctx):
        """静态等长验证段：批量短卷积，递推融合且逐行落槽，无 host 同步。"""
        rows = ctx.gdn_row_seg
        batch = len(ctx.gdn_row_bases)
        base = ctx.gdn_row_bases[0]
        assert ctx.gdn_row_bases == tuple(base + i * rows for i in range(batch))
        conv_out = self.backend.gdn_conv(mixed_qkv, self.conv1d.weight, self.conv_cache,
                                         ctx.gdn_carry_slots, rows=rows, base=base)
        q, k, v = conv_out.split([self.key_dim, self.key_dim, self.value_dim], dim=-1)
        q = q.reshape(-1, self.num_k_heads, self.head_k_dim)
        k = k.reshape(-1, self.num_k_heads, self.head_k_dim)
        v = v.reshape(-1, self.num_v_heads, self.head_v_dim)
        if self.gqa_ratio > 1:
            q = q.repeat_interleave(self.gqa_ratio, dim=1)
            k = k.repeat_interleave(self.gqa_ratio, dim=1)
        g, beta = _gdn_gating(a, b, self.A_log, self.dt_bias)
        q = _l2norm(q).float() * self.head_k_dim ** -0.5
        k = _l2norm(k).float()
        return self.backend.gdn_recurrent(q, k, v, g, beta, self.ssm_cache,
                                          ctx.gdn_carry_slots, rows=rows, base=base)


class Qwen3_5SparseMoeBlock(Qwen3MoeSparseMoeBlock):
    """复用三后端路由专家前向，再加入 Qwen3.5 的门控共享专家。"""

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.shared_expert = Qwen3MLP(config.hidden_size, config.shared_expert_intermediate_size, config.hidden_act)
        self.shared_expert_gate = ReplicatedLinear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        routed = super().forward(hidden_states)
        return routed + self.shared_expert(hidden_states) * torch.sigmoid(self.shared_expert_gate(hidden_states))


class Qwen3_5DecoderLayer(nn.Module):

    def __init__(self, config, layer_type: str, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config)
        elif layer_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config)
        else:
            raise ValueError(f"无效 layer_type: {layer_type}")
        # MLP 与 Qwen3 dense 同构（gate/up 6144 + silu），直接复用
        self.mlp = (Qwen3_5SparseMoeBlock(config, layer_idx) if getattr(config, "num_experts", 0) > 0
                    else Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act))
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        if self.layer_type == "full_attention":
            hidden_states = self.self_attn(positions, hidden_states)
        else:
            hidden_states = self.linear_attn(hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3_5Model(nn.Module):

    is_hybrid = True  # 混合架构契约（阶段 1b 引擎判定用）

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, layer_type, idx) for idx, layer_type in enumerate(config.layer_types)]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # EAGLE/DFlash aux 隐层捕获索引：语义 = 捕获指定 0-based 层的【输出】隐状态
        # （区别于 qwen3.py 的层输入口径），由引擎在启用 spec 时设置；
        # 空元组 = 普通路径，forward 返回单个 hidden。
        self.aux_hidden_state_layers: tuple[int, ...] = ()
        # ---- 混合架构契约（阶段 1b 引擎接线用）----
        # 全注意力层的全局层号 / 数量（KV 池按此分配与 wiring，修复模块序号错位）
        self.attention_layer_ids = tuple(
            i for i, lt in enumerate(config.layer_types) if lt == "full_attention"
        )
        self.num_attention_layers = len(self.attention_layer_ids)
        # GDN 层全局层号 + 按序模块引用（引擎按 GDN 层序注入 conv/ssm 状态池切片）
        self.linear_attn_layer_indices = tuple(
            i for i, lt in enumerate(config.layer_types) if lt == "linear_attention"
        )
        self.gdn_layers = [self.layers[i].linear_attn for i in self.linear_attn_layer_indices]

    def get_gdn_state_spec(self) -> dict:
        """GDN 状态池规格（形状从 text_config 计算，不硬编码；每槽）。

        conv 状态 [conv_dim, kernel-1]（vllm causal_conv1d 的 raw 窗口约定），
        ssm 状态 [v_heads, head_k, head_v]（transformers 递归状态 [B, H, K, V]）；
        conv 用模型 dtype，ssm 用 config mamba_ssm_dtype（Qwen3.5-2B = float32）。
        """
        cfg = self.config
        conv_dim = (
            2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim
            + cfg.linear_num_value_heads * cfg.linear_value_head_dim
        )
        ssm_dtype_name = getattr(cfg, "mamba_ssm_dtype", "float32")
        return {
            "conv_shape": (conv_dim, cfg.linear_conv_kernel_dim - 1),
            "ssm_shape": (cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim),
            "conv_dtype": self.embed_tokens.weight.dtype,
            "ssm_dtype": {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }.get(ssm_dtype_name, torch.float32),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        capture = bool(self.aux_hidden_state_layers)
        aux_states = {} if capture else None
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            # 层输出口径（区别于 qwen3.py 的层输入口径）：记录该层完整输出（含残差），
            # 对齐 vllm EagleModelMixin._maybe_add_hidden_state = hidden_states + residual
            #（models/interfaces.py L1349-1359）。
            if capture and idx in self.aux_hidden_state_layers:
                aux_states[idx] = hidden_states if residual is None else hidden_states + residual
        hidden_states, _ = self.norm(hidden_states, residual)
        if capture:
            # 按 draft/配置声明的顺序拼接，不把任意显式层列表擅自排序。
            return hidden_states, torch.cat([aux_states[idx] for idx in self.aux_hidden_state_layers], dim=-1)
        return hidden_states


class Qwen3_5ForCausalLM(nn.Module):

    is_hybrid = True
    packed_modules_mapping = PACKED_MODULES_MAPPING
    _EXPERT_FIELDS = Qwen3MoeForCausalLM._EXPERT_FIELDS

    def __init__(self, hf_config) -> None:
        super().__init__()
        # 多模态壳（Qwen3_5ForConditionalGeneration）取 text_config；纯 text config 原样
        config = getattr(hf_config, "text_config", hf_config)
        self.config = config
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def load_weights(self, weights) -> None:
        """vllm 式权重装载（阶段 1b 由引擎对本类调用，替代 utils.loader.load_model，
        原因见模块 docstring）。weights: Iterable[(name, tensor)]。

        映射逻辑（对齐 vllm Qwen3_5Model.hf_to_vllm_mapper + Qwen3_5ForCausalLMBase
        .packed_modules_mapping 的最小子集）：
        - 跳过 mtp.* / model.visual.* / lm_head.*；
        - 剥离 model.language_model. 前缀 → 本模型 model.* 参数路径；
        - q/k/v → qkv_proj 分片、gate/up → gate_up_proj 分片、in_proj_z → in_proj_qkvz
          第 4 分片、in_proj_b/a → in_proj_ba 第 1/2 分片；
        - in_proj_qkv 单张量 [q|k|v] 拆三份装 in_proj_qkvz 前 3 分片；
        - conv1d（检查点 [dim,1,K] 3D / 兼容 [dim,K] 2D）→ narrow 成 [dim_local,1,K]；
          A_log/dt_bias 逐头 narrow。
        """
        key_dim = self.config.linear_num_key_heads * self.config.linear_key_head_dim
        value_dim = self.config.linear_num_value_heads * self.config.linear_value_head_dim
        tp_rank = self.model.embed_tokens.tp_rank

        for name, loaded_weight in weights:
            if name.startswith(_SKIP_PREFIXES):
                continue
            if name.startswith(_TEXT_PREFIX):
                name = name[len(_TEXT_PREFIX):]
            elif name.startswith("language_model."):
                name = name[len("language_model."):]
            elif name.startswith("model."):
                name = name[len("model."):]
            if name.startswith("lm_head."):
                if not self.config.tie_word_embeddings:
                    default_weight_loader(self.get_parameter(name), loaded_weight)
                continue
            if ".mlp.experts." in name:
                if not name.endswith((".weight", ".weight_scale", ".weight_offset")):
                    name += ".weight"
                Qwen3MoeForCausalLM._load_expert_weight(self, "model." + name, loaded_weight)
                continue
            # ---- GDN 特例（packed_modules_mapping 一格一 shard 表达不了的三类）----
            if name.endswith("linear_attn.in_proj_qkv.weight"):
                # 检查点单张量 [q|k|v]（key_dim*2+value_dim 行）→ qkvz 分片 0/1/2
                param = self.get_parameter("model." + name.replace("in_proj_qkv", "in_proj_qkvz"))
                offset = 0
                for shard_id, size in enumerate((key_dim, key_dim, value_dim)):
                    param.weight_loader(param, loaded_weight[offset:offset + size], shard_id)
                    offset += size
                continue
            if name.endswith("linear_attn.conv1d.weight"):
                param = self.get_parameter("model." + name)
                param.weight_loader(param, loaded_weight)
                continue
            if name.endswith("linear_attn.A_log") or name.endswith("linear_attn.dt_bias"):
                param = self.get_parameter("model." + name)
                shard = param.data.size(0)
                param.data.copy_(loaded_weight.narrow(0, tp_rank * shard, shard))
                continue
            # ---- 常规：packed 映射或默认装载（与 utils.loader.load_model 同构）----
            for ckpt_sub, (param_sub, shard_id) in PACKED_MODULES_MAPPING.items():
                if ckpt_sub in name:
                    param = self.get_parameter("model." + name.replace(ckpt_sub, param_sub))
                    param.weight_loader(param, loaded_weight, shard_id)
                    break
            else:
                param = self.get_parameter("model." + name)
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
