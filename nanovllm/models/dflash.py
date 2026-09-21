"""DFlash 块草稿模型（z-lab DFlash / vllm model_executor/models/qwen3_dflash.py
的最小等价实现，阶段2 eager 路径）。

结构（Qwen3.5-2B-Dflash checkpoint：model_type=qwen3，5 层 decoder，hidden 2048，
GQA 16Q/4KV 头，head_dim 128，rope theta 1e7 全旋转，q_norm/k_norm 标准 RMSNorm
——draft 是普通 qwen3 注意力，非 target 的 Gemma 式/Gemma 门控）：
- fc:          Linear(len(target_layer_ids)*hidden -> hidden)，aux 拼接投影；
- hidden_norm: RMSNorm(hidden)，作用在 fc 输出上（上下文 KV 投影用，见
               vllm qwen3_dflash._project_context_kv：先 fc 后 hidden_norm）；
- layers.N:    标准 qwen3 decoder 层（pre-norm 残差 + 自注意力 + Qwen3MLP）；
- norm:        最终 RMSNorm；
- 无 embed/lm_head 权重：embed_tokens 共享 target（构造传入，同 eagle3.py 模式），
  lm_head 直接用 target 的 lm_head 权重（dflash 词表 = target 词表，无 d2t 映射，
  对齐 vllm load_dflash_model 的 lm_head 共享分支）。

与 EAGLE3 draft 的两个关键差异：
1. 非因果块注意力：查询行（anchor + mask×k）对 [上下文 KV + block 自身 KV] 全量
   注意（无 causal mask，z-lab block 口径：mask 行互见 + 全上下文可见；vllm
   draft attention metadata causal=False）。
2. 上下文 KV 注入式缓存：上下文 K/V 不是 embed 推理产物，而是由 target aux 隐层
   重投影而来——fc(concat aux) → hidden_norm → 每层各自 k_proj/v_proj → k_norm →
   rope(真实位置) → 写入草稿缓存。CUDA/CoreX 使用按序列分配的稠密缓冲；310P
   使用每层独立的 FRACTAL_NZ paged KV，复用 target block-table/slot 编号并通过
   非因果 SplitFuse 读取。block 查询槽在下一轮由已提交行的 aux 重投影覆盖，
   因此被拒提案不会进入有效上下文。

forward_block 形状流转（n=批内序列数，rows=k+1=1 anchor + k mask 查询行）：
  embeds [n, rows, h]（行 0 = anchor = 最后已提交 token 的 target embed；
                       行 1..k = embed(mask_token_id)，无独立 mask_embedding.pt）
  每层：qkv_proj([n*rows, h]) → 拆 q/k/v → q_norm/k_norm（head_dim 内）→
        rope(positions 展平 [n*rows]) →
        k_full = cat(上下文 K [n, ctx_len_i, kvh, hd] 右 padding 至 max_ctx,
                     自身 K [n, rows, kvh, hd])   （v 维同理）
        sdpa（scale=head_dim^-0.5，attn_mask=[n,1,rows,max_ctx+rows]：padding 位
        -inf、查询行互见全 True；GQA 以 repeat_interleave 展开，规模小且 eager
        阶段优先确定性）→ o_proj → 残差流
  输出：行 0 = anchor 丢弃（z-lab：anchor 只作条件不产出提案），行 1..k →
        final norm → F.linear(·, target lm_head.weight) → logits [n*k, vocab]
"""

import os
from glob import glob

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nanovllm.backends import get_backend
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ReplicatedLinear, QKVParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.models.qwen3 import PACKED_MODULES_MAPPING, Qwen3MLP
from nanovllm.utils.loader import default_weight_loader


class DFlashBlockAttention(nn.Module):
    """block 草稿注意力：非因果、上下文 KV 注入。

    CUDA/CoreX 保留稠密上下文 + torch SDPA；310P 将投影上下文与查询 K/V 写入
    私有分页 NZ 缓存，并用一次非因果 SplitFuse 计算整个查询块。
    """

    def __init__(self, config, rotary_emb) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        assert self.total_num_heads % tp_size == 0
        assert self.total_num_kv_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.backend = get_backend()
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        # 各层 rope 参数一致（vllm _build_fused_kv_buffers 同款前提），共享同一
        # get_rope 实例（lru_cache 按 (head,rotary,max_pos,base) 去重）。
        self.rotary_emb = rotary_emb
        # 标准 RMSNorm（draft model_type=qwen3，非 target 的 Gemma 式 +1 口径）
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        # 310P DFlash uses a private paged FRACTAL_NZ cache per draft layer.
        # Other backends leave these empty and retain the dense-SDPA path.
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(
        self,
        hidden_states: torch.Tensor,      # [n*rows, h]
        n: int,
        rows: int,
        positions: torch.Tensor,          # [n*rows]
        k_ctx: torch.Tensor,              # [n, max_ctx, kvh, hd]
        v_ctx: torch.Tensor,
        mask: torch.Tensor,               # [n, 1, rows, max_ctx+rows] bool
        query_slots: torch.Tensor | None = None,
        block_tables: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
        max_seq_len: int | None = None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q = qkv[..., :self.q_size].view(n, rows, self.num_heads, self.head_dim)
        k = qkv[..., self.q_size:self.q_size + self.kv_size].view(n, rows, self.num_kv_heads, self.head_dim)
        v = qkv[..., self.q_size + self.kv_size:].view(n, rows, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        # rope：展平批维（positions 与行一一对应）
        q, k = self.rotary_emb(
            positions,
            q.reshape(n * rows, self.num_heads, self.head_dim),
            k.reshape(n * rows, self.num_kv_heads, self.head_dim),
        )
        if (query_slots is not None
                and getattr(self.backend, "supports_dflash_paged", False)
                and self.k_cache.numel() and self.v_cache.numel()):
            q_flat = q.reshape(n * rows, self.num_heads, self.head_dim)
            k_flat = k.reshape(n * rows, self.num_kv_heads, self.head_dim)
            v_flat = v.reshape(n * rows, self.num_kv_heads, self.head_dim)
            self.backend.store_kvcache(
                k_flat, v_flat, self.k_cache, self.v_cache, query_slots
            )
            o = self.backend.dflash_paged_attention(
                q_flat,
                self.k_cache,
                self.v_cache,
                block_tables,
                context_lens,
                rows,
                self.scaling,
                max_seq_len,
            )
            return self.o_proj(o.reshape(n * rows, self.q_size))
        q = q.view(n, rows, self.num_heads, self.head_dim).transpose(1, 2)      # [n, H, rows, hd]
        k = k.view(n, rows, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [n, kvh, rows, hd]
        v = v.transpose(1, 2)
        # K/V = cat(上下文, 自身查询行)：查询行互见 + 全上下文可见（非因果）
        k_full = torch.cat([k_ctx.transpose(1, 2), k], dim=2)   # [n, kvh, ctx+rows, hd]
        v_full = torch.cat([v_ctx.transpose(1, 2), v], dim=2)
        rep = self.num_heads // self.num_kv_heads
        if rep > 1:
            # GQA 展开（16Q/4KV → 4×）；sdpa enable_gqa 的后端门控行为随版本变化，
            # eager 阶段显式展开换确定性（规模 ≤ [n,16,ctx+16,128] bf16，~MB 级）。
            k_full = k_full.repeat_interleave(rep, dim=1)
            v_full = v_full.repeat_interleave(rep, dim=1)
        o = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=mask, scale=self.scaling)
        o = o.transpose(1, 2).reshape(n * rows, self.q_size)
        return self.o_proj(o)


class DFlashDecoderLayer(nn.Module):
    """标准 qwen3 decoder 层（权重名 layers.N.*，经 packed_modules_mapping 重映射）。"""

    def __init__(self, config, rotary_emb) -> None:
        super().__init__()
        self.self_attn = DFlashBlockAttention(config, rotary_emb)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, residual, positions, n, rows, k_ctx, v_ctx, mask,
                query_slots=None, block_tables=None, context_lens=None,
                max_seq_len=None):
        # 与 qwen3.Qwen3DecoderLayer 同构（延迟残差 + 融合 add_rms_norm）
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            hidden_states, n, rows, positions, k_ctx, v_ctx, mask,
            query_slots=query_slots,
            block_tables=block_tables,
            context_lens=context_lens,
            max_seq_len=max_seq_len,
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class DFlashDraftModel(nn.Module):

    packed_modules_mapping = PACKED_MODULES_MAPPING

    def __init__(self, config, target_embed_tokens: nn.Module,
                 target_lm_head_weight: torch.Tensor, max_position: int | None = None) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        dfc = getattr(config, "dflash_config", None) or {}
        assert dfc, (
            "DFlash draft config is missing dflash_config "
            "(block_size/mask_token_id/target_layer_ids)."
        )
        self.target_layer_ids = tuple(dfc["target_layer_ids"])
        self.mask_token_id = int(dfc["mask_token_id"])
        # embedding / lm_head 与 target 共享（checkpoint 无此权重；dflash 无独立
        # draft 词表与 d2t，直接在 target 词表上采样）。
        self.embed_tokens = target_embed_tokens
        self.lm_head_weight = target_lm_head_weight
        # fc 输入宽度 = len(target_layer_ids) * hidden（vllm fc_input_size 同式）
        self.fc = ReplicatedLinear(len(self.target_layer_ids) * hidden_size, hidden_size, bias=False)
        self.hidden_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        # rope：head_dim 全旋转（qwen3 半分式惯例），theta 取 rope_parameters（新式
        # dict）兼容旧式顶层字段；max_position 由引擎按 max_model_len 收紧（位置
        # 恒 < max_model_len，省 262144 全长 cos/sin 缓存的 ~134MB fp32）。
        rp = getattr(config, "rope_parameters", None) or {}
        theta = rp.get("rope_theta", getattr(config, "rope_theta", 1000000.0))
        head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
        if max_position is None:
            max_position = config.max_position_embeddings
        self.rotary_emb = get_rope(head_dim, head_dim, max_position, theta)
        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config, self.rotary_emb) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.backend = get_backend()
        self.use_paged_kv = bool(getattr(self.backend, "supports_dflash_paged", False))
        # 上下文 KV 稠密缓冲占位（引擎注入真实缓冲，镜像 qwen3_5 conv_cache/ssm_cache）
        self.ctx_k_cache = torch.tensor([])
        self.ctx_v_cache = torch.tensor([])

    def _build_fused_kv_buffers(self) -> None:
        """权重加载后构建上下文投影的融合缓冲（对齐 vllm _build_fused_kv_buffers）：
        各层 k/v_proj 拼成单 GEMM 权重 + 缓存 mask token 的 embed 行。"""
        attns = [layer.self_attn for layer in self.layers]
        a0 = attns[0]
        # 各层注意力形状一致（融合 GEMM 的前提，vllm 同款校验）
        for a in attns[1:]:
            assert (a.kv_size == a0.kv_size and a.head_dim == a0.head_dim
                    and a.num_kv_heads == a0.num_kv_heads), (
                        "All DFlash attention layers must use identical shapes."
                    )
        # qkv_proj.weight 行序 [q | k | v]，取 k|v 段逐层拼接 → [L*2*kv, hidden]
        self._fused_kv_weight = torch.cat(
            [a.qkv_proj.weight[a.q_size:] for a in attns], dim=0).contiguous()
        self._num_layers = len(attns)
        self._kv_size = a0.kv_size
        self._num_kv_heads = a0.num_kv_heads
        self._head_dim = a0.head_dim
        # mask 查询行的 embed：直接查词表行（本 checkpoint 无独立 mask_embedding.pt；
        # vllm 对带 mask_embedding.pt 的 checkpoint 另有替换分支，此处不支持并在
        # load_dflash_draft 中对异常权重名报错）
        self._mask_embed = self.embed_tokens(torch.tensor(
            [self.mask_token_id], dtype=torch.int64,
            device=self.embed_tokens.weight.device))  # [1, h]

    def project_context_kv(self, aux: torch.Tensor, positions: torch.Tensor,
                           slots: torch.Tensor) -> None:
        """上下文 KV 投影 + 写稠密缓冲（对齐 vllm precompute_and_store_context_kv）。

        aux [T, len(target_layer_ids)*h]（target 0-based 层输出 aux 拼接，拼接顺序
        = 层号升序 = fc 训练顺序）；positions [T]（真实序列位置）；slots [T]（稠密
        缓冲的槽号，与行一一对应）。链路：fc → hidden_norm → 融合 k/v GEMM → 逐层
        k_norm（与 block 查询行同一 RMSNorm 模块，数值一致）→ rope(真实位置) →
        ctx_{k,v}_cache[slot, pos, layer]。（v 不做 norm/rope，q 不参与投影。）
        """
        T = aux.size(0)
        x = F.linear(aux, self.fc.weight)        # [T, h]
        x = self.hidden_norm(x)                  # [T, h]
        kv = F.linear(x, self._fused_kv_weight)  # [T, L*2*kv]
        kv = kv.view(T, self._num_layers, 2, self._num_kv_heads, self._head_dim)
        k_all = kv[:, :, 0]                      # [T, L, kvh, hd]
        v_all = kv[:, :, 1]
        L = self._num_layers
        hd = self._head_dim
        for lid in range(L):
            k_lid = k_all[:, lid]  # [T, kvh, hd]
            k_lid = self.layers[lid].self_attn.k_norm(k_lid)
            # rope 只作用于 k（q/v 不参与上下文注入）；同张量传入 q/k 位取 k 输出
            k_lid, _ = self.rotary_emb(positions, k_lid, k_lid)
            if self.use_paged_kv:
                attn = self.layers[lid].self_attn
                self.backend.store_kvcache(
                    k_lid.reshape(T, self._num_kv_heads, hd),
                    v_all[:, lid],
                    attn.k_cache,
                    attn.v_cache,
                    slots,
                )
            else:
                # 高级索引写：[slots[t], positions[t]] 各行互异（同序列位置不重、槽不共）
                self.ctx_k_cache[slots, positions, lid] = k_lid.reshape(T, self._num_kv_heads, hd)
                self.ctx_v_cache[slots, positions, lid] = v_all[:, lid]

    def forward_block(self, last_token_ids: torch.Tensor, positions: torch.Tensor,
                      ctx_slots: torch.Tensor, ctx_lens: torch.Tensor,
                      max_ctx: int, rows: int,
                      query_slots: torch.Tensor | None = None,
                      block_tables: torch.Tensor | None = None,
                      total_lens: torch.Tensor | None = None) -> torch.Tensor:
        """block 草稿前向（形状流转见模块 docstring）。

        last_token_ids [n]（anchor = 最后已提交 token）；positions [n, rows] =
        [l'-1 .. l'+k-1]；ctx_slots/ctx_lens [n] 设备张量（ctx_len_i = l'_i - 1，
        上下文覆盖 0..l'-2）；max_ctx 为 host int（批内 ctx_len 最大值，调用方算好
        免设备同步）。返回 logits [n*k, vocab]，行序 (i, j) → i*k + j（第 i 序列的
        rows 1..k 展平，采样参数按 repeat_interleave(k) 对齐）。
        """
        n = last_token_ids.size(0)
        k = rows - 1
        # 输入行：[anchor embed, mask×k]（mask 行 = embed(mask_token_id)）
        anchor = self.embed_tokens(last_token_ids)                         # [n, h]
        embeds = torch.cat([anchor.unsqueeze(1),
                            self._mask_embed.expand(n, k, -1)], dim=1)     # [n, rows, h]
        flat = embeds.reshape(n * rows, -1)
        pos_flat = positions.reshape(-1)
        # 上下文 gather（右 padding 至 max_ctx；padding 行由 -inf mask 屏蔽。缓冲
        # 零初始化保证未写行为有限值，掩码后 0 权重 × 有限 v 无 NaN 传播）
        if self.use_paged_kv:
            assert query_slots is not None and block_tables is not None and total_lens is not None
            k_all = v_all = mask = None
        else:
            k_all = self.ctx_k_cache[ctx_slots, :max_ctx]   # [n, max_ctx, L, kvh, hd]
            v_all = self.ctx_v_cache[ctx_slots, :max_ctx]
            # 非因果 mask：上下文段 j < ctx_len_i 可见，自身段（后 rows 列）全可见
            ar = torch.arange(max_ctx, device=flat.device)
            ctx_ok = ar[None, :] < ctx_lens[:, None]        # [n, max_ctx]
            mask = torch.cat(
                [ctx_ok[:, None, None, :].expand(n, 1, rows, max_ctx),
                 torch.ones(n, 1, rows, rows, dtype=torch.bool, device=flat.device)],
                dim=-1)                                     # [n, 1, rows, max_ctx+rows]
        residual = None
        hidden = flat
        for lid, layer in enumerate(self.layers):
            hidden, residual = layer(hidden, residual, pos_flat, n, rows,
                                     None if k_all is None else k_all[:, :, lid],
                                     None if v_all is None else v_all[:, :, lid], mask,
                                     query_slots=query_slots,
                                     block_tables=block_tables,
                                     context_lens=total_lens,
                                     max_seq_len=max_ctx)
        hidden, _ = self.norm(hidden, residual)
        # 行 0 = anchor 丢弃（z-lab：anchor 只作条件），行 1..k 过 target lm_head
        h = hidden.view(n, rows, -1)[:, 1:].reshape(n * k, -1)
        return F.linear(
            h, getattr(self, "lm_head_weight_nz", self.lm_head_weight)
        )                                               # [n*k, vocab]


def load_dflash_draft(model: DFlashDraftModel, path: str) -> None:
    """DFlash draft 权重装载（与 load_eagle3_draft 同构）。

    键名直配无前缀剥离（checkpoint 顶层即 fc./hidden_norm./norm./layers.N.*）；
    q/k/v → qkv_proj 分片、gate/up → gate_up_proj 分片（PACKED_MODULES_MAPPING），
    其余（fc/hidden_norm/norm/layernorm/o_proj/down_proj）直装。不支持
    mask_embedding.pt 覆盖与 d2t 映射（本类 checkpoint 两者皆无）。
    """
    bin_path = os.path.join(path, "pytorch_model.bin")
    if os.path.exists(bin_path):
        state = torch.load(bin_path, map_location="cpu", weights_only=True)

        def items():
            yield from state.items()
    else:
        from safetensors import safe_open
        files = glob(os.path.join(path, "*.safetensors"))
        assert files, f"No DFlash draft weight files found in {path}."

        def items():
            for file in files:
                with safe_open(file, "pt", "cpu") as f:
                    for name in f.keys():
                        yield name, f.get_tensor(name)

    for name, tensor in items():
        target_name = name
        shard_id = None
        for k, (v, shard) in model.packed_modules_mapping.items():
            if k in name:
                target_name = name.replace(k, v)
                shard_id = shard
                break
        param = model.get_parameter(target_name)
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        if shard_id is not None:
            weight_loader(param, tensor, shard_id)
        else:
            weight_loader(param, tensor)
