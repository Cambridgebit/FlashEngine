
import os
from glob import glob

import torch
import torch.distributed as dist
from torch import nn

from flashengine.backends import get_backend
from flashengine.layers.layernorm import RMSNorm
from flashengine.layers.linear import ReplicatedLinear, QKVParallelLinear, RowParallelLinear
from flashengine.layers.rotary_embedding import get_rope
from flashengine.models.qwen3 import PACKED_MODULES_MAPPING, Qwen3MLP
from flashengine.speculative.registry import block_draft_attention
from flashengine.utils.loader import (
    default_weight_loader,
    iter_with_progress,
    safetensors_total_bytes,
)


def block_attention_mask(ctx_lens, max_ctx, rows, causal=False, window=None):
    """按每条序列的逻辑位置构造掩码，不能把稠密缓存的 padding 算进窗口。"""
    device = ctx_lens.device
    ctx_pos = torch.arange(max_ctx, device=device)[None, :].expand(ctx_lens.numel(), -1)
    query_pos = ctx_lens[:, None] + torch.arange(rows, device=device)[None, :]
    key_pos = torch.cat((ctx_pos, query_pos), dim=-1)
    valid = torch.cat((ctx_pos < ctx_lens[:, None],
                       torch.ones_like(query_pos, dtype=torch.bool)), dim=-1)
    delta = query_pos[:, :, None] - key_pos[:, None, :]
    mask = valid[:, None, :].expand(-1, rows, -1)
    if causal:
        mask = mask & (delta >= 0)
    if window is not None:
        mask = mask & (delta < window)
        if not causal:
            mask = mask & (-delta < window)
    return mask[:, None]


class DFlashBlockAttention(nn.Module):

    def __init__(self, config, rotary_emb, layer_idx=0) -> None:
        super().__init__()
        self.is_causal, self.sliding_window = block_draft_attention(config)[layer_idx]
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
        self.rotary_emb = rotary_emb
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
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
        dense_context_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q = qkv[..., :self.q_size].view(n, rows, self.num_heads, self.head_dim)
        k = qkv[..., self.q_size:self.q_size + self.kv_size].view(n, rows, self.num_kv_heads, self.head_dim)
        v = qkv[..., self.q_size + self.kv_size:].view(n, rows, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(
            positions,
            q.reshape(n * rows, self.num_heads, self.head_dim),
            k.reshape(n * rows, self.num_kv_heads, self.head_dim),
        )
        if (query_slots is not None and not self.is_causal and self.sliding_window is None
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
        if (getattr(self.backend, "supports_dflash_dense_varlen", False)
                and dense_context_lens is not None):
            o = self.backend.dflash_dense_attention(
                q.view(n, rows, self.num_heads, self.head_dim),
                k.view(n, rows, self.num_kv_heads, self.head_dim), v,
                k_ctx, v_ctx, dense_context_lens, self.scaling,
                causal=self.is_causal, window=self.sliding_window)
            return self.o_proj(o.reshape(n * rows, self.q_size))
        if self.is_causal or self.sliding_window is not None:
            if dense_context_lens is None:
                raise ValueError("DFlash 窗口注意力缺少上下文长度")
            mask = block_attention_mask(dense_context_lens, k_ctx.shape[1], rows,
                                        self.is_causal, self.sliding_window)
        q = q.view(n, rows, self.num_heads, self.head_dim).transpose(1, 2)      # [n, H, rows, hd]
        k = k.view(n, rows, self.num_kv_heads, self.head_dim).transpose(1, 2)   # [n, kvh, rows, hd]
        v = v.transpose(1, 2)
        k_full = torch.cat([k_ctx.transpose(1, 2), k], dim=2)   # [n, kvh, ctx+rows, hd]
        v_full = torch.cat([v_ctx.transpose(1, 2), v], dim=2)
        rep = self.num_heads // self.num_kv_heads
        if rep > 1:
            k_full = k_full.repeat_interleave(rep, dim=1)
            v_full = v_full.repeat_interleave(rep, dim=1)
        o = self.backend.dense_noncausal_attention(
            q, k_full, v_full, mask, self.scaling)
        o = o.transpose(1, 2).reshape(n * rows, self.q_size)
        return self.o_proj(o)


class DFlashDecoderLayer(nn.Module):

    def __init__(self, config, rotary_emb, layer_idx=0) -> None:
        super().__init__()
        self.self_attn = DFlashBlockAttention(config, rotary_emb, layer_idx)
        self.mlp = Qwen3MLP(config.hidden_size, config.intermediate_size, config.hidden_act)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, residual, positions, n, rows, k_ctx, v_ctx, mask,
                query_slots=None, block_tables=None, context_lens=None,
                max_seq_len=None, dense_context_lens=None):
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
            dense_context_lens=dense_context_lens,
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
        self.sample_from_anchor = False
        self.embed_tokens = target_embed_tokens
        self.lm_head_weight = target_lm_head_weight
        # 缩减词表（speculators 格式）：draft 词表小于目标词表，d2t 为增量映射表
        # （target_id = draft_id + d2t[draft_id]，与 EAGLE3 同口径）；全词表时
        # 增量恒为 0，映射退化为恒等。
        draft_vocab = int(getattr(config, "draft_vocab_size", None) or config.vocab_size)
        self.draft_vocab_size = draft_vocab
        self.reduced_vocab = draft_vocab != int(config.vocab_size)
        self.register_buffer("d2t", torch.zeros(draft_vocab, dtype=torch.int64))
        self.register_buffer("t2d", torch.zeros(int(config.vocab_size), dtype=torch.bool))
        self.fc = ReplicatedLinear(len(self.target_layer_ids) * hidden_size, hidden_size, bias=False)
        self.hidden_norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        rp = getattr(config, "rope_parameters", None) or {}
        theta = rp.get("rope_theta", getattr(config, "rope_theta", 1000000.0))
        head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
        if max_position is None:
            max_position = config.max_position_embeddings
        self.rotary_emb = get_rope(head_dim, head_dim, max_position, theta)
        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config, self.rotary_emb, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.backend = get_backend()
        self.use_paged_kv = bool(
            getattr(self.backend, "supports_dflash_paged", False)
            and not any(causal or window is not None
                        for causal, window in block_draft_attention(config)))
        self.ctx_k_cache = torch.tensor([])
        self.ctx_v_cache = torch.tensor([])

    def _build_fused_kv_buffers(self) -> None:
        attns = [layer.self_attn for layer in self.layers]
        a0 = attns[0]
        for a in attns[1:]:
            assert (a.kv_size == a0.kv_size and a.head_dim == a0.head_dim
                    and a.num_kv_heads == a0.num_kv_heads), (
                        "All DFlash attention layers must use identical shapes."
                    )
        self._fused_kv_weight = torch.cat(
            [a.qkv_proj.weight[a.q_size:] for a in attns], dim=0).contiguous()
        self._num_layers = len(attns)
        self._kv_size = a0.kv_size
        self._num_kv_heads = a0.num_kv_heads
        self._head_dim = a0.head_dim
        self._mask_embed = self.embed_tokens(torch.tensor(
            [self.mask_token_id], dtype=torch.int64,
            device=self.embed_tokens.weight.device))  # [1, h]

    def query_rows(self, num_tokens: int) -> int:
        return num_tokens + (not self.sample_from_anchor)

    def sample_proposals(self, logits, anchor_ids, num_tokens, temperatures,
                         top_ps, top_ks, sample):
        ids = sample(
            logits,
            temperatures.repeat_interleave(num_tokens),
            top_ps.repeat_interleave(num_tokens),
            top_ks.repeat_interleave(num_tokens),
        )
        return ids.view(anchor_ids.numel(), num_tokens)

    def map_to_target_ids(self, draft_ids: torch.Tensor) -> torch.Tensor:
        # d2t 为增量映射表：target = draft + d2t[draft]（全词表时增量为 0）。
        return draft_ids + self.d2t[draft_ids]

    def project_context_kv(self, aux: torch.Tensor, positions: torch.Tensor,
                           slots: torch.Tensor) -> None:
        T = aux.size(0)
        x = self.backend.linear(aux, self.fc.weight)  # [T, h]
        x = self.hidden_norm(x)                  # [T, h]
        kv = self.backend.linear(x, self._fused_kv_weight)  # [T, L*2*kv]
        kv = kv.view(T, self._num_layers, 2, self._num_kv_heads, self._head_dim)
        k_all = kv[:, :, 0]                      # [T, L, kvh, hd]
        v_all = kv[:, :, 1]
        L = self._num_layers
        hd = self._head_dim
        for lid in range(L):
            k_lid = k_all[:, lid]  # [T, kvh, hd]
            k_lid = self.layers[lid].self_attn.k_norm(k_lid)
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
                self.ctx_k_cache[slots, positions, lid] = k_lid.reshape(T, self._num_kv_heads, hd)
                self.ctx_v_cache[slots, positions, lid] = v_all[:, lid]

    def forward_block(self, last_token_ids: torch.Tensor, positions: torch.Tensor,
                      ctx_slots: torch.Tensor, ctx_lens: torch.Tensor,
                      max_ctx: int, rows: int,
                      query_slots: torch.Tensor | None = None,
                      block_tables: torch.Tensor | None = None,
                      total_lens: torch.Tensor | None = None,
                      sample_from_anchor: bool = False) -> torch.Tensor:
        n = last_token_ids.size(0)
        k = rows - 1
        anchor = self.embed_tokens(last_token_ids)                         # [n, h]
        embeds = torch.cat([anchor.unsqueeze(1),
                            self._mask_embed.expand(n, rows - 1, -1)], dim=1)
        flat = embeds.reshape(n * rows, -1)
        pos_flat = positions.reshape(-1)
        if self.use_paged_kv:
            assert query_slots is not None and block_tables is not None and total_lens is not None
            k_all = v_all = mask = None
        else:
            k_all = self.ctx_k_cache[ctx_slots, :max_ctx]
            v_all = self.ctx_v_cache[ctx_slots, :max_ctx]
            ar = torch.arange(max_ctx, device=flat.device)
            ctx_ok = (ar[None, None, :] < ctx_lens[:, None, None]).expand(
                n, rows, max_ctx)
            mask = torch.cat(
                [ctx_ok[:, None, :, :],
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
                                     max_seq_len=max_ctx, dense_context_lens=ctx_lens)
        hidden, _ = self.norm(hidden, residual)
        h = hidden.view(n, rows, -1)
        h = (h if sample_from_anchor else h[:, 1:]).reshape(-1, h.size(-1))
        return self.backend.linear(
            h, getattr(self, "lm_head_weight_nz", self.lm_head_weight)
        )                                               # [n*k, vocab]


def load_dflash_draft(model: DFlashDraftModel, path: str) -> None:
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

    total_bytes = safetensors_total_bytes(path)
    for name, tensor in iter_with_progress(items(), "Draft model loading (DFlash)",
                                           total_bytes):
        bare = name.removeprefix("model.")
        if bare in ("d2t", "t2d"):
            buf = getattr(model, bare)
            buf.data.copy_(tensor.to(buf.device))
            continue
        if name == "embed_tokens.weight":
            if model.embed_tokens.weight.shape != tensor.shape:
                raise ValueError("DFlash embedding size differs from the target vocabulary")
            # 不做默认随机初始化：ARM CPU 上 6 亿元素 normal_ 约 40s（实测），
            # 且随后会被 checkpoint 覆盖；直接在目标设备上建空权重交给下方加载器。
            _emb = nn.Embedding(tensor.shape[0], tensor.shape[1],
                                dtype=model.fc.weight.dtype,
                                _weight=torch.empty(tensor.shape,
                                                    dtype=model.fc.weight.dtype,
                                                    device=model.fc.weight.device))
            model.embed_tokens = _emb
        if name == "lm_head.weight":
            if model.lm_head_weight.shape != tensor.shape:
                if not (model.reduced_vocab
                        and tensor.shape[0] == model.draft_vocab_size):
                    raise ValueError(
                        "DFlash LM-head size differs from the target vocabulary")
            model.lm_head_weight = nn.Parameter(torch.empty(
                tensor.shape, device=model.fc.weight.device,
                dtype=model.fc.weight.dtype))
            name = "lm_head_weight"
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
