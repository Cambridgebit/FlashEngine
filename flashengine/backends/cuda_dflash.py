"""DFlash 稠密上下文打包：去掉固定窗口空洞，交给非因果 FlashAttention GQA。"""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_kernel(KC, VC, K, V, Lengths, Cu, KO, VO,
                 CS0: tl.constexpr, CS1: tl.constexpr, CS2: tl.constexpr,
                 KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
                 VS0: tl.constexpr, VS1: tl.constexpr, VS2: tl.constexpr,
                 H: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr,
                 BLOCK: tl.constexpr):
    seq = tl.program_id(0)
    offset = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    time = offset // (H * D)
    head = (offset // D) % H
    dim = offset % D
    length = tl.load(Lengths + seq)
    start = tl.load(Cu + seq)
    prefix = time < length
    query = (time >= length) & (time < length + ROWS)
    ctx_offset = seq * CS0 + time * CS1 + head * CS2 + dim
    kc = tl.load(KC + ctx_offset, mask=prefix, other=0)
    vc = tl.load(VC + ctx_offset, mask=prefix, other=0)
    token = seq * ROWS + time - length
    kb = tl.load(K + token * KS0 + head * KS1 + dim * KS2, mask=query, other=0)
    vb = tl.load(V + token * VS0 + head * VS1 + dim * VS2, mask=query, other=0)
    destination = (start + time) * H * D + head * D + dim
    tl.store(KO + destination, tl.where(prefix, kc, kb), mask=prefix | query)
    tl.store(VO + destination, tl.where(prefix, vc, vb), mask=prefix | query)


def attention(varlen, query, key, value, ctx_key, ctx_value, lengths, scale):
    """只打包有效前缀与本轮 query KV；不复制/展开 GQA 的重复 heads。"""
    batch, rows, heads, dim = query.shape
    kv_heads = key.shape[-2]
    max_ctx = ctx_key.shape[1]
    lengths = lengths.to(dtype=torch.int32)
    cu_q = torch.arange(batch + 1, dtype=torch.int32, device=query.device) * rows
    cu_k = torch.cat((torch.zeros(1, dtype=torch.int32, device=query.device),
                      (lengths + rows).cumsum(0, dtype=torch.int32)))
    packed_k = torch.empty((batch * (max_ctx + rows), kv_heads, dim), dtype=key.dtype, device=key.device)
    packed_v = torch.empty_like(packed_k)
    assert ctx_key.stride(-1) == ctx_value.stride(-1) == 1
    assert ctx_key.stride() == ctx_value.stride()
    k = key.reshape(batch * rows, kv_heads, dim)
    v = value.reshape(batch * rows, kv_heads, dim)
    _pack_kernel[(batch, triton.cdiv((max_ctx + rows) * kv_heads * dim, 256))](
        ctx_key, ctx_value, k, v, lengths, cu_k, packed_k, packed_v,
        *ctx_key.stride()[:3], *k.stride(), *v.stride(), kv_heads, dim, rows, 256)
    return varlen(query.reshape(batch * rows, heads, dim), packed_k, packed_v,
                  cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                  max_seqlen_q=rows, max_seqlen_k=max_ctx + rows,
                  softmax_scale=scale, causal=False)
