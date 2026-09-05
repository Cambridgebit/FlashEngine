import torch
import triton
import triton.language as tl

from flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

from .base import BackendBase


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)

    slot = tl.load(slot_mapping_ptr + idx)

    if slot == -1:
        return

    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)

    cache_offsets = slot * D + tl.arange(0, D)

    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


class CUDABackend(BackendBase):

    name = "cuda"
    # flash-attn/图内采样捕获已在 CUDA 实证。
    supports_spec_graph = True

    def store_kvcache(
        self,
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
    ):
        N, num_heads, head_dim = key.shape
        D = num_heads * head_dim
        assert key.stride(-1) == 1 and value.stride(-1) == 1
        assert key.stride(1) == head_dim and value.stride(1) == head_dim
        assert k_cache.stride(1) == D and v_cache.stride(1) == D
        assert slot_mapping.numel() == N
        store_kvcache_kernel[(N,)](
            key,
            key.stride(0),
            value,
            value.stride(0),
            k_cache,
            v_cache,
            slot_mapping,
            D,
        )

    def prefill_attention(
        self,
        q,
        k,
        v,
        context,
        scale,
    ):
        return flash_attn_varlen_func(
            q,
            k,
            v,
            max_seqlen_q=context.max_seqlen_q,
            cu_seqlens_q=context.cu_seqlens_q,
            max_seqlen_k=context.max_seqlen_k,
            cu_seqlens_k=context.cu_seqlens_k,
            softmax_scale=scale,
            causal=True,
            block_table=context.block_tables,
        )

    def decode_attention(
        self,
        q,
        k_cache,
        v_cache,
        context,
        scale,
    ):
        return flash_attn_with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=scale,
            causal=True,
        )
