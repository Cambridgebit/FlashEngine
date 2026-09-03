import torch
import torch.distributed as dist

import ixformer.inference.functions as ixf_ops

from ixformer.contrib.vllm_flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

from .base import BackendBase


class CoreXBackend(BackendBase):

    name = "corex"

    def store_kvcache(
        self,
        key,
        value,
        k_cache,
        v_cache,
        slot_mapping,
    ):
        assert key.is_cuda
        assert value.is_cuda
        assert k_cache.is_cuda
        assert v_cache.is_cuda

        # ixformer 的 vllm_reshape_and_cache 要求 slot_mapping 为 int64，
        # 而 nano-vLLM 引擎（Context.slot_mapping）生成 int32。
        if slot_mapping.dtype != torch.long:
            slot_mapping = slot_mapping.to(torch.long)

        ixf_ops.reshape_and_cache_flash(
            key,
            value,
            k_cache,
            v_cache,
            slot_mapping,
            "auto",
            1.0,
            1.0,
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
            max_context_len=context.max_context_len,
            softmax_scale=scale,
            causal=True,
        )

    def get_kv_cache_shape(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        # ixformer paged (HND) 布局：
        # [2, num_layers, num_blocks, num_kv_heads, block_size, head_dim]
        return (
            2,
            num_layers,
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
        )

    def all_reduce(self, tensor):
        # 第一版默认走 CoreX PyTorch/NCCL compatibility path。
        #
        # IXFormer distributed 等单卡和 NCCL 路径稳定后再打开：
        #
        #   NANOVLLM_COREX_USE_IXFD=1
        #
        # 时再切换 ixformer.distributed.all_reduce。
        dist.all_reduce(tensor)
        return tensor
