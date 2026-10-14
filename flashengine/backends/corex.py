import torch

import ixformer.inference.functions as ixf_ops

from ixformer.contrib.vllm_flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

from .base import BackendBase


class CoreXBackend(BackendBase):

    name = "corex"
    dist_name = "nccl"
    supports_spec_graph = True
    supports_device_sampling_rng = True  # CUDA 兼容设备 RNG；Graph 不复用 CPU 噪声。
    supports_moe = True  # 专家前向用 BackendBase 的纯 torch 实现（可被 CUDA Graph 捕获）。
    supports_int8_dequant = True  # Quark int8 检查点加载期反量化（无 int8 算子）。

    def runtime_info(self) -> dict[str, str]:
        return {
            "attention_backend": "IXFormer",
            "graph_backend": "CUDA Graph compatibility layer",
            "kv_cache_layout": "HND",
            "fused_ops": "IXFormer attention and KV-cache ops",
        }

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

    def graph_slot_mapping_dtype(self):
        return torch.long

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
            use_cuda_graph=context.use_cuda_graph,
        )

    def get_kv_cache_shape(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        return (
            2,
            num_layers,
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
        )
