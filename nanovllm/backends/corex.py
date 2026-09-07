import torch

import ixformer.inference.functions as ixf_ops

from ixformer.contrib.vllm_flash_attn import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
)

from .base import BackendBase


class CoreXBackend(BackendBase):

    name = "corex"
    # 厂商 torch 提供 NCCL 兼容层，V1.3 实测路径；TP all_reduce 暂继承 base 的
    # NCCL 路径（ixformer.distributed 等单卡/NCCL 路径稳定后，再经
    # NANOVLLM_COREX_USE_IXFD=1 切换）。
    dist_name = "nccl"
    # spec 图捕获 CoreX 实测可成功（2026-09 实机验证：单图整轮一次通过，
    # E=2.050 / TPS 103.7）；尝试失败仍自动降级 eager spec 轮（见
    # capture_spec_cudagraph 的 except）。
    supports_spec_graph = True

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

    def graph_slot_mapping_dtype(self):
        # CUDA Graph 的 static slot_mapping buffer 直接用 int64，
        # 让 reshape_and_cache_flash 在 capture 区间内不需要 dtype 转换
        # （dtype 转换发生在 graph.replay() 之前的 runtime → static copy）。
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
