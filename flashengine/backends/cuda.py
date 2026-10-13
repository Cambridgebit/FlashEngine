import os

import torch
import triton
import triton.language as tl

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
    supports_spec_graph = True
    supports_device_sampling_rng = True
    supports_moe = True  # 专家前向用 BackendBase 的纯 torch 实现（可被 CUDA Graph 捕获）。
    supports_int8_dequant = True  # Quark int8 检查点加载期反量化（无 int8 算子）。

    def __init__(self):
        self.attention_backend = os.getenv("FLASHENGINE_CUDA_ATTENTION", "flash_attn").lower()
        if self.attention_backend not in ("flash_attn", "flashinfer"):
            raise ValueError("FLASHENGINE_CUDA_ATTENTION 必须为 flash_attn 或 flashinfer")
        self.flashinfer = None
        if self.attention_backend == "flashinfer":
            from .flashinfer import FlashInferAttention

            self.flashinfer = FlashInferAttention()
            # 投机图在图内改变上下文；FlashInfer 的 CPU plan 尚未接入整轮图。
            self.supports_spec_graph = False
        else:
            from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache

            self._varlen = flash_attn_varlen_func
            self._with_kvcache = flash_attn_with_kvcache

    def runtime_info(self):
        info = super().runtime_info()
        info["attention_backend"] = self.attention_backend
        if self.flashinfer is not None:
            info["attention_kernel"] = "FlashInfer fa2 (NHD, tensor cores)"
            info["spec_graph"] = "disabled: eager spec with CPU plan"
        return info

    def create_decode_graph_executor(self, plan):
        if self.flashinfer is not None:
            from .flashinfer import FlashInferDecodeGraphRunner

            return FlashInferDecodeGraphRunner(plan, self.flashinfer)
        return None

    def shutdown(self):
        if self.flashinfer is not None:
            self.flashinfer.close()
        super().shutdown()

    def validate_config(self, config) -> None:
        if self.flashinfer is not None:
            if config.tensor_parallel_size != 1:
                raise NotImplementedError("FlashInfer 注意力适配当前要求 tensor_parallel_size=1")
            if config.spec_model and not config.enforce_eager:
                print("[FlashEngine] FlashInfer 投机解码使用 eager；整轮投机图尚未接入。", flush=True)
        elif config.kvcache_block_size % 256:
            raise ValueError(
                "CUDA flash-attn paged KV requires block_size to be a power "
                "of two and a multiple of 256 (256, 512, 1024, ...)."
            )

    def graph_capture_block(self, num_blocks: int) -> int:
        return 0

    def graph_capture_context_len(self) -> int:
        return 1

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
        if self.flashinfer is not None:
            return self.flashinfer.prefill(q, k, v, context, scale)
        return self._varlen(
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
        if self.flashinfer is not None:
            return self.flashinfer.decode(q, k_cache, v_cache, context, scale)
        if context.use_cuda_graph:
            if context.cu_seqlens_q is None or context.cu_seqlens_k is None:
                raise RuntimeError(
                    "CUDA Graph decode requires cumulative KV-length metadata"
                )
            return self._varlen(
                q,
                k_cache,
                v_cache,
                cu_seqlens_q=context.cu_seqlens_q,
                cu_seqlens_k=context.cu_seqlens_k,
                max_seqlen_q=1,
                max_seqlen_k=context.max_context_len,
                softmax_scale=scale,
                causal=True,
                block_table=context.block_tables,
            )
        return self._with_kvcache(
            q.unsqueeze(1),
            k_cache,
            v_cache,
            cache_seqlens=context.context_lens,
            block_table=context.block_tables,
            softmax_scale=scale,
            causal=True,
        )
