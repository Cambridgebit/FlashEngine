from abc import ABC, abstractmethod

import torch


class BackendBase(ABC):

    name = "base"

    @abstractmethod
    def store_kvcache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        raise NotImplementedError

    @abstractmethod
    def prefill_attention(
        self,
        q,
        k,
        v,
        context,
        scale,
    ):
        raise NotImplementedError

    @abstractmethod
    def decode_attention(
        self,
        q,
        k_cache,
        v_cache,
        context,
        scale,
    ):
        raise NotImplementedError

    def get_kv_cache_shape(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
    ):
        # 默认 CUDA 布局：
        # [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        return (
            2,
            num_layers,
            num_blocks,
            block_size,
            num_kv_heads,
            head_dim,
        )

    def graph_slot_mapping_dtype(self):
        # CUDA Graph 的 static slot_mapping buffer dtype。
        # 默认 int32（CUDA triton store kernel 的约定 dtype，-1 padding 跳过）；
        # 若 vendor kernel 要求 int64，由 backend override（见 CoreXBackend）。
        return torch.int32

    def all_reduce(self, tensor):
        import torch.distributed as dist

        dist.all_reduce(tensor)
        return tensor
