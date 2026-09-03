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

    def all_reduce(self, tensor):
        import torch.distributed as dist

        dist.all_reduce(tensor)
        return tensor
