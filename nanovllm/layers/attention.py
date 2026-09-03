import torch
from torch import nn

from nanovllm.backends import get_backend
from nanovllm.utils.context import get_context


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

        self.k_cache = self.v_cache = torch.tensor([])

        self.backend = get_backend()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        context = get_context()

        k_cache = self.k_cache
        v_cache = self.v_cache

        if k_cache.numel() and v_cache.numel():

            self.backend.store_kvcache(
                k,
                v,
                k_cache,
                v_cache,
                context.slot_mapping,
            )

        if context.is_prefill:

            if context.block_tables is not None:    # prefix cache
                k = k_cache
                v = v_cache

            return self.backend.prefill_attention(
                q,
                k,
                v,
                context,
                self.scale,
            )

        return self.backend.decode_attention(
            q,
            k_cache,
            v_cache,
            context,
            self.scale,
        )
