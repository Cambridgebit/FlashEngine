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

    # torch.compile 时保持 eager（graph-break）：对齐 vLLM PIECEWISE 思路，
    # IXFormer/flash-attn 自定义算子在 dynamo 下只作 fallback，不参与编译融合。
    @torch.compiler.disable

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

            if context.block_tables is not None:    # prefix cache / spec verify
                # 310P spec verify 专径（T13）：verify 与 prefix-hit 同为 prefill 形态
                # + 恒传 block_tables，但其分页 KV 消费形态 TND prefill 算子不接受
                # ——后端实现 paged_verify（duck-typing 探测，CUDA/CoreX 无此方法）
                # 且本步确为 verify（spec_verify_q_lens>0，_sg_verify 组装时传入）时
                # 分流；普通 prefix-hit prefill 行为不变（310P 由
                # supports_prefix_cache=False 全局排除，不会到达此分支）。
                if hasattr(self.backend, "paged_verify") and context.spec_verify_q_lens > 0:
                    return self.backend.paged_verify(q, k_cache, v_cache, context, self.scale)
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
