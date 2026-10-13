from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    max_context_len: int | None = None
    use_cuda_graph: bool = False
    block_size: int = 0
    rope_cache_key: object | None = None
    rope_cos: torch.Tensor | None = None
    rope_sin: torch.Tensor | None = None
    host_seq_lens_q: torch.Tensor | None = None
    # qlens/mask、展开后的 per-query PA 表）。这些量只由 set_context 传入的
    attn_scratch: dict | None = None
    prefill_cache_hit: bool = False
    prefix_cache_hit_blocks: int = 0
    prefill_query_tokens: int = 0
    prefix_reused_tokens: int = 0
    splitfuse_used: bool = False
    prefix_recomputed: bool = False

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    *,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_context_len=None,
    use_cuda_graph=False,
    block_size=0,
    host_seq_lens_q=None,
    prefill_cache_hit=False,
    prefix_cache_hit_blocks=0,
    prefill_query_tokens=0,
    prefix_reused_tokens=0,
    prefix_recomputed=False,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_context_len=max_context_len,
        use_cuda_graph=use_cuda_graph,
        block_size=block_size,
        host_seq_lens_q=host_seq_lens_q,
        prefill_cache_hit=prefill_cache_hit,
        prefix_cache_hit_blocks=prefix_cache_hit_blocks,
        prefill_query_tokens=prefill_query_tokens,
        prefix_reused_tokens=prefix_reused_tokens,
        prefix_recomputed=prefix_recomputed,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
