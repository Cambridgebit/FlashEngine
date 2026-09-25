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
    # CUDA Graph capture 期间置 True（capture 后由 graph.replay() 驱动，
    # 不再走 python）；普通 eager decode 保持 False。
    use_cuda_graph: bool = False
    # KV block 大小：npu 融合算子 decode 路径消费；CUDA 路径无读者，恒传值无害。
    block_size: int = 0
    # 310P fused RoPE: the first attention layer materializes full-dim cos/sin
    # once per model forward; subsequent layers reuse the fixed-address slices.
    rope_cache_key: object | None = None
    rope_cos: torch.Tensor | None = None
    rope_sin: torch.Tensor | None = None
    # Host copy of per-request prefill query lengths. Ascend kernels consume
    # this metadata on the host; retaining it here avoids an NPU->CPU copy (and
    # therefore a full stream synchronization) in every attention layer.
    host_seq_lens_q: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    max_context_len=None,
    use_cuda_graph=False,
    block_size=0,
    host_seq_lens_q=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        max_context_len=max_context_len,
        use_cuda_graph=use_cuda_graph,
        block_size=block_size,
        host_seq_lens_q=host_seq_lens_q,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
