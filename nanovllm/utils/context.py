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
    # spec verify 每序列 q 长度（EAGLE3 verify 专用，310P paged_verify 分流标记）：
    # 0=非 verify；>0 时=rows=k+1（每序列 verify q 长度均质，见 model_runner._sg_verify）。
    # 仅 310P 后端消费：attention 层 prefix-hit 分支按本字段 duck-typing 分流到
    # backend.paged_verify；CUDA/CoreX 后端无该方法照走原分支，行为零变化。
    spec_verify_q_lens: int = 0

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, max_context_len=None, use_cuda_graph=False, block_size=0, spec_verify_q_lens=0):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, max_context_len, use_cuda_graph, block_size, spec_verify_q_lens)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
