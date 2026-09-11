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
    # ---- hybrid（Qwen3.5）GDN 线性注意力的每序列状态槽元数据 ----
    # gdn_state_slots：int32 [n_seqs]，各序列在 GDN 状态池中的槽号（所有 GDN 层
    # 共用同一槽号，模型侧按 [layer, slot] 索引池切片）；
    # gdn_has_initial：bool [n_seqs]，True=从槽读初始状态（decode 单行消费主槽），
    # False=零初始化（prefill 终态写槽；preempt 重 prefill 同口径重建）。
    # qwen3 路径不设置（恒 None），注意力层无读者，行为不变。
    gdn_state_slots: torch.Tensor | None = None
    gdn_has_initial: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, max_context_len=None, use_cuda_graph=False, block_size=0, gdn_state_slots=None, gdn_has_initial=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, max_context_len, use_cuda_graph, block_size, gdn_state_slots, gdn_has_initial)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
