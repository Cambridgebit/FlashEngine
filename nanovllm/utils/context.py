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
    # ---- DFlash verify 的 GDN 'row' 模式（阶段2）----
    # gdn_state_mode：None/'final' = 现状语义（prefill 只写末态 / decode 单步读写
    # 同槽）；'row' = verify 逐行落槽（段内每行终态写 状态槽 = gdn_state_slots[i]
    # + 行内偏移，行 0 从 gdn_carry_slots[i] 读初始态）。
    # gdn_carry_slots：int32 [n_seqs]，'row' 模式行 0 的读槽（'final' 模式忽略）。
    # qwen3 / 非 spec hybrid 路径不设置（恒 None），行为不变。
    gdn_state_mode: str | None = None
    gdn_carry_slots: torch.Tensor | None = None
    # ---- DFlash verify 'row' 模式的图捕获静态段（阶段3）----
    # gdn_row_bases：host list[int]，图捕获期各序列的写基址（= max_num_seqs +
    # i*(k+1)，按池静态布局、只依赖批内序号 i 与桶常量——bake 进图即正确，replay
    # 时 python 不再运行）；gdn_row_seg：int，图捕获期段长（= k+1，常量）。
    # 二者非 None/非零时 _forward_rows 走静态模式（跳过 cu_seqlens/槽位的
    # tolist host 同步——图捕获期禁止 D2H）；eager 路径不设置（恒 None），
    # 循环体逐位一致，行为零变化。
    gdn_row_bases: list | None = None
    gdn_row_seg: int = 0
    # 310P fused RoPE: the first attention layer materializes full-dim cos/sin
    # once per model forward; subsequent layers reuse the fixed-address slices.
    rope_cache_key: object | None = None
    rope_cos: torch.Tensor | None = None
    rope_sin: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, max_context_len=None, use_cuda_graph=False, block_size=0, gdn_state_slots=None, gdn_has_initial=None, gdn_state_mode=None, gdn_carry_slots=None, gdn_row_bases=None, gdn_row_seg=0):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, max_context_len, use_cuda_graph, block_size, gdn_state_slots, gdn_has_initial, gdn_state_mode, gdn_carry_slots, gdn_row_bases, gdn_row_seg)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
