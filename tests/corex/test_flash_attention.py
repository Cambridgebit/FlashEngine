import pytest

ixformer = pytest.importorskip("ixformer")

import torch

from nanovllm.backends import get_backend
from nanovllm.utils.context import Context


# ixformer paged (HND) layout：
# k_cache / v_cache 为 [num_blocks, num_kv_heads, block_size, head_dim]，
# slot = block * block_size + token_in_block。
HEAD_DIM = 128
BLOCK_SIZE = 256
DTYPE = torch.float16


def _dense_attention(q, k, v, scale, mask):
    """参考实现：fp32 计算，支持 GQA。

    q: [Q, num_heads, head_dim]
    k / v: [L, num_kv_heads, head_dim]
    mask: [Q, L] bool，True 表示允许 attend
    """
    qf = q.float()
    kf = k.float()
    vf = v.float()

    rep = qf.size(1) // kf.size(1)
    assert qf.size(1) % kf.size(1) == 0
    if rep > 1:
        kf = kf.repeat_interleave(rep, dim=1)
        vf = vf.repeat_interleave(rep, dim=1)

    assert qf.size(0) == mask.size(0)
    assert kf.size(0) == mask.size(1)

    # 显式 batched matmul（H 在前），避免 einsum 隐式行为差异
    # scores: [H, Q, L]
    scores = torch.bmm(
        qf.transpose(0, 1).contiguous(),
        kf.transpose(0, 1).transpose(-1, -2).contiguous(),
    )
    scores = scores.transpose(0, 1) * scale    # [Q, H, L]

    mask_b = mask[:, None, :].expand(scores.shape)
    scores = scores.masked_fill(~mask_b, float("-inf"))
    probs = torch.softmax(scores, dim=-1)    # fp32 softmax

    # 最终输出恢复为输入 Attention dtype（v.dtype = fp16），
    # 与 IXFormer 的输出 dtype 一致（assert_close 默认检查 dtype）
    out = torch.bmm(
        probs.transpose(0, 1).contiguous(),
        vf.transpose(0, 1).contiguous(),
    ).transpose(0, 1)

    return out.to(v.dtype)


def _causal_mask(cu_seqlens_q, cu_seqlens_k, device):
    """prefill 的 causal mask（与 FA 的 bottom-right 对齐语义一致）。

    q 行 i（属于第 s 个 segment，长度 q_len = qe-qs）对应 k segment
    [ks, ke)（长度 k_len = ke-ks）。FA 的 causal 对齐方式是底部右对齐：
    q 行 i 可 attend 的 k 行范围为
        [ks, ks + (i - qs) + (k_len - q_len)]
    即 segment 等长时 attend [ks, i]；prefix cache 场景 k_len > q_len 时，
    新 token 额外 attend 缓存的 prefix（全局位置 ≤ ks + (i - qs) + (k_len - q_len)）。
    """
    Q = int(cu_seqlens_q[-1])
    L = int(cu_seqlens_k[-1])
    mask = torch.zeros(Q, L, dtype=torch.bool, device=device)
    for s in range(len(cu_seqlens_q) - 1):
        qs, qe = int(cu_seqlens_q[s]), int(cu_seqlens_q[s + 1])
        ks, ke = int(cu_seqlens_k[s]), int(cu_seqlens_k[s + 1])
        offset = (ke - ks) - (qe - qs)
        for i in range(qs, qe):
            hi = min(ke, ks + (i - qs) + offset + 1)
            mask[i, ks:hi] = True
    return mask


def _fill_block(cache, block, rows):
    """把连续 rows [L, kv_heads, head_dim] 写入 block 的 offset 0..L-1。"""
    rows = rows.permute(1, 0, 2)    # [kv_heads, L, head_dim]
    cache[block, :, : rows.size(1)] = rows
    return cache


def _cache_rows(cache, block_table, length):
    """按 block_table 把某条 sequence 的前 length 行取出来（CoreX HND layout）。"""
    chunks = []
    remaining = length
    for block in block_table.flatten().tolist():
        if remaining <= 0:
            break
        take = min(remaining, BLOCK_SIZE)
        chunk = cache[block, :, :take, :].permute(1, 0, 2)    # [take, kv_heads, head_dim]
        chunks.append(chunk)
        remaining -= take
    assert remaining == 0
    return torch.cat(chunks, dim=0)


def _test_backend():
    if get_backend().name != "corex":
        pytest.skip("requires the corex backend")
    return get_backend()


def test_prefill_varlen_multi_seq():
    """不同 context length 的多个 sequence 的 varlen prefill（无 prefix cache）。"""
    backend = _test_backend()
    scale = 1.0 / (HEAD_DIM ** 0.5)

    num_heads, num_kv_heads = 8, 2
    lengths = [13, 37]
    total = sum(lengths)

    q = torch.randn(total, num_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(total, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn_like(k)

    # cu_seqlens 必须是 cumulative sum：lengths=[13, 37] -> [0, 13, 50]
    cu_list = [0]
    for length in lengths:
        cu_list.append(cu_list[-1] + length)
    cu_seqlens = torch.tensor(cu_list, dtype=torch.int32, device="cuda")
    context = Context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths),
        block_tables=None,
    )

    out = backend.prefill_attention(q, k, v, context, scale)
    mask = _causal_mask(cu_seqlens, cu_seqlens, "cuda")
    ref = _dense_attention(q, k, v, scale, mask)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_decode_paged_gqa_cross_block():
    """paged decode：GQA + 跨 block + 不同 context length。"""
    backend = _test_backend()
    scale = 1.0 / (HEAD_DIM ** 0.5)

    num_heads, num_kv_heads = 8, 2
    num_blocks = 8

    lengths = [300, 5]    # 300 > block_size，覆盖两个 block
    # seq0 用 block 3/7，seq1 用 block 1 —— 故意错开，验证 block_table 语义
    block_tables = torch.tensor([[3, 7], [1, -1]], dtype=torch.int32, device="cuda")

    k_cache = torch.zeros(num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM, dtype=DTYPE, device="cuda")
    v_cache = torch.zeros_like(k_cache)

    for seq_id, (length, blocks) in enumerate(zip(lengths, [[3, 7], [1]])):
        seq_k = torch.randn(length, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
        seq_v = torch.randn_like(seq_k)
        offset = 0
        for block in blocks:
            take = min(length - offset, BLOCK_SIZE)
            _fill_block(k_cache, block, seq_k[offset : offset + take])
            _fill_block(v_cache, block, seq_v[offset : offset + take])
            offset += take

    context_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    q = torch.randn(len(lengths), num_heads, HEAD_DIM, dtype=DTYPE, device="cuda")

    context = Context(
        is_prefill=False,
        context_lens=context_lens,
        block_tables=block_tables,
        max_context_len=max(lengths),
    )
    out = backend.decode_attention(q, k_cache, v_cache, context, scale)

    ref = torch.empty_like(out)
    for seq_id, length in enumerate(lengths):
        seq_k = _cache_rows(k_cache, block_tables[seq_id], length)
        seq_v = _cache_rows(v_cache, block_tables[seq_id], length)
        mask = torch.ones(1, length, dtype=torch.bool, device="cuda")
        ref[seq_id : seq_id + 1] = _dense_attention(
            q[seq_id : seq_id + 1], seq_k, seq_v, scale, mask
        )

    # ixformer 的 paged decode 实测输出 fp32（此前 fp32 参考即通过），
    # 而 prefill 输出 fp16 —— 在比较侧统一对齐到 kernel 实际输出 dtype
    if ref.dtype != out.dtype:
        ref = ref.to(out.dtype)
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)


def test_prefill_with_prefix_cache_block_table():
    """prefix cache prefill：已有 20 token 缓存 + 本次 8 个新 token。

    与 nano-vLLM 行为一致：先 store 新 token 的 KV，再对整段
    (cached + new) KV 用 block_table 做 varlen attention。
    """
    backend = _test_backend()
    scale = 1.0 / (HEAD_DIM ** 0.5)

    num_heads, num_kv_heads = 8, 2
    num_blocks = 8
    cached_len, new_len = 20, 8
    length = cached_len + new_len
    block = 5

    k_cache = torch.zeros(num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM, dtype=DTYPE, device="cuda")
    v_cache = torch.zeros_like(k_cache)

    # 上一步已缓存的 prefix（offset 0..19）
    prefix_k = torch.randn(cached_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    prefix_v = torch.randn_like(prefix_k)
    _fill_block(k_cache, block, prefix_k)
    _fill_block(v_cache, block, prefix_v)

    # 本次 step 的新 token，先按 nano-vLLM 方式 store 进 cache（引擎侧 int32 slot）
    new_k = torch.randn(new_len, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    new_v = torch.randn_like(new_k)
    slot_mapping = block * BLOCK_SIZE + torch.arange(cached_len, length, dtype=torch.int32, device="cuda")
    backend.store_kvcache(new_k, new_v, k_cache, v_cache, slot_mapping)

    cu_seqlens_q = torch.tensor([0, new_len], dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.tensor([0, length], dtype=torch.int32, device="cuda")
    block_tables = torch.tensor([[block]], dtype=torch.int32, device="cuda")
    context = Context(
        is_prefill=True,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=new_len,
        max_seqlen_k=length,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
    )

    # q 只包含本次的新 token；k/v 走 cache（与 Attention.forward 一致）
    q = torch.randn(new_len, num_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    out = backend.prefill_attention(q, k_cache, v_cache, context, scale)

    # 参考：cached prefix 与 store 后的 new KV 拼接成完整 dense KV
    full_k = _cache_rows(k_cache, block_tables[0], length)
    full_v = _cache_rows(v_cache, block_tables[0], length)
    assert torch.equal(full_k[:cached_len], prefix_k)
    assert torch.equal(full_k[cached_len:], new_k)

    mask = _causal_mask(cu_seqlens_q, cu_seqlens_k, "cuda")
    ref = _dense_attention(q, full_k, full_v, scale, mask)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
