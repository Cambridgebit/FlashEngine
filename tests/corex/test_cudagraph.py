"""CoreX Attention CUDA Graph 一致性测试（CUDA Graph 适配指导 §22）。

capture 区间 = store_kvcache + decode_attention，即引擎 decode step 的
等价物（model forward 在 capture 期间会执行这两步）：

- Decode
- GQA（8 heads / 2 kv heads）
- Paged KV cache（HND layout，同一地址跨 replay 复用，不复制）
- static slot_mapping（backend dtype：int64，capture 内不做 dtype 转换；
  runtime int32 → static int64 的转换发生在 replay 前的 copy）
- static block_table / context_lens / max_context_len

每次 replay 前只更新 static buffer 内容（copy_ / fill_），然后
graph.replay()；逐轮与 eager 路径 + dense fp32 参考比对。
"""
import pytest

ixformer = pytest.importorskip("ixformer")

import torch

from nanovllm.backends import get_backend
from nanovllm.utils.context import Context


# ixformer paged (HND) layout：
# k_cache / v_cache 为 [num_blocks, num_kv_heads, block_size, head_dim]
HEAD_DIM = 128
BLOCK_SIZE = 256
DTYPE = torch.float16
# capture 时固定的静态上界（对应引擎的 config.max_model_len）
MAX_CONTEXT_LEN = 512


def _dense_attention(q, k, v, scale, mask):
    """参考实现：fp32 计算，支持 GQA（与 test_flash_attention.py 相同 helper）。"""
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

    scores = torch.bmm(
        qf.transpose(0, 1).contiguous(),
        kf.transpose(0, 1).transpose(-1, -2).contiguous(),
    ).transpose(0, 1) * scale    # [Q, H, L]

    mask_b = mask[:, None, :].expand(scores.shape)
    scores = scores.masked_fill(~mask_b, float("-inf"))
    probs = torch.softmax(scores, dim=-1)

    out = torch.bmm(
        probs.transpose(0, 1).contiguous(),
        vf.transpose(0, 1).contiguous(),
    ).transpose(0, 1)

    return out.to(v.dtype)


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


def test_decode_gqa_paged_cuda_graph():
    backend = _test_backend()
    scale = 1.0 / (HEAD_DIM ** 0.5)

    num_heads, num_kv_heads = 8, 2
    num_blocks = 8
    bs = 2

    # 初始已缓存：seq0 = 300 token（block 3 满 256 + block 7 前 44），
    # seq1 = 5 token（block 1）。每轮 decode step 各 append 1 个新 token。
    cache_lens = [300, 5]
    block_tables = torch.tensor([[3, 7], [1, -1]], dtype=torch.int32, device="cuda")

    k_cache = torch.zeros(num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM, dtype=DTYPE, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    for seq_id, (length, blocks) in enumerate(zip(cache_lens, [[3, 7], [1]])):
        seq_k = torch.randn(length, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
        seq_v = torch.randn_like(seq_k)
        offset = 0
        for block in blocks:
            take = min(length - offset, BLOCK_SIZE)
            _fill_block(k_cache, block, seq_k[offset : offset + take])
            _fill_block(v_cache, block, seq_v[offset : offset + take])
            offset += take

    # ---- CUDA Graph static buffers（地址 capture 后固定，只能 copy_ 内容）----
    # slot_mapping 直接用 backend.graph_slot_mapping_dtype()（CoreX: int64），
    # capture 内不做 dtype 转换。
    slot_static = torch.full((bs,), -1, dtype=backend.graph_slot_mapping_dtype(), device="cuda")
    key_static = torch.zeros(bs, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    value_static = torch.zeros_like(key_static)
    q_static = torch.zeros(bs, num_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
    context_lens_static = torch.zeros(bs, dtype=torch.int32, device="cuda")
    block_tables_static = torch.full(
        (bs, block_tables.size(1)), -1, dtype=torch.int32, device="cuda"
    )
    block_tables_static.copy_(block_tables)
    # ixformer 的 flash_attn_with_kvcache 契约：q 4D 进 → 4D 出，
    # 输出永远带 seqlen 维 [bs, seqlen, nheads, headdim]，且内部用
    # empty_like(q) 作为输出 buffer → dtype 跟随 q（fp16）。
    # static output buffer 必须按 kernel 的真实输出 shape/dtype 分配。
    out_static = torch.zeros(bs, 1, num_heads, HEAD_DIM, dtype=DTYPE, device="cuda")

    # Graph Context（use_cuda_graph=True）与 eager Context（False）
    # 共享同一批 static tensor，仅 flag 不同。
    graph_context = Context(
        is_prefill=False,
        slot_mapping=slot_static,
        context_lens=context_lens_static,
        block_tables=block_tables_static,
        max_context_len=MAX_CONTEXT_LEN,
        use_cuda_graph=True,
    )
    eager_context = Context(
        is_prefill=False,
        slot_mapping=slot_static,
        context_lens=context_lens_static,
        block_tables=block_tables_static,
        max_context_len=MAX_CONTEXT_LEN,
        use_cuda_graph=False,
    )

    # 引擎 runtime（int32 slots）→ graph static（int64）的转换发生在 copy 里
    def update_step(q, keys, values, slots, lens):
        q_static.copy_(q)
        key_static.copy_(keys)
        value_static.copy_(values)
        slot_static.copy_(slots)
        context_lens_static.copy_(lens)

    def eager_step():
        backend.store_kvcache(key_static, value_static, k_cache, v_cache, slot_static)
        return backend.decode_attention(q_static, k_cache, v_cache, eager_context, scale)

    def captured_step():
        backend.store_kvcache(key_static, value_static, k_cache, v_cache, slot_static)
        out_static.copy_(backend.decode_attention(q_static, k_cache, v_cache, graph_context, scale))

    def dense_decode(lens):
        """从 cache 读回完整 KV，逐 seq 做 dense 参考。

        lens 必须是 python int 列表（比较 / 切片 / torch.ones 维度
        都按 int 语义处理；0-dim CUDA tensor 会破坏这些操作）。
        """
        lens = [int(length) for length in lens]
        ref = torch.empty(bs, num_heads, HEAD_DIM, dtype=DTYPE, device="cuda")
        for seq_id, length in enumerate(lens):
            seq_k = _cache_rows(k_cache, block_tables_static[seq_id], length)
            seq_v = _cache_rows(v_cache, block_tables_static[seq_id], length)
            mask = torch.ones(1, length, dtype=torch.bool, device="cuda")
            ref[seq_id : seq_id + 1] = _dense_attention(
                q_static[seq_id : seq_id + 1], seq_k, seq_v, scale, mask
            )
        return ref

    def assert_store_wrote(keys, values, slots):
        """graph replay 的 store 必须真的写进 KV cache 对应 slot。"""
        for t in range(bs):
            b, o = divmod(int(slots[t]), BLOCK_SIZE)
            torch.testing.assert_close(k_cache[b, :, o, :], keys[t], atol=0, rtol=0)
            torch.testing.assert_close(v_cache[b, :, o, :], values[t], atol=0, rtol=0)

    # 两个 decode step：每 step 每 seq 追加 1 个 token
    steps = []
    for step_id, lens in enumerate(([301, 6], [302, 7])):
        # 续写位置 = 各 seq 最后一个物理 block 的下一行：
        # seq0 的续写块是 block 7（300 token 占满 block 3 后落到 block 7），
        # 续写行 44/45；seq1 的续写块是 block 1，续写行 5/6。
        rows_in_block = [44 + step_id, 5 + step_id]
        slots = []
        for seq_id, row_in_block in enumerate(rows_in_block):
            bt_row = block_tables[seq_id]
            last_block = int(bt_row[(bt_row != -1).nonzero()[-1]])
            slots.append(last_block * BLOCK_SIZE + row_in_block)
        steps.append({
            # context_lens static buffer 是 int32 tensor：这里直接给 tensor，
            # 避免 copy_ 收到 python list
            "lens": torch.tensor(lens, dtype=torch.int32, device="cuda"),
            "slots": torch.tensor(slots, dtype=torch.int32, device="cuda"),    # 引擎 runtime dtype
            "q": torch.randn(bs, num_heads, HEAD_DIM, dtype=DTYPE, device="cuda"),
            "keys": torch.randn(bs, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"),
            "values": torch.randn(bs, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda"),
        })

    # ---- Warmup（第一 step 内容）：vendor kernel lazy init / workspace
    # 必须在正式 capture 前完成，不能在 capture 区间内发生。----
    s0 = steps[0]
    update_step(s0["q"], s0["keys"], s0["values"], s0["slots"], s0["lens"])
    eager_step()
    torch.cuda.synchronize()

    # ---- Capture（capture 区间的 python 与 warmup 完全一致）----
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_step()
    torch.cuda.synchronize()

    # ---- 逐轮 replay：eager vs graph vs dense 三方一致 ----
    for step in steps:
        update_step(step["q"], step["keys"], step["values"], step["slots"], step["lens"])

        # vendor kernel 输出带 seqlen 维 [bs, 1, h, d]：取 [:, 0] 后与 3D 参考对齐
        eager_out = eager_step()[:, 0]      # eager：store + decode（graph 外）
        graph.replay()                      # graph：capture 的 store + decode
        graph_out = out_static[:, 0].clone()

        ref = dense_decode(step["lens"])
        if ref.dtype != eager_out.dtype:
            ref = ref.to(eager_out.dtype)

        # graph 输出必须与 eager 一致，且与 dense 参考一致
        torch.testing.assert_close(graph_out, eager_out, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(graph_out, ref, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(eager_out, ref, atol=1e-2, rtol=1e-2)

        # store 落盘校验：本轮 slot 行 == 本轮 key/value
        assert_store_wrote(step["keys"], step["values"], step["slots"])
