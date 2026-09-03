import pytest

ixformer = pytest.importorskip("ixformer")

import torch

import ixformer.inference.functions as ops

from nanovllm.backends import get_backend


BLOCK_SIZE = 256
KV_HEADS = 2
HEAD_DIM = 128
BLOCKS = 4


def _make_cache():
    # ixformer paged (HND) layout: [num_blocks, num_kv_heads, block_size, head_dim]
    return torch.zeros(
        BLOCKS,
        KV_HEADS,
        BLOCK_SIZE,
        HEAD_DIM,
        device="cuda",
        dtype=torch.float16,
    )


def _check_rows(key, value, k_cache, v_cache, slots):
    """按 block_idx / block_offset 显式检查（不依赖 view flatten）。

    slot = block_idx * block_size + block_offset，
    第 t 个 token 写入 slot_mapping[t] 对应位置。
    """
    for t in range(key.size(0)):
        slot = int(slots[t])
        b, o = divmod(slot, BLOCK_SIZE)
        torch.testing.assert_close(k_cache[b, :, o, :], key[t])
        torch.testing.assert_close(v_cache[b, :, o, :], value[t])


def test_kv_cache_write():
    """reshape_and_cache_flash 的真实 ABI：slot_mapping 必须为 int64。"""

    tokens = 3

    key = torch.randn(
        tokens,
        KV_HEADS,
        HEAD_DIM,
        device="cuda",
        dtype=torch.float16,
    )

    value = torch.randn_like(key)

    k_cache = _make_cache()
    v_cache = _make_cache()

    slots = torch.tensor(
        [
            0,
            257,
            600,
        ],
        device="cuda",
        dtype=torch.int64,
    )

    ops.reshape_and_cache_flash(
        key,
        value,
        k_cache,
        v_cache,
        slots,
        "auto",
        1.0,
        1.0,
    )

    _check_rows(key, value, k_cache, v_cache, slots)


def test_kv_cache_write_backend_int32_slots():
    """nano-vLLM 引擎输入的是 int32 slot_mapping（Context.slot_mapping）。

    CoreXBackend 内部必须转成 ixformer 需要的 int64。
    """
    backend = get_backend()
    if backend.name != "corex":
        pytest.skip("requires the corex backend")

    tokens = 3

    key = torch.randn(
        tokens,
        KV_HEADS,
        HEAD_DIM,
        device="cuda",
        dtype=torch.float16,
    )

    value = torch.randn_like(key)

    k_cache = _make_cache()
    v_cache = _make_cache()

    # 引擎侧 int32 slot_mapping
    slots = torch.tensor(
        [
            0,
            257,
            600,
        ],
        device="cuda",
        dtype=torch.int32,
    )

    backend.store_kvcache(key, value, k_cache, v_cache, slots)

    _check_rows(key, value, k_cache, v_cache, slots)
