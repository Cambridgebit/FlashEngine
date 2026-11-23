"""vLLM / xgrammar 口径的打包 token 位图（int32 bitmask）与图外 apply。

约定与 xgrammar 一致：bitmask 是 ``int32[rows, ceil(vocab/32)]``，token ``32*word+i``
对应第 ``word`` 个 int32 的第 ``i`` 位，**置 1 = 允许**；apply 时把 0 的位置写 ``-inf``。

- 主机侧：语法机按当前状态填一张 ``[rows, words]`` 的位图（``BitmaskApplier.fill``），
  一次 H2D 拷到设备；
- 设备侧：``apply`` 用一个 Triton kernel 把不允许的位置写 ``-inf``（与 vLLM
  ``_apply_grammar_bitmask_kernel`` 同构；非 CUDA 后端退回纯 torch 展开）。

位图只在图外落：图内采样（整轮 spec 图）拿不到动态 mask，所以带约束的请求走 eager
spec 轮（图外采样），这也是 vLLM 的边界——它的 CUDA Graph 只包前向。
"""

from __future__ import annotations

import numpy as np
import torch

try:  # CUDA 后端才需要；CPU/CoreX/310P 走 torch 回退
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - 依赖环境
    _HAS_TRITON = False

_BLOCK = 8192


def words_for(vocab_size: int) -> int:
    return (int(vocab_size) + 31) // 32


def pack_mask(mask: np.ndarray) -> np.ndarray:
    """bool[vocab] → int32[words]（little-endian 位序，与 xgrammar 一致）。"""
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    words = words_for(mask.shape[0])
    padded = np.zeros(words * 32, dtype=bool)
    padded[:mask.shape[0]] = mask
    packed = np.packbits(padded, bitorder="little")       # uint8[words*4]
    return packed.view(np.int32)                           # 小端主机上按 int32 读


def unpack_mask(packed: np.ndarray, vocab_size: int) -> np.ndarray:
    """int32[words] → bool[vocab]（给测试/回退用）。"""
    packed = np.asarray(packed, dtype=np.int32).reshape(-1)
    bytes_ = packed.view(np.uint8)
    bits = np.unpackbits(bytes_, bitorder="little")
    return bits[:vocab_size].astype(bool)


if _HAS_TRITON:

    @triton.jit
    def _apply_bitmask_kernel(
        logits_ptr, logits_stride,
        bitmask_ptr, bitmask_stride,
        vocab_size,
        BLOCK_SIZE: tl.constexpr,
    ):
        """逐行把 bitmask 的 0 位写成 -inf（行号 = logits 行号）。"""
        row = tl.program_id(0)
        block = tl.program_id(1)
        word_offset = (block * BLOCK_SIZE) // 32 + tl.arange(0, BLOCK_SIZE // 32)
        packed = tl.load(
            bitmask_ptr + row * bitmask_stride + word_offset,
            mask=word_offset < bitmask_stride,
            other=0,
        )
        # bit == 0 → 不允许（tl 里 True 表示要写 -inf）
        blocked = ((packed[:, None] >> tl.arange(0, 32)[None, :]) & 1) == 0
        blocked = blocked.reshape(BLOCK_SIZE)
        offset = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        tl.store(
            logits_ptr + row * logits_stride + offset,
            -float("inf"),
            mask=blocked & (offset < vocab_size),
        )


class BitmaskApplier:
    """持有 ``[max_rows, words]`` 的 int32 设备位图；fill 在主机侧填、apply 在设备侧落。"""

    def __init__(self, max_rows: int, vocab_size: int, device=None,
                 dtype=torch.float32):
        self.max_rows = int(max_rows)
        self.vocab_size = int(vocab_size)
        self.words = words_for(self.vocab_size)
        self.device = torch.device(device) if device is not None else None
        self.dtype = dtype
        self._host = np.full((self.max_rows, self.words), -1, dtype=np.int32)
        # 非 CUDA 后端用「已展开的 bool 允许表」：310P 上 aclnn 的 RightShift 不支持
        # 位图广播（实测 `Shape of out should be [1, 7760*32]` 类 161002），展开后的
        # 形状与 logits 同宽，masked_fill 一条算子即可。
        self._allowed_host = np.ones((self.max_rows, self.vocab_size), dtype=bool)
        self._allowed_device = None
        self._device = None
        self._pinned = None
        self._triton_ready = False

    # -- 设备张量惰性分配 ------------------------------------------------

    def _ensure_device(self, device):
        if self.device is None:
            self.device = torch.device(device)
        if self._device is not None and self._device.device == self.device:
            return
        self._device = torch.empty(
            (self.max_rows, self.words), dtype=torch.int32, device=self.device)
        use_pin = self.device.type == "cuda" and torch.cuda.is_available()
        self._pinned = (torch.empty((self.max_rows, self.words), dtype=torch.int32,
                                    pin_memory=True) if use_pin else None)
        self._triton_ready = bool(
            use_pin and _HAS_TRITON and self.device.type == "cuda")
        self._allowed_device = (
            None if self._triton_ready else
            torch.empty((self.max_rows, self.vocab_size), dtype=torch.bool,
                        device=self.device))

    # -- 填充与 apply -----------------------------------------------------

    def fill(self, rows, device=None) -> int:
        """rows: 每行 None（全允许）或 ``(constraint, state)``；返回行数。"""
        n = len(rows)
        if n > self.max_rows:
            raise RuntimeError(
                f"位图行数 {n} 超过容量 {self.max_rows}；请调大 max_rows")
        host = self._host[:n]
        allowed_host = self._allowed_host[:n]
        for index, row in enumerate(rows):
            if row is None:
                host[index].fill(-1)             # 全部允许
                allowed_host[index].fill(True)
            else:
                constraint, state = row
                packed = constraint.bitmask(state)
                host[index] = packed
                allowed_host[index] = unpack_mask(packed, self.vocab_size)
        self._ensure_device(device)
        src = torch.from_numpy(host)
        if self._pinned is not None:
            self._pinned[:n].copy_(src)
            self._device[:n].copy_(self._pinned[:n], non_blocking=True)
        else:
            self._device[:n].copy_(src)
        if self._allowed_device is not None:
            self._allowed_device[:n].copy_(torch.from_numpy(allowed_host))
        return n

    def apply(self, logits: torch.Tensor, n: int) -> torch.Tensor:
        """返回落好 -inf 的新 logits（不原地改输入：宿主路径可能拿到 inference tensor）。"""
        out = logits.clone()
        if n <= 0:
            return out
        if self._triton_ready:
            grid = (n, triton.cdiv(self.vocab_size, _BLOCK))
            _apply_bitmask_kernel[grid](
                out, out.stride(0),
                self._device, self._device.stride(0),
                self.vocab_size, BLOCK_SIZE=_BLOCK,
            )
            return out
        # 回退：主机侧展开成 bool 允许表（CPU/CoreX/310P 都能跑，行的语义一致）
        out[:n] = out[:n].masked_fill(~self._allowed_device[:n], float("-inf"))
        return out
