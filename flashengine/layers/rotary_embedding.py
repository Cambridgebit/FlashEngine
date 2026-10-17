from weakref import WeakValueDictionary

import torch
from torch import nn

from flashengine.backends import get_backend


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        self.backend = get_backend()
        inv_freq = 1.0 / (
            base
            ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        positions = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", positions, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.backend.rope(positions, query, key, self.cos_sin_cache)


_ROPE_CACHE = WeakValueDictionary()


def _cached_rope(cls, *args):
    # 同模型各层共享，但缓存不延长设备张量寿命，也不跨设备/类型复用。
    key = (cls, args, str(torch.get_default_device()), torch.get_default_dtype())
    rope = _ROPE_CACHE.get(key)
    if rope is None:
        rope = cls(*args)
        _ROPE_CACHE[key] = rope
    return rope


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    return _cached_rope(RotaryEmbedding, head_size, rotary_dim, max_position, base)


class PartialRotaryEmbedding(nn.Module):
    """部分旋转 rope（Qwen3.5 text-only 口径）：前 rotary_dim 维 NeoX 旋转，其余直通。

    实现方式选型（方案 i，kernel 变体）而非加载期权重置换，理由：
    1. 检查点/transformers 本就是 NeoX 半分配对（modeling_qwen3_5.py apply_rotary_pos_emb
       用 rotate_half + cat((freqs, freqs))），q/k 权重无需任何交织→半分置换；
    2. q_norm/k_norm 作用在完整 head_dim 且先 norm 后 rope，若置换 q/k 权重的 64 维
       rotary 段，norm 权重虽是逐头标量不受影响，但置换器与权重加载顺序耦合、增加出错面；
    3. 直接复用 backend.rope 融合点（Ascend 等后端覆盖 rope 时自动受益）。
    mrope_interleaved 说明：该开关只影响多模态 T/H/W 三个位置行的频率交织
    （transformers apply_interleaved_mrope / vllm MRotaryEmbeddingInterleaved），
    text-only 三行位置相同，交织为恒等变换——text-only 语义 = 本类的普通 partial rope。
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert 0 < rotary_dim <= head_size, f"rotary_dim={rotary_dim} 越界（head_size={head_size}）"
        self.rotary_dim = rotary_dim
        self.backend = get_backend()
        # 缓存布局与现有 RotaryEmbedding 一致（cos | sin 拼接，positions 索引行），
        # 仅频率数 = rotary_dim // 2（Qwen3.5-2B：head 256 × partial 0.25 → 64 维旋转、32 频率）。
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 只旋转前 rotary_dim 维（transformers apply_rotary_pos_emb：q_rot/q_pass 切分），
        # 其余维度直通；q/k 同处理、v/o 不动。
        q_rot = query[..., :self.rotary_dim].contiguous()
        k_rot = key[..., :self.rotary_dim].contiguous()
        q_rot, k_rot = self.backend.rope(positions, q_rot, k_rot, self.cos_sin_cache)
        query = torch.cat((q_rot, query[..., self.rotary_dim:]), dim=-1)
        key = torch.cat((k_rot, key[..., self.rotary_dim:]), dim=-1)
        return query, key


def get_partial_rope(head_size, rotary_dim, max_position, base):
    return _cached_rope(PartialRotaryEmbedding, head_size, rotary_dim, max_position, base)
