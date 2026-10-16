"""Ascend 310P W8A8 int8 量化线性层（Qwen3 dense / Qwen3 MoE 目标）。

仅支持单卡（TP=1）：量化伴生参数（weight_scale/weight_offset/quant_bias/deq_scale/
input_scale/input_offset）按整表装载，不做分片；TP>1 由 support_matrix 拒绝。

方案口径对齐 vLLM-Ascend 310P（_310p/quantization/methods/w8a8_*.py）：
- "W8A8"（静态，attention qkv/o）：per-tensor 激活量化（npu_quantize + 倒数尺度）
  + per-channel 权重；运行时 npu_quant_matmul(weight, deq_scale, bias=quant_bias)。
- "W8A8_DYNAMIC"（动态，MoE 专家/可选线性）：npu_dynamic_quant 逐 token 量化激活 +
  npu_quant_matmul(weight, weight_scale, pertoken_scale=)。

实现方式：`quantized_class(base_cls, scheme)` 生成 LinearBase 子类——复用父类的
TP 装载/mapping 逻辑（权重与尺度挂同一 weight_loader），只替换权重 dtype 与 forward。
这样 qwen3.py / dflash.py 的线性层构造代码零改动。
"""

from functools import lru_cache

import torch
from torch import nn

from flashengine.backends import get_backend

W8A8_STATIC = "W8A8"
W8A8_DYNAMIC = "W8A8_DYNAMIC"
SUPPORTED_SCHEMES = (W8A8_STATIC, W8A8_DYNAMIC)


def _ignore_shard_loader(param, loaded_weight, shard_id=None):
    """per-tensor 参数（input_scale/input_offset）的装载器：忽略 packed 分片号。"""
    param.data.copy_(loaded_weight)


def _attach(param, loader):
    # 尺度类参数复用权重分片装载器（TP=1 下即整表拷贝；保留分片语义以便未来 TP）。
    param.weight_loader = loader


def _setup_quant_params(module, scheme: str) -> None:
    if scheme not in SUPPORTED_SCHEMES:
        raise NotImplementedError(f"Unsupported Ascend W8A8 scheme: {scheme}")
    weight = module.weight
    out_size, in_size = weight.shape
    out_dtype = weight.dtype
    weight_loader = getattr(weight, "weight_loader", None)
    # 非浮点 dtype 的 Parameter 必须显式 requires_grad=False（int8/int32/int64 不可求导）。
    module.weight = nn.Parameter(torch.empty(out_size, in_size, dtype=torch.int8),
                                 requires_grad=False)
    if weight_loader is not None:
        _attach(module.weight, weight_loader)
    module.quant_scheme_name = scheme
    module.quant_out_dtype = out_dtype
    if scheme == W8A8_STATIC:
        module.weight_scale = nn.Parameter(torch.empty(out_size, 1, dtype=out_dtype))
        module.weight_offset = nn.Parameter(torch.empty(out_size, 1, dtype=out_dtype))
        module.quant_bias = nn.Parameter(torch.empty(out_size, dtype=torch.int32),
                                         requires_grad=False)
        module.deq_scale = nn.Parameter(torch.empty(out_size, dtype=torch.int64),
                                        requires_grad=False)
        # per-tensor 激活参数（整表，不分片）。
        module.input_scale = nn.Parameter(torch.empty(1, dtype=out_dtype))
        module.input_offset = nn.Parameter(torch.empty(1, dtype=torch.int8),
                                           requires_grad=False)
        for name in ("weight_scale", "weight_offset", "quant_bias", "deq_scale"):
            if weight_loader is not None:
                _attach(getattr(module, name), weight_loader)
        _attach(module.input_scale, _ignore_shard_loader)
        _attach(module.input_offset, _ignore_shard_loader)
    else:
        module.weight_scale = nn.Parameter(torch.empty(out_size, 1, dtype=torch.float32))
        module.weight_offset = nn.Parameter(torch.empty(out_size, 1, dtype=torch.float32))
        if weight_loader is not None:
            _attach(module.weight_scale, weight_loader)
            _attach(module.weight_offset, weight_loader)


def _quant_forward(module, x: torch.Tensor) -> torch.Tensor:
    backend = get_backend()
    if module.quant_scheme_name == W8A8_STATIC:
        return backend.quant_linear_static(
            x, module.weight, module.deq_scale, module.quant_bias,
            module.aclnn_input_scale, module.aclnn_input_scale_reciprocal,
            module.aclnn_input_offset, module.quant_out_dtype, module.bias,
        )
    return backend.quant_linear_dynamic(
        x, module.weight, module.weight_scale, module.quant_out_dtype, module.bias,
    )


@lru_cache(maxsize=None)
def quantized_class(base_cls, scheme: str):
    """生成 base_cls 的 W8A8 子类（按 (base_cls, scheme) 去重）。"""

    class _QuantLinear(base_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            _setup_quant_params(self, scheme)

        def forward(self, x):
            return _quant_forward(self, x)

    _QuantLinear.__name__ = f"{base_cls.__name__}_{scheme}"
    _QuantLinear.__qualname__ = _QuantLinear.__name__
    return _QuantLinear
