import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def iter_safetensors(path: str):
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                yield name, f.get_tensor(name)


def dequantize_int8_weight(
    qweight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_zero_point: torch.Tensor | None,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    """Quark W8A8 `real_quantized` 权重的 per-channel 反量化。

    权重 [out, in] + 逐输出通道 scale/zero_point（ch_axis=0，shape [out]）：
    `W = (qweight - zero_point) * scale`，沿 dim=1 广播。中间在 fp32 计算，
    最后转回 dtype（None = 保持 fp32，交给卸载器的 copy_ 转换）。
    """
    w = qweight.to(torch.float32)
    if weight_zero_point is not None:
        w = w - weight_zero_point.to(torch.float32).reshape(-1, 1)
    w = w * weight_scale.to(torch.float32).reshape(-1, 1)
    return w if dtype is None else w.to(dtype)


def iter_safetensors_dequant(path: str, dtype: torch.dtype | None = None):
    """流式读取，把 Quark int8 权重与其 scale/zero_point 合并为反量化权重。

    仅对 int8 权重生效（fp16/bf16 权重原样透传）；`*.weight_scale` /
    `*.weight_zero_point` 伴生键被吸收、不再单独产出。单遍、无整表缓存。
    """
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            names = set(f.keys())
            for name in f.keys():
                if name.endswith((".weight_scale", ".weight_zero_point")):
                    continue
                if name.endswith(".weight"):
                    base = name[: -len(".weight")]
                    scale_name = base + ".weight_scale"
                    if scale_name in names:
                        weight = f.get_tensor(name)
                        if weight.dtype == torch.int8:
                            zero_point_name = base + ".weight_zero_point"
                            zero_point = (
                                f.get_tensor(zero_point_name)
                                if zero_point_name in names
                                else None
                            )
                            yield name, dequantize_int8_weight(
                                weight, f.get_tensor(scale_name), zero_point, dtype
                            )
                            continue
                yield name, f.get_tensor(name)


def load_model(model: nn.Module, path: str, weights=None):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    source = weights if weights is not None else iter_safetensors(path)
    for weight_name, tensor in source:
        for k in packed_modules_mapping:
            if k in weight_name:
                v, shard_id = packed_modules_mapping[k]
                param_name = weight_name.replace(k, v)
                param = model.get_parameter(param_name)
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, tensor, shard_id)
                break
        else:
            param = model.get_parameter(weight_name)
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)
