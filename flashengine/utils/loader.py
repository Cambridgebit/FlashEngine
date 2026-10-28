import json
import os
import struct
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from tqdm import tqdm

# fp16_only 模式的加载期浮点校验开关。默认关闭：主项目要保留 W8A8 int8 权重与
# Quark 加载期反量化，默认路径上逐张量校验会直接打断它们。
_REQUIRE_FLOAT_WEIGHTS = False


def set_require_float_weights(enabled: bool) -> None:
    """fp16_only 模式打开加载期浮点校验；默认关闭，避免影响 W8A8/int8 反量化。"""
    global _REQUIRE_FLOAT_WEIGHTS
    _REQUIRE_FLOAT_WEIGHTS = bool(enabled)


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def require_float_weight(tensor: torch.Tensor) -> None:
    """fp16_only 加载期守卫：参与计算的权重必须来自浮点检查点。

    由模块级开关 `_REQUIRE_FLOAT_WEIGHTS` 控制（`Config.fp16_only` 为真时经
    `set_require_float_weights(True)` 打开），默认关闭，因此不会阻断 W8A8 int8
    权重与 Quark `dequantize_int8_weight` 反量化路径；反量化后的权重本身已是
    浮点，即使开关打开也能通过。
    """
    if not tensor.is_floating_point():
        raise ValueError(
            f"fp16_only 拒绝整数或量化计算权重：dtype={tensor.dtype}"
        )


def iter_safetensors(path: str):
    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if _REQUIRE_FLOAT_WEIGHTS:
                    require_float_weight(tensor)
                yield name, tensor


def safetensors_total_bytes(path: str, skip=None) -> int | None:
    """只读 safetensors 头部，统计权重字节总数（用作加载进度条总量）。

    手工解析头部 JSON（格式 v1：8 字节小端长度 + JSON），不加载任何张量；
    目录里没有 .safetensors（如 pytorch_model.bin）或解析失败时返回 None，
    调用方退化为"只显示已加载张量数"的进度条。skip(name) 可排除已知不加载
    的键（如 DSpark 的 t2d/d2t）。
    """
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        return None
    total = 0
    for filepath in files:
        try:
            with open(filepath, "rb") as stream:
                header_len = struct.unpack("<Q", stream.read(8))[0]
                header = json.loads(stream.read(header_len))
        except (OSError, ValueError, struct.error):
            return None
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if skip is not None and skip(name):
                continue
            begin, end = info["data_offsets"]
            total += end - begin
    return total


def iter_with_progress(weights, label: str, total_bytes: int | None = None):
    """包装权重迭代器，用 tqdm 进度条显示加载进度。

    8B 级权重在板上加载需几十秒，期间完全静默会被误认为卡死；目标模型与
    草稿模型的加载循环都套它输出进度。total_bytes 由调用方经
    safetensors_total_bytes() 从文件头取得（进度按已加载字节走）；拿不到
    总量时（pytorch_model.bin、加载期反量化等）退化为按张量数计数。
    """
    if total_bytes:
        pbar = tqdm(total=total_bytes, desc=label, unit="B", unit_scale=True,
                    unit_divisor=1024, dynamic_ncols=True)
    else:
        pbar = tqdm(desc=label, unit="tensor", dynamic_ncols=True)
    count = 0
    nbytes = 0
    try:
        for name, tensor in weights:
            count += 1
            nbytes += tensor.numel() * tensor.element_size()
            pbar.update(tensor.numel() * tensor.element_size() if total_bytes else 1)
            pbar.set_postfix_str(f"{count} tensors, {nbytes / 2**30:.2f} GiB",
                                 refresh=False)
            yield name, tensor
    finally:
        if total_bytes and pbar.n >= total_bytes:
            pbar.n = total_bytes
        if not total_bytes:
            pbar.total = count
        pbar.refresh()
        pbar.close()


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
        # 自定义 weights 源（iter_with_progress / 模型自带加载器）也过一遍守卫；
        # 默认关闭，故 W8A8 int8 与 Quark 反量化路径不受影响。
        if _REQUIRE_FLOAT_WEIGHTS:
            require_float_weight(tensor)
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
