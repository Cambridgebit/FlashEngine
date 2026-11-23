"""PEFT 适配器读取与「适配器模块名 → 引擎 Linear 子分片」解析。

适配器由 PEFT/ms-swift 导出，张量名是 HF state_dict 路径（去掉
``base_model.model.``），例如``model.language_model.layers.0.linear_attn.
in_proj_qkv``；引擎的模块树是 ``model.layers.0.linear_attn.in_proj_qkvz``
（qkv/z/ba 已经融合）。因此这里做两件事：

1. 名字归一化到引擎树（去掉多模态壳的 ``language_model`` 段）；
2. 按 ``PACKED_MODULES_MAPPING`` 把非融合名映射到（融合模块, 分片号），并把
   ``in_proj_qkv`` 这种「单张量 → 前三分片」的特例拆成多段（A 共用、B 按行切）。
"""

import json
import os

import torch

# 与 flashengine/models/qwen3_5.py:PACKED_MODULES_MAPPING 保持一致的子集；
# 这里只描述「适配器里的名字 → 引擎融合模块的哪个分片」。
FUSED_SUBMODULES = {
    "q_proj": ("qkv_proj", 0),
    "k_proj": ("qkv_proj", 1),
    "v_proj": ("qkv_proj", 2),
    "gate_proj": ("gate_up_proj", 0),
    "up_proj": ("gate_up_proj", 1),
    "in_proj_z": ("in_proj_qkvz", 3),
    "in_proj_b": ("in_proj_ba", 0),
    "in_proj_a": ("in_proj_ba", 1),
}

# 「一个检查点张量覆盖融合模块的前 N 个分片」（B 按各分片行宽切分，A 共享）。
SPLIT_SUBMODULES = {
    "in_proj_qkv": ("in_proj_qkvz", (0, 1, 2)),
}

# 适配器里出现、但引擎侧不支持的模块（视觉塔 / MTP / 卷积等）一律跳过并告警。
_SKIP_SUFFIXES = ("visual.", "mtp.", "conv1d", "lm_head", "embed_tokens")


def _engine_path(hf_name: str) -> str:
    """HF state_dict 路径 → 引擎模块路径。"""
    name = hf_name
    if name.startswith("base_model.model."):
        name = name[len("base_model.model."):]
    if name.startswith("model.language_model."):
        name = "model." + name[len("model.language_model."):]
    elif name.startswith("language_model."):
        name = "model." + name[len("language_model."):]
    elif name.startswith("model."):
        pass  # Qwen3 dense：model.layers.*，原样
    else:
        name = "model." + name
    return name


def read_adapter_config(path: str) -> dict:
    """读 adapter_config.json，返回 (rank, alpha, scaling_factor)。"""
    cfg_path = os.path.join(path, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"LoRA 适配器缺少 adapter_config.json: {path}")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    rank = int(cfg.get("r", 8))
    alpha = int(cfg.get("lora_alpha", rank))
    use_rslora = bool(cfg.get("use_rslora", False))
    scaling = alpha / (rank ** 0.5) if use_rslora else alpha / rank
    return {
        "rank": rank,
        "alpha": alpha,
        "scaling": scaling,
        "use_rslora": use_rslora,
        "target_modules": cfg.get("target_modules"),
        "path": path,
    }


def load_adapter_tensors(path: str) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """读 adapter_model.safetensors，返回 {引擎模块路径: (A[r,in], B[out,r])}。

    张量保持 fp32（CPU），由 manager 在装载进设备栈时再折缩放并转 dtype。
    """
    from safetensors import safe_open

    model_file = os.path.join(path, "adapter_model.safetensors")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(f"LoRA 适配器缺少 adapter_model.safetensors: {path}")
    a_map: dict[str, torch.Tensor] = {}
    b_map: dict[str, torch.Tensor] = {}
    with safe_open(model_file, "pt") as handle:
        for key in handle.keys():
            if key.endswith(".lora_A.weight"):
                a_map[_engine_path(key[: -len(".lora_A.weight")])] = handle.get_tensor(key).float()
            elif key.endswith(".lora_B.weight"):
                b_map[_engine_path(key[: -len(".lora_B.weight")])] = handle.get_tensor(key).float()
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for module in a_map:
        if module not in b_map:
            raise ValueError(f"LoRA 适配器 {path} 中 {module} 只有 A 没有 B")
        if any(tag in module for tag in _SKIP_SUFFIXES):
            continue
        out[module] = (a_map[module], b_map[module])
    return out
