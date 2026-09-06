import os

import torch


_backend = None


def detect_backend():
    env = os.getenv("NANOVLLM_BACKEND", "auto").lower()

    if env != "auto":
        return env

    # CoreX 的 PyTorch 环境存在此标记。
    if getattr(torch, "corex", False) is True:
        return "corex"

    # torch_npu 导入后注册 torch.npu 模块——存在且可用即 NPU 环境
    # （优先级在 corex 标记之后）。
    if hasattr(torch, "npu") and torch.npu.is_available():
        return "ascend310p"

    return "cuda"


def get_backend():
    global _backend

    if _backend is not None:
        return _backend

    name = detect_backend()

    if name == "cuda":
        from .cuda import CUDABackend

        _backend = CUDABackend()

    elif name == "corex":
        from .corex import CoreXBackend

        _backend = CoreXBackend()

    elif name == "ascend310p":
        from .ascend import Ascend310PBackend

        _backend = Ascend310PBackend()

    else:
        raise RuntimeError(
            f"Unsupported nano-vLLM backend: {name}"
        )

    return _backend
