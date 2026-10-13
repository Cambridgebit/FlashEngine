import os

import torch


_backend = None


def detect_backend():
    env = os.getenv("FLASHENGINE_BACKEND", "auto").lower()

    if env != "auto":
        return env

    if getattr(torch, "corex", False) is True:
        return "corex"

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
            f"Unsupported FlashEngine backend: {name}"
        )

    return _backend
