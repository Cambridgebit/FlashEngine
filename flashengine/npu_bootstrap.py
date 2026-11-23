"""NPU 设备预初始化（必须在 transformers 之前，且只做一次）。

现场实测（Atlas 300I Duo / 310P / CANN 9.0.T3 / torch_npu 2.9.0.post2.dev /
驱动 24.1.rc4.b999）：
- 进程里第一次 `torch.npu.set_device()` 会走 `torch_npu._lazy_init()`
  → `_npu_init()` → `aclSetCompileopt(ACL_ALLOW_HF32, ...)`；
- 只要这一步发生在 `import transformers` 之后，或由引擎里某处延迟触发，
  该 compileopt 就返回 500001，进程直接失败，并且**会把驱动态搞坏**：
  此后同机上所有进程（包括裸 `set_device`）都持续失败，直到重启板卡；
- 反过来，进程一开始就 `import torch; import torch_npu; torch.npu.set_device(0)`
  总是成功（实测多次）。

因此本模块在 `import flashengine` 时立即做一次干净的初始化；成功之后引擎里
再调 `set_device` 会先被 `npu_device_initialized()` 拦掉，不再触发第二次
`_npu_init`。`FLASHENGINE_NPU_BOOTSTRAP=0` 可关闭（非 Ascend 机器自动 no-op）。
"""

import os

_DISABLED = os.environ.get("FLASHENGINE_NPU_BOOTSTRAP", "1").strip().lower() in (
    "0", "false", "no", "off",
)

_done = False
_device = -1


def npu_device_initialized() -> int:
    """返回已被本模块初始化的设备号，未初始化返回 -1。"""
    return _device if _done else -1


def ensure_npu_device_before_transformers() -> bool:
    global _done, _device
    if _done or _DISABLED:
        return _done
    try:
        # 顺序必须是 torch → torch_npu（与板上实测可用的裸序列逐字节一致）；
        # 先 torch_npu 再 torch 在本机驱动上会走到 HF32 compileopt 失败。
        import torch
        import torch_npu  # noqa: F401  仅 Ascend 环境存在
    except Exception:
        return False
    try:
        device = int(os.environ.get("LOCAL_RANK", "0") or 0)
        torch.npu.set_device(device)
        _done = True
        _device = device
        print(f"[FlashEngine] NPU device {device} 预初始化完成", flush=True)
        return True
    except Exception as exc:  # 非 NPU 机器或驱动不可用：不阻断后续流程
        print(f"[FlashEngine] NPU 预初始化失败：{str(exc)[:160]}", flush=True)
        return False
