"""锁页（pinned / host-mapped）CPU 缓冲开关。

默认开启；`FLASHENGINE_PIN_MEMORY=0` 关闭。310P 上某些驱动/固件组合不支持
torch_npu 在首次 host-mapped 分配时下发的 ACL_ALLOW_HF32 compileopt
（报 500001），此时锁页分配会直接抛 RuntimeError，需要退化为普通 CPU 缓冲。
锁页只是 H2D 拷贝的加速手段，关掉后语义不变。
"""

import os

PIN_MEMORY = os.environ.get("FLASHENGINE_PIN_MEMORY", "1").strip().lower() not in (
    "0", "false", "no", "off",
)
