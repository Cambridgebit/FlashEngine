"""当前步的适配器路由上下文。

路由以**主机侧算好的 0/1 掩码**形式下发：``[rows, max_loras*rank]`` fp16，
第 r 行只在「该行 token 所属适配器的 rank 列段」上为 1。选这个形态而不是
「整数下标 + 图内 ``==`` 比较」，是因为 310P 的 NPUGraph 里一旦出现整数
比较/类型提升（int32 vs int64 广播），回放出的掩码与捕获期语义不一致，
直接表现为投机接受长度塌到 1；改成静态 fp16 buffer 后图内只剩一次
``mul``，捕获/回放形状与算子完全一致。

单进程单驱动线程，用模块级变量而不是 threading.local 就够；NPUGraph 捕获时
这里持有的是图内静态 buffer，回放前由 ModelRunner 用 ``copy_`` 更新内容，
只要保持「同一个张量对象」即可。
"""

_ACTIVE = None


def set_active_mask(mask):
    global _ACTIVE
    _ACTIVE = mask


def get_active_mask():
    return _ACTIVE


def clear_active_mask():
    global _ACTIVE
    _ACTIVE = None
