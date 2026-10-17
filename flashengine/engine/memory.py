"""跨后端的启动显存预算；这是容量规划，不是进程显存硬隔离。"""

from dataclasses import asdict, dataclass
import math


class MemoryBudgetError(RuntimeError):
    pass


def validate_memory_options(utilization, budget_bytes, reserve_bytes, workspace_bytes,
                            graph_bytes, blocks):
    if isinstance(utilization, bool) or not math.isfinite(utilization) or not 0 < utilization <= 1:
        raise ValueError("gpu_memory_utilization 必须在 (0, 1] 内")
    for name, value in (("memory_budget_bytes", budget_bytes),
                        ("device_memory_reserve_bytes", reserve_bytes),
                        ("memory_workspace_bytes", workspace_bytes),
                        ("graph_memory_reserve_bytes", graph_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} 必须为非负整数字节数")
    if blocks != -1 and (isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 1):
        raise ValueError("num_kvcache_blocks 必须为 -1 或正整数")


@dataclass(frozen=True)
class MemoryPlan:
    total_bytes: int
    free_bytes: int
    resident_bytes: int
    workspace_bytes: int
    graph_reserve_bytes: int
    device_reserve_bytes: int
    engine_budget_bytes: int
    block_bytes: int
    num_blocks: int

    def snapshot(self):
        return {**asdict(self), "kv_bytes": self.block_bytes * self.num_blocks}


def plan_kv_memory(*, total, free, resident, workspace, block_bytes, utilization,
                   budget_bytes=0, reserve_bytes=0, graph_bytes=0,
                   requested_blocks=-1, minimum_blocks=1):
    """同时满足整卡占用、外部预留和本引擎预算；手动块数也须通过检查。"""
    validate_memory_options(utilization, budget_bytes, reserve_bytes, workspace,
                            graph_bytes, requested_blocks)
    if total <= 0 or not 0 <= free <= total or resident < 0 or block_bytes <= 0:
        raise ValueError("显存快照或 KV 块大小无效")
    available = min(free - reserve_bytes, int(total * utilization) - (total - free))
    if budget_bytes:
        available = min(available, budget_bytes - resident)
    available -= workspace + graph_bytes
    capacity = max(0, available // block_bytes)
    count = capacity if requested_blocks == -1 else requested_blocks
    if count < minimum_blocks or count > capacity:
        raise MemoryBudgetError(
            f"KV 显存预算不足：请求 {count} 块，可容纳 {capacity} 块，"
            f"至少需要 {minimum_blocks} 块；每块 {block_bytes} 字节，"
            f"常驻={resident}，工作区={workspace}，Graph预留={graph_bytes}，"
            f"外部预留={reserve_bytes}，引擎预算={budget_bytes}。"
            "请降低并发/上下文或调整预算。")
    return MemoryPlan(total, free, resident, workspace, graph_bytes, reserve_bytes,
                      budget_bytes, block_bytes, count)
