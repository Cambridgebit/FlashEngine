
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DecodeGraphPlan:
    model: Any
    config: Any
    device: str
    block_size: int
    max_batch_size: int
    max_block_columns: int
    warmup_iters: int
    state_runtime: Any = None


class DecodeGraphExecutor:
    @property
    def capture_sizes(self) -> tuple[int, ...]:
        raise NotImplementedError

    def capture_all(self) -> None:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError

    def try_run(self, seqs):
        raise NotImplementedError

    def snapshot(self) -> dict:
        """图执行统计快照（replay/fallback/bucket_hits/captured 等）。

        默认返回空字典：未实现统计的执行器不会让运行快照接口报错，调用方按
        「缺键即未采集」处理即可。
        """
        return {}

    def reset(self) -> None:
        raise NotImplementedError
