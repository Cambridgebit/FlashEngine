
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
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
