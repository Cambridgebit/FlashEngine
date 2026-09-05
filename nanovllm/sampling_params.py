from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    # temperature == 0（含极小值）时启用 greedy 采样，此时 top_p/top_k 不生效。
    max_tokens: int = 64
    ignore_eos: bool = False
    # top_k <= 0 表示不限制候选数；top_p == 1.0 表示不做核采样截断。
    top_p: float = 1.0
    top_k: int = 0

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature must be non-negative (0 = greedy)"
        assert 0.0 < self.top_p <= 1.0
        assert self.top_k >= 0
