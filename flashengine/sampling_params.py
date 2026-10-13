from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_p: float = 1.0
    top_k: int = 0

    def __post_init__(self):
        assert self.temperature >= 0.0, "temperature must be non-negative (0 = greedy)"
        assert 0.0 < self.top_p <= 1.0
        assert self.top_k >= 0
