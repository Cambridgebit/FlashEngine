import math
from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_p: float = 1.0
    top_k: int = 0

    def __post_init__(self):
        # 显式异常而非 assert：assert 在 python -O 下会被整体去掉，非法采样参数
        # 会一路带进 kernel（NaN temperature、max_tokens=0 等），症状是空输出
        # 或满屏重复 token，难以回溯到请求参数。
        if (type(self.temperature) not in (int, float)
                or not math.isfinite(self.temperature)
                or self.temperature < 0):
            raise ValueError("temperature 必须为有限非负数，0 表示 greedy")
        if (type(self.top_p) not in (int, float)
                or not math.isfinite(self.top_p)
                or not 0.0 < self.top_p <= 1.0):
            raise ValueError("top_p 必须在 (0, 1] 内")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("top_k 必须是非负整数")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError("max_tokens 必须是正整数")
        if type(self.ignore_eos) is not bool:
            raise ValueError("ignore_eos 必须是布尔值")
