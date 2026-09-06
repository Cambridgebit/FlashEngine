import torch
from torch import nn

from nanovllm.backends import get_backend


class SiluAndMul(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.backend = get_backend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 融合点走 backend：CUDA/CoreX = 编译版纯 torch（位等保真），
        # Ascend = npu_swiglu（若厂商 torch 提供，否则回退默认实现）。
        return self.backend.silu_and_mul(x)
