import torch
from torch import nn

from flashengine.backends import get_backend


class SiluAndMul(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.backend = get_backend()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backend.silu_and_mul(x)
