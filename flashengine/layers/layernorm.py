import torch
from torch import nn

from flashengine.backends import get_backend


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.backend = get_backend()

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.backend.rms_norm(x, self.weight, self.eps)
        return self.backend.add_rms_norm(x, residual, self.weight, self.eps)
