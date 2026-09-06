import torch
from torch import nn

from nanovllm.backends import get_backend


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
        # 融合点走 backend：CUDA/CoreX = 编译版纯 torch（位等保真，实现见 base.py），
        # Ascend = npu_rms_norm / npu_add_rms_norm 融合算子（见 ascend.py）。
        if residual is None:
            return self.backend.rms_norm(x, self.weight, self.eps)
        else:
            return self.backend.add_rms_norm(x, residual, self.weight, self.eps)
