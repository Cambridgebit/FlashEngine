import torch
from torch import nn
import torch.nn.functional as F

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


class GemmaRMSNorm(nn.Module):
    """Gemma 式 RMSNorm：out = rmsnorm(x) * (1 + weight)，weight 检查点存零基增量。

    Qwen3.5 全部 norm（input/post_attention/final + q_norm/k_norm）均为此式。
    语义出处：transformers Qwen3_5RMSNorm.forward（fp32 归一化后乘 1+weight 再回投，
    与 vllm GemmaRMSNorm 同义）。与现有 RMSNorm 的差异：weight+1、零基初始化。

    residual 融合签名与现有 RMSNorm 对齐（层代码可直接复用 qwen3.py 写法）；
    注意相加用模型 dtype（transformers 解码层 `residual + hidden_states` 同款），
    有意区别于 backend.add_rms_norm 的 fp32 相加——Qwen3.5 以 transformers 为金标准。
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        # 零基初始化：checkpoint 保存的是 (weight) 增量，运行时 +1（transformers
        # Qwen3_5RMSNorm weight= zeros 同款）。
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def _gemma_norm(self, x: torch.Tensor) -> torch.Tensor:
        # 逐行对表 transformers Qwen3_5RMSNorm.forward：
        # output = _norm(x.float()); output = output * (1.0 + weight.float());
        # return output.type_as(x)
        orig_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * (1.0 + self.weight.float())
        return x.to(orig_dtype)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._gemma_norm(x)
        # 融合加残差（模型 dtype 相加，见类注释）；返回 (norm 后, 新残差)。
        summed = x + residual
        return self._gemma_norm(summed), summed


class RMSNormGated(nn.Module):
    """门控 RMSNorm（GDN 输出归一化）：out = rmsnorm(x) * silu(z)（norm_before_gate=True）。

    作用域 = 每 (token, v-head) 的 head_v_dim（vllm qwen_gdn_linear_attn.py 构造
    RMSNormGated(head_v_dim, norm_before_gate=True, activation=output_gate_type)），
    即逐头在 value head 维内归一化，不是对拼接后的 value_dim 整体归一化。
    数值顺序逐行对表 transformers Qwen3_5RMSNormGated.forward（金标准）：
    fp32 归一化 → 乘 weight 回投 input dtype → 乘 fp32 silu(gate) → 回投。
    （vllm RMSNormGated.forward_static 同序，仅 weight 乘法保留在 fp32——此处从
    transformers 口径。）norm_before_gate 语义：norm 在前、gate 在后（vllm
    layernorm.py L238-241 文档：True → out = norm(x) * act(z)）。
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        activation: str = "silu",
    ) -> None:
        super().__init__()
        self.eps = eps
        assert activation in ("silu", "swish", "sigmoid"), f"不支持的 gate 激活 {activation}"
        self.activation = activation
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # x/gate: [..., hidden_size]（调用方保证已按 head 展平）
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        # norm before gate（先归一化、后门控）
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x.to(orig_dtype)
        act = F.sigmoid if self.activation == "sigmoid" else F.silu
        x = x * act(gate.to(torch.float32))
        return x.to(orig_dtype)
