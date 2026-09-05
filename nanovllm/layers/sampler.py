import torch
from torch import nn


class Sampler(nn.Module):
    """支持 greedy（temperature==0）、top-k、top-p 的批量采样。

    所有算子均无数据依赖的动态 shape，可被 CUDA Graph 安全捕获；
    随机性来自 exponential_（PyTorch CUDA RNG 是 graph-safe 的，replay 每次产生新随机数）。
    """

    def forward_native(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
    ):
        # greedy 行用恒等温度走同一条计算路径，最后用 argmax 覆盖结果，
        # 保证 graph 捕获与批量混合（部分 greedy 部分采样）都无需分支。
        greedy = temperatures <= 1e-10
        t = torch.where(greedy, torch.ones_like(temperatures), temperatures)
        probs = torch.softmax(logits.float() / t.unsqueeze(dim=1), dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        vocab = probs.size(-1)
        # top-p：保留累计概率（含当前项）不超过 top_p 的前缀，至少保留第 1 个。
        mask_p = (cum - sorted_probs) > top_ps.unsqueeze(dim=1)
        # top-k：top_k 不生效时调用方传入词表大小，掩码恒为假。
        ranks = torch.arange(vocab, device=logits.device)
        mask_k = ranks.unsqueeze(dim=0) >= top_ks.unsqueeze(dim=1)
        sorted_probs = sorted_probs.masked_fill(mask_p | mask_k, 0)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        # Gumbel 技巧：argmax(p / Exp(1)) ~ Categorical(p)，p=0 的项恒不中。
        sample_ids = (probs / torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        greedy_ids = logits.argmax(dim=-1)
        return torch.where(greedy, greedy_ids, sample_ids)

    # 不使用 @torch.compile：torch 2.11 Inductor 对 forward_native 中
    # zeros_like().scatter_ 的 codegen 有 bug（get_block_shape NoneType），
    # 任意 batch>1 的编译都会崩；Sampler 开销小且在 CUDA Graph 内被整体捕获，保持 eager 即可。
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
    ):
        return self.forward_native(logits, temperatures, top_ps, top_ks)
