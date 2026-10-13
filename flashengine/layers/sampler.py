import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
        self.fast_top_p = True

    def _forward_fast(self, logits, t, greedy, top_ks):
        scaled = logits.float() / t.unsqueeze(dim=1)
        idx = scaled.argsort(dim=-1, descending=True)
        ranks = torch.empty_like(idx)
        ranks.scatter_(1, idx, torch.arange(idx.size(1), device=idx.device).expand_as(idx))
        scaled = scaled.masked_fill_(ranks >= top_ks.unsqueeze(dim=1), float("-inf"))
        e = torch.empty(scaled.shape, dtype=scaled.dtype, device='cpu').exponential_().to(scaled.device)
        e.log_().neg_().mul_(greedy.logical_not().unsqueeze(1))
        scaled = scaled.add_(e)
        return scaled.argmax(dim=-1)

    def _forward_legacy(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
    ):
        greedy = temperatures <= 1e-10
        t = torch.where(greedy, torch.ones_like(temperatures), temperatures)
        probs = torch.softmax(logits.float() / t.unsqueeze(dim=1), dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        vocab = probs.size(-1)
        mask_p = (cum - sorted_probs) > top_ps.unsqueeze(dim=1)
        ranks = torch.arange(vocab, device=logits.device)
        mask_k = ranks.unsqueeze(dim=0) >= top_ks.unsqueeze(dim=1)
        sorted_probs = sorted_probs.masked_fill(mask_p | mask_k, 0)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
        _noise = torch.empty(probs.shape, dtype=probs.dtype, device='cpu').exponential_(1).to(probs.device).clamp_min_(1e-10)
        sample_ids = (probs / _noise).argmax(dim=-1)
        greedy_ids = logits.argmax(dim=-1)
        return torch.where(greedy, greedy_ids, sample_ids)

    def forward_native(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
    ):
        if self.fast_top_p:
            greedy = temperatures <= 1e-10
            t = torch.where(greedy, torch.ones_like(temperatures), temperatures)
            return self._forward_fast(logits, t, greedy, top_ks)
        return self._forward_legacy(logits, temperatures, top_ps, top_ks)

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
    ):
        return self.forward_native(logits, temperatures, top_ps, top_ks)
