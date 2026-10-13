import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self, *, device_rng: bool = True):
        super().__init__()
        self.fast_top_p = True
        self.device_rng = device_rng

    def _exponential_noise(self, reference):
        # CUDA RNG 可捕获且每次 replay 推进；CPU 随机数不会进入设备图。
        if self.device_rng:
            return torch.empty_like(reference).exponential_().clamp_min_(1e-10)
        return torch.empty(reference.shape, dtype=reference.dtype, device="cpu").exponential_().clamp_min_(1e-10).to(reference.device)

    def _filter_logits(self, logits, t, top_ps, top_ks):
        scaled = logits.float() / t.unsqueeze(dim=1)
        sorted_logits, sorted_idx = scaled.sort(dim=-1, descending=True)
        ranks = torch.arange(sorted_logits.size(-1), device=logits.device)
        sorted_logits.masked_fill_(ranks.unsqueeze(0) >= top_ks.unsqueeze(1), float("-inf"))
        # 先应用 top-k，再按剩余候选归一化计算 nucleus，保留首个跨过阈值的 token。
        probs = sorted_logits.softmax(dim=-1)
        cumulative = probs.cumsum(dim=-1)
        # 移位而非 cumulative-probs，避免浮点相减在阈值处误保留候选。
        before = torch.cat((torch.zeros_like(cumulative[:, :1]), cumulative[:, :-1]), dim=-1)
        sorted_logits.masked_fill_(before >= top_ps.unsqueeze(1), float("-inf"))
        return sorted_logits, sorted_idx

    def _forward_fast(self, logits, t, greedy, top_ps, top_ks):
        sorted_logits, sorted_idx = self._filter_logits(logits, t, top_ps, top_ks)
        noise = self._exponential_noise(sorted_logits).log_().neg_()
        selected = (sorted_logits + noise).argmax(dim=-1, keepdim=True)
        sampled = sorted_idx.gather(1, selected).squeeze(1)
        return torch.where(greedy, logits.argmax(dim=-1), sampled)

    def _forward_legacy(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
    ):
        greedy = temperatures <= 1e-10
        t = torch.where(greedy, torch.ones_like(temperatures), temperatures)
        sorted_logits, sorted_idx = self._filter_logits(logits, t, top_ps, top_ks)
        probs = sorted_logits.softmax(dim=-1)
        selected = (probs / self._exponential_noise(probs)).argmax(dim=-1, keepdim=True)
        sample_ids = sorted_idx.gather(1, selected).squeeze(1)
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
            return self._forward_fast(logits, t, greedy, top_ps, top_ks)
        return self._forward_legacy(logits, temperatures, top_ps, top_ks)

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
    ):
        return self.forward_native(logits, temperatures, top_ps, top_ks)
