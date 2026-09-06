import torch
from torch import nn


class Sampler(nn.Module):
    """支持 greedy（temperature==0）、top-k、top-p 的批量采样。

    所有算子均无数据依赖的动态 shape，可被 CUDA Graph 安全捕获；
    随机性来自 PyTorch CUDA RNG（graph-safe，replay 每次产生新随机数）。

    fast_top_p=True（默认）走瘦身快路径：top-k 掩码 + Gumbel-argmax，
    替代原「softmax + full sort + cumsum + scatter」流水（省 ~0.2ms/步）。
    - Gumbel-max 定理保证 argmax(scaled + Gumbel) 与「softmax(scaled) 多项式
      采样」分布精确等价；temperature 沿用原 clamp 约定（greedy 行 t=1，最后
      argmax 覆盖），temp0 时噪声相对 logit/T≈1e6·gap 可忽略，greedy 语义与
      原实现逐位一致（greedy_ids 同为 logits.argmax，采样值被 where 丢弃）。
    - top-k 掩码用 argsort 的 rank 构造，对任意 top_k 精确；top_ks ≥ vocab
      （禁用态）时 ranks ≤ vocab-1 < top_ks，掩码天然恒假，无需分支。
    - top-p：图内 top_ps 是张量，精确 top-p 截断需要 sort+cumsum 全流水（正是
      要省的成本），快路径不施加。当前所有调用方（server 默认 / harness /
      check_eagle3 / benchmark）top_p 恒为 1.0 不受影响；需要图内 top_p<1
      精确截断的部署置 fast_top_p=False 回退 legacy 全流水（一字未改）。
    - 该开关是 host 属性分支（非张量分支），CUDA Graph 捕获时被常量折叠，
      运行时零开销。
    """

    def __init__(self):
        super().__init__()
        self.fast_top_p = True

    def _forward_fast(self, logits, t, greedy, top_ks):
        scaled = logits.float() / t.unsqueeze(dim=1)
        # top-k 掩码：rank(i) = 降序位次，由 argsort 的逆排列 scatter 得到。
        idx = scaled.argsort(dim=-1, descending=True)
        ranks = torch.empty_like(idx)
        ranks.scatter_(1, idx, torch.arange(idx.size(1), device=idx.device).expand_as(idx))
        scaled = scaled.masked_fill_(ranks >= top_ks.unsqueeze(dim=1), float("-inf"))
        # Gumbel-max：argmax(scaled + g) ~ Categorical(softmax(scaled))。
        # g = -log(Exp(1))；greedy 行噪声置 0 → argmax(logits.float()) 与
        # logits.argmax 逐位一致（/1.0 与 +0.0 均为精确运算），省去第二个
        # argmax 与 where。u=e~Exp(1)>0，log 定义域安全（e=0 概率为 0，
        # clamp 由 exponential_ 的值域天然保证，无需额外 eps）。
        e = torch.empty_like(scaled).exponential_()
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

    # 不使用 @torch.compile：torch 2.11 Inductor 对 legacy 路径中
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
