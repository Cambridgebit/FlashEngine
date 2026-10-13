"""Ascend 310P MoE 算子链契约测试（无需 NPU 硬件，纯 torch 桩）。

现有 310P MoE 路径（`backends/ascend.py`）依赖 4 个 torch_npu 算子：
`npu_moe_gating_top_k_softmax` / `npu_moe_init_routing_v2` / `npu_grouped_matmul` /
`npu_moe_token_unpermute`（+ `npu_swiglu`）。本测试用纯 torch 桩替换它们，
直接调用**真实的** `Ascend310PBackend.moe_router` / `optimize_moe_weights` /
`fused_moe`，验证接线（游程排序 / 权重转置 / gate|up 次序 / unpermute 索引 /
top-k 加权求和）与朴素 MoE 数值一致。

桩语义取自 vllm-ascend 310P 参考实现：
- `_310p/fused_moe/grouped_topk_router.py`（fp16 分块 gating，返回**未归一化**
  softmax top-k，故 `norm_topk_prob` 的再归一在调用侧完成）；
- `_310p/fused_moe/token_dispatcher.py`（`npu_moe_init_routing_v2` + cumsum
  group_list_type=0）；
- `ops/fused_moe/token_dispatcher.py`（unpermute 前对 `expanded_row_idx` 取 abs）。
"""

import unittest

import torch
import torch.nn.functional as F

from flashengine.backends import ascend as ascend_mod
from flashengine.backends.ascend import Ascend310PBackend


class _StubNpu:
    """torch_npu MoE 算子的纯 torch 参考实现（仅覆盖测试用到的签名）。"""

    ACL_FORMAT_FRACTAL_NZ = 29

    @staticmethod
    def npu_format_cast(weight, acl_format, **kwargs):
        return weight  # 纯 torch 桩：NZ 视为无操作，逻辑形状不变

    @staticmethod
    def npu_moe_gating_top_k_softmax(router_logits, finished=None, k=None):
        probs = torch.softmax(router_logits.float(), dim=-1)
        weights, ids = torch.topk(probs, k, dim=-1)
        return weights.to(router_logits.dtype), ids.to(torch.int32), None

    @staticmethod
    def npu_moe_init_routing_v2(x, expert_idx, active_num, expert_num,
                                drop_pad_mode, active_expert_range, quant_mode,
                                row_idx_type):
        flat = expert_idx.reshape(-1).to(torch.long)
        order = torch.argsort(flat, stable=True)
        expanded = x.repeat_interleave(expert_idx.shape[-1], dim=0)[order]
        counts = torch.bincount(flat, minlength=expert_num)
        group_list = torch.cumsum(counts, dim=0).to(torch.int32)
        return expanded, order, group_list, None

    @staticmethod
    def npu_grouped_matmul(x, weight, split_item, group_list_type, group_type,
                           group_list):
        # weight 布局 = [E, K, N]（310P 预转置后的 [E,hidden,inter]），x=[M,K]。
        tokens, experts = x[0], weight[0]
        out = torch.empty(tokens.shape[0], experts.shape[2],
                          dtype=tokens.dtype, device=tokens.device)
        start = 0
        for expert in range(experts.shape[0]):
            end = int(group_list[expert])
            if end > start:
                out[start:end] = tokens[start:end] @ experts[expert]
            start = end
        return [out]

    @staticmethod
    def npu_swiglu(x):
        gate, up = x.chunk(2, dim=-1)
        return F.silu(gate) * up

    @staticmethod
    def npu_moe_token_unpermute(permuted_tokens, sorted_indices, probs):
        top_k = probs.shape[-1]
        weights = probs.reshape(-1)[sorted_indices.to(torch.long)]
        out = permuted_tokens.new_zeros(probs.shape[0], permuted_tokens.shape[-1])
        out.index_add_(
            0, sorted_indices.to(torch.long) // top_k, permuted_tokens * weights.unsqueeze(-1))
        return out


class _FakeExperts(torch.nn.Module):
    is_moe_experts = True
    quantized = False

    def __init__(self, w13, w2):
        super().__init__()
        self.w13_weight = torch.nn.Parameter(w13.clone())
        self.w2_weight = torch.nn.Parameter(w2.clone())


def _naive_moe(x, w13, w2, topk_weights, topk_ids):
    """朴素参考：逐 token 逐 top-k 累加（布局 w13=[E,2I,H]、w2=[E,H,I]）。"""
    out = torch.zeros_like(x)
    for t in range(x.shape[0]):
        for k in range(topk_ids.shape[-1]):
            expert = int(topk_ids[t, k])
            gate_up = w13[expert] @ x[t]
            act = F.silu(gate_up[: gate_up.shape[0] // 2]) * gate_up[gate_up.shape[0] // 2:]
            out[t] += topk_weights[t, k] * (w2[expert] @ act)
    return out


class AscendMoeContractTests(unittest.TestCase):

    def setUp(self):
        self._saved = ascend_mod._npu
        ascend_mod._npu = _StubNpu()
        self.backend = object.__new__(Ascend310PBackend)
        self.backend._moe_ops_ok = True  # 跳过 _require_moe_ops 的算子存在性检查

    def tearDown(self):
        ascend_mod._npu = self._saved

    def test_moe_router_matches_softmax_topk(self):
        torch.manual_seed(0)
        # 310P 的 gating 算子仅吃 fp16，故与代码内 .to(float16) 的精度对齐。
        logits = torch.randn(9, 4, dtype=torch.float16)
        for renormalize in (True, False):
            weights, ids = self.backend.moe_router(logits, top_k=2, renormalize=renormalize)
            ref = torch.softmax(logits.float(), dim=-1)
            ref_w, ref_ids = torch.topk(ref, 2, dim=-1)
            if renormalize:
                ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)
            self.assertTrue(torch.allclose(weights.float(), ref_w, atol=1e-3))
            self.assertTrue(torch.equal(ids, ref_ids.to(torch.int32)))

    def test_optimize_transposes_expert_weights(self):
        experts = _FakeExperts(torch.randn(4, 32, 16), torch.randn(4, 16, 16))
        self.backend.optimize_moe_weights(torch.nn.ModuleList([experts]))
        self.assertEqual(tuple(experts.w13_weight.shape), (4, 16, 32))  # [E,2I,H]→[E,H,2I]
        self.assertEqual(tuple(experts.w2_weight.shape), (4, 16, 16))   # [E,H,I]→[E,I,H]

    def test_fused_moe_matches_naive_after_optimize(self):
        torch.manual_seed(1)
        hidden, inter, experts_n, tokens, top_k = 16, 16, 4, 7, 2
        x = torch.randn(tokens, hidden)
        w13 = torch.randn(experts_n, 2 * inter, hidden)
        w2 = torch.randn(experts_n, hidden, inter)
        topk_ids = torch.randint(0, experts_n, (tokens, top_k), dtype=torch.int32)
        topk_weights = torch.rand(tokens, top_k)
        ref = _naive_moe(x, w13, w2, topk_weights, topk_ids)

        module = _FakeExperts(w13, w2)
        model = torch.nn.ModuleList([module])
        self.backend.optimize_moe_weights(model)  # 转置 + “NZ”（桩等价原张量）
        out = self.backend.fused_moe(
            x, module.w13_weight, module.w2_weight, topk_weights, topk_ids)
        self.assertEqual(tuple(out.shape), (tokens, hidden))
        self.assertTrue(torch.allclose(out, ref, atol=1e-5),
                        f"max diff {((out - ref).abs().max()).item()}")


if __name__ == "__main__":
    unittest.main()
