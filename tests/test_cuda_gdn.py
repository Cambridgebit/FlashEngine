"""CUDA GDN 融合算子：递推、逐行状态、动态图元数据与 padding 回归。"""

import unittest
import torch


@unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA / Triton")
class CUDAGDNContracts(unittest.TestCase):
    def test_prefill_recurrent_final_state_matches_chunk_reference(self):
        from flashengine.backends.cuda_gdn import recurrent
        from flashengine.models.qwen3_5 import chunk_gated_delta_rule, _l2norm
        for length in (5, 67, 128):
            q, k = [torch.randn(length, 2, 128, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
            v = torch.randn_like(q)
            g = -torch.rand(length, 2, device="cuda")
            beta = torch.rand_like(g).bfloat16()
            pool = torch.randn(3, 2, 128, 128, device="cuda")
            before = pool.clone()
            expected, final = chunk_gated_delta_rule(q[None], k[None], v[None], g[None], beta[None], initial_state=pool[1])
            actual = recurrent((_l2norm(q).float() / 128 ** 0.5).contiguous(), _l2norm(k).float().contiguous(),
                               v, g, beta, pool, torch.tensor([1], device="cuda", dtype=torch.int32),
                               rows=length, initial=torch.tensor([True], device="cuda"))
            torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
            torch.testing.assert_close(pool[1], final, atol=3e-5, rtol=3e-5)
            torch.testing.assert_close(pool[[0, 2]], before[[0, 2]], atol=0, rtol=0)

    def test_spec_greedy_and_sampling_graphs_keep_sampling_semantics(self):
        from types import SimpleNamespace
        from flashengine.engine.model_runner import ModelRunner
        from flashengine.layers.sampler import Sampler
        runner = SimpleNamespace(_spec_capture_greedy=False, _spec_batch_greedy=False,
                                 graph_sample=True, sampler=Sampler())
        logits = torch.zeros(512, 4, device="cuda")
        temp = torch.ones(512, device="cuda")
        topp = torch.ones(512, device="cuda")
        topk = torch.full((512,), 4, device="cuda", dtype=torch.int64)
        ModelRunner._sg_sample(runner, logits, temp, topp, topk, True)
        sampling_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(sampling_graph):
            sampling_ids = ModelRunner._sg_sample(runner, logits, temp, topp, topk, True)
        runner._spec_capture_greedy = True
        greedy_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(greedy_graph):
            greedy_ids = ModelRunner._sg_sample(runner, logits, temp, topp, topk, True)
        logits[:, 2].fill_(0.1)
        greedy_graph.replay()
        self.assertTrue(bool((greedy_ids == 2).all()))
        sampling_graph.replay()
        first = sampling_ids.clone()
        sampling_graph.replay()
        self.assertFalse(torch.equal(first, sampling_ids))
        topk.fill_(1)
        sampling_graph.replay()
        self.assertTrue(bool((sampling_ids == 2).all()))

    def test_device_commit_selects_row_without_touching_dummy(self):
        from flashengine.backends.cuda_gdn import commit
        pool = torch.randn(3, 18, 2, 8, 8, device="cuda")
        expected = pool.clone()
        counts = torch.zeros(4, 5, device="cuda", dtype=torch.int64)
        counts[:, 3] = torch.tensor([3, 1, 2, 1], device="cuda")
        targets = torch.tensor([2, 0, 17, 17], device="cuda", dtype=torch.int32)
        expected[:, 2] = expected[:, 6]
        expected[:, 0] = expected[:, 7]
        commit(pool, counts, targets, 3, 4)
        torch.testing.assert_close(pool, expected, rtol=0, atol=0)

    def test_draft_varlen_gqa_ignores_unused_context_and_replays_lengths(self):
        from flash_attn import flash_attn_varlen_func
        from flashengine.backends.cuda_dflash import attention
        import torch.nn.functional as F
        batch, rows, heads, kv_heads, dim, max_ctx = 3, 8, 8, 2, 64, 24
        q = torch.randn(batch, rows, heads, dim, device="cuda", dtype=torch.bfloat16)
        k, v = [torch.randn(batch, rows, kv_heads, dim, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        kc, vc = [torch.randn(batch, max_ctx, 3, kv_heads, dim, device="cuda", dtype=torch.bfloat16)[:, :, 1] for _ in range(2)]
        lengths = torch.tensor([0, 3, 17], device="cuda", dtype=torch.int64)
        def reference():
            mask = torch.cat(((torch.arange(max_ctx, device="cuda")[None, :] < lengths[:, None]),
                              torch.ones(batch, rows, device="cuda", dtype=torch.bool)), dim=-1)[:, None, None, :]
            full_k = torch.cat((kc, k), dim=1).transpose(1, 2).repeat_interleave(heads // kv_heads, dim=1)
            full_v = torch.cat((vc, v), dim=1).transpose(1, 2).repeat_interleave(heads // kv_heads, dim=1)
            return F.scaled_dot_product_attention(q.transpose(1, 2), full_k, full_v, attn_mask=mask).transpose(1, 2).reshape(-1, heads, dim)
        attention(flash_attn_varlen_func, q, k, v, kc, vc, lengths, dim ** -0.5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = attention(flash_attn_varlen_func, q, k, v, kc, vc, lengths, dim ** -0.5)
        for values in ([0, 3, 17], [24, 0, 8]):
            lengths.copy_(torch.tensor(values, device="cuda"))
            expected = reference()
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0.02, atol=0.008)

    def test_fused_norm_preserves_intermediate_rounding(self):
        from flashengine.backends.cuda_gdn import norm
        import torch.nn.functional as F
        torch.manual_seed(9)
        for dtype in (torch.float16, torch.bfloat16):
            x = torch.randn(7, 2048, device="cuda", dtype=dtype)
            weight = torch.randn(2048, device="cuda", dtype=dtype) * 0.1
            residual = torch.randn_like(x)
            summed = x + residual
            expected = summed.float() * torch.rsqrt(summed.float().square().mean(-1, keepdim=True) + 1e-6)
            expected = (expected * (1 + weight.float())).to(dtype)
            actual, actual_residual = norm(x, weight, 1e-6, residual=residual)
            torch.testing.assert_close(actual_residual, summed, atol=0, rtol=0)
            torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)
            for activation in ("sigmoid", "silu"):
                gate = torch.randn_like(x)
                expected = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
                expected = (weight * expected.to(dtype))
                expected = (expected * (F.sigmoid(gate.float()) if activation == "sigmoid" else F.silu(gate.float()))).to(dtype)
                actual = norm(x, weight, 1e-6, gate=gate, activation=activation)
                torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)

    def test_layer_batched_conv_and_core_match_reference(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch.nn.functional as F
        from flashengine.backends.cuda_gdn import recurrent, conv
        from flashengine.models.qwen3_5 import Qwen3_5GatedDeltaNet
        backend = SimpleNamespace(linear=F.linear, gdn_causal_conv1d=lambda x, w: F.conv1d(x, w, groups=w.shape[0]),
                                  supports_fused_gdn=False, gdn_recurrent=recurrent, gdn_conv=conv)
        config = SimpleNamespace(hidden_size=8, linear_num_key_heads=2, linear_num_value_heads=2,
                                 linear_key_head_dim=128, linear_value_head_dim=128,
                                 linear_conv_kernel_dim=4, hidden_act="silu", rms_norm_eps=1e-6)
        with patch("flashengine.backends._backend", backend), patch("torch.distributed.get_rank", return_value=0), patch("torch.distributed.get_world_size", return_value=1):
            layer = Qwen3_5GatedDeltaNet(config).to(device="cuda", dtype=torch.bfloat16)
        layer.A_log.data = torch.randn(2, device="cuda")
        layer.dt_bias.data = torch.randn(2, device="cuda")
        layer.conv_cache = torch.randn(36, layer.conv_dim, 3, device="cuda", dtype=torch.bfloat16)
        layer.ssm_cache = torch.randn(36, 2, 128, 128, device="cuda")
        inputs = torch.randn(32, layer.conv_dim, device="cuda", dtype=torch.bfloat16)
        b, a = [torch.randn(32, 2, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        context = SimpleNamespace(gdn_row_bases=(3, 19), gdn_row_seg=16,
                                  gdn_carry_slots=torch.tensor([2, 0], device="cuda", dtype=torch.int32))
        before_conv, before_ssm = layer.conv_cache.clone(), layer.ssm_cache.clone()
        expected = layer._forward_rows(inputs, b, a, context)
        expected_conv, expected_ssm = layer.conv_cache.clone(), layer.ssm_cache.clone()
        layer.conv_cache.copy_(before_conv)
        layer.ssm_cache.copy_(before_ssm)
        backend.supports_fused_gdn = True
        actual = layer._forward_rows(inputs, b, a, context)
        torch.testing.assert_close(layer.conv_cache, expected_conv, rtol=0, atol=0)
        torch.testing.assert_close(layer.ssm_cache, expected_ssm, rtol=3e-5, atol=3e-5)
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.008)

    def reference(self, q, k, v, g, beta, cache, carry, rows, base, initial=None):
        outputs = torch.empty_like(v)
        for seq in range(len(carry)):
            state = cache[int(carry[seq])].clone().float()
            if initial is not None and not bool(initial[seq]):
                state.zero_()
            for row in range(rows):
                t = seq * rows + row
                state = state * g[t].exp()[:, None, None]
                memory = (state * k[t, :, :, None]).sum(-2)
                delta = (v[t].float() - memory) * beta[t, :, None]
                state = state + k[t, :, :, None] * delta[:, None, :]
                outputs[t] = (state * q[t, :, :, None]).sum(-2)
                if base is not None:
                    cache[base + t] = state
                elif int(carry[seq]) != cache.shape[0] - 1:
                    cache[int(carry[seq])] = state
        return outputs

    def test_multistep_state_and_strided_values(self):
        from flashengine.backends.cuda_gdn import recurrent
        torch.manual_seed(8)
        for rows, dk, dv in ((1, 4, 4), (3, 48, 96), (16, 128, 128)):
            with self.subTest(rows=rows, dk=dk, dv=dv):
                batch, heads = 3, 2
                q = torch.randn(batch * rows, heads, dk, device="cuda") / dk ** 0.5
                k = torch.nn.functional.normalize(torch.randn_like(q), dim=-1)
                # 模拟 qkv 拆分后的非连续 token/head 步长。
                raw = torch.randn(heads, dv * 2, batch * rows, device="cuda", dtype=torch.bfloat16)
                v = raw[:, :dv].permute(2, 0, 1)
                g = -torch.rand(batch * rows, heads, device="cuda")
                beta = torch.rand_like(g).bfloat16()
                cache = torch.randn(batch + batch * rows + 1, heads, dk, dv, device="cuda")
                expected_cache = cache.clone()
                carry = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
                expected = self.reference(q, k, v, g, beta, expected_cache, carry, rows, batch)
                actual = recurrent(q, k, v, g, beta, cache, carry, rows=rows, base=batch)
                torch.testing.assert_close(cache, expected_cache, atol=2e-5, rtol=3e-5)
                torch.testing.assert_close(actual, expected, atol=0.008, rtol=0.008)

    def test_decode_graph_slot_changes_initial_mask_and_dummy(self):
        from flashengine.backends.cuda_gdn import recurrent
        batch, heads, dk, dv = 3, 2, 32, 32
        q, k = [torch.randn(batch, heads, dk, device="cuda") / dk ** 0.5 for _ in range(2)]
        v = torch.randn(batch, heads, dv, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(batch, heads, device="cuda")
        beta = torch.rand_like(g)
        cache = torch.randn(5, heads, dk, dv, device="cuda")
        carry = torch.tensor([0, 4, 4], device="cuda", dtype=torch.int32)
        initial = torch.tensor([True, False, False], device="cuda")
        recurrent(q, k, v, g, beta, cache, carry, initial=initial)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = recurrent(q, k, v, g, beta, cache, carry, initial=initial)
        for values, flags in (([2, 1, 4], [True, False, False]), ([0, 4, 4], [True, False, False])):
            carry.copy_(torch.tensor(values, device="cuda", dtype=torch.int32))
            initial.copy_(torch.tensor(flags, device="cuda"))
            expected_cache = cache.clone()
            expected = self.reference(q, k, v, g, beta, expected_cache, carry, 1, None, initial)
            graph.replay()
            torch.testing.assert_close(cache, expected_cache, atol=2e-5, rtol=3e-5)
            torch.testing.assert_close(output, expected, atol=0.008, rtol=0.008)


if __name__ == "__main__":
    unittest.main()
