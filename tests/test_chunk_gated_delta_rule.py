"""Bitwise regression against the pre-padding-optimization GDN chunk rule.

Run: python -m unittest tests.test_chunk_gated_delta_rule -v
The two functions are loaded from the model source without importing the model's
accelerator and distributed dependencies; their actual function bodies are used.
"""
import ast
import __future__
from pathlib import Path
import unittest

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None


def _load_model_functions(scan=False, scan_group=128, mm_half=False):
    source = Path(__file__).resolve().parents[1] / "flashengine" / "models" / "qwen3_5.py"
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_l2norm", "chunk_gated_delta_rule", "_gdn_mm"}]
    if len(functions) != 3:
        raise AssertionError("GDN functions missing from model source")
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"torch": torch, "F": F, "_GDN_SCAN": scan, "_GDN_SCAN_GROUP": scan_group,
                 "_GDN_MM_HALF": mm_half}
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace["chunk_gated_delta_rule"]


def _before_optimization(query, key, value, g, beta, chunk_size=64,
                         initial_state=None, use_qk_l2norm_in_kernel=True):
    """Frozen pre-change implementation, including all padded substitution rows."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = query * torch.rsqrt((query * query).sum(dim=-1, keepdim=True) + 1e-6)
        key = key * torch.rsqrt((key * key).sum(dim=-1, keepdim=True) + 1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    query = query * (1 / (query.shape[-1] ** 0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value).unsqueeze(0)
    )
    core_attn_out = torch.zeros_like(value)
    for i in range(total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length].transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out.squeeze(0), last_recurrent_state.squeeze(0)


def _inputs(length, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, length, 2)
    tensors = tuple(torch.randn(*shape, dim, generator=generator).to(dtype) for dim in (4, 4, 3))
    tensors += (-torch.rand(*shape, generator=generator), torch.rand(*shape, generator=generator).to(dtype))
    return tensors, generator


@unittest.skipIf(torch is None, "requires PyTorch")
class ChunkGatedDeltaRuleTest(unittest.TestCase):
    def setUp(self):
        previous_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)

    def test_matches_pre_optimization_bitwise(self):
        current = _load_model_functions()
        for length in (1, 2, 15, 63, 64, 65):
            for with_initial_state in (False, True):
                for dtype in (torch.float32, torch.float16, torch.bfloat16):
                    for normalize in (False, True):
                        with self.subTest(length=length, initial=with_initial_state,
                                          dtype=dtype, normalize=normalize):
                            inputs, generator = _inputs(length, dtype, 1000 + length)
                            initial = (torch.randn(2, 4, 3, generator=generator)
                                       if with_initial_state else None)
                            initial_copy = initial.clone() if initial is not None else None
                            expected = _before_optimization(*inputs, initial_state=initial,
                                use_qk_l2norm_in_kernel=normalize)
                            actual = current(*inputs, initial_state=initial,
                                use_qk_l2norm_in_kernel=normalize)
                            for name, got, want in zip(("output", "state"), actual, expected):
                                self.assertEqual(got.shape, want.shape, name)
                                self.assertEqual(got.dtype, want.dtype, name)
                                self.assertTrue(torch.isfinite(got).all().item(), name)
                                self.assertTrue(torch.equal(got.contiguous().view(torch.uint8),
                                                            want.contiguous().view(torch.uint8)), name)
                            if initial is not None:
                                self.assertTrue(torch.equal(initial, initial_copy), "initial state mutated")


    def test_scan_variant_matches_row_loop(self):
        """chunk-parallel scan 路线与逐 chunk 递推只差浮点结合顺序。"""
        row = _load_model_functions(scan=False)
        scan = _load_model_functions(scan=True)
        # 分组边界（64 chunk = 一条组，正好覆盖 length>64 的多组串接）
        grouped = _load_model_functions(scan=True, scan_group=2)
        for length in (1, 2, 63, 64, 65, 129, 200):
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                with self.subTest(length=length, dtype=dtype):
                    inputs, generator = _inputs(length, dtype, 2000 + length)
                    initial = torch.randn(2, 4, 3, generator=generator) * 0.1
                    want = row(*inputs, initial_state=initial.clone())
                    got = scan(*inputs, initial_state=initial.clone())
                    tol = 1e-5 if dtype == torch.float32 else 6e-2
                    grouped_out = grouped(*inputs, initial_state=initial.clone())
                    for variant, result in (("grouped", grouped_out), ("scan", got)):
                        for name, a, b in zip(("output", "state"), result, want):
                            self.assertEqual(a.shape, b.shape, name)
                            self.assertTrue(torch.isfinite(a).all().item(), name)
                            diff = (a.double() - b.double()).abs().max().item()
                            self.assertLessEqual(diff, tol, f"{variant} {name} {dtype} {length}: {diff:.3e}")

    def test_inline_checkpoints_match_segmented_prefill(self):
        inputs, generator = _inputs(193, torch.float16, 52)
        initial = torch.randn(2, 4, 3, generator=generator).half()
        offsets = (64, 128, 192)
        for scan in (False, True):
            for mm_half in (False, True):
                with self.subTest(scan=scan, mm_half=mm_half):
                    rule = _load_model_functions(scan=scan, scan_group=2, mm_half=mm_half)
                    state_dtype = torch.float16 if mm_half else torch.float32
                    expected_states = {}
                    expected_output = []
                    state = initial.to(state_dtype)
                    begin = 0
                    for end in (*offsets, 193):
                        output, state = rule(*(x[:, begin:end] for x in inputs), initial_state=state)
                        expected_output.append(output)
                        state = state.to(state_dtype)
                        expected_states[end] = state
                        begin = end

                    captured = {}

                    def capture(offset, state):
                        stored = state.to(state_dtype)
                        captured[offset] = stored
                        return stored.to(state.dtype)

                    output, state = rule(*inputs, initial_state=initial, state_checkpoints=offsets,
                                         capture_state=capture)
                    self.assertEqual(tuple(captured), offsets)
                    tolerance = 1e-3 if mm_half else 1e-5
                    for offset, saved in captured.items():
                        torch.testing.assert_close(saved, expected_states[offset], atol=tolerance, rtol=0)
                    torch.testing.assert_close(state.to(state_dtype), expected_states[193], atol=tolerance, rtol=0)
                    torch.testing.assert_close(output, torch.cat(expected_output), atol=tolerance, rtol=0)


if __name__ == "__main__":
    unittest.main()
