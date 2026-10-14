import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

from flashengine.backends.ascend import Ascend310PBackend
from flashengine.backends.base import BackendBase
from flashengine.engine.recurrent_state import RecurrentStateRuntime
from flashengine.models.qwen3_5 import chunk_gated_delta_rule
from flashengine.models.registry import resolve_target_model


class Qwen35Contracts(unittest.TestCase):
    def test_host_greedy_bypasses_sort_and_random_sampling(self):
        from flashengine.engine.model_runner import ModelRunner
        sampler = unittest.mock.Mock(return_value=torch.tensor([0, 1]))
        runner = SimpleNamespace(sampler=sampler, prepare_sample=lambda seqs: (1, 2, 3))
        logits = torch.tensor([[0., 2.], [3., 1.]])
        seqs = [SimpleNamespace(temperature=0.), SimpleNamespace(temperature=0.)]
        self.assertEqual(ModelRunner._sample(runner, logits, seqs).tolist(), [1, 0])
        sampler.assert_not_called()
        seqs[1].temperature = 0.7
        self.assertEqual(ModelRunner._sample(runner, logits, seqs).tolist(), [0, 1])
        sampler.assert_called_once()
    def test_dense_draft_padding_never_uses_negative_physical_slots(self):
        from flashengine.engine.model_runner import ModelRunner
        sizes = {"v_slots": 12, "v_ctx_lens": 12, "v_pos": 12, "df_ctx_pos": 12,
                 "df_ctx_phys_slots": 12, "df_l1": 4, "df_c": 4, "df_ctxslot": 4,
                 "df_total_lens": 4, "v_cu_k": 5, "btables": 12, "v_btables_exp": 36}
        storage = {name: torch.full((size,), 777, dtype=torch.int64) for name, size in sizes.items()}
        blob = SimpleNamespace(view_i64=lambda name, shape=None: storage[name].view(shape) if shape else storage[name])
        runner = SimpleNamespace(config=SimpleNamespace(spec_num_tokens=2), _pin_M=4, _pin_W=3,
                                 _sblob=blob, block_size=256, dflash_paged=False, _df_pad_ctx_slot=4,
                                 _graph_pad_block=lambda: -1, _graph_pad_slot=lambda: -1, _graph_pad_ctx=lambda: 0,
                                 _df_query_rows=lambda: 3)
        ModelRunner._reset_spec_padding(runner, 3, 4, True)
        self.assertTrue(bool((storage["v_slots"][:9] == 777).all()))
        self.assertEqual(storage["v_slots"][9:].tolist(), [-1, -1, -1])
        self.assertEqual(storage["df_ctx_phys_slots"][9:].tolist(), [4, 4, 4])

    @unittest.skipUnless(torch.cuda.is_available(), "需要 CUDA pinned allocator")
    def test_metadata_device_is_not_inferred_from_default_cpu(self):
        from flashengine.engine.model_runner import MetaBlob
        with torch.device("cpu"):
            blob = MetaBlob([("ids", 4)], [("temps", 4)], device="cuda")
            blob.view_i64("ids").fill_(7)
            blob.h2d()
            self.assertEqual(blob.device_i64().device.type, "cuda")
            self.assertEqual(blob.device_i64().cpu().tolist(), [7] * 4)

    def test_gdn_chunked_prefill_preserves_conv_and_ssm(self):
        from flashengine.models.qwen3_5 import Qwen3_5GatedDeltaNet
        from flashengine.utils.context import get_context, reset_context, set_context
        backend = SimpleNamespace(linear=F.linear, gdn_causal_conv1d=lambda x, w: F.conv1d(x, w, groups=w.shape[0]))
        config = SimpleNamespace(hidden_size=8, linear_num_key_heads=2, linear_num_value_heads=2,
                                 linear_key_head_dim=4, linear_value_head_dim=4,
                                 linear_conv_kernel_dim=4, hidden_act="silu", rms_norm_eps=1e-6)
        with patch("flashengine.backends._backend", backend), patch("torch.distributed.get_rank", return_value=0), patch("torch.distributed.get_world_size", return_value=1):
            layer = Qwen3_5GatedDeltaNet(config)
        for parameter in layer.parameters():
            parameter.data.normal_(std=0.1)
        layer.conv_cache = torch.zeros(2, layer.conv_dim, 3)
        layer.ssm_cache = torch.zeros(2, 2, 4, 4)
        inputs = torch.randn(11, 8)

        def run(values, initial):
            set_context(True, cu_seqlens_q=torch.tensor([0, values.shape[0]]))
            context = get_context()
            context.gdn_state_slots = torch.tensor([0])
            context.gdn_has_initial = torch.tensor([initial])
            return layer(values)

        expected = run(inputs, False)
        expected_conv, expected_ssm = layer.conv_cache.clone(), layer.ssm_cache.clone()
        actual = torch.cat((run(inputs[:4], False), run(inputs[4:], True)))
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(layer.conv_cache, expected_conv)
        torch.testing.assert_close(layer.ssm_cache, expected_ssm, atol=1e-5, rtol=1e-4)
        reset_context()

    def test_adapter_resolves_shell_text_and_output_aux(self):
        text = SimpleNamespace(model_type="qwen3_5_text", num_hidden_layers=24)
        adapter = resolve_target_model(SimpleNamespace(model_type="qwen3_5", text_config=text))
        self.assertIs(adapter.model_config(SimpleNamespace(text_config=text)), text)
        draft = SimpleNamespace(dflash_config={"target_layer_ids": [1, 6, 11, 16, 21]})
        self.assertEqual(adapter.resolve_aux_layers("dflash", draft, (), text), (1, 6, 11, 16, 21))
        self.assertTrue(adapter.recurrent_state)
        with self.assertRaises(NotImplementedError):
            adapter.validate_spec_method("eagle3")

    def test_scheduler_does_not_preempt_inflight_prefill(self):
        from flashengine.engine.scheduler import Scheduler
        from flashengine.engine.sequence import Sequence
        from flashengine.sampling_params import SamplingParams
        config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=128,
                                 enable_chunked_prefill=True, max_num_partial_prefills=1,
                                 eos=-1, kvcache_block_size=256, max_model_len=512,
                                 num_kvcache_blocks=1, enable_prefix_cache=False,
                                 keep_last_block_unhashed=False,
                                 backend_impl=SimpleNamespace(reserve_dummy_kv_block=lambda: False))
        scheduler = Scheduler(config)
        first = Sequence([1, 2, 3], SamplingParams(temperature=0, max_tokens=8))
        second = Sequence([4, 5, 6], SamplingParams(temperature=0, max_tokens=8))
        scheduler.add(first)
        scheduler.add(second)
        selected, prefill = scheduler.schedule()
        self.assertTrue(prefill)
        self.assertEqual(selected, [first])
        self.assertEqual(len(first.block_table), 1)
        self.assertEqual(scheduler.stats["preemptions"], 0)
        scheduler.postprocess(selected, [7], True)
        selected, prefill = scheduler.schedule()
        scheduler.postprocess(selected, [8], prefill)
        released = []
        scheduler.on_preempt = lambda seq: released.append(seq.seq_id)
        selected, prefill = scheduler.schedule()
        self.assertTrue(prefill)
        self.assertEqual(selected, [second])
        self.assertEqual(released, [first.seq_id])
        self.assertFalse(first.block_table)

    def test_310p_short_conv_and_page_size(self):
        backend = Ascend310PBackend.__new__(Ascend310PBackend)
        inputs = torch.randn(2, 6, 13)
        weight = torch.randn(6, 1, 4)
        torch.testing.assert_close(backend.gdn_causal_conv1d(inputs, weight),
                                   F.conv1d(inputs, weight, groups=6), atol=1e-6, rtol=1e-5)
        config = SimpleNamespace(kvcache_block_size=256,
                                 model_config=SimpleNamespace(head_dim=256, hidden_size=2048, num_attention_heads=8),
                                 tensor_parallel_size=1, is_moe=False)
        backend.normalize_config(config)
        backend.validate_config(config)
        self.assertEqual(config.kvcache_block_size, 64)

    def test_gdn_chunk_matches_recurrent_reference(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
        torch.manual_seed(3)
        q, k, v = [torch.randn(1, 11, 2, 4) for _ in range(3)]
        g = -torch.rand(1, 11, 2)
        beta = torch.rand(1, 11, 2)
        initial = torch.randn(2, 4, 4)
        out, state = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=initial)
        expected, expected_state = torch_recurrent_gated_delta_rule(
            q, k, v, g, beta, initial_state=initial[None], output_final_state=True,
            use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(out, expected[0], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(state, expected_state[0], atol=2e-6, rtol=1e-5)

    def test_acceptance_restores_only_accepted_row_and_releases_slots(self):
        layers = [SimpleNamespace(), SimpleNamespace()]
        spec = {"conv_shape": (3, 2), "ssm_shape": (2, 4, 4),
                "conv_dtype": torch.float32, "ssm_dtype": torch.float32}
        model = SimpleNamespace(model=SimpleNamespace(gdn_layers=layers, get_gdn_state_spec=lambda: spec))
        config = SimpleNamespace(backend_impl=SimpleNamespace(device_name="cpu"), max_num_seqs=2,
                                 spec_model="draft", spec_num_tokens=2)
        runtime = RecurrentStateRuntime(model, config)
        seqs = [SimpleNamespace(seq_id=10), SimpleNamespace(seq_id=20)]
        slots = [runtime.get_slot(seq) for seq in seqs]
        for pool in (runtime.conv_pool, runtime.ssm_pool):
            for index in range(runtime.slot_count):
                pool[:, index].fill_(index)
        runtime.commit_verify(seqs, [2, 1])
        for layer in layers:
            self.assertTrue(bool((layer.conv_cache[slots[0]] == 3).all()))
            self.assertTrue(bool((layer.ssm_cache[slots[1]] == 5).all()))
        runtime.drop(10)
        self.assertEqual(runtime.get_slot(SimpleNamespace(seq_id=30)), slots[0])
        runtime.clear()
        self.assertEqual(runtime.snapshot()["used_slots"], 0)

    def test_shared_moe_matches_naive_and_fused_weight_loading(self):
        from flashengine.models.qwen3_5 import Qwen3_5SparseMoeBlock, Qwen3_5ForCausalLM
        from flashengine.models.qwen3_moe import Qwen3MoeForCausalLM
        backend = SimpleNamespace(linear=F.linear, silu_and_mul=lambda x: F.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1])
        backend.moe_router = lambda logits, k, renorm: BackendBase.moe_router(backend, logits, k, renorm)
        backend.fused_moe = lambda *args: BackendBase.fused_moe(backend, *args)
        backend._MOE_GATHER_BYTES = 256 << 20
        config = SimpleNamespace(hidden_size=8, num_experts=3, num_experts_per_tok=2,
                                 moe_intermediate_size=6, shared_expert_intermediate_size=5,
                                 hidden_act="silu", norm_topk_prob=True)
        with patch("flashengine.backends._backend", backend), patch("torch.distributed.get_rank", return_value=0), patch("torch.distributed.get_world_size", return_value=1):
            block = Qwen3_5SparseMoeBlock(config, 0)
        for parameter in block.parameters():
            parameter.data.normal_(std=0.1)
        inputs = torch.randn(3, 8)
        weights, ids = backend.moe_router(block.gate(inputs), 2, True)
        expected = torch.zeros_like(inputs)
        for token in range(inputs.shape[0]):
            for branch in range(2):
                expert = int(ids[token, branch])
                gu = F.linear(inputs[token], block.experts.w13_weight[expert])
                gate, up = gu.chunk(2)
                expected[token] += F.linear(F.silu(gate) * up, block.experts.w2_weight[expert]) * weights[token, branch]
        expected += block.shared_expert(inputs) * torch.sigmoid(block.shared_expert_gate(inputs))
        torch.testing.assert_close(block(inputs), expected, atol=1e-6, rtol=1e-5)
        wrapper = SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=block)]),
                                  _EXPERT_FIELDS=Qwen3_5ForCausalLM._EXPERT_FIELDS)
        fused = torch.randn_like(block.experts.w13_weight)
        Qwen3MoeForCausalLM._load_expert_weight(wrapper, "model.layers.0.mlp.experts.gate_up_proj.weight", fused)
        torch.testing.assert_close(block.experts.w13_weight, fused)
