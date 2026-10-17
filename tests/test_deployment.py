"""部署契约的纯主机回归，不导入 torch/transformers。"""
import json
from pathlib import Path
import struct
import tempfile
import unittest

from flashengine.deployment import normalize_model, parse_json, publish_contracts, safetensors_header, sha256, weight_plan, expected_weights


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = dict(model_type="qwen3", architectures=["Qwen3ForCausalLM"], hidden_size=8, intermediate_size=16, vocab_size=20, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=4)

    def tensor_file(self, tensors):
        header, offset = {}, 0
        for name, shape in tensors.items():
            count = 2
            for dim in shape:
                count *= dim
            header[name] = dict(dtype="F16", shape=shape, data_offsets=[offset, offset + count])
            offset += count
        encoded = json.dumps(header).encode()
        path = self.root / "model.safetensors"
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
        return path

    def test_valid_complete_weights(self):
        cfg = normalize_model(self.raw)
        tensors = expected_weights(cfg)
        self.tensor_file(tensors)
        plan, _ = weight_plan(self.root, cfg)
        self.assertEqual(len(plan["entries"]), len(tensors))

    def test_missing_weight_is_rejected(self):
        cfg = normalize_model(self.raw)
        tensors = expected_weights(cfg)
        del tensors["norm.weight"]
        self.tensor_file(tensors)
        with self.assertRaisesRegex(ValueError, "缺失"):
            weight_plan(self.root, cfg)

    def test_shape_and_truncation(self):
        path = self.tensor_file({"w": [3, 4]})
        self.assertEqual(safetensors_header(path)["w"]["bytes"], 24)
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            safetensors_header(path)

    def test_moe_and_wrong_arch_rejected(self):
        with self.assertRaises(ValueError):
            normalize_model(dict(self.raw, model_type="qwen3_moe"))
        with self.assertRaises(ValueError):
            normalize_model(dict(self.raw, num_experts=4))

    def test_invalid_gqa(self):
        with self.assertRaises(ValueError):
            normalize_model(dict(self.raw, num_key_value_heads=3))

    def test_moe_mixed_layers_and_expert_layouts(self):
        raw = dict(self.raw, model_type="qwen3_moe", architectures=["Qwen3MoeForCausalLM"], num_experts=4, num_experts_per_tok=2, moe_intermediate_size=6, decoder_sparse_step=2)
        for fused in (False, True):
            cfg = normalize_model(raw)
            if fused:
                cfg["expert_layouts"] = {"1": "fused"}
            tensors = expected_weights(cfg)
            self.assertIn("layers.0.mlp.gate_proj.weight", tensors)
            self.assertIn("layers.1.mlp.gate.weight", tensors)
            self.tensor_file(tensors)
            plan, _ = weight_plan(self.root, cfg)
            self.assertEqual(len(plan["entries"]), len(tensors))
            self.assertEqual(cfg["expert_layouts"]["1"], "fused" if fused else "separate")

    def test_moe_invalid_routes(self):
        raw = dict(self.raw, model_type="qwen3_moe", architectures=["Qwen3MoeForCausalLM"], num_experts=4, num_experts_per_tok=2, moe_intermediate_size=6)
        for extra in (dict(num_experts_per_tok=5), dict(decoder_sparse_step=0), dict(mlp_only_layers=[2]), dict(num_shared_experts=1)):
            with self.assertRaises(ValueError):
                normalize_model(dict(raw, **extra))

    def test_strict_json(self):
        for text in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.assertRaises(ValueError):
                parse_json(text)

    def test_duplicate_safetensors_header(self):
        path = self.root / "duplicate.safetensors"
        for encoded in (b'{"w":{},"w":{}}', b'{"__metadata__":{"a":1}}', b' {"__metadata__":{}}'):
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded)
            with self.assertRaises(ValueError):
                safetensors_header(path)

    def test_contract_archive_is_reproducible_and_immutable(self):
        source = self.root / "source"
        source.mkdir()
        (source / "a.json").write_text('{"version":1}', encoding="utf-8")
        a, b = self.root / "a.tar.gz", self.root / "b.tar.gz"
        self.assertEqual(publish_contracts(source, a), publish_contracts(source, b))
        self.assertEqual(sha256(a), sha256(b))
        with self.assertRaises(FileExistsError):
            publish_contracts(source, a)

    def test_replay_crash_output_is_preserved(self):
        from scripts.check_native_replay import parse_native_output
        for stdout in ("", "Segmentation fault", "[]", "null", "{invalid"):
            result, error = parse_native_output(stdout)
            self.assertEqual(result, {})
            self.assertTrue(error)
        result, error = parse_native_output('{"passed":false,"cases":[]}')
        self.assertIsNone(error)
        self.assertFalse(result["passed"])

    def test_spec_logits_are_not_global_token_samples(self):
        from scripts.check_native_replay import compare
        samples = [dict(values=[[3., 2.], [5., 4.]], ids=[[1, 2], [1, 2]])] * 2
        golden = dict(cases=[dict(output_token_ids=[1, 2], rounds=[], samples=samples)])
        actual = dict(cases=[dict(token_ids=[1, 3], passed=False, rounds=[], samples=samples)])
        diagnostic = compare(golden, actual)[0]
        self.assertEqual(diagnostic["common_prefix"], 1)
        self.assertIn("note", diagnostic["first_difference"]["oracle"])
        self.assertNotIn("top2_margin", diagnostic["first_difference"]["oracle"])


if __name__ == "__main__":
    unittest.main()
