"""``scripts.merge_lora`` 的 CPU 契约测试：数学等价、分片自洽、错误输入干净失败。

运行（仓库根目录，需要 torch + safetensors；无依赖时整体 skip）::

    python -m unittest tests.test_merge_lora -v
"""

import json
import os
import tempfile
import unittest

try:
    import torch
    from safetensors.torch import load_file, save_file

    _HAS_DEPS = True
except Exception:  # pragma: no cover - 依赖缺失时 skip
    torch = None
    _HAS_DEPS = False

from scripts import merge_lora


def _write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _make_base(root: str, dtype=None) -> dict:
    """两层小基座：两个分片 + index + config + 一个辅助文件。"""
    dtype = dtype or torch.float32
    shard1 = {
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight":
            torch.randn(12, 8, dtype=torch.float32).to(dtype),
        "model.language_model.layers.0.mlp.down_proj.weight":
            torch.randn(8, 16, dtype=torch.float32).to(dtype),
    }
    shard2 = {
        "model.language_model.embed_tokens.weight":
            torch.randn(32, 8, dtype=torch.float32).to(dtype),
        "mtp.layers.0.self_attn.q_proj.weight":
            torch.randn(8, 8, dtype=torch.float32).to(dtype),
    }
    weight_map = {}
    for index, (name, tensors) in enumerate(
            (("model.safetensors-00001-of-00002.safetensors", shard1),
             ("model.safetensors-00002-of-00002.safetensors", shard2))):
        save_file(tensors, os.path.join(root, name))
        for key in tensors:
            weight_map[key] = name
    _write_json(os.path.join(root, "model.safetensors.index.json"),
                {"metadata": {"total_size": 0}, "weight_map": weight_map})
    _write_json(os.path.join(root, "config.json"),
                {"architectures": ["Qwen3_5ForConditionalGeneration"],
                 "text_config": {"dtype": "bfloat16", "num_hidden_layers": 2}})
    with open(os.path.join(root, "chat_template.jinja"), "w", encoding="utf-8") as handle:
        handle.write("{{ messages }}\n")
    return {"shard1": shard1, "shard2": shard2}


def _make_adapter(root: str, *, r=2, alpha=4, use_dora=False, missing_b=False,
                  unmatched=False) -> dict:
    a_qkv = torch.randn(r, 8)
    b_qkv = torch.randn(12, r)
    a_down = torch.randn(r, 16)
    b_down = torch.randn(8, r)
    tensors = {
        "base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_A.weight": a_qkv,
        "base_model.model.model.language_model.layers.0.linear_attn.in_proj_qkv.lora_B.weight": b_qkv,
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight": a_down,
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight": b_down,
        # 引擎同口径跳过的模块（conv1d 不是 Linear）
        "base_model.model.model.language_model.layers.0.linear_attn.conv1d.lora_A.weight":
            torch.randn(r, 4),
        "base_model.model.model.language_model.layers.0.linear_attn.conv1d.lora_B.weight":
            torch.randn(4, r),
    }
    if missing_b:
        tensors.pop(
            "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_B.weight"
        )
    if unmatched:
        tensors["base_model.model.model.language_model.layers.9.mlp.down_proj.lora_A.weight"] = \
            torch.randn(r, 16)
        tensors["base_model.model.model.language_model.layers.9.mlp.down_proj.lora_B.weight"] = \
            torch.randn(8, r)
    if use_dora:
        tensors[
            "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_magnitude_vector"
        ] = torch.randn(8, 1)
    save_file(tensors, os.path.join(root, merge_lora.ADAPTER_WEIGHT_FILE))
    _write_json(os.path.join(root, merge_lora.ADAPTER_CONFIG_FILE), {
        "peft_type": "LORA",
        "r": r,
        "lora_alpha": alpha,
        "use_rslora": False,
        "use_dora": use_dora,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "/models/base",
        "target_modules": "^(model\\.language_model.*)$",
    })
    return {"a_qkv": a_qkv, "b_qkv": b_qkv, "a_down": a_down, "b_down": b_down}


@unittest.skipUnless(_HAS_DEPS, "需要 torch 与 safetensors")
class MergeLoraTest(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.base_dir = os.path.join(self.root, "base")
        self.adapter_dir = os.path.join(self.root, "adapter")
        self.out_dir = os.path.join(self.root, "out")
        os.makedirs(self.base_dir)
        os.makedirs(self.adapter_dir)
        self.base = _make_base(self.base_dir)
        self.weights = _make_adapter(self.adapter_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, *extra: str) -> int:
        argv = ["--base", self.base_dir, "--adapter", self.adapter_dir,
                "--out", self.out_dir, "--verify-tensors", "0", *extra]
        return merge_lora.main(argv)

    def test_merge_math_and_shards(self) -> None:
        # 分片阈值压到 1 KiB，强制产出多片，验证 index 与分片自洽。
        self.assertEqual(self._run("--dtype", "float32", "--shard-size-gb", "0.000001"), 0)
        index = json.load(open(os.path.join(self.out_dir, "model.safetensors.index.json"),
                               encoding="utf-8"))
        shard_names = sorted(set(index["weight_map"].values()))
        self.assertGreater(len(shard_names), 1)
        for name in shard_names:
            self.assertTrue(os.path.isfile(os.path.join(self.out_dir, name)))
        merged = {}
        for name in shard_names:
            merged.update(load_file(os.path.join(self.out_dir, name)))
        qkv_key = "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
        down_key = "model.language_model.layers.0.mlp.down_proj.weight"
        scaling = 4 / 2
        expected_qkv = self.base["shard1"][qkv_key] + scaling * (
            self.weights["b_qkv"] @ self.weights["a_qkv"])
        expected_down = self.base["shard1"][down_key] + scaling * (
            self.weights["b_down"] @ self.weights["a_down"])
        self.assertTrue(torch.allclose(merged[qkv_key], expected_qkv, atol=1e-5))
        self.assertTrue(torch.allclose(merged[down_key], expected_down, atol=1e-5))
        # 未参与融合的张量按原值透传
        emb = "model.language_model.embed_tokens.weight"
        self.assertTrue(torch.equal(merged[emb], self.base["shard2"][emb]))
        mtp = "mtp.layers.0.self_attn.q_proj.weight"
        self.assertTrue(torch.equal(merged[mtp], self.base["shard2"][mtp]))
        self.assertEqual(len(index["weight_map"]), 4)
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, "chat_template.jinja")))
        config = json.load(open(os.path.join(self.out_dir, "config.json"), encoding="utf-8"))
        self.assertEqual(config["text_config"]["dtype"], "float32")

    def test_dtype_follows_flag(self) -> None:
        self.assertEqual(self._run("--dtype", "float16"), 0)
        index = json.load(open(os.path.join(self.out_dir, "model.safetensors.index.json"),
                               encoding="utf-8"))
        merged = {}
        for name in sorted(set(index["weight_map"].values())):
            merged.update(load_file(os.path.join(self.out_dir, name)))
        for key in ("model.language_model.layers.0.mlp.down_proj.weight",
                    "model.language_model.embed_tokens.weight",
                    "mtp.layers.0.self_attn.q_proj.weight"):
            self.assertEqual(merged[key].dtype, torch.float16, key)
        config = json.load(open(os.path.join(self.out_dir, "config.json"), encoding="utf-8"))
        self.assertEqual(config["text_config"]["dtype"], "float16")

    def test_check_mode_writes_nothing(self) -> None:
        self.assertEqual(self._run("--check"), 0)
        self.assertFalse(os.path.exists(self.out_dir))

    def test_unmatched_module_fails(self) -> None:
        shutil_rm = os.path.join(self.adapter_dir, merge_lora.ADAPTER_WEIGHT_FILE)
        os.remove(shutil_rm)
        _make_adapter(self.adapter_dir, unmatched=True)
        self.assertEqual(self._run("--dtype", "float32"), 2)
        self.assertFalse(os.path.exists(self.out_dir))

    def test_dora_rejected(self) -> None:
        os.remove(os.path.join(self.adapter_dir, merge_lora.ADAPTER_WEIGHT_FILE))
        _make_adapter(self.adapter_dir, use_dora=True)
        self.assertEqual(self._run("--dtype", "float32"), 2)

    def test_missing_pair_rejected(self) -> None:
        os.remove(os.path.join(self.adapter_dir, merge_lora.ADAPTER_WEIGHT_FILE))
        _make_adapter(self.adapter_dir, missing_b=True)
        self.assertEqual(self._run("--dtype", "float32"), 2)

    def test_non_empty_out_dir_needs_overwrite(self) -> None:
        os.makedirs(self.out_dir)
        with open(os.path.join(self.out_dir, "keep.txt"), "w", encoding="utf-8") as handle:
            handle.write("x")
        self.assertEqual(self._run("--dtype", "float32"), 2)
        self.assertEqual(self._run("--dtype", "float32", "--overwrite"), 0)

    def test_scaling_patterns(self) -> None:
        meta = {"rank": 8, "alpha": 32, "use_rslora": False,
                "rank_pattern": {}, "alpha_pattern": {}}
        self.assertAlmostEqual(merge_lora.adapter_scaling(meta, "x"), 4.0)
        meta_rs = dict(meta, use_rslora=True)
        self.assertAlmostEqual(merge_lora.adapter_scaling(meta_rs, "x"), 32 / (8 ** 0.5))
        meta_pat = dict(meta, rank_pattern={"^a\\..*": 4}, alpha_pattern={"^a\\..*": 8})
        self.assertAlmostEqual(merge_lora.adapter_scaling(meta_pat, "a.q_proj"), 2.0)



def _make_moe_base(root: str) -> dict:
    """MoE 小基座：shared_expert + shared_expert_gate + 逐专家 experts.<i>。"""
    hidden, shared_inter, expert_inter, experts = 8, 6, 5, 2
    tensors = {
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight":
            torch.randn(shared_inter, hidden),
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight":
            torch.randn(shared_inter, hidden),
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight":
            torch.randn(hidden, shared_inter),
        "model.language_model.layers.0.mlp.shared_expert_gate.weight":
            torch.randn(1, hidden),
        "model.language_model.embed_tokens.weight": torch.randn(32, hidden),
    }
    for expert in range(experts):
        for proj, shape in (("gate_proj", (expert_inter, hidden)),
                            ("up_proj", (expert_inter, hidden)),
                            ("down_proj", (hidden, expert_inter))):
            tensors[f"model.language_model.layers.0.mlp.experts.{expert}.{proj}.weight"] = \
                torch.randn(*shape)
    save_file(tensors, os.path.join(root, "model.safetensors"))
    _write_json(os.path.join(root, "config.json"),
                {"architectures": ["Qwen3_5MoeForConditionalGeneration"],
                 "text_config": {"dtype": "bfloat16", "num_hidden_layers": 1}})
    return tensors


def _make_moe_adapter(root: str, *, r: int = 4, alpha: int = 8) -> dict:
    """仿 Qwen3.5-35B-A3B：shared_expert 三项 + shared_expert_gate + 一个逐专家目标。"""
    hidden, shared_inter, expert_inter = 8, 6, 5
    prefix = "base_model.model.model.language_model.layers.0.mlp."
    # 只造 A/B；被融合的基座张量由 _make_moe_base 生成，断言里从基座取。
    shapes = {
        "shared_expert.gate_proj": (shared_inter, hidden),
        "shared_expert.up_proj": (shared_inter, hidden),
        "shared_expert.down_proj": (hidden, shared_inter),
        "shared_expert_gate": (1, hidden),
        "experts.0.gate_proj": (expert_inter, hidden),
    }
    tensors = {}
    expected = {}
    for module, (out_features, in_features) in shapes.items():
        a = torch.randn(r, in_features)
        b = torch.randn(out_features, r)
        tensors[prefix + module + ".lora_A.weight"] = a
        tensors[prefix + module + ".lora_B.weight"] = b
        expected[module] = (a, b)
    save_file(tensors, os.path.join(root, merge_lora.ADAPTER_WEIGHT_FILE))
    _write_json(os.path.join(root, merge_lora.ADAPTER_CONFIG_FILE), {
        "peft_type": "LORA",
        "r": r,
        "lora_alpha": alpha,
        "use_rslora": False,
        "use_dora": False,
        "bias": "none",
        "fan_in_fan_out": False,
        "modules_to_save": [],
        "target_parameters": None,
        "base_model_name_or_path": "/models/Qwen3.5-35B-A3B",
        "target_modules": "^(model\\.language_model.*)$",
    })
    return expected


@unittest.skipUnless(_HAS_DEPS, "需要 torch 与 safetensors")
class MergeLoraMoeTest(unittest.TestCase):
    """MoE 基座（Qwen3.5-35B-A3B 形状）：shared_expert / shared_expert_gate / 逐专家。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.base_dir = os.path.join(self.root, "base")
        self.adapter_dir = os.path.join(self.root, "adapter")
        self.out_dir = os.path.join(self.root, "out")
        os.makedirs(self.base_dir)
        os.makedirs(self.adapter_dir)
        self.base = _make_moe_base(self.base_dir)
        self.expected = _make_moe_adapter(self.adapter_dir, r=4, alpha=8)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_shared_expert_and_expert_merge(self) -> None:
        argv = ["--base", self.base_dir, "--adapter", self.adapter_dir,
                "--out", self.out_dir, "--dtype", "float32", "--verify-tensors", "0"]
        self.assertEqual(merge_lora.main(argv), 0)
        out_index = json.load(open(
            os.path.join(self.out_dir, "model.safetensors.index.json"), encoding="utf-8"))
        merged = load_file(os.path.join(
            self.out_dir, sorted(set(out_index["weight_map"].values()))[0]))
        scaling = 8 / 4
        for module, (a, b) in self.expected.items():
            key = f"model.language_model.layers.0.mlp.{module}.weight"
            expected = self.base[key] + scaling * (b @ a)
            self.assertTrue(torch.allclose(merged[key], expected, atol=1e-5), key)
        # 未被适配器命中的专家保持原值
        untouched = "model.language_model.layers.0.mlp.experts.1.up_proj.weight"
        self.assertTrue(torch.equal(merged[untouched], self.base[untouched]))

    def test_audit_reports_moe_experts(self) -> None:
        report = merge_lora.audit_adapter(self.adapter_dir)
        self.assertEqual(report["modules"], 5)
        self.assertEqual(report["moe_expert_modules"], 1)
        self.assertEqual(report["skipped"], [])
        self.assertEqual(report["verdict"], "OK")

    def test_audit_does_not_need_base_or_out(self) -> None:
        self.assertEqual(merge_lora.main(["--adapter", self.adapter_dir, "--audit-adapter"]), 0)
        self.assertFalse(os.path.exists(self.out_dir))


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_HAS_DEPS, "需要 torch 与 safetensors")
class MergeLoraAutoTest(unittest.TestCase):
    """默认行为：dtype 自动、校验自动、--out 可省略。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.base_dir = os.path.join(self.root, "base")
        self.adapter_dir = os.path.join(self.root, "adapter")
        os.makedirs(self.base_dir)
        os.makedirs(self.adapter_dir)
        self.base = _make_base(self.base_dir)
        self.weights = _make_adapter(self.adapter_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_detect_dtype_prefers_declared_field(self) -> None:
        name, source = merge_lora.detect_base_dtype(self.base_dir)
        self.assertEqual(name, "bfloat16")
        self.assertIn("text_config", source)

        config_path = os.path.join(self.base_dir, "config.json")
        config = json.load(open(config_path, encoding="utf-8"))
        config["torch_dtype"] = "float16"
        _write_json(config_path, config)
        name, source = merge_lora.detect_base_dtype(self.base_dir)
        self.assertEqual(name, "float16")
        self.assertIn("config.json", source)

    def test_detect_dtype_falls_back(self) -> None:
        empty_dir = os.path.join(self.root, "empty")
        os.makedirs(empty_dir)
        name, source = merge_lora.detect_base_dtype(empty_dir)
        self.assertEqual(name, "bfloat16")
        self.assertIn("兜底", source)

    def test_module_pattern_normalizes_layer_index(self) -> None:
        self.assertEqual(
            merge_lora.module_pattern("model.language_model.layers.17.mlp.down_proj"),
            "model.language_model.layers.N.mlp.down_proj")

    def test_verify_spec_auto_covers_every_pattern(self) -> None:
        planned = {
            "k.0.a": ("model.layers.0.self_attn.q_proj", 4.0),
            "k.0.b": ("model.layers.1.self_attn.q_proj", 4.0),
            "k.0.c": ("model.layers.0.mlp.down_proj", 4.0),
            "k.0.d": ("model.layers.1.mlp.down_proj", 4.0),
            "k.0.e": ("model.layers.0.mlp.up_proj", 4.0),
        }
        picks = merge_lora.resolve_verify_spec("auto", list(planned), planned)
        patterns = {merge_lora.module_pattern(planned[key][0]) for key in picks}
        self.assertEqual(len(patterns), 3)
        self.assertEqual(merge_lora.resolve_verify_spec("all", list(planned), planned),
                         sorted(planned))
        self.assertEqual(merge_lora.resolve_verify_spec("off", list(planned), planned), [])
        self.assertEqual(len(merge_lora.resolve_verify_spec("2", list(planned), planned)), 2)
        with self.assertRaises(merge_lora.MergeError):
            merge_lora.resolve_verify_spec("many", list(planned), planned)

    def test_defaults_need_only_base_adapter(self) -> None:
        code = merge_lora.main(["--base", self.base_dir, "--adapter", self.adapter_dir])
        self.assertEqual(code, 0)
        out_dir = self.adapter_dir + "-merged"
        self.assertTrue(os.path.isfile(os.path.join(out_dir, "model.safetensors.index.json")))
        config = json.load(open(os.path.join(out_dir, "config.json"), encoding="utf-8"))
        self.assertEqual(config["text_config"]["dtype"], "bfloat16")
        from safetensors.torch import load_file

        merged = {}
        for shard in sorted(set(
                json.load(open(os.path.join(out_dir, "model.safetensors.index.json"),
                               encoding="utf-8"))["weight_map"].values())):
            merged.update(load_file(os.path.join(out_dir, shard)))
        self.assertEqual(merged["model.language_model.layers.0.mlp.down_proj.weight"].dtype,
                         torch.bfloat16)

    def test_auto_verify_runs_and_passes(self) -> None:
        code = merge_lora.main(["--base", self.base_dir, "--adapter", self.adapter_dir,
                                "--out", os.path.join(self.root, "out"), "--dtype", "float32"])
        self.assertEqual(code, 0)
