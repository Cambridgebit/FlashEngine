"""新增 profile 的结构及拒绝行为，纯主机测试。"""
import unittest
from flashengine.deployment import normalize_model, expected_weights, normalize_quantization


class NativeProfilesTests(unittest.TestCase):
    def test_prefix_default_and_combination_contract(self):
        from pathlib import Path
        from flashengine.deployment import read_json
        root=Path(__file__).resolve().parents[1]/"contracts"
        self.assertTrue(read_json(root/"engine-config.schema.json")["properties"]["enable_prefix_cache"]["default"])
        profiles=read_json(root/"catalog.json")["profiles"]
        self.assertEqual(profiles["qwen35-moe-v1"]["speculative"],["dflash"])
        self.assertTrue(profiles["paged-prefix-v1"]["speculative"])

    def test_trace_rejects_refill_overwrite(self):
        from scripts.native_bench_common import checked_greedy_round
        value=checked_greedy_round([16,28],[16,28,18],[16,28,18],8)
        self.assertEqual(value["accepted_count"],2)
        with self.assertRaises(ValueError):
            checked_greedy_round([11,28],[16,28,18],[16,28,18],8)
        with self.assertRaises(ValueError):
            checked_greedy_round([16],[16],[16],8)

    def test_hybrid_dflash_last_layer_aux(self):
        from flashengine.deployment import configure_dflash
        dc=dict(hidden_size=128,vocab_size=512)
        raw=dict(dflash_config=dict(target_layer_ids=[0,3],mask_token_id=1,block_size=4))
        target=dict(self.base(),model_type="qwen3_5")
        configure_dflash(dc,raw,target,2)
        self.assertEqual(dc["target_layer_ids"],[0,3])
        with self.assertRaises(ValueError):configure_dflash(dict(hidden_size=128,vocab_size=512),raw,self.base(),2)

    def base(self):
        return dict(model_type="qwen3",architectures=["Qwen3ForCausalLM"],hidden_size=128,intermediate_size=256,vocab_size=512,num_hidden_layers=4,num_attention_heads=1,num_key_value_heads=1,head_dim=128)

    def test_eagle_layout_and_reduced_vocab(self):
        raw=dict(self.base(),architectures=["LlamaForCausalLMEagle3"],num_hidden_layers=1,draft_vocab_size=64)
        cfg=normalize_model(raw,draft=True);weights=expected_weights(cfg,draft=True)
        self.assertEqual(weights["midlayer.self_attn.q_proj.weight"],[128,256])
        self.assertEqual(weights["lm_head.weight"],[64,128])
        self.assertEqual(weights["d2t"],[64])
        self.assertNotIn("midlayer.self_attn.q_norm.weight",weights)
        with self.assertRaises(ValueError):normalize_model(dict(raw,num_hidden_layers=2),draft=True)

    def test_hybrid_shapes_and_text_shell(self):
        text=dict(self.base(),model_type="qwen3_5_text",layer_types=["linear_attention","full_attention"]*2,linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=16,linear_conv_kernel_dim=4,rope_parameters=dict(partial_rotary_factor=.25))
        cfg=normalize_model(dict(model_type="qwen3_5",text_config=text));weights=expected_weights(cfg)
        self.assertEqual(cfg["rotary_dim"],32)
        self.assertEqual(weights["layers.0.linear_attn.conv1d.weight"],[48,1,4])
        self.assertNotIn("layers.0.self_attn.q_proj.weight",weights)
        self.assertEqual(weights["layers.1.self_attn.q_proj.weight"],[256,128])
        with self.assertRaises(ValueError):normalize_model(dict(model_type="qwen3_5",text_config=dict(text,linear_num_value_heads=3,linear_num_key_heads=2)))

    def test_quantization_profile_is_explicit(self):
        quant=dict(quant_method="quark",config_groups=dict(x=dict(weights=dict(num_bits=8,type="int",strategy="channel"))))
        self.assertEqual(normalize_quantization(quant),"quark-int8-dequant-v1")
        self.assertEqual(normalize_model(dict(self.base(),quantization_config=quant))["quantization"],"quark-int8-dequant-v1")
        for bad in (dict(quant_method="awq"),dict(quant_method="quark"),dict(quant_method="quark",config_groups=dict(x=dict(weights=dict(num_bits=4,type="int",strategy="channel"))))):
            with self.assertRaises(ValueError):normalize_quantization(bad)


if __name__=="__main__":unittest.main()
