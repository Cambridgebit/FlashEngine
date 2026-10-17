"""构造小型确定性 MoE/DFlash 功能夹具；不是训练模型或性能证据。"""
import argparse
from pathlib import Path
import shutil

from flashengine.deployment import expected_weights, normalize_model, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tokenizer_source", help="本地 Qwen3 tokenizer 目录")
    parser.add_argument("output", help="新的夹具根目录")
    parser.add_argument("--fused", action="store_true", help="目标专家权重使用融合 3D 布局")
    parser.add_argument("--transition", choices=("partial_eos", "accepted_eos"), help="确定性 token 转移，覆盖部分接受和轮内 EOS 截断")
    args = parser.parse_args()
    import torch
    from safetensors.torch import save_file
    root = Path(args.output)
    if root.exists():
        raise ValueError("拒绝覆盖夹具")
    raw = dict(model_type="qwen3_moe", architectures=["Qwen3MoeForCausalLM"], hidden_size=128, intermediate_size=384, moe_intermediate_size=256, num_hidden_layers=4, num_attention_heads=1, num_key_value_heads=1, head_dim=128, vocab_size=151936, num_experts=4, num_experts_per_tok=2, decoder_sparse_step=2, mlp_only_layers=[], norm_topk_prob=True, attention_bias=False, hidden_act="silu", max_position_embeddings=512, rms_norm_eps=1e-6, rope_theta=1000000, tie_word_embeddings=False, dtype="bfloat16", eos_token_id=151645)
    draft = dict(raw, model_type="qwen3", architectures=["DFlashDraftModel"], num_hidden_layers=1, tie_word_embeddings=True, block_size=4, dflash_config=dict(block_size=4, mask_token_id=151640, target_layer_ids=[0, 2]))
    for key in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "decoder_sparse_step", "mlp_only_layers", "norm_topk_prob"):
        draft.pop(key)
    generator = torch.Generator().manual_seed(20261016)
    for kind, config in (("target", raw), ("draft", draft)):
        directory = root / kind
        directory.mkdir(parents=True)
        normalized = normalize_model(config, draft=kind == "draft")
        if kind == "draft":
            normalized["target_layer_ids"] = [0, 2]
        elif args.fused:
            normalized["expert_layouts"] = {"1": "fused", "3": "fused"}
        weights = {}
        for name, shape in expected_weights(normalized, draft=kind == "draft").items():
            if "norm" in name:
                value = torch.ones(shape)
            elif name == "embed_tokens.weight":
                if args.transition:
                    value = torch.zeros(shape)
                    value[:, 1 if args.transition == "accepted_eos" else 0] = 1.
                    value[42].zero_(); value[42, 1] = 1.
                    value[43].zero_(); value[43, 2] = 1.
                    value[151645].zero_(); value[151645, 2] = 1.
                    value[151640].zero_(); value[151640, 2 if args.transition == "accepted_eos" else 1] = 1.
                else:
                    value = torch.rand(shape, generator=generator) * .04 + .01
            elif name == "lm_head.weight":
                value = torch.zeros(shape)
                if args.transition:
                    value[42, 0], value[43, 1], value[151645, 2] = 10., 10., 10.
                else:
                    value[42].fill_(1.)
            else:
                value = torch.zeros(shape) if args.transition else torch.randn(shape, generator=generator) * .01
            destination = name if kind == "draft" or name == "lm_head.weight" else "model." + name
            weights[destination] = value.to(torch.bfloat16).contiguous()
        save_file(weights, directory / "model.safetensors")
        write_json(directory / "config.json", config)
        if kind == "target":
            for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json", "vocab.json", "merges.txt"):
                source = Path(args.tokenizer_source) / name
                if source.is_file():
                    shutil.copyfile(source, directory / name)
    write_json(root / "fixture.json", dict(synthetic=True, purpose="仅功能与状态验收；不用于真实性能", transition=args.transition, constant_output_token=None if args.transition else 42, fused_experts=args.fused, seed=20261016))
    print(f"功能夹具已创建：{root}；不是训练模型。")


if __name__ == "__main__":
    main()
