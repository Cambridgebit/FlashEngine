"""为真实目标创建小型随机 DFlash，用于回滚/状态测试，绝不用于接受率或加速收益证明。"""
import argparse
from pathlib import Path
from flashengine.deployment import normalize_model,read_json,expected_weights,write_json,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("target");p.add_argument("output");a=p.parse_args()
    import torch
    from safetensors.torch import save_file
    target=normalize_model(read_json(Path(a.target)/"config.json"));h=target["hidden_size"];vocab=target["vocab_size"]
    if h%128:raise ValueError("此合成器只生成 head_dim=128 且隐藏宽度整除的草稿")
    raw=dict(model_type="qwen3",architectures=["DFlashDraftModel"],hidden_size=h,intermediate_size=256,num_hidden_layers=1,num_attention_heads=h//128,num_key_value_heads=1,head_dim=128,vocab_size=vocab,max_position_embeddings=target["max_position_embeddings"],rms_norm_eps=target["rms_norm_eps"],rope_theta=target["rope_theta"],tie_word_embeddings=True,hidden_act="silu",attention_bias=False,dtype="float16",block_size=4,dflash_config=dict(block_size=4,mask_token_id=vocab-1,target_layer_ids=[0,target["num_hidden_layers"]-2]))
    cfg=normalize_model(raw,draft=True);cfg["target_layer_ids"]=raw["dflash_config"]["target_layer_ids"];rng=torch.Generator().manual_seed(20261018);weights={}
    for key,shape in expected_weights(cfg,draft=True).items():weights[key]=(torch.ones(shape) if "norm" in key else torch.randn(shape,generator=rng)*.002).half()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);save_file(weights,root/"model.safetensors");write_json(root/"config.json",raw);write_json(root/"SYNTHETIC.json",dict(synthetic=True,target_config_sha256=sha256(Path(a.target)/"config.json"),purpose="仅状态回滚与正确性测试，不证明训练质量或性能"));print(root)


if __name__=="__main__":main()
