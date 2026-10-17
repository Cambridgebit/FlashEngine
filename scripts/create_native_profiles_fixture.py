"""创建 Prefix/EAGLE3/Quark/Qwen3.5 的小型功能夹具，不证明训练模型精度或收益。"""
import argparse
from pathlib import Path
import shutil
from flashengine.deployment import expected_weights, normalize_model, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("tokenizer_source");p.add_argument("output");args=p.parse_args()
    import torch
    from safetensors.torch import save_file
    root=Path(args.output)
    if root.exists():raise ValueError("拒绝覆盖夹具")
    root.mkdir(parents=True)
    base=dict(model_type="qwen3",architectures=["Qwen3ForCausalLM"],hidden_size=128,intermediate_size=256,num_hidden_layers=4,num_attention_heads=1,num_key_value_heads=1,head_dim=128,vocab_size=151936,attention_bias=False,hidden_act="silu",max_position_embeddings=512,rms_norm_eps=1e-6,rope_theta=1000000,tie_word_embeddings=False,dtype="float16",eos_token_id=151645)
    rng=torch.Generator().manual_seed(20261017)
    def create(name,raw,draft=False,hybrid=False):
        directory=root/name;directory.mkdir();cfg=normalize_model(raw,draft=draft);weights={}
        for key,shape in expected_weights(cfg,draft=draft).items():
            if key=="d2t":value=torch.full(shape,10,dtype=torch.int64)
            elif key=="fc.weight":value=torch.cat([torch.eye(128),torch.zeros(128,256)],1)
            elif "norm" in key:value=torch.zeros(shape) if hybrid and ".linear_attn.norm." not in key else torch.ones(shape)
            elif key=="embed_tokens.weight":
                value=torch.rand(shape,generator=rng)*.04+.01
                if hybrid:
                    value[52].fill_(-.04);value[53].fill_(.04)
            elif key=="lm_head.weight":
                value=torch.zeros(shape);value[42 if draft else 52].fill_(1.)
                if hybrid:value[53].fill_(-1.)
            elif key.endswith("A_log"):value=torch.zeros(shape)
            else:value=torch.randn(shape,generator=rng)*.01
            key=key if draft or key=="lm_head.weight" else "model."+key
            weights[key]=value if value.dtype==torch.int64 or key.endswith(("A_log","dt_bias")) else value.to(torch.float16)
        save_file(weights,directory/"model.safetensors")
        saved=raw
        if hybrid:
            moe=raw["model_type"]=="qwen3_5_moe"
            saved=dict(model_type="qwen3_5_moe" if moe else "qwen3_5",architectures=["Qwen3_5MoeForConditionalGeneration" if moe else "Qwen3_5ForConditionalGeneration"],text_config=dict(raw,model_type="qwen3_5_moe_text" if moe else "qwen3_5_text"))
        write_json(directory/"config.json",saved)
        if not draft:
            for filename in ("tokenizer.json","tokenizer_config.json","chat_template.jinja","special_tokens_map.json","vocab.json","merges.txt"):
                source=Path(args.tokenizer_source)/filename
                if source.is_file():shutil.copyfile(source,directory/filename)
        return weights
    floating=create("dense",base)
    eagle=dict(base,architectures=["LlamaForCausalLMEagle3"],num_hidden_layers=1,draft_vocab_size=64,aux_hidden_state_layer_ids=[0,1,2]);create("eagle",eagle,draft=True)
    quant=dict(quant_method="quark",config_groups=dict(x=dict(weights=dict(num_bits=8,type="int",strategy="channel"))))
    quantized={};reference={}
    for name,value in floating.items():
        if value.ndim==2 and name.endswith(".weight") and "embed_tokens" not in name and "lm_head" not in name:
            scale=(value.float().abs().amax(1)/100).clamp_min(1e-8);zero=torch.full((value.shape[0],),3,dtype=torch.int8);q=(value.float()/scale[:,None]+zero[:,None]).round().clamp(-128,127).to(torch.int8)
            quantized[name]=q;quantized[name.removesuffix(".weight")+".weight_scale"]=scale;quantized[name.removesuffix(".weight")+".weight_zero_point"]=zero
            reference[name]=((q.float()-zero[:,None])*scale[:,None]).to(torch.float16)
        else:quantized[name]=reference[name]=value
    for name,weights,raw in (("quark",quantized,dict(base,quantization_config=quant)),("quark-reference",reference,base)):
        directory=root/name;directory.mkdir();save_file(weights,directory/"model.safetensors");write_json(directory/"config.json",raw)
        for filename in ("tokenizer.json","tokenizer_config.json","chat_template.jinja","special_tokens_map.json","vocab.json","merges.txt"):
            source=root/"dense"/filename
            if source.exists():shutil.copyfile(source,directory/filename)
    hybrid=dict(base,model_type="qwen3_5",architectures=["Qwen3_5ForCausalLM"],layer_types=["linear_attention","full_attention"]*2,linear_num_key_heads=1,linear_num_value_heads=2,linear_key_head_dim=16,linear_value_head_dim=16,linear_conv_kernel_dim=4,rope_parameters=dict(rope_theta=10000000,partial_rotary_factor=.25),mamba_ssm_dtype="float32")
    create("hybrid",hybrid,hybrid=True)
    hybrid_moe=dict(hybrid,model_type="qwen3_5_moe",architectures=["Qwen3_5MoeForCausalLM"],num_experts=4,num_experts_per_tok=2,moe_intermediate_size=64,shared_expert_intermediate_size=64,decoder_sparse_step=1,mlp_only_layers=[],norm_topk_prob=True)
    create("hybrid-moe",hybrid_moe,hybrid=True)
    write_json(root/"fixture.json",dict(synthetic=True,purpose="功能、状态与载入差分；不证明训练模型收益",eagle_output=52,eagle_draft_token=42,d2t_offset=10,hybrid_nonconstant=True))
    print(root)


if __name__=="__main__":main()
