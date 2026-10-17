"""为目标×投机×前缀组合建立非恒定输出夹具；不证明训练模型精度或性能。"""
import argparse
from pathlib import Path
import shutil
from flashengine.deployment import read_json,write_json,normalize_model,expected_weights


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("profiles");p.add_argument("moe_target");p.add_argument("output");a=p.parse_args()
    import torch
    from safetensors.torch import load_file,save_file
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);source=Path(a.profiles)
    for label,origin in (("dense",source/"dense"),("moe",Path(a.moe_target)),("hybrid",source/"hybrid"),("hybrid-moe",source/"hybrid-moe")):
        dest=root/label;shutil.copytree(origin,dest);weights={k:v.clone() for k,v in load_file(dest/"model.safetensors").items()}
        embed=weights["model.embed_tokens.weight"].float();embed[52].fill_(-.04);embed[53].fill_(.04)
        weights["model.embed_tokens.weight"]=embed.half();head=torch.zeros_like(weights["lm_head.weight"]);head[52].fill_(1);head[53].fill_(-1);weights["lm_head.weight"]=head
        save_file(weights,dest/"model.safetensors")
    shutil.copytree(source/"eagle",root/"eagle")
    raw=read_json(source/"dense/config.json");raw.update(architectures=["DFlashDraftModel"],num_hidden_layers=1,tie_word_embeddings=True,block_size=4,dflash_config=dict(block_size=4,mask_token_id=151640,target_layer_ids=[0,2]))
    cfg=normalize_model(raw,draft=True);cfg["target_layer_ids"]=[0,2];generator=torch.Generator().manual_seed(20261018);weights={}
    for name,shape in expected_weights(cfg,draft=True).items():
        if "norm" in name:value=torch.ones(shape)
        elif name=="fc.weight":value=torch.cat([torch.eye(128),torch.zeros(128,128)],1)
        else:value=torch.randn(shape,generator=generator)*.01
        weights[name]=value.half()
    (root/"dflash").mkdir();save_file(weights,root/"dflash/model.safetensors");write_json(root/"dflash/config.json",raw)
    write_json(root/"fixture.json",dict(synthetic=True,nonconstant=True,output_tokens=[52,53],purpose="组合功能、前缀恢复和生命周期回归，不是训练或性能模型"));print(root)


if __name__=="__main__":main()
