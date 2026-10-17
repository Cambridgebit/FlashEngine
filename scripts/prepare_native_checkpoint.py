"""离线将 EAGLE3 的 weights_only pickle 权重转换为原生 safetensors，不改原目录。"""
import argparse
from pathlib import Path
import shutil
from flashengine.deployment import write_json, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("source");p.add_argument("output");args=p.parse_args()
    source,output=Path(args.source),Path(args.output)
    if output.exists():raise ValueError("拒绝覆盖转换目录")
    import torch
    from safetensors.torch import save_file
    state=torch.load(source/"pytorch_model.bin",map_location="cpu",weights_only=True)
    if not isinstance(state,dict) or not all(isinstance(k,str) and isinstance(v,torch.Tensor) for k,v in state.items()):raise ValueError("检查点必须为纯张量字典")
    output.mkdir(parents=True);save_file({k:v.contiguous().clone() for k,v in state.items()},output/"model.safetensors");shutil.copyfile(source/"config.json",output/"config.json")
    write_json(output/"conversion.json",dict(source_sha256=sha256(source/"pytorch_model.bin"),weights_sha256=sha256(output/"model.safetensors"),weights_only=True))
    print(output)


if __name__=="__main__":main()
