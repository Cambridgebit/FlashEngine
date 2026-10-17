"""导出真实 Python 后端基础算子的舍入参考，不进入部署运行时。"""
import argparse
import os
from pathlib import Path

from flashengine.deployment import source_state, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="新的 safetensors 算子参考文件")
    parser.add_argument("--cache-dir", help="将离线 Python 编译缓存保存在指定开发目录")
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise ValueError("拒绝覆盖算子参考")
    if args.cache_dir:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = args.cache_dir
    import torch
    from safetensors.torch import save_file
    from flashengine.backends.base import BackendBase
    torch.manual_seed(20261017)
    tensors, cases = {}, []
    for dtype in (torch.float16, torch.bfloat16):
        for width in (128, 2048):
            prefix = f"{str(dtype).removeprefix('torch.')}_{width}"
            x = torch.randn((8, width), device="cuda", dtype=dtype)
            residual = torch.randn_like(x)
            weight = torch.randn((width,), device="cuda", dtype=dtype)
            gu = torch.randn((8, width * 2), device="cuda", dtype=dtype)
            output, new_residual = BackendBase.add_rms_norm(x, residual, weight, 1e-6)
            values = dict(x=x, residual=residual, weight=weight, gu=gu, norm=BackendBase.rms_norm(x, weight, 1e-6), add_norm=output, new_residual=new_residual, silu=BackendBase.silu_and_mul(gu))
            for name, value in values.items():
                tensors[prefix + "." + name] = value.detach().cpu().contiguous()
            # 同时记录 cast 位置的区分度，不能只用全 1 的 norm 权重做验收。
            xf = x.float()
            normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
            early = normalized.to(dtype) * weight
            late = (normalized * weight.float()).to(dtype)
            silu_early = torch.nn.functional.silu(gu[:, :width]) * gu[:, width:]
            silu_late = (torch.nn.functional.silu(gu[:, :width].float()) * gu[:, width:].float()).to(dtype)
            cases.append(dict(prefix=prefix, dtype=str(dtype), width=width, norm_early_diff=int((values["norm"] != early).sum()), norm_late_diff=int((values["norm"] != late).sum()), silu_early_diff=int((values["silu"] != silu_early).sum()), silu_late_diff=int((values["silu"] != silu_late).sum())))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, path)
    write_json(path.with_suffix(".json"), dict(cases=cases, oracle=source_state(), torch=torch.__version__, device=torch.cuda.get_device_name()))
    for case in cases:
        print(case)


if __name__ == "__main__":
    main()
