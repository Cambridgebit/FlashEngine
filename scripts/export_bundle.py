"""将 Qwen3 dense/MoE、Qwen3.5 文本、DFlash/EAGLE3 检查点打包为原生工件。"""
import argparse
from flashengine.deployment import export_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="本地 Qwen3 dense/MoE 或 Qwen3.5 文本模型目录")
    parser.add_argument("output", help="尚不存在的输出目录")
    parser.add_argument("--draft", help="本地适配草稿目录：Qwen3 支持 DFlash1/EAGLE3，Qwen3.5 仅 DFlash1；须匹配目标宽度、词表与 aux 层")
    parser.add_argument("--spec-num-tokens", type=int, help="明确 k 值；不使用旧 CLI 的 3/4 sentinel")
    parser.add_argument("--hardlink", action="store_true", help="显式使用同文件系统硬链接；不得修改源权重")
    parser.add_argument("--verifier", help="flashengine-native 原生校验器路径")
    args = parser.parse_args()
    export_bundle(args.model, args.output, draft=args.draft, spec_num_tokens=args.spec_num_tokens, hardlink=args.hardlink, verifier=args.verifier)
    print(f"已导出 bundle：{args.output}；未指定 --verifier 时仍须执行原生 verify。")


if __name__ == "__main__":
    main()
