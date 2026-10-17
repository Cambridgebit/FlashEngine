"""对固定种子的 Unicode 语料执行 Python/C++ tokenizer 差分。"""
import argparse
from pathlib import Path
import random
import subprocess
import tempfile

from flashengine.deployment import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="原始模型目录")
    parser.add_argument("bundle", help="原生部署 bundle")
    parser.add_argument("native", help="原生可执行文件")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rng = random.Random(20261016)
    alphabet = "abcXYZ019!?_ \t\r\n中文日本語αβéößعربىहिन्दी😀👩‍💻\u0301\u00a0\u2003"
    corpus = ["", "e\u0301", "é", "<think>test</think>", "<|im_start|>user\n你好<|im_end|>"]
    corpus += ["".join(rng.choice(alphabet) for _ in range(rng.randrange(1, 180))) for _ in range(200)]
    cases = []
    for text in corpus:
        ids = tok.encode(text, add_special_tokens=False)
        cases.append(dict(text=text, token_ids=ids, decoded=tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)))
    with tempfile.TemporaryDirectory(prefix="fe-tokenizer-") as tmp:
        path = Path(tmp) / "cases.json"
        write_json(path, cases)
        result = subprocess.run([args.native, "tokenize-file", args.bundle, str(path)], text=True, capture_output=True)
        if result.returncode:
            print(result.stdout)
            print(result.stderr)
            raise SystemExit(result.returncode)
    print(f"tokenizer Unicode 差分：{len(cases)}/{len(cases)} PASS")


if __name__ == "__main__":
    main()
