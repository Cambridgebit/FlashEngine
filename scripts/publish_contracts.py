"""生成不可变且可重复的契约发布包。"""
import argparse
from pathlib import Path
from flashengine.deployment import publish_contracts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", help="新 tar.gz 文件路径，不覆盖既有发布物")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "contracts"
    print(publish_contracts(source, args.output))


if __name__ == "__main__":
    main()
