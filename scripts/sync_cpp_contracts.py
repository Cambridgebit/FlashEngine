"""将不可变契约快照同步到独立 C++ 仓并写入可核验的锁文件。"""
import argparse
from pathlib import Path
import shutil

from flashengine.deployment import CONTRACT_VERSION, git_sha, publish_contracts, read_json, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpp_repo", help="独立 C++ Git 仓目录")
    parser.add_argument("artifact", help="新的契约 tar.gz 发布路径")
    parser.add_argument("--upgrade", action="store_true", help="显式晋升新版本；旧快照和锁文件保留")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "contracts"
    cpp = Path(args.cpp_repo).resolve()
    if not (cpp / ".git").exists():
        raise ValueError("目标必须是独立 Git 仓")
    destination = cpp / "vendor" / (f"contracts-{CONTRACT_VERSION}" if args.upgrade else "contracts")
    if destination.exists():
        raise ValueError("拒绝覆盖既有契约快照，请显式进行版本更新")
    lock = cpp / "contracts.lock"
    old = read_json(lock) if lock.exists() else None
    if old and (not args.upgrade or old["contract_version"] == CONTRACT_VERSION):
        raise ValueError("已有契约锁；只允许显式升级到新版本")
    if old:
        archived = cpp / "contracts.history" / (old["contract_version"] + ".lock")
        if archived.exists():
            raise ValueError("历史契约锁已存在，拒绝覆盖")
        archived.parent.mkdir(exist_ok=True)
        shutil.copyfile(lock, archived)
    artifact = Path(args.artifact).resolve()
    artifact.parent.mkdir(parents=True, exist_ok=True)
    digest = publish_contracts(source, artifact)
    shutil.copytree(source, destination)
    files = {p.relative_to(destination).as_posix(): sha256(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    write_json(lock, dict(contract_version=CONTRACT_VERSION, oracle_git_sha=git_sha(), directory=destination.relative_to(cpp).as_posix(), artifact=artifact.name, artifact_sha256=digest, files=files))
    print(f"契约快照已锁定：{digest}")


if __name__ == "__main__":
    main()
