"""记录原生程序依赖、源码和设备环境，拒绝 Python/Rust 动态依赖。"""
import argparse
import hashlib
import importlib.metadata
import platform
import shutil
from pathlib import Path
import subprocess

from flashengine.deployment import read_json, sha256, write_json


def command_evidence(command):
    """保留探测失败原文；缺设备工具不伪装成成功验收。"""
    if not shutil.which(command[0]):
        return dict(command=command, available=False)
    process = subprocess.run(command, capture_output=True, text=True, timeout=30)
    return dict(command=command, available=True, exit_code=process.returncode, stdout=process.stdout.strip(), stderr=process.stderr.strip())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native", help="原生 ELF 可执行文件")
    parser.add_argument("cpp_repo", help="独立 C++ 仓")
    parser.add_argument("output", help="证据 JSON")
    parser.add_argument("--flashinfer-include", help="编译使用的 FlashInfer 原生头文件根目录")
    args = parser.parse_args()
    if Path(args.output).exists():
        raise ValueError("构建证据已存在，拒绝覆盖")
    cpp = Path(args.cpp_repo).resolve()
    capabilities = read_capabilities(args.native)
    backend = capabilities["compiled_backend"]
    linked = subprocess.check_output(["ldd", args.native], text=True)
    if any(name in linked.lower() for name in ("libpython", "libtorch_python", "librust")):
        raise RuntimeError("原生运行时引入了禁止的语言动态依赖")
    if "not found" in linked:
        raise RuntimeError("部署程序有未解析的动态库依赖")
    lock = read_json(cpp / "contracts.lock")
    tree = hashlib.sha256()
    files = {}
    for directory in (cpp / "src", cpp / "include", cpp / lock.get("directory", "vendor/contracts"), cpp / "tests"):
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                relative = path.relative_to(cpp).as_posix()
                digest = sha256(path)
                files[relative] = digest
                tree.update(relative.encode() + digest.encode())
    for name in ("CMakeLists.txt", "CMakePresets.json", "dependencies.lock.json", "contracts.lock"):
        files[name] = sha256(cpp / name)
        tree.update(name.encode() + files[name].encode())
    headers = {}
    for name, directory in (("minja", cpp / "third_party/minja/include"), ("flashinfer", Path(args.flashinfer_include) if args.flashinfer_include else None)):
        if directory is not None:
            headers[name] = {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    dependencies = {}
    for line in linked.splitlines():
        target = line.split("=>", 1)[-1].strip().split()[0] if line.strip() else ""
        if target.startswith("/"):
            path = Path(target)
            dependencies[str(path)] = sha256(path)
    try:
        flashinfer = importlib.metadata.version("flashinfer-python")
    except importlib.metadata.PackageNotFoundError:
        flashinfer = None
    device_commands = {"cuda": ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], "corex": ["ixsmi"], "ascend310p": ["npu-smi", "info"]}
    device = command_evidence(device_commands[backend]) if backend in device_commands else None
    evidence = dict(binary_sha256=sha256(args.native), source_tree_sha256=tree.hexdigest(), sources=files, native_headers=headers, dynamic_dependencies=dependencies, ldd=linked, flashinfer_version=flashinfer, contract=lock, capabilities=capabilities, architecture=platform.machine(), platform=platform.platform(), elf=command_evidence(["readelf", "-h", args.native]), device=device, release_ready=False)
    write_json(args.output, evidence)
    print("原生依赖检查通过，构建证据已保存；此报告不授予 production 状态。")


def read_capabilities(native):
    """读取二进制自己的编译能力，不能按机器存在的工具猜后端。"""
    import json
    return json.loads(subprocess.check_output([native, "capabilities"], text=True, timeout=30))


if __name__ == "__main__":
    main()
