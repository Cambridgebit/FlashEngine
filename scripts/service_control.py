"""Linux 容器内的服务托管：只管理本目录、记录过启动时间的单个进程。

安全边界（误停别的服务是这里唯一的风险）：
- 状态文件只记录**本框架**启动的子进程：``python -u -m flashengine serve --config ...``；
- 停止前同时核对 PID、内核启动计数（``/proc/<pid>/stat`` 第 22 字段）、cwd、完整 argv
  与记录里的源码根目录，任意一项不符即拒绝发送信号（PID 被复用时不误杀）；
- 有 ``os.pidfd_open`` 时经 pidfd 发 SIGTERM，彻底避开「校验后 PID 被复用」的竞态；
- 不扫描全局进程、不发送 SIGKILL，也不停止占用端口的其他服务。

本模块只依赖标准库与 ``flashengine.version``（该模块纯标准库、不 import torch），
``python -c "import scripts.service_control"`` 在无 torch 的机器上可用。
"""

import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
import uuid

from flashengine.version import SERVICE_NAME, VERSION


ROOT = Path(__file__).resolve().parents[1]
# 运行期状态（PID/日志/一次性配置）默认落在源码根的 runtime/；可用环境变量改到 tmpfs。
RUNTIME_DIR_ENV = "FLASHENGINE_RUNTIME_DIR"


def runtime_dir(root=ROOT):
    folder = Path(os.environ.get(RUNTIME_DIR_ENV) or (Path(root) / "runtime"))
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def process_identity(pid):
    """读取 PID 的启动计数、cwd 与 argv；进程不存在或已成僵尸时返回 None。

    ``start_ticks`` 是内核为每个进程分配的启动时间（jiffies），PID 复用时必然不同，
    因此它是「同一个进程」最可靠的判据。
    """
    try:
        proc = Path("/proc") / str(pid)
        values = (proc / "stat").read_text().rsplit(") ", 1)[1].split()
        if values[0] == "Z":
            return None
        return {"pid": int(pid), "start_ticks": values[19],
                "cwd": str((proc / "cwd").resolve(strict=True)),
                "command": (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")}
    except (OSError, IndexError, ValueError):
        return None


def owned_process(state, current, root=ROOT):
    """身份、工作目录、命令及源码根目录全部匹配才拥有停止权限。"""
    return bool(state and current and state.get("pid", 0) > 1
                and state.get("root") == str(Path(root).resolve())
                and state.get("pid") == current.get("pid")
                and state.get("start_ticks") == current.get("start_ticks")
                and current.get("cwd") == str(Path(root).resolve())
                and state.get("command") == current.get("command")
                and current.get("command", [])[1:4] == ["-u", "-m", "flashengine"]
                and current.get("command", [])[4:5] == ["serve"])


def ready(state):
    """访问 /ready 并核对 PID、服务名与版本，避免把别的端口服务当成已就绪。"""
    try:
        with urlopen(f"http://{state.get('probe_host', '127.0.0.1')}:{state['port']}/ready",
                     timeout=2) as response:
            value = json.load(response)
        return (value.get("status") == "ok" and value.get("pid") == state["pid"]
                and value.get("service") == SERVICE_NAME and value.get("version") == VERSION)
    except (OSError, ValueError, KeyError):
        return False


def stop_owned(state):
    """重新校验进程身份后经 pidfd 或 PID 发送 SIGTERM，不做全局 kill。"""
    descriptor = os.pidfd_open(state["pid"]) if hasattr(os, "pidfd_open") else None
    try:
        if not owned_process(state, process_identity(state["pid"])):
            raise RuntimeError("进程身份发生变化，拒绝发送停止信号")
        if descriptor is not None and hasattr(signal, "pidfd_send_signal"):
            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
        else:
            os.kill(state["pid"], signal.SIGTERM)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def control(command, settings, timeout):
    """执行本框架服务的 start/status/stop，保存状态并等待就绪或退出。"""
    if sys.platform != "linux":
        raise RuntimeError("服务启停命令需要在 Linux 开发板/容器内执行")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout 必须为有限正数")
    import fcntl

    folder = runtime_dir()
    state_path = folder / "service.json"
    with (folder / "service.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text()) if state_path.is_file() else None
        current = process_identity(state["pid"]) if state else None
        owned = owned_process(state, current)
        if command == "status":
            value = {"running": owned, "ready": ready(state) if owned else False,
                     "pid": state["pid"] if state else None, "log": state.get("log") if state else None}
            print("FRAMEWORK_SERVICE_STATUS", json.dumps(value, ensure_ascii=False), flush=True)
            return 0 if owned and value["ready"] else 1
        if command == "stop":
            if not current:
                print("FRAMEWORK_SERVICE_NOT_RUNNING", flush=True)
                return 0
            if not owned:
                raise RuntimeError("记录的 PID 已被其他进程复用或身份不符，拒绝停止")
            stop_owned(state)
        elif command == "start":
            if current:
                if not owned:
                    raise RuntimeError("状态记录的进程身份异常，拒绝覆盖状态文件")
                print(f"FRAMEWORK_SERVICE_ALREADY_RUNNING PID={state['pid']} LOG={state['log']}", flush=True)
                return 0 if ready(state) else 1
            # 不停止占用端口的其他服务；绑定检查失败则直接报告。
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind((settings.host, settings.port))
            config_path = folder / "service_config.json"
            config_path.write_text(json.dumps(settings.as_dict(), ensure_ascii=False, indent=2),
                                   encoding="utf-8")
            log_path = folder / ("server_" + uuid.uuid4().hex[:12] + ".log")
            args = [sys.executable, "-u", "-m", "flashengine", "serve", "--config", str(config_path)]
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            with log_path.open("ab", buffering=0) as log:
                process = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True,
                                           env=environment)
            identity = None
            for _ in range(50):
                identity = process_identity(process.pid)
                if identity and identity["command"] == args:
                    break
                time.sleep(.02)
            if not identity or identity["command"] != args:
                raise RuntimeError(f"未能确认新进程身份，请检查日志：{log_path}")
            state = {**identity, "root": str(ROOT.resolve()), "port": settings.port,
                     "log": str(log_path),
                     "probe_host": "127.0.0.1" if settings.host == "0.0.0.0" else settings.host}
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"FRAMEWORK_SERVICE_STARTING PID={state['pid']} LOG={log_path}", flush=True)
    # 就绪/退出等待不持有文件锁，允许用户在启动期间查询或停止。
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = process_identity(state["pid"])
        if command == "stop":
            if not owned_process(state, current):
                print(f"FRAMEWORK_SERVICE_STOPPED PID={state['pid']}", flush=True)
                return 0
        else:
            if not owned_process(state, current):
                raise RuntimeError(f"服务在就绪前退出，请检查日志：{state['log']}")
            if ready(state):
                print(f"FRAMEWORK_SERVICE_READY PID={state['pid']} PORT={state['port']} "
                      f"LOG={state['log']}", flush=True)
                return 0
        time.sleep(1)
    if command == "start":
        raise RuntimeError(f"等待就绪超时；未强制杀进程。可用 status/stop 管理，日志：{state['log']}")
    raise RuntimeError(f"优雅停止超时；未发送 SIGKILL。请检查日志：{state['log']}")
