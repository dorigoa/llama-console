from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from config import Settings
from logging_utils import emit, LogSink

settings = Settings()


class RpcStartupError(RuntimeError):
    pass


def tcp_connect(host: str, port: int, timeout_seconds: int = 2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def ensure_remote_rpc(server: str, timeout: int, rpc_host: str, rpc_port: int, log_sink: LogSink = None) -> None:
    #emit(f"Checking remote RPC endpoint {rpc_host}:{rpc_port}", log_sink)
    emit(f"Stopping any remotely running rpc-server...")
    #if tcp_connect(rpc_host, rpc_port, timeout):
    #    emit("Remote RPC is already reachable", log_sink)
    #    return

    remote_kill_cmd = (
        f"killall -9 {Path(settings.rpc_server_path).name}"
    )
    emit(f"ssh {server} {remote_kill_cmd}", log_sink)
    subprocess.run(["ssh", server, remote_kill_cmd], check=False)

    emit(f"Starting remote RPC through SSH host {server}", log_sink)

    time.sleep(3)

    remote_cmd = (
        f"nohup {settings.rpc_server_path} "
        f"--host '{rpc_host}' "
        f"--port '{rpc_port}' "
        "-c >/dev/null 2>&1 &"
    )

    emit(f"ssh {server} {remote_cmd}", log_sink)
    subprocess.run(["ssh", server, remote_cmd], check=True)

    emit("Waiting for remote RPC to become reachable...", log_sink)

    for attempt in range(1, 11):
        if tcp_connect(rpc_host, rpc_port, 2):
            emit("Remote RPC is now reachable", log_sink)
            return
        emit(f"RPC not reachable yet, attempt {attempt}/10", log_sink)
        time.sleep(3)

    raise RpcStartupError("Remote RPC did not become reachable. Stop.")
