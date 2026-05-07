from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from config import settings
from logging_utils import emit, LogSink

#settings = Settings()


class RpcStartupError(RuntimeError):
    pass


def tcp_connect(host: str, port: int, timeout_seconds: int = 2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False


def ensure_remote_rpc(server: str, timeout: int, RPC_HOST: str, RPC_PORT: int, log_sink: LogSink = None) -> None:
    emit(f"Stopping any remotely running rpc-server...")
    
    remote_kill_cmd = (
        f"killall -9 {Path(settings.RPC_SERVER_PATH).name}"
    )
    emit(f"ssh {server} {remote_kill_cmd}", log_sink)
    subprocess.run(["ssh", server, remote_kill_cmd], check=False)

    emit(f"Starting remote RPC through SSH host {server}", log_sink)

    time.sleep(3)

    remote_cmd = (
        f"LLAMA_CACHE={settings.RPC_CACHE_PATH} nohup {settings.RPC_SERVER_PATH} "
        f"--host '{RPC_HOST}' "
        f"--port '{RPC_PORT}' "
        "-c >/dev/null 2>&1 &"
    )

    emit(f"ssh {server} {remote_cmd}", log_sink)
    subprocess.run(["ssh", server, remote_cmd], check=True)

    emit("Waiting for remote RPC to become reachable...", log_sink)

    for attempt in range(1, 11):
        if tcp_connect(RPC_HOST, RPC_PORT, 2):
            emit("Remote RPC is now reachable", log_sink)
            return
        emit(f"RPC not reachable yet, attempt {attempt}/10", log_sink)
        time.sleep(3)

    raise RpcStartupError("Remote RPC did not become reachable. Stop.")
