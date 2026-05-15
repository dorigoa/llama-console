from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from config_manager import get_settings
from object_models import Server
from logging_utils import emit, LogSink, setup_console_logging
#import re

settings = get_settings()

logger = setup_console_logging()

class RpcStartupError(RuntimeError):
    pass

#__________________________________________________________________________________________
def tcp_connect(host: str, port: int, timeout_seconds: int = 2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False

#__________________________________________________________________________________________
def ensure_remote_rpc(timeout: int, 
                      rpc: Server,
                      log_sink: LogSink = None, 
                      ) -> None:
    
    if not rpc:
        raise RpcStartupError("Passed None rpc argument to ensure_remote_rpc")

    if rpc.platform == "Windows":
        for attempt in range(1, 11):
            if tcp_connect(rpc.hostname, rpc.tcpport, 2):
                emit(f"-> Remote RPC on {rpc.hostname}:{rpc.tcpport} is now reachable", log_sink)
                return
            emit(f"-> RPC on on {rpc.hostname}:{rpc.tcpport} not reachable yet, attempt {attempt}/10 to start it", log_sink)
            
            remote_cmd = (f'schtasks /Create /TN llama-rpc-server-manual /TR "C:\\llama.cpp\\build\\bin\\Release\\rpc-server.exe --host {RPC_HOST} --port {RPC_PORT} -c" /SC ONCE /ST 23:59 /F')
            emit(f"-> Executing {remote_cmd}", log_sink)
            subprocess.run(["ssh", rpc.hostname, remote_cmd], check=True)
            
            remote_cmd = ('schtasks /Run /TN llama-rpc-server-manual')
            emit(f"-> Executing {remote_cmd}", log_sink)
            subprocess.run(["ssh", rpc.hostname, remote_cmd], check=True)
            time.sleep(3)
        raise RpcStartupError("Remote RPC did not become reachable. Stop.")
    
    if rpc.platform == "Linux" or rpc.platform == "Darwin":
        emit(f"-> Stopping any remotely running rpc-server...")
        remote_kill_cmd = (
            f'killall -9 {Path(rpc.binarypath).name}'
        )
        emit(f"-> Executing ssh {rpc.hostname} {remote_kill_cmd}", log_sink)
        subprocess.run(["ssh", rpc.hostname, remote_kill_cmd], check=False)
        emit(f"-> Starting remote RPC through SSH host {rpc.hostname}", log_sink)
        time.sleep(3)
        remote_cmd = (
            f"LLAMA_CACHE={rpc.cachepath} nohup {rpc.binarypath} "
            f"--host '{rpc.hostname}' "
            f"--port '{rpc.tcpport}' "
            "-c >/dev/null 2>&1 &"
        )
        emit(f"-> Executing ssh {rpc.hostname} {remote_cmd}", log_sink)
        subprocess.run(["ssh", rpc.hostname, remote_cmd], check=True)

        emit("-> Waiting for remote RPC to become reachable...", log_sink)

        for attempt in range(1, 11):
            if tcp_connect(rpc.hostname, rpc.tcpport, 2):
                emit("-> Remote RPC is now reachable", log_sink)
                return
            emit(f"-> RPC not reachable yet, attempt {attempt}/10", log_sink)
            time.sleep(3)

        raise RpcStartupError("Remote RPC did not become reachable. Stop.")
    
    