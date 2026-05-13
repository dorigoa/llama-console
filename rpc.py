from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from config_manager import get_settings
from logging_utils import emit, LogSink, setup_console_logging
import re

settings = get_settings()

logger = setup_console_logging()

class RpcStartupError(RuntimeError):
    pass

#__________________________________________________________________________________________
# def _is_valid_rpc_list(text: str) -> bool:
#     pattern = r"""
#         ^                                   
#         (?:\d{1,3}(?:\.\d{1,3}){3}:\d+)    
#         (?:\s*,\s*\d{1,3}(?:\.\d{1,3}){3}:\d+)*  
#         $                                   
#     """
#     return re.fullmatch(pattern, text.strip(), re.VERBOSE) is not None

#__________________________________________________________________________________________
def tcp_connect(host: str, port: int, timeout_seconds: int = 2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False

#__________________________________________________________________________________________
def ensure_remote_rpc(timeout: int, 
                      RPC_HOST: str, 
                      RPC_PORT: int, 
                      platform: str, 
                      log_sink: LogSink = None, 
                      ) -> None:
    
    #if run_local_only:


    if platform == "Windows":
        for attempt in range(1, 11):
            if tcp_connect(RPC_HOST, RPC_PORT, 2):
                #logger.info(f"DEBUG - Remote RPC on {RPC_HOST}:{RPC_PORT} is now reachable")
                emit(f"Remote RPC on {RPC_HOST}:{RPC_PORT} is now reachable", log_sink)
                return
            emit(f"RPC on on {RPC_HOST}:{RPC_PORT} not reachable yet, attempt {attempt}/10 to start it", log_sink)
            
            remote_cmd = (f'schtasks /Create /TN llama-rpc-server-manual /TR "C:\\llama.cpp\\build\\bin\\Release\\rpc-server.exe --host {RPC_HOST} --port {RPC_PORT} -c" /SC ONCE /ST 23:59 /F')
            logger.info(f"DEBUG - Executing {remote_cmd}")
            subprocess.run(["ssh", RPC_HOST, remote_cmd], check=True)
            remote_cmd = ('schtasks /Run /TN llama-rpc-server-manual')
            logger.info(f"DEBUG - Executing {remote_cmd}")
            subprocess.run(["ssh", RPC_HOST, remote_cmd], check=True)
            time.sleep(3)
        raise RpcStartupError("Remote RPC did not become reachable. Stop.")
    
    if platform == "Linux" or platform == "Darwin":
        emit(f"Stopping any remotely running rpc-server...")
        remote_kill_cmd = (
            f'killall -9 {settings.RPC_SERVER_PATH["Linux"]}'
        )
        emit(f"ssh {RPC_HOST} {remote_kill_cmd}", log_sink)
        subprocess.run(["ssh", RPC_HOST, remote_kill_cmd], check=False)
        emit(f"Starting remote RPC through SSH host {RPC_HOST}", log_sink)
        time.sleep(3)
        remote_cmd = (
            f"LLAMA_CACHE={settings.RPC_CACHE_PATH} nohup {settings.RPC_SERVER_PATH['Linux']} "
            f"--host '{RPC_HOST}' "
            f"--port '{RPC_PORT}' "
            "-c >/dev/null 2>&1 &"
        )
        emit(f"ssh {RPC_HOST} {remote_cmd}", log_sink)
        subprocess.run(["ssh", RPC_HOST, remote_cmd], check=True)

        emit("Waiting for remote RPC to become reachable...", log_sink)

        for attempt in range(1, 11):
            if tcp_connect(RPC_HOST, RPC_PORT, 2):
                emit("Remote RPC is now reachable", log_sink)
                return
            emit(f"RPC not reachable yet, attempt {attempt}/10", log_sink)
            time.sleep(3)

        raise RpcStartupError("Remote RPC did not become reachable. Stop.")
    
    