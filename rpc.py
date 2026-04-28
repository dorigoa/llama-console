from __future__ import annotations

from logzero import logger

import subprocess
import socket
import time

#from .config import RpcConfig
#from .netcheck import tcp_connect

#________________________________________________________________________________________
class RpcStartupError(RuntimeError):
    pass

#________________________________________________________________________________________
def tcp_connect(host: str, port: int, timeout_seconds: int = 2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True
    except OSError:
        return False

#________________________________________________________________________________________
def ensure_remote_rpc( server: str, timeout: int, rpc_host: str, rpc_port: int ) -> None:
    # The original shell script first checks rpc_host_ssh:rpc_port, then waits on
    # rpc_server:rpc_port after starting the remote process. Here we check the
    # actual RPC endpoint first, which is normally the semantically correct test.
    if tcp_connect( server, rpc_port, timeout ):
        logger.info("Remote RPC is Alive!")
        return

    logger.info("Starting remote RPC...")

    remote_cmd = (
        f"nohup /usr/local/bin/rpc-server "
        f"--host '{rpc_host}' "
        f"--port '{rpc_port}' "
        "-c >/dev/null 2>&1 &"
    )

    subprocess.run(["ssh", server, remote_cmd], check=True)

    logger.info("Waiting for remote RPC to become reachable...")

    for _ in range(10):
        if tcp_connect( rpc_host, rpc_port, 2 ):
            logger.info("Remote RPC is now Alive!")
            return
        time.sleep(3)

    raise RpcStartupError("Remote RPC did not become reachable. Stop.")
