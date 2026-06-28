import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from model import Model, rpc_server

_TIMEOUT = 2.0

_RPC_START_POLL_INTERVAL = 2
_RPC_START_TIMEOUT      = 20

#___________________________________________________________________________________
def _tcp_reachable(addr: rpc_server) -> bool:
    try:
        with socket.create_connection((addr.IP, addr.PORT), timeout=_TIMEOUT):
            return True
    except OSError:
        return False

#___________________________________________________________________________________
def unreachable_rpc_servers(servers: list[rpc_server]) -> list[rpc_server]:
    """Return the rpc_server entries that do not respond to a TCP ping.

    Returns [] immediately if servers is empty.
    Checks are run in parallel (one thread per server).
    """
    if not servers:
        return []

    dead: list[rpc_server] = []
    with ThreadPoolExecutor(max_workers=len(servers)) as pool:
        future_to_addr = {pool.submit(_tcp_reachable, s): s for s in servers}
        for future in as_completed(future_to_addr):
            addr = future_to_addr[future]
            try:
                alive = future.result()
            except Exception:
                alive = False
            if not alive:
                dead.append(addr)

    return dead

#___________________________________________________________________________________
def start_rpc_server(addr: rpc_server) -> bool:
    """Start rpc-server on the remote host via SSH in the background.

    </dev/null detaches stdin so nohup does not stay bound to the SSH session.
    Returns True if SSH exited with code 0.
    """
    remote_cmd = (
        f"LLAMA_CACHE={addr.cachepath} "
        f"nohup {addr.bin} --host 0.0.0.0 --port {addr.PORT} -c  "
        f">/tmp/rpc-server.out 2>&1 </dev/null &"
    )
    print(f"  SSH: {addr.remuser}@{addr.IP}  \"{remote_cmd}\"", file=sys.stderr)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", f"{addr.remuser}@{addr.IP}", remote_cmd],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        print(f"  SSH stderr: {result.stderr.strip()}", file=sys.stderr)
    return result.returncode == 0

#___________________________________________________________________________________
def wait_for_rpc_servers(servers: list[rpc_server]) -> list[rpc_server]:
    """Poll until all servers become reachable or the timeout expires.

    Returns the list of servers still unreachable when the deadline is reached.
    """
    deadline = time.monotonic() + _RPC_START_TIMEOUT
    remaining = list(servers)
    while remaining and time.monotonic() < deadline:
        time.sleep(_RPC_START_POLL_INTERVAL)
        remaining = [a for a in remaining if not _tcp_reachable(a)]
        if remaining:
            addrs = ", ".join(f"{a.IP}:{a.PORT}" for a in remaining)
            print(f"  Still unreachable: {addrs}", file=sys.stderr)
    return remaining

#___________________________________________________________________________________
if __name__ == "__main__":
    from model import load_models

    if len(sys.argv) < 2:
        print("Usage: python rpc_check.py <config.json>")
        sys.exit(1)

    for m in load_models(sys.argv[1]):
        down = unreachable_rpc_servers(m.rpcservers)
        if not m.rpcservers:
            print(f"{m.model_name}: no rpc server configured")
        elif down:
            addrs = ", ".join(f"{a.IP}:{a.PORT} ({a.remuser}@{a.bin})" for a in down)
            print(f"{m.model_name}: not reachable server -> {addrs}")
        else:
            print(f"{m.model_name}: all rpc servers reachable")
