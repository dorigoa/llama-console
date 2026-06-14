import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from model import Model, rpc_server

_TIMEOUT = 2.0   # secondi per il tentativo di connessione TCP

_RPC_START_POLL_INTERVAL = 2   # secondi tra un tentativo e l'altro
_RPC_START_TIMEOUT      = 20  # secondi massimi di attesa

#___________________________________________________________________________________
def _tcp_reachable(addr: rpc_server) -> bool:
    try:
        with socket.create_connection((addr.IP, addr.PORT), timeout=_TIMEOUT):
            return True
    except OSError:
        return False

#___________________________________________________________________________________
def unreachable_rpc_servers(model: Model) -> list[rpc_server]:
    """Ritorna gli rpc_server di model che non rispondono al ping TCP.

    Se la lista rpcservers è vuota ritorna [] senza effettuare alcun tentativo.
    I check vengono eseguiti in parallelo (un thread per server).
    """
    servers = model.rpcservers
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
    """Avvia rpc-server sul host remoto via SSH in background.

    </dev/null stacca stdin così nohup non rimane agganciato alla sessione SSH.
    Ritorna True se SSH ha restituito exit code 0.
    """
    remote_cmd = (
        f"LLAMA_CACHE={addr.cachepath} "
        f"nohup {addr.bin} --host 0.0.0.0 --port {addr.PORT} -c  "
        f">/dev/null 2>&1 </dev/null &"
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
    """Attende con polling che i server diventino raggiungibili.

    Ritorna la lista di quelli ancora non raggiungibili allo scadere del timeout.
    """
    deadline = time.monotonic() + _RPC_START_TIMEOUT
    remaining = list(servers)
    while remaining and time.monotonic() < deadline:
        time.sleep(_RPC_START_POLL_INTERVAL)
        remaining = [a for a in remaining if not _tcp_reachable(a)]
        if remaining:
            addrs = ", ".join(f"{a.IP}:{a.PORT}" for a in remaining)
            print(f"  Ancora non raggiungibili: {addrs}", file=sys.stderr)
    return remaining


#___________________________________________________________________________________
if __name__ == "__main__":
    from model import load_models

    if len(sys.argv) < 2:
        print("Usage: python rpc_check.py <config.json>")
        sys.exit(1)

    for m in load_models(sys.argv[1]):
        down = unreachable_rpc_servers(m)
        if not m.rpcservers:
            print(f"{m.model_name}: no rpc server configured")
        elif down:
            addrs = ", ".join(f"{a.IP}:{a.PORT} ({a.remuser}@{a.bin})" for a in down)
            print(f"{m.model_name}: not reachable server -> {addrs}")
        else:
            print(f"{m.model_name}: all rpc servers reachable")
