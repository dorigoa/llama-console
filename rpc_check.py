import socket
import shlex
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from model import Model, rpc_server

_TIMEOUT = 2.0

_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]

_RPC_START_POLL_INTERVAL = 2
_RPC_START_TIMEOUT      = 20

#___________________________________________________________________________________
def _tcp_reachable(addr: rpc_server, exec_host: str | None = None) -> bool:
    """True if addr.IP:addr.PORT accepts a TCP connection.

    If exec_host is given, the probe is performed FROM that host over SSH
    (using bash's /dev/tcp), because only that host can reach the RPC network.
    Otherwise the probe is a local socket connection.
    """
    if exec_host:
        probe = f"timeout {int(_TIMEOUT)} bash -c 'exec 3<>/dev/tcp/{addr.IP}/{addr.PORT}'"
        try:
            r = subprocess.run(
                ["ssh", *_SSH_OPTS, exec_host, probe],
                capture_output=True, text=True, timeout=_TIMEOUT + 8,
            )
        except subprocess.TimeoutExpired:
            return False
        return r.returncode == 0
    try:
        with socket.create_connection((addr.IP, addr.PORT), timeout=_TIMEOUT):
            return True
    except OSError:
        return False

#___________________________________________________________________________________
def unreachable_rpc_servers(servers: list[rpc_server], exec_host: str | None = None) -> list[rpc_server]:
    """Return the rpc_server entries that do not respond to a TCP ping.

    Returns [] immediately if servers is empty.
    Checks are run in parallel (one thread per server).
    If exec_host is given, probes originate from that host (see _tcp_reachable).
    """
    if not servers:
        return []

    dead: list[rpc_server] = []
    with ThreadPoolExecutor(max_workers=len(servers)) as pool:
        future_to_addr = {pool.submit(_tcp_reachable, s, exec_host): s for s in servers}
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
def start_rpc_server(addr: rpc_server, exec_host: str | None = None) -> bool:
    """Start rpc-server on the RPC node via SSH in the background.

    </dev/null detaches stdin so nohup does not stay bound to the SSH session.

    If exec_host is given, the SSH to the RPC node is issued FROM that host
    (nested SSH: laptop -> exec_host -> remuser@addr.IP), because only exec_host
    has network reachability to the RPC servers. Otherwise the SSH is direct.
    Returns True if SSH exited with code 0.
    """
    remote_cmd = (
        f"LLAMA_CACHE={addr.cachepath} "
        f"nohup {addr.bin} --host 0.0.0.0 --port {addr.PORT} -c  "
        f">/tmp/rpc-server.out 2>&1 </dev/null &"
    )
    if exec_host:
        # Run the rpc-node SSH on exec_host; quote remote_cmd so the exec_host
        # shell forwards it verbatim to the inner ssh as a single argument.
        inner = f"ssh {' '.join(_SSH_OPTS)} {addr.remuser}@{addr.IP} {shlex.quote(remote_cmd)}"
        argv = ["ssh", *_SSH_OPTS, exec_host, inner]
        shown = f"{exec_host} -> {addr.remuser}@{addr.IP}"
    else:
        argv = ["ssh", *_SSH_OPTS, f"{addr.remuser}@{addr.IP}", remote_cmd]
        shown = f"{addr.remuser}@{addr.IP}"
    print(f"  SSH: {shown}  \"{remote_cmd}\"", file=sys.stderr)
    result = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        print(f"  SSH stderr: {result.stderr.strip()}", file=sys.stderr)
    return result.returncode == 0

#___________________________________________________________________________________
def kill_rpc_server(addr: rpc_server, exec_host: str | None = None) -> bool:
    """Kill rpc-server on the RPC node with `killall rpc-server`.

    `killall` is available on both Linux (psmisc) and macOS/BSD, so the same
    command works regardless of the node OS. If exec_host is given, the SSH to
    the RPC node is issued FROM that host (nested SSH: laptop -> exec_host ->
    remuser@addr.IP), because only exec_host can reach the RPC network.
    Otherwise the SSH is direct.

    Returns True if the node was reached and killall ran (rc 0 = killed,
    rc 1 = no matching process / already stopped). Returns False on SSH/contact
    failure (rc 255 or other).
    """
    remote_cmd = "killall rpc-server"
    if exec_host:
        inner = f"ssh {' '.join(_SSH_OPTS)} {addr.remuser}@{addr.IP} {shlex.quote(remote_cmd)}"
        argv = ["ssh", *_SSH_OPTS, exec_host, inner]
        shown = f"{exec_host} -> {addr.remuser}@{addr.IP}"
    else:
        argv = ["ssh", *_SSH_OPTS, f"{addr.remuser}@{addr.IP}", remote_cmd]
        shown = f"{addr.remuser}@{addr.IP}"
    print(f"  SSH: {shown}  \"{remote_cmd}\"", file=sys.stderr)
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        print(f"  SSH timeout contacting {shown}", file=sys.stderr)
        return False
    if result.returncode not in (0, 1):
        print(f"  SSH stderr: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True

#___________________________________________________________________________________
def wait_for_rpc_servers(servers: list[rpc_server], exec_host: str | None = None) -> list[rpc_server]:
    """Poll until all servers become reachable or the timeout expires.

    Returns the list of servers still unreachable when the deadline is reached.
    If exec_host is given, probes originate from that host (see _tcp_reachable).
    """
    deadline = time.monotonic() + _RPC_START_TIMEOUT
    remaining = list(servers)
    while remaining and time.monotonic() < deadline:
        time.sleep(_RPC_START_POLL_INTERVAL)
        remaining = [a for a in remaining if not _tcp_reachable(a, exec_host)]
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
