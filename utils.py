import os
import ping3
import signal
import statistics

from config_manager import get_settings

settings = get_settings()

def get_all_rpc_servers() -> list[str]:
    all_endpoints = []
    for rpc_server in settings.RPC_SERVERS:
        all_endpoints.append(f"{rpc_server.hostname}:{rpc_server.tcpport}")
    return all_endpoints

#_____________________________________________________________________________
def _format_context_size(value: int) -> str:
    """Human-readable label for context size values."""
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)

#_____________________________________________________________________________
def configured_context_options() -> dict[int, str]:
    """Return NiceGUI select options for context sizes.
    NiceGUI expects dict options in the form {value: label}; therefore
    the selected value remains the integer context size, while the UI
    shows the compact label such as "32k".
    """
    values = settings.CONTEXT_SIZE_OPTIONS
    
    return {int(v): _format_context_size(int(v)) for v in values}
    
#_____________________________________________________________________________
def kill_pids_sync(pids: list[int], *, terminate_timeout: float = 10.0) -> tuple[list[int], list[str]]:
    """Terminate, then force-kill if required. Returns affected PIDs and error strings."""
    import time

    current_pid = os.getpid()
    targets = [pid for pid in pids if pid != current_pid]
    killed: list[int] = []
    errors: list[str] = []

    if not targets:
        return killed, ["No killable PID found"]

    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            killed.append(pid)
        except PermissionError as exc:
            errors.append(f"PID {pid}: permission denied while sending SIGTERM: {exc}")
        except OSError as exc:
            errors.append(f"PID {pid}: failed to send SIGTERM: {exc}")

    end_time = time.monotonic() + terminate_timeout
    while time.monotonic() < end_time:
        alive: list[int] = []
        for pid in targets:
            if pid in killed:
                continue
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                killed.append(pid)
            except PermissionError:
                alive.append(pid)
        if not alive:
            return killed, errors
        time.sleep(0.2)

    for pid in targets:
        if pid in killed:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            killed.append(pid)
        except PermissionError as exc:
            errors.append(f"PID {pid}: permission denied while sending SIGKILL: {exc}")
        except OSError as exc:
            errors.append(f"PID {pid}: failed to send SIGKILL: {exc}")

    return killed, errors
  
#_____________________________________________________________________________
def ping(hostname: str, count: int = 4):
    latencies=[]
    #print(f"Pinging {hostname}")
    for _ in range(count):
        latency = ping3.ping(hostname, timeout=5)
        if latency is not None and latency > 0:
            latencies.append(latency)
    #print(f"{hostname}: {latency}")
    if latencies:
        return statistics.mean(latencies)
    return None