#!/usr/bin/env python3
"""Launch llama-server for a given model defined in models.json."""


import os
import re
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from logzero import logger
from model import Model, load_models
from config_manager import get_settings
from rpc_check import unreachable_rpc_servers, start_rpc_server, wait_for_rpc_servers

MODELS_JSON = Path(__file__).parent / "models.json"

_CSV_TOKENS = re.compile(r"[A-Za-z0-9]+(?:,[A-Za-z0-9]+)*")

settings = get_settings()

#___________________________________________________________________________________
def valid_csv_tokens(s) -> bool:
    return isinstance(s, str) and _CSV_TOKENS.fullmatch(s) is not None

#___________________________________________________________________________________
def _build_command(binary: str, model: Model, devices: str = "", ctx: int | None = None) -> list[str]:
    cmd = [binary, "-m", str(model.model_path), "-c", str(ctx if ctx is not None else model.ctxsize)]

    if model.fitt:
        cmd += ["-fitt", model.fitt]

    if model.rpcservers:
        rpc_list = ",".join(f"{s.IP}:{s.PORT}" for s in model.rpcservers)
        cmd += ["--rpc", rpc_list]

    if devices:
        cmd += ["--device", devices]

    if model.mmproj_path and str(model.mmproj_path).lower() not in ("none", "null", ""):
        cmd += ["--mmproj", str(model.mmproj_path)]

    data = {"reasoning_effort": model.reasoning}

    cmd += ["--host", settings.ADDRESS_BIND]
    cmd += ["--port", str(settings.PORT_BIND)]
    cmd += ["--split-mode", "layer"]
    cmd += ["--metrics"]
    cmd += ["--jinja"]
    cmd += ["-fa", "on"]
    cmd += ["-fit", "on"] # Using "on" makes the rpc/Vulkan on PC with 2 NVidia cards crash
    cmd += ["-fitc", "8192"]
    cmd += ["-np", "1"]
    cmd += ["--no-warmup"]
    cmd += ["--temp", str(model.temperature)]
    cmd += ["--top-p", str(model.top_p)]
    cmd += ["--top-k", str(model.top_k)]
    cmd += ["--chat-template-kwargs", f"'{json.dumps(data)}'"]
    cmd += ["--seed", "123456789"]
    if model.kvquant:
        cmd += ["-ctk", model.kvquant]
        cmd += ["-ctv", model.kvquant]
    cmd += ["--alias", str(model.alias)]
    if model.min_p >= 0:
        cmd += ["--min-p", str(model.min_p)]
    #if model.extras:
    #    cmd += model.extras
    if model.ub:
        cmd += ["-ub", str(model.ub)]
    if model.b:
        cmd += ["-b", str(model.b)]

    return cmd

#___________________________________________________________________________________
def _ssh_dest() -> str | None:
    if not settings.LLAMA_SERVER_HOST:
        return None
    if settings.LLAMA_SERVER_USER:
        return f"{settings.LLAMA_SERVER_USER}@{settings.LLAMA_SERVER_HOST}"
    return settings.LLAMA_SERVER_HOST

#___________________________________________________________________________________
class ServerHostUnreachable(Exception):
    """Raised when LLAMA_SERVER_HOST cannot be contacted over SSH (vs. the
    server process simply not running)."""

#___________________________________________________________________________________
def _server_location() -> str:
    return _ssh_dest() or "localhost"

#___________________________________________________________________________________
def _pgrep_pattern() -> str:
    """Regex for pgrep/pkill -f that matches the llama-server command line but
    NOT the wrapping shell/ssh that carries this very pattern (classic [x] trick)."""
    b = settings.LLAMA_SERVER_BIN
    return f"[{b[0]}]{b[1:]}" if b else b

#___________________________________________________________________________________
def _run_on_server(shell_cmd: str, timeout: int = 15) -> subprocess.CompletedProcess:
    """Run shell_cmd on LLAMA_SERVER_HOST via SSH if configured, else locally.

    Raises ServerHostUnreachable if SSH itself fails (connection refused,
    timeout, auth/host error). SSH reserves exit code 255 for its own failures,
    while pgrep/pkill only ever return 0/1/2/3, so 255 unambiguously means the
    host was not reached rather than 'no process found'."""
    ssh_dest = _ssh_dest()
    if ssh_dest:
        argv = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_dest, shell_cmd]
    else:
        argv = ["bash", "-c", shell_cmd]
    try:
        logger.debug(f"Running command {argv}")
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        if ssh_dest:
            raise ServerHostUnreachable(f"timeout after {timeout}s contacting {ssh_dest}") from e
        raise
    if ssh_dest and r.returncode == 255:
        detail = r.stderr.strip() or "connection failed"
        raise ServerHostUnreachable(f"SSH error contacting {ssh_dest}: {detail}")
    return r

#___________________________________________________________________________________
def _server_pids() -> list[str]:
    """PIDs of the running llama-server process(es), or [] if none.

    May raise ServerHostUnreachable (propagated from _run_on_server)."""
    r = _run_on_server(f"pgrep -f -- '{_pgrep_pattern()}'")
    return [p for p in r.stdout.split() if p.strip().isdigit()]

#___________________________________________________________________________________
def report_server_status() -> bool:
    """Print whether llama-server is running. Return True if running."""
    where = _server_location()
    pids = _server_pids()
    if pids:
        logger.info(f"llama-server is RUNNING on {where} (pid(s): {', '.join(pids)})")
        return True
    logger.warning(f"llama-server is NOT running on {where}")
    return False

#___________________________________________________________________________________
def stop_server() -> bool:
    """Kill the llama-server process on the server. Return True if stopped."""
    where = _server_location()
    pids = _server_pids()
    if not pids:
        logger.warning(f"No llama-server process found on {where}.")
        return True

    pattern = _pgrep_pattern()
    logger.debug(f"Sending SIGTERM to llama-server on {where} (pid(s): {', '.join(pids)})...")
    _run_on_server(f"pkill -TERM -f -- '{pattern}'")

    time.sleep(2)
    pids = _server_pids()
    if pids:
        logger.debug(f"Still running (pid(s): {', '.join(pids)}); sending SIGKILL...")
        _run_on_server(f"pkill -KILL -f -- '{pattern}'")
        time.sleep(1)
        pids = _server_pids()

    if pids:
        logger.error(f"Error: could not stop llama-server on {where} (pid(s): {', '.join(pids)}).", file=sys.stderr)
        return False
    print(f"llama-server stopped on {where}.")
    return True

#___________________________________________________________________________________
def _run_server_action(action) -> "None":
    """Run a server-management action and exit: 0 = ok, 1 = not ok,
    2 = LLAMA_SERVER_HOST unreachable."""
    try:
        ok = action()
    except ServerHostUnreachable as e:
        logger.error(f"Error: host unreachable — {e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if ok else 1)

#___________________________________________________________________________________
def start_model(
    model_name: str | None,
    dry_run: bool = False,
    only_rpc: bool = False,
    only_list_devs: bool = False,
    list_models: bool = False,
    kill_server: bool = False,
    server_status: bool = False,
    override_temp: float | None = None,
    override_top_p: float | None = None,
    override_top_k: int | None = None,
    override_min_p: float | None = None,
    override_devices: str | None = None,
    override_fitt: str | None = None,
    override_ctx: int | None = None
) -> None:
    if server_status:
        _run_server_action(report_server_status)

    if kill_server:
        _run_server_action(stop_server)

    models = load_models(MODELS_JSON, remote_host=settings.LLAMA_SERVER_HOST, remote_user=settings.LLAMA_SERVER_USER)

    if list_models:
        models_info = []
        for m in models:
            m_info = f"{m.model_name} ({int(m.size_gib)} GiB - {len(m.rpcservers)} RPC)"
            models_info.append(m_info)
        #print(f"Available models:\n  {"\n  ".join(m.model_name for m in models)}")
        print(f"Available models:\n  {"\n  ".join(m_info for m_info in models_info)}")
        sys.exit(0)

    if not model_name:
        print("Error: model_name is required", file=sys.stderr)
        sys.exit(1)

    model = next((m for m in models if m.model_name == model_name), None)
    if model is None:
        available = "\n  ".join(m.model_name for m in models)
        logger.error(f"Error: model '{model_name}' not found in {MODELS_JSON}.\n\nAvailable models:\n  {available}", file=sys.stderr)
        sys.exit(1)

    if (override_fitt is None) != (override_devices is None):
        logger.error(f"If override-devices is specified than also override-fitt must be specified, and vice versa")
        sys.exit(1)

    if (override_fitt):
        if not (valid_csv_tokens(override_fitt) and valid_csv_tokens(override_devices)):
            logger.error(f"Invalid format of {override_devices} or {override_fitt}")
            sys.exit(1)
        model.fitt = override_fitt

    if override_temp is not None:
        model.temperature = override_temp
    if override_top_p is not None:
        model.top_p = override_top_p
    if override_top_k is not None:
        model.top_k = override_top_k
    if override_min_p is not None:
        model.min_p = override_min_p
    if override_ctx is not None:
        model.ctxsize = override_ctx

    binary = settings.LLAMA_SERVER_BIN
    ssh_dest = _ssh_dest()
    if ssh_dest:
        cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_dest, "test", "-f", binary]
        logger.debug(f"Executing {cmd}")
        r = subprocess.run(
            cmd,
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.error(f"Error: llama-server binary not found at '{binary}' on {ssh_dest}", file=sys.stderr)
            sys.exit(1)
    else:
        if not Path(binary).is_file():
            logger.error(f"Error: llama-server binary not found at '{binary}'", file=sys.stderr)
            sys.exit(1)

    if not dry_run and model.rpcservers and not override_devices:
        logger.debug("Checking rpc servers...", flush=True)
        dead = unreachable_rpc_servers(model.rpcservers)
        if dead:
            for addr in dead:
                logger.error(f"RPC server {addr.IP}:{addr.PORT} unreachable — starting via SSH as {addr.remuser}...", flush=True)
                start_rpc_server(addr)

            still_dead = wait_for_rpc_servers(dead)
            if still_dead:
                addrs = ", ".join(f"{a.IP}:{a.PORT}" for a in still_dead)
                logger.error(f"Error: RPC server(s) still unreachable after start attempt: {addrs}", flush=True)
                sys.exit(1)

        logger.info("All RPC servers reachable.", flush=True)
        if only_rpc:
            logger.warning("'--only-start-rpc' specified. Gracefully exiting.")
            sys.exit(0)

    devices = ""
    if dry_run:
        devices = "SKIP(DRYRUN)"
    else:
        if not override_devices:
            if model.rpcservers:
                rpc_list = ",".join(f"{s.IP}:{s.PORT}" for s in model.rpcservers)
                logger.debug("Running list-devices...", flush=True)
                if ssh_dest:
                    result = subprocess.run(
                        ["ssh", ssh_dest, binary, "--rpc", rpc_list, "--list-devices"],
                        capture_output=True,
                        text=True,
                    )
                else:
                    result = subprocess.run(
                        f"{binary} --rpc {rpc_list} --list-devices",
                        shell=True,
                        capture_output=True,
                        text=True,
                    )
                if result.stdout:
                    logger.debug(result.stdout, end="", flush=True)
                if result.stderr:
                    logger.debug(result.stderr, end="", flush=True)
                # filter and join device names
                raw_devices = [
                    line.replace(":", "").split()[0]
                    for line in result.stdout.splitlines()
                    if line.strip()
                    and "Available" not in line
                    and " 0 MiB free" not in line
                ]
                devices = ",".join(raw_devices)
                logger.info(f"Using devices: {devices or '(none)'}", flush=True)
        else:
            devices = override_devices

        if only_list_devs:
            logger.warning("'--only-list-devices' specified. Gracefully exiting.")
            sys.exit(0)

    cmd = _build_command(binary, model, devices, override_ctx)
    if ssh_dest:
        exec_cmd = ["ssh", ssh_dest] + cmd
    else:
        exec_cmd = cmd
    logger.debug("Command:", " ".join(exec_cmd), flush=True)

    if dry_run:
        return

    os.execvp(exec_cmd[0], exec_cmd)

#___________________________________________________________________________________
def main() -> None:
    parser = argparse.ArgumentParser(description="Launch llama-server for a model defined in models.json")
    parser.add_argument("model_name", nargs="?", help="Model name as listed in models.json (without .gguf extension)")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    parser.add_argument("--only-start-rpc", action="store_true", help="Just start the RPC remote servers and exit")
    parser.add_argument("--only-check-rpc", action="store_true", help="Just check that RPC remote servers and reachable")
    parser.add_argument("--only-list-devices", action="store_true", help="Just retrieve the list of GPU devices from local and remote RPC servers")
    parser.add_argument("--list-models", action="store_true", help="Print the available models and exit")
    parser.add_argument("--kill-server", action="store_true", help="Kill the llama-server process on LLAMA_SERVER_HOST and exit")
    parser.add_argument("--server-status", action="store_true", help="Check whether llama-server is running on LLAMA_SERVER_HOST and exit")
    parser.add_argument("--override-temp", type=float, default=None, metavar="FLOAT")
    parser.add_argument("--override-top-p", type=float, default=None, metavar="FLOAT")
    parser.add_argument("--override-top-k", type=int, default=None, metavar="INT")
    parser.add_argument("--override-min-p", type=float, default=None, metavar="FLOAT")
    parser.add_argument("--override-devices", type=str, default=None, metavar="STR")
    parser.add_argument("--override-fitt", type=str, default=None, metavar="STR")
    parser.add_argument("--override-ctx", type=int, default=None, metavar="INT")


    args = parser.parse_args()
    start_model(
        args.model_name,
        dry_run=args.dry_run,
        only_rpc=args.only_start_rpc,
        only_list_devs=args.only_list_devices,
        list_models=args.list_models,
        kill_server=args.kill_server,
        server_status=args.server_status,
        override_temp=args.override_temp,
        override_top_p=args.override_top_p,
        override_top_k=args.override_top_k,
        override_min_p=args.override_min_p,
        override_devices=args.override_devices,
        override_fitt=args.override_fitt,
        override_ctx=args.override_ctx
    )

#___________________________________________________________________________________
if __name__ == "__main__":
    main()
