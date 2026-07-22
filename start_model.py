#!/usr/bin/env python3
"""Launch llama-server for a given model defined in models.json.

When LLAMA_SERVER_HOST is set the server is **detached** (nohup + background) so
this script exits immediately.  Log output goes to a well-known log file that the
UI can poll via SSH.
"""


import os
import re
import shlex
import sys
import json
import time
import argparse
import requests
import subprocess
from pathlib import Path
from logzero import logger
from model import Model, load_models
from config_manager import get_settings
from rpc_check import unreachable_rpc_servers, start_rpc_server, wait_for_rpc_servers, kill_rpc_server

LLAMA_LOG_FILE = "/tmp/llama-server.log"  # shared path for polling the output
LLAMA_BOOT_LOG = "/tmp/llama-server.boot.log"  # startup stdout/stderr (crash diagnostics)

_CSV_TOKENS = re.compile(r"[A-Za-z0-9]+(?:,[A-Za-z0-9]+)*")

settings = get_settings()

#___________________________________________________________________________________
def valid_csv_tokens(s) -> bool:
    return isinstance(s, str) and _CSV_TOKENS.fullmatch(s) is not None

#___________________________________________________________________________________
def _get_first_model_name(endpoint: str) -> tuple[str,int] | None:

    url = f"http://{endpoint}/models"
    try:
        response = requests.get(url)
        response.raise_for_status()  # raise an exception for HTTP codes 4xx, 5xx
        data = response.json()
        # Let's make sure the structure is as expected
        if isinstance(data, dict) and 'models' in data and isinstance(data['models'], list) and len(data['models']) > 0 and 'data' in data and len(data['data']) > 0:
            first_model = data['models'][0]
            first_data  = data['data'][0]#['meta']#['n_ctx']
            if isinstance(first_model, dict) and 'name' in first_model and isinstance(first_data, dict) and 'meta' in first_data:
                return (first_model['name'], first_data['meta']['n_ctx'])
            else:
                return None
        else:
            raise ValueError("JSON response has a bad structure: missing or empty 'models' and/or 'data or they are not lists")
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"HTTP request error: {e}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"JSON parsing error: {e}") from e

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

    #cmd += ["--chat-template", "chatml"]
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
    cmd += ["--chat-template-kwargs", json.dumps(data)]
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
    # Log file for detached execution (UI polls it)
    cmd += ["--log-file", LLAMA_LOG_FILE]

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
        print(f"llama-server is RUNNING on {where} (pid(s): {', '.join(pids)})")
        try:
            model, ctxsize = _get_first_model_name(f"{settings.LLAMA_SERVER_HOST}:{settings.LLAMA_SERVER_PORT}")
        except RuntimeError as e:
            print(f"llama-server is RUNNING but not ready yet: {e}")
        else:
            if model:
                print(f"Running model: {model} - CTX Size: {ctxsize}")
        return True
    print(f"llama-server is NOT RUNNING on {where}")
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
def tail_log(lines: int = 50, follow: bool = True) -> int:
    """Stream llama-server's log file from the server to this terminal.

    Uses `tail` over SSH (or locally) on LLAMA_LOG_FILE. With follow=True it uses
    `tail -F` (follow by name + retry), so it survives the file being rotated or
    recreated on a server restart, and keeps streaming until the user hits Ctrl-C.

    Returns the tail/ssh exit code (0 on a clean Ctrl-C)."""
    where = _server_location()
    ssh_dest = _ssh_dest()

    flag = "-F" if follow else ""
    tail_cmd = f"tail -n {int(lines)} {flag} {shlex.quote(LLAMA_LOG_FILE)}".replace("  ", " ")

    if follow:
        logger.info(f"Following {LLAMA_LOG_FILE} on {where} (Ctrl-C to stop)...")
    else:
        logger.info(f"Last {int(lines)} lines of {LLAMA_LOG_FILE} on {where}:")

    if ssh_dest:
        # -t allocates a remote pty so Ctrl-C is forwarded and the remote `tail`
        # is torn down cleanly instead of lingering.
        argv = ["ssh", "-t", "-o", "ConnectTimeout=5", ssh_dest, tail_cmd]
    else:
        argv = ["bash", "-c", tail_cmd]

    logger.debug(f"Running command {argv}")
    try:
        r = subprocess.run(argv)
    except KeyboardInterrupt:
        return 0
    return r.returncode

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
def _launch_detached(cmd: list[str], ssh_dest: str | None) -> None:
    """Start llama-server in the background, detached from this session, then
    VERIFY it actually stayed up before exiting.

    Detach mechanics: redirect ALL of stdin/stdout/stderr away from the SSH
    channel. Redirecting only stdin (an older behaviour) left stdout/stderr
    wired to the SSH pipe, so ssh kept the channel open (this script would hang
    waiting for EOF) and, once the connection was torn down, llama-server could
    be killed by SIGHUP/SIGPIPE. `nohup` ignores SIGHUP; </dev/null frees the
    ssh channel so ssh returns immediately. Since ssh is invoked without a PTY
    there is no controlling terminal, so nohup alone is enough to survive the
    parent shell exiting.

    `setsid` additionally starts a brand-new session (extra isolation), but it
    is util-linux only and does NOT exist on macOS. We therefore detect it at
    runtime in the remote shell and use `setsid nohup` when present, falling
    back to plain `nohup` otherwise — portable across Linux and macOS.

    stdout/stderr go to LLAMA_BOOT_LOG (not /dev/null): an early crash (bad
    argument, missing shared library, OOM) happens before llama-server opens
    --log-file, so without a boot log it would leave no trace at all. The
    server's own runtime output still lands in LLAMA_LOG_FILE via --log-file,
    which --tail-log can follow.
    """
    server_args = " ".join(shlex.quote(a) for a in cmd)
    boot_log = shlex.quote(LLAMA_BOOT_LOG)
    # D = 'setsid nohup' where setsid exists (Linux), else 'nohup' (macOS).
    # $D is intentionally unquoted so it word-splits into 1 or 2 words.
    detach = "D=nohup; command -v setsid >/dev/null 2>&1 && D='setsid nohup';"
    remote_cmd = f"{detach} $D {server_args} >{boot_log} 2>&1 </dev/null &"

    where = ssh_dest or "localhost"
    logger.info(f"Starting detached llama-server on {where} ...")
    logger.info(f"Log file: {LLAMA_LOG_FILE}  (boot log: {LLAMA_BOOT_LOG})")

    if ssh_dest:
        argv = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", ssh_dest, remote_cmd]
    else:
        argv = ["bash", "-c", remote_cmd]
    r = subprocess.run(argv, capture_output=True, text=True)
    if r.returncode != 0:
        logger.error(f"Failed to launch llama-server on {where}: {r.stderr.strip()}")
        sys.exit(1)

    # ssh/bash returns as soon as the background shell forks; that says nothing
    # about whether llama-server survived. Poll pgrep to confirm it is alive.
    for _ in range(10):
        time.sleep(1)
        try:
            pids = _server_pids()
        except ServerHostUnreachable as e:
            logger.warning(f"Cannot verify server yet: {e}")
            continue
        if pids:
            logger.info(f"llama-server RUNNING on {where} (pid(s): {', '.join(pids)}).")
            sys.exit(0)

    # Not alive: it crashed at startup. Show the boot log so the user knows why.
    logger.error(f"llama-server did NOT stay up on {where}. Boot log ({LLAMA_BOOT_LOG}):")
    boot = _run_on_server(f"tail -n 40 {shlex.quote(LLAMA_BOOT_LOG)} 2>/dev/null || true")
    logger.error("\n" + (boot.stdout.rstrip() if boot.stdout.strip() else "(boot log empty)"))
    sys.exit(1)

#___________________________________________________________________________________
def start_model(
    model_name: str | None,
    dry_run: bool = False,
    only_rpc: bool = False,
    only_check_rpc: bool = False,
    only_list_devs: bool = False,
    list_models: bool = False,
    kill_server: bool = False,
    server_status: bool = False,
    follow_log: bool = False,
    tail_lines: int = 50,
    kill_rpc: bool = False,
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

    if follow_log:
        try:
            rc = tail_log(lines=tail_lines, follow=True)
        except ServerHostUnreachable as e:
            logger.error(f"Error: host unreachable — {e}")
            sys.exit(2)
        sys.exit(rc)

    models = load_models(settings.MODELS_JSON, remote_host=settings.LLAMA_SERVER_HOST, remote_user=settings.LLAMA_SERVER_USER)

    if list_models:
        models_info = []
        for m in models:
            m_info = f"{m.model_name} ({int(m.size_gib) if m.size_gib is not None else '?'} GiB - {len(m.rpcservers)} RPC)"
            models_info.append(m_info)
        # print("Available models:\n  " + "\n  ".join(m.model_name for m in models))
        print(
            "Available models:\n  "
            + "\n  ".join(m_info for m_info in models_info)
        )
        sys.exit(0)

    if not model_name:
        print("Error: model_name is required", file=sys.stderr)
        sys.exit(1)

    model = next((m for m in models if m.model_name == model_name), None)
    if model is None:
        available = "\n  ".join(m.model_name for m in models)
        logger.error(f"Error: model '{model_name}' not found in {settings.MODELS_JSON}.\n\nAvailable models:\n  {available}", file=sys.stderr)
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

    if only_check_rpc:
        # Check only: report which RPC servers are not running and exit.
        # Never start them (that's what --only-start-rpc is for).
        if not model.rpcservers:
            logger.info(f"Model '{model.model_name}' has no RPC servers configured.")
            sys.exit(0)
        logger.debug(f"Checking rpc servers (via {ssh_dest or 'localhost'})...")
        dead = unreachable_rpc_servers(model.rpcservers, exec_host=ssh_dest)
        if dead:
            for addr in dead:
                logger.info(f"RPC server {addr.IP}:{addr.PORT} is NOT running")
            sys.exit(1)
        logger.info("All RPC servers reachable.")
        sys.exit(0)

    if kill_rpc:
        # Kill rpc-server on every RPC node of the model (killall rpc-server),
        # originating from LLAMA_SERVER_HOST. Never starts anything.
        if not model.rpcservers:
            logger.info(f"Model '{model.model_name}' has no RPC servers configured.")
            sys.exit(0)
        failed = []
        for addr in model.rpcservers:
            via = f"{ssh_dest} -> " if ssh_dest else ""
            logger.info(f"Killing rpc-server on {via}{addr.remuser}@{addr.IP} ...")
            if not kill_rpc_server(addr, exec_host=ssh_dest):
                failed.append(f"{addr.IP}:{addr.PORT}")
        if failed:
            logger.error(f"Could not kill rpc-server on: {', '.join(failed)}")
            sys.exit(1)
        logger.info("rpc-server killed on all RPC nodes (or already stopped).")
        sys.exit(0)

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

    if not dry_run and not only_list_devs and model.rpcservers and not override_devices:
        # RPC probing/starting must originate from LLAMA_SERVER_HOST: only it can
        # reach the RPC network. When ssh_dest is None (local llama-server), the
        # operations run locally as before.
        # Skipped for --only-list-devices: that option must NOT start anything.
        logger.debug(f"Checking rpc servers (via {ssh_dest or 'localhost'})...")
        dead = unreachable_rpc_servers(model.rpcservers, exec_host=ssh_dest)
        if dead:
            for addr in dead:
                via = f"{ssh_dest} -> " if ssh_dest else ""
                logger.warning(f"RPC server {addr.IP}:{addr.PORT} unreachable — starting via SSH as {via}{addr.remuser}...")
                start_rpc_server(addr, exec_host=ssh_dest)

            still_dead = wait_for_rpc_servers(dead, exec_host=ssh_dest)
            if still_dead:
                addrs = ", ".join(f"{a.IP}:{a.PORT}" for a in still_dead)
                logger.error(f"Error: RPC server(s) still unreachable after start attempt: {addrs}")
                sys.exit(1)

        logger.info("All RPC servers reachable.")
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
                logger.debug("Running list-devices...")
                if ssh_dest:
                    result = subprocess.run(
                        ["ssh", ssh_dest, binary, "--rpc", rpc_list, "--list-devices"],
                        capture_output=True,
                        text=True,
                    )
                else:
                    result = subprocess.run(
                        #f"{binary} --rpc {rpc_list} --list-devices",
                        [binary, "--rpc", rpc_list, "--list-devices"],
                        shell=True,
                        capture_output=True,
                        text=True,
                    )
                if result.stdout:
                    logger.debug(result.stdout)
                if result.stderr:
                    logger.debug(result.stderr)
                # filter and join device names
                raw_devices = [
                    line.replace(":", "").split()[0]
                    for line in result.stdout.splitlines()
                    if line.strip()
                    and "Available" not in line
                    and " 0 MiB free" not in line
                ]
                devices = ",".join(raw_devices)
                logger.info(f"Using devices: {devices or '(none)'}")
        else:
            devices = override_devices

        if only_list_devs:
            logger.warning("'--only-list-devices' specified. Gracefully exiting.")
            sys.exit(0)

    cmd = _build_command(binary, model, devices, override_ctx)
    logger.debug(f"Command: {cmd}")

    if dry_run:
        # Still show the command with --log-file for reference
        print("Dry-run command:")
        print("  " + " ".join(shlex.quote(a) for a in cmd))
        return

    # DETACH: start the server in the background, verify it stayed up, then exit.
    _launch_detached(cmd, ssh_dest)

#___________________________________________________________________________________
def main() -> None:
    parser = argparse.ArgumentParser(description="Launch llama-server for a model defined in models.json")
    parser.add_argument("model_name", nargs="?", help="Model name as listed in models.json (without .gguf extension)")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    parser.add_argument("--only-start-rpc", action="store_true", help="Just start the RPC remote servers and exit")
    parser.add_argument("--only-check-rpc", action="store_true", help="Just check that RPC remote servers and reachable")
    parser.add_argument("--only-list-devices", action="store_true", help="Just retrieve the list of GPU devices from local and remote RPC servers (does NOT start RPC servers)")
    parser.add_argument("--kill-rpc-server", action="store_true", help="Run 'killall rpc-server' on every RPC node of the model (via LLAMA_SERVER_HOST) and exit")
    parser.add_argument("--list-models", action="store_true", help="Print the available models and exit")
    parser.add_argument("--kill-server", action="store_true", help="Kill the llama-server process on LLAMA_SERVER_HOST and exit")
    parser.add_argument("--server-status", action="store_true", help="Check whether llama-server is running on LLAMA_SERVER_HOST and exit")
    parser.add_argument("--tail-log", action="store_true", help=f"Follow (tail -F) llama-server's log file ({LLAMA_LOG_FILE}) on LLAMA_SERVER_HOST until Ctrl-C, then exit")
    parser.add_argument("--tail-lines", "-n", type=int, default=50, metavar="INT", help="Number of trailing log lines to show before following (default: 50)")
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
        only_check_rpc=args.only_check_rpc,
        only_list_devs=args.only_list_devices,
        list_models=args.list_models,
        kill_server=args.kill_server,
        server_status=args.server_status,
        follow_log=args.tail_log,
        tail_lines=args.tail_lines,
        kill_rpc=args.kill_rpc_server,
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
