#!/usr/bin/env python3
"""Launch llama-server for a given model defined in models.json."""

import argparse
import os
import subprocess
import sys
import json
from pathlib import Path
from model import Model, load_models
from config_manager import get_settings
from rpc_check import unreachable_rpc_servers, start_rpc_server, wait_for_rpc_servers

MODELS_JSON = Path(__file__).parent / "models.json"

settings = get_settings()

#___________________________________________________________________________________
def _build_command(binary: str, model: Model, devices: str = "") -> list[str]:
    cmd = [binary, "-m", str(model.model_path), "-c", str(model.ctxsize)]

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
    cmd += ["--chat-template-kwargs", json.dumps(data)]
    cmd += ["-ctk", "q4_0"]
    cmd += ["-ctv", "q8_0"]
    cmd += ["--alias", str(model.alias)]
    if model.min_p >= 0:
        cmd += ["--min-p", str(model.min_p)]
    if model.extras:
        cmd += model.extras

    return cmd

#___________________________________________________________________________________
def start_model(
    model_name: str | None,
    dry_run: bool = False,
    list_models: bool = False,
    override_temp: float | None = None,
    override_top_p: float | None = None,
    override_top_k: int | None = None,
    override_min_p: float | None = None,
) -> None:
    models = load_models(MODELS_JSON)

    if list_models:
        print(f"Available models:\n  {"\n  ".join(m.model_name for m in models)}")
        sys.exit(0)

    if not model_name:
        print("Error: model_name is required", file=sys.stderr)
        sys.exit(1)

    model = next((m for m in models if m.model_name == model_name), None)
    if model is None:
        available = "\n  ".join(m.model_name for m in models)
        print(f"Error: model '{model_name}' not found in {MODELS_JSON}.\n\nAvailable models:\n  {available}", file=sys.stderr)
        sys.exit(1)

    if override_temp is not None:
        model.temperature = override_temp
    if override_top_p is not None:
        model.top_p = override_top_p
    if override_top_k is not None:
        model.top_k = override_top_k
    if override_min_p is not None:
        model.min_p = override_min_p

    if not dry_run:
        print("Checking rpc servers...", flush=True)
        dead = unreachable_rpc_servers(model)
        if dead:
            for addr in dead:
                print(f"RPC server {addr.IP}:{addr.PORT} unreachable — starting via SSH as {addr.remuser}...", flush=True)
                start_rpc_server(addr)

            still_dead = wait_for_rpc_servers(dead)
            if still_dead:
                addrs = ", ".join(f"{a.IP}:{a.PORT}" for a in still_dead)
                print(f"Error: RPC server(s) still unreachable after start attempt: {addrs}", flush=True)
                sys.exit(1)

        print("All RPC servers reachable.", flush=True)

    binary = settings.LLAMA_SERVER_BIN

    if not Path(binary).is_file():
        print(f"Error: llama-server binary not found at '{binary}'", file=sys.stderr)
        sys.exit(1)

    devices = ""
    if dry_run:
        devices = "SKIP_DRYRUN"
    else:
        if model.rpcservers:
            rpc_list = ",".join(f"{s.IP}:{s.PORT}" for s in model.rpcservers)
            print("Running list-devices...", flush=True)
            result = subprocess.run(
                f"{binary} --rpc {rpc_list} --list-devices",
                shell=True,
                capture_output=True,
                text=True,
            )
            if result.stdout:
                print(result.stdout, end="", flush=True)
            if result.stderr:
                print(result.stderr, end="", flush=True)
            # filter and join device names
            raw_devices = [
                line.replace(":", "").split()[0]
                for line in result.stdout.splitlines()
                if line.strip()
                and "Available" not in line
                and " 0 MiB free" not in line
            ]
            devices = ",".join(raw_devices)
            print(f"Using devices: {devices or '(none)'}", flush=True)

    cmd = _build_command(binary, model, devices)
    print("Command:", " ".join(cmd), flush=True)

    if dry_run:
        return

    os.execvp(cmd[0], cmd)

#___________________________________________________________________________________
def main() -> None:
    parser = argparse.ArgumentParser(description="Launch llama-server for a model defined in models.json")
    parser.add_argument("model_name", nargs="?", help="Model name as listed in models.json (without .gguf extension)")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    parser.add_argument("--list-models", action="store_true", help="Print the available models and exit")
    parser.add_argument("--override-temp", type=float, default=None, metavar="FLOAT")
    parser.add_argument("--override-top-p", type=float, default=None, metavar="FLOAT")
    parser.add_argument("--override-top-k", type=int, default=None, metavar="INT")
    parser.add_argument("--override-min-p", type=float, default=None, metavar="FLOAT")
    args = parser.parse_args()
    start_model(
        args.model_name,
        dry_run=args.dry_run,
        list_models=args.list_models,
        override_temp=args.override_temp,
        override_top_p=args.override_top_p,
        override_top_k=args.override_top_k,
        override_min_p=args.override_min_p,
    )

#___________________________________________________________________________________
if __name__ == "__main__":
    main()
