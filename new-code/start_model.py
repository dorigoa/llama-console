#!/usr/bin/env python3
"""Launch llama-server for a given model defined in models.json."""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from model import Model, load_models
from config_manager import get_settings
from rpc_check import unreachable_rpc_servers, start_rpc_server, wait_for_rpc_servers

MODELS_JSON = Path(__file__).parent / "models.json"

settings = get_settings()

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
    cmd += ["-fit", "on"]
    cmd += ["-fitc", "8192"]
    cmd += ["-np", "1"]
    cmd += ["--cache-type-k", "q8_0"]
    cmd += ["--cache-type-v", "q8_0"]
    cmd += ["--no-warmup"]
    cmd += ["--temp", str(model.temperature)]
    cmd += ["--top-p", str(model.top_p)]
    cmd += ["--top-k", str(model.top_k)]
    cmd += ["--chat-template-kwargs", f"\"{str(data)}\""]
    cmd += ["--alias", str(model.alias)]
    if model.min_p >= 0:
        cmd += ["--min-p", str(model.min_p)]

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch llama-server for a model defined in models.json")
    parser.add_argument("model_name", nargs="?", help="Model name as listed in models.json (without .gguf extension)")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it")
    parser.add_argument("--list-models", action="store_true", help="Print the available models and exit")
    args = parser.parse_args()

    models = load_models(MODELS_JSON)

    if args.list_models:
        print(f"Available models:\n  {"\n  ".join(m.model_name for m in models)}")
        sys.exit(0)

    if not args.model_name:
        parser.error("model_name is required")

    model = next((m for m in models if m.model_name == args.model_name), None)
    if model is None:
        available = "\n  ".join(m.model_name for m in models)
        print(f"Error: model '{args.model_name}' not found in {MODELS_JSON}.\n\nAvailable models:\n  {available}", file=sys.stderr)
        sys.exit(1)

    
    dead = unreachable_rpc_servers(model)
    if dead:
        for addr in dead:
            print(f"RPC server {addr.IP}:{addr.PORT} non raggiungibile — avvio via SSH come {addr.remuser}...", file=sys.stderr)
            start_rpc_server(addr)

        still_dead = wait_for_rpc_servers(dead)
        if still_dead:
            addrs = ", ".join(f"{a.IP}:{a.PORT}" for a in still_dead)
            print(f"Error: RPC server(s) ancora non raggiungibili dopo il tentativo di avvio: {addrs}", file=sys.stderr)
            sys.exit(1)

        print("Tutti i server RPC sono raggiungibili.", file=sys.stderr)

    

    binary = settings.LLAMA_SERVER_BIN

    if not Path(binary).is_file():
        print(f"Error: llama-server binary not found at '{binary}'", file=sys.stderr)
        sys.exit(1)

    devices = ""
    if model.rpcservers:
        rpc_list = ",".join(f"{s.IP}:{s.PORT}" for s in model.rpcservers)
        result = subprocess.run(
            f"{binary} --rpc {rpc_list} --list-devices"
            " | grep -v 'Available'"
            " | grep -v ' 0 MiB free'"
            " | sed 's+:++g'"
            " | awk '{{t=t\",\"$1}}END{{print t}}'"
            " | sed 's+,++'",
            shell=True,
            capture_output=True,
            text=True,
        )
        devices = result.stdout.strip()

    cmd = _build_command(binary, model, devices)
    print("Command:", " ".join(cmd))

    if args.dry_run:
        return

    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    main()
