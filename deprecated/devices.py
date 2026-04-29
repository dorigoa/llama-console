from __future__ import annotations

import re
import subprocess

#from .config import RpcConfig, LlamaServerConfig

#________________________________________________________________________________________
class DeviceDiscoveryError(RuntimeError):
    pass

#________________________________________________________________________________________
def list_usable_devices( rpc_server: str, rpc_port: int ) -> str:
    cmd = [
        #server_config.llama_server_bin,
        "/usr/local/bin/llama-server",
        "--rpc",
        f"{rpc_server}:{rpc_port}",
        "--list-devices",
    ]

    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    devices: list[str] = []
    in_available = False

    for raw_line in proc.stdout.splitlines():
        line = raw_line.rstrip()

        if line.startswith("Available devices"):
            in_available = True
            continue

        if not in_available:
            continue

        if not line.strip():
            continue

        if " 0 MiB free" in line:
            continue

        first_field = line.split()[0]
        device_id = first_field.split(":", 1)[0]
        devices.append(device_id)

    if not devices:
        raise DeviceDiscoveryError("No usable GPU/device found. Stop.")

    return ",".join(devices)
