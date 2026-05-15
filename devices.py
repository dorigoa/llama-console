from __future__ import annotations

import subprocess
from collections.abc import Callable
from object_models import Server, ServerType
from config_manager import get_settings
from logging_utils import emit, LogSink

settings = get_settings()#Settings()

#__________________________________________________________________________________________
class DeviceDiscoveryError(RuntimeError):
    pass

#__________________________________________________________________________________________
def list_local_usable_devices(local: Server, log_sink: LogSink = None) -> str:
    if local:
        if local.type != ServerType.LLAMASERVER:
            raise DeviceDiscoveryError(f"Provided server for local device discovery is not LLAMASERVER type")
        cmd = [
            settings.LLAMA_SERVER.binarypath,
            "--list-devices",
        ]
        emit(f"-> Discovering devices: {' '.join(cmd)}", log_sink)
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if proc.returncode != 0:
            raise DeviceDiscoveryError(f"Device discovery command failed with return code {proc.returncode}")
        devices: list[str] = []
        in_available = False

        for raw_line in proc.stdout.splitlines():
            line = raw_line.rstrip()

            if line.startswith("Available devices"):
                in_available = True
                continue

            if not in_available or not line.strip():
                continue

            if " 0 MiB free" in line:
                continue

            first_field = line.split()[0]
            device_id = first_field.split(":", 1)[0]
            devices.append(device_id)

        if not devices:
            raise DeviceDiscoveryError("No usable GPU/device found. Stop.")

        result = ",".join(devices)
        emit(f"-> Usable devices: {result}", log_sink)
        return result
    else:
        raise DeviceDiscoveryError(f"Provided a None LLAMASERVER")
    
#__________________________________________________________________________________________
def list_remote_usable_devices(rpcs: list[Server], log_sink: LogSink = None) -> str:

    if not rpcs:
        raise DeviceDiscoveryError(f"Provided a None list of RPC servers")

    all = []
    for rpc in rpcs:
        all.append( f"{rpc.hostname}:{rpc.tcpport}" )

    cmd = [
        settings.LLAMA_SERVER.binarypath,
        "--rpc",
        f"{','.join(all)}",
        "--list-devices",
    ]
    
    emit(f"-> Discovering devices: {' '.join(cmd)}", log_sink)

    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if proc.returncode != 0:
        raise DeviceDiscoveryError(f"Device discovery command failed with return code {proc.returncode}")

    devices: list[str] = []
    in_available = False

    for raw_line in proc.stdout.splitlines():
        line = raw_line.rstrip()

        if line.startswith("Available devices"):
            in_available = True
            continue

        if not in_available or not line.strip():
            continue

        if " 0 MiB free" in line:
            continue

        first_field = line.split()[0]
        device_id = first_field.split(":", 1)[0]
        devices.append(device_id)

    if not devices:
        raise DeviceDiscoveryError("No usable GPU/device found. Stop.")

    result = ",".join(devices)
    emit(f"-> Usable devices: {result}", log_sink)
    return result
