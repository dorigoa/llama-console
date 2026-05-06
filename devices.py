from __future__ import annotations

import subprocess
from collections.abc import Callable

from config import settings
from logging_utils import emit, LogSink

#settings = Settings()


class DeviceDiscoveryError(RuntimeError):
    pass


def list_usable_devices(rpc_server: str, rpc_port: int, log_sink: LogSink = None) -> str:
    cmd = [
        settings.llama_server_path,
        "--rpc",
        f"{rpc_server}:{rpc_port}",
        "--list-devices",
    ]

    emit(f"Discovering devices: {' '.join(cmd)}", log_sink)

    proc = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if proc.stdout:
        for line in proc.stdout.splitlines():
            emit(f"[list-devices] {line}", log_sink)

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
    emit(f"Usable devices: {result}", log_sink)
    return result
