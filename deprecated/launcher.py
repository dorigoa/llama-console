from __future__ import annotations

import shlex
import subprocess
import sys
import webbrowser
from pathlib import Path
from logzero import logger

from config import Settings

settings = Settings()

#________________________________________________________________________________________
def build_llama_command( llama_server_bin: str, 
                        rpc_server: str,
                        rpc_port: int,
                        gguf_file: str,
                        mmproj_file: str,
                        devices: str,
                        sslkeyfile,
                        sslcertfile,
                        tensorsplit,
                        splitmode,
                        ctxsize) -> list[str]:
    #server = config.server
    #rpc = config.rpc

    cmd: list[str] = []
    #if server.use_sudo:
    cmd.append("sudo")

    cmd.extend(
        [
            llama_server_bin,
            "--host", settings.llama_server_host,
            "--port", settings.llama_server_port,
            "-m", str(gguf_file),
            "--rpc", f"{rpc_server}:{rpc_port}",
            "--device", devices,
            "--split-mode", splitmode,
            "--tensor-split", tensorsplit,
            "-ngl", settings.llama_param['ngl'],
            "--fit", settings.llama_param['fit'],
            "-c", ctxsize,
            "-t", settings.llama_param['threads'],
            "-tb", settings.llama_param['threadsbunch'],
            "--parallel", settings.llama_param['parallel'],
            "--ssl-key-file", sslkeyfile,
            "--ssl-cert-file", sslcertfile,
        ]
    )

    if mmproj_file:
        cmd.extend(["--mmproj", mmproj_file])

    return cmd

#________________________________________________________________________________________
def print_command(cmd: list[str]) -> None:
    logger.info(f"Executing: [{" ".join(cmd)}]")

#________________________________________________________________________________________
def validate_ssl_files(key_file: Path, cert_file: Path) -> None:
    missing = [str(p) for p in (key_file, cert_file) if not p.exists()]
    if missing:
        joined = "\n".join(f"  {p}" for p in missing)
        raise FileNotFoundError(f"Missing SSL file(s):\n{joined}")
