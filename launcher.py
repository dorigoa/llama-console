from __future__ import annotations

import shlex
from pathlib import Path

from config import Settings

settings = Settings()


def build_llama_command(
    llama_server_bin: str,
    rpc_server: str,
    rpc_port: int,
    gguf_file: str,
    mmproj_file: str | None,
    devices: str,
    sslkeyfile: str,
    sslcertfile: str,
    tensorsplit: str,
    splitmode: str,
    ctxsize: str,
    *,
    listen_host: str | None = None,
    listen_port: int | str | None = None,
    #use_sudo: bool = True,
) -> list[str]:
    cmd: list[str] = []

    #if use_sudo:
    #    cmd.append("sudo")

    cmd.extend([
        llama_server_bin,
        "--host", str(listen_host or settings.llama_server_host),
        "--port", str(listen_port or settings.llama_server_port),
        "-m", str(gguf_file),
        "--rpc", f"{rpc_server}:{rpc_port}",
        "--device", str(devices),
        "--split-mode", str(splitmode),
        "--tensor-split", str(tensorsplit),
        "-ngl", str(settings.llama_param["ngl"]),
        "--fit", str(settings.llama_param["fit"]),
        "-c", str(ctxsize),
        "-t", str(settings.llama_param["threads"]),
        "-tb", str(settings.llama_param["threadsbunch"]),
        "--parallel", str(settings.llama_param["parallel"]),
        "--ssl-key-file", str(sslkeyfile),
        "--ssl-cert-file", str(sslcertfile),
    ])

    if mmproj_file:
        cmd.extend(["--mmproj", str(mmproj_file)])

    return cmd


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)


def validate_ssl_files(key_file: Path, cert_file: Path) -> None:
    missing = [str(p) for p in (key_file, cert_file) if not p.exists()]
    if missing:
        joined = "\n".join(f"  {p}" for p in missing)
        raise FileNotFoundError(f"Missing SSL file(s):\n{joined}")
