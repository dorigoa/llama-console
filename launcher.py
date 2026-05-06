from __future__ import annotations

import shlex
from pathlib import Path
from config import Settings
from logging_utils import emit, LogSink, setup_console_logging

settings = Settings()

#_____________________________________________________________________________
def build_llama_command(
    llama_server_bin: str,
    rpc_server: str,
    rpc_port: int,
    gguf_file: str,
    mmproj_file: str | None,
    devices: str,
    tensorsplit: str,
    splitmode: str,
    ctxsize: str,
    *,
    listen_host: str | None = None,
    listen_port: int | str | None = None,
) -> list[str]:
    cmd: list[str] = []

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
    ])

    if mmproj_file:
        cmd.extend(["--mmproj", str(mmproj_file)])

    return cmd

#_____________________________________________________________________________
def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(x)) for x in cmd)

#_____________________________________________________________________________
def validate_ssl_files(key_file: Path, cert_file: Path) -> None:
    missing = [str(p) for p in (key_file, cert_file) if not p.exists()]
    if missing:
        joined = "\n".join(f"  {p}" for p in missing)
        raise FileNotFoundError(f"Missing SSL file(s):\n{joined}")

#_____________________________________________________________________________
def get_llama_command(model_folder: Path, log_sink: LogSink = None, **kwargs) -> list[str]:
    """Build the llama-server command without executing it."""
    import devices
    import launcher
    import model_finder
    import rpc

    model_folder = Path(model_folder).expanduser().resolve()
    emit(f"Selected model folder: {model_folder}", log_sink)

    rpc_host = kwargs.get("rpc_host", settings.rpc_host)
    rpc_server = kwargs.get("rpc_server", settings.rpc_host)
    rpc_port = int(kwargs.get("rpc_port", settings.rpc_port))

    files = model_finder.discover_model_files(model_folder)
    emit(f"Model name   : {files.model_name}", log_sink)
    emit(f"GGUF model   : {files.gguf}", log_sink)
    emit(f"MMProj       : {files.mmproj if files.mmproj else 'none'}", log_sink)

    rpc.ensure_remote_rpc(rpc_host, 5, rpc_server, rpc_port, log_sink=log_sink)
    gpus = devices.list_usable_devices(rpc_server, rpc_port, log_sink=log_sink)

    cmd = launcher.build_llama_command(
        settings.llama_server_path,
        rpc_server,
        rpc_port,
        str(files.gguf),
        str(files.mmproj) if files.mmproj else None,
        gpus,
        str(kwargs.get("tensorsplit", settings.llama_param["tensorsplit"])),
        str(kwargs.get("splitmode", settings.llama_param["defaultsplitmode"])),
        str(kwargs.get("ctxsize", settings.AVAILABLE_MODELS[files.model_name]["ctxsize"])),
        listen_host=kwargs.get("listen_host", settings.llama_server_host),
        listen_port=kwargs.get("listen_port", settings.llama_server_port),
    )

    emit(f"Command: {launcher.format_command(cmd)}", log_sink)
    return cmd
