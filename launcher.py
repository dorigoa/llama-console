from __future__ import annotations

import shlex
from pathlib import Path
from config_manager import get_settings
from logging_utils import emit, LogSink#, setup_console_logging
import model_utils

settings = get_settings()

def build_llama_command(
    llama_server_bin: str,
    rpc_server: str | None,
    RPC_PORT: int | None,
    gguf_file: str,
    mmproj_file: str | None,
    devices: str,
    tensorsplit: str,
    splitmode: str,
    ctxsize: str,
    temperature: str | float | None ,
    top_p: float,
    top_k: int,
    load_mmproj: bool,
    *,
    listen_host: str | None = None,
    listen_port: int | str | None = None,
) -> list[str]:
    cmd: list[str] = []

    cmd.extend([
        llama_server_bin,
        "--host", str(listen_host or settings.LLAMA_SERVER_HOST),
        "--port", str(listen_port or settings.LLAMA_SERVER_PORT),
        "-m", str(gguf_file),
    ])

    if rpc_server is not None and RPC_PORT is not None:
        cmd.extend(["--rpc", f"{rpc_server}:{RPC_PORT}",
                   "--split-mode", str(splitmode),
                   "--tensor-split", str(tensorsplit),
                   ])

    cmd.extend([
        "--device", str(devices),
        "--jinja",
        "-ngl", str(settings.DEFAULT_NGL),
        "--fit", str(settings.DEFAULT_FIT),
        "-c", str(ctxsize),
        "-t", str(settings.DEFAULT_THREADS),
        "-tb", str(settings.DEFAULT_THREAD_BUNCHES),
        "--parallel", str(settings.DEFAULT_PARALLEL),
        "--top-p", top_p,
        "--top-k", top_k,
    ])

    # cmd.extend([
    #     llama_server_bin,
    #     "--host", str(listen_host or settings.LLAMA_SERVER_HOST),
    #     "--port", str(listen_port or settings.LLAMA_SERVER_PORT),
    #     "-m", str(gguf_file),
    #     "--rpc", f"{rpc_server}:{RPC_PORT}",
    #     "--device", str(devices),
    #     "--jinja",
    #     "--split-mode", str(splitmode),
    #     "--tensor-split", str(tensorsplit),
    #     "-ngl", str(settings.DEFAULT_NGL),
    #     "--fit", str(settings.DEAFULT_FIT),
    #     "-c", str(ctxsize),
    #     "-t", str(settings.DEFAULT_THREADS),
    #     "-tb", str(settings.DEFAULT_THREAD_BUNCHES),
    #     "--parallel", str(settings.DEFAULT_PARALLEL),
    #     "--top-p", top_p,
    #     "--top-k", top_k,
    # ])

    # if rpc_server is not None and RPC_PORT is not None:
    #     cmd.extend(["--rpc", f"{rpc_server}:{RPC_PORT}"])

    if temperature is not None:
        cmd.extend(["--temp", f"{float(temperature):.1f}"])

    if mmproj_file and load_mmproj:
        cmd.extend(["--mmproj", str(mmproj_file)])

    return [str(arg) for arg in cmd]

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
def get_llama_command(model_folder: Path, 
                      log_sink: LogSink = None, 
                      run_local_only: bool = False,
                      **kwargs) -> list[str]:
    """Build the llama-server command without executing it."""
    import devices
    import launcher
    import model_finder
    import rpc

    model_folder = Path(model_folder).expanduser().resolve()
    emit(f"Selected model folder: {model_folder}", log_sink)

    RPC_HOST = kwargs.get("RPC_HOST", settings.RPC_HOST)
    rpc_server = kwargs.get("rpc_server", settings.RPC_HOST)
    RPC_PORT = int(kwargs.get("RPC_PORT", settings.RPC_PORT))

    if run_local_only:
        rpc_server = None
        RPC_PORT = None

    files = model_finder.discover_model_files(model_folder)
    emit(f"Model name   : {files.model_name}", log_sink)
    emit(f"GGUF model   : {files.gguf}", log_sink)
    emit(f"MMProj       : {files.mmproj if files.mmproj else 'none'}", log_sink)

    if not run_local_only:
        rpc.ensure_remote_rpc(RPC_HOST, 5, rpc_server, RPC_PORT, log_sink=log_sink)
    gpus = devices.list_usable_devices(rpc_server, RPC_PORT, log_sink=log_sink)
    
    cmd = launcher.build_llama_command(
        settings.LLAMA_SERVER_PATH,
        rpc_server,
        RPC_PORT,
        str(files.gguf),
        str(files.mmproj) if files.mmproj else None,
        gpus,
        str(kwargs.get("tensorsplit", settings.DEFAULT_SHARD_BALANCE)),
        str(kwargs.get("splitmode", settings.DEFAULT_SPLIT_MODE)),
        str(kwargs.get("ctxsize", model_utils.AVAILABLE_MODELS[files.model_name]["ctxsize"])),
        kwargs.get("temperature", None),
        kwargs.get("top_p", None),
        kwargs.get("top_k", None),
        kwargs.get("load_mmproj", False),
        listen_host=kwargs.get("listen_host", settings.LLAMA_SERVER_HOST),
        listen_port=kwargs.get("listen_port", settings.LLAMA_SERVER_PORT),
    )

    emit(f"Command: {launcher.format_command(cmd)}", log_sink)
    return cmd
