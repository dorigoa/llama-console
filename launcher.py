from __future__ import annotations

import shlex
from pathlib import Path
from config_manager import get_settings
from logging_utils import emit, LogSink, setup_console_logging

settings = get_settings()

logger = setup_console_logging()

def build_llama_command(
    #llama_server_bin: str,
    rpc_server: str | None,
    gguf_file: str,
    mmproj_file: str | None,
    devices: str,
    tensorsplit: str,
    #splitmode: str,
    ctxsize: int,
    temperature: float,
    top_p: float,
    top_k: int,
    load_mmproj: bool,
) -> list[str]:
    cmd: list[str] = []

    cmd.extend([
        settings.LLAMA_SERVER_PATH,
        "--host", settings.LLAMA_SERVER_HOST,
        "--port", settings.LLAMA_SERVER_PORT,
        "-m", str(gguf_file),
    ])

    if rpc_server is not None:
        cmd.extend(["--rpc", f"{rpc_server}",
                   "--split-mode", str(settings.DEFAULT_SPLIT_MODE),
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

    if temperature is not None:
        cmd.extend(["--temp", f"{float(temperature):.1f}"])

    if mmproj_file and load_mmproj:
        cmd.extend(["--mmproj", str(mmproj_file)])

    return [str(arg) for arg in cmd]

#_____________________________________________________________________________
#def format_command(cmd: list[str]) -> str:
#    return " ".join(shlex.quote(str(x)) for x in cmd)

#_____________________________________________________________________________
# def validate_ssl_files(key_file: Path, cert_file: Path) -> None:
#     missing = [str(p) for p in (key_file, cert_file) if not p.exists()]
#     if missing:
#         joined = "\n".join(f"  {p}" for p in missing)
#         raise FileNotFoundError(f"Missing SSL file(s):\n{joined}")

#_____________________________________________________________________________
def get_llama_command(model_folder: Path, 
                      log_sink: LogSink = None, 
                      run_local_only: bool = False,
                      tensorsplit: str = "1,1",
                      ctxsize: int = 32768,
                      temperature: float = 0.5,
                      top_p: float = 0.8,
                      top_k: int = 40,
                      load_mmproj: bool = False,
                      ) -> list[str]:
    
    import devices
    import launcher
    import model_finder
    import rpc

    model_folder = Path(model_folder).expanduser().resolve()
    emit(f"Selected model folder: {model_folder}", log_sink)

    if not run_local_only:
        for rpc_server in settings.RPC_SERVERS:
            logger.info(f"DEBUG - rpc.ensure {rpc_server.hostname}:{rpc_server.tcpport}/{rpc_server.platform}")
            rpc.ensure_remote_rpc(5, rpc_server.hostname, rpc_server.tcpport, rpc_server.platform, log_sink=log_sink)

    files = model_finder.discover_model_files(model_folder)
    emit(f"Model name   : {files.model_name}", log_sink)
    emit(f"GGUF model   : {files.gguf}", log_sink)
    emit(f"MMProj       : {files.mmproj if files.mmproj else 'none'}", log_sink)

    all_endpoints = []
    if not run_local_only:    
        for rpc_server in settings.RPC_SERVERS:
            all_endpoints.append(f"{rpc_server.hostname}:{rpc_server.tcpport}")
        gpus = devices.list_usable_devices(",".join(all_endpoints), log_sink=log_sink)
    else:
        gpus = devices.list_usable_devices(None, log_sink=log_sink)
    
    cmd = launcher.build_llama_command(
        ",".join(all_endpoints),
        str(files.gguf),
        str(files.mmproj) if files.mmproj else None,
        gpus,
        tensorsplit,
        ctxsize,
        temperature,
        top_p,
        top_k,
        load_mmproj,
    )

    return cmd
