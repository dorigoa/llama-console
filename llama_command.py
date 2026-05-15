from __future__ import annotations
import shlex
from pathlib import Path
from config_manager import get_settings
from logging_utils import emit, LogSink, setup_console_logging

import devices
from model_finder import ModelFiles
import rpc

settings = get_settings()

logger = setup_console_logging()

#_____________________________________________________________________________
def get_llama_command(
        #model_folder: Path, 
        files: ModelFiles = None,              
        log_sink: LogSink = None, 
        run_local_only: bool = False,
        tensorsplit: str = "1,1",
        ctxsize: int = 32768,
        temperature: float = 0.5,
        top_p: float = 0.8,
        top_k: int = 40,
        load_mmproj: bool = False,
        ) -> list[str]:
    
    

    #model_folder = Path(model_folder).expanduser().resolve()
    #emit(f"-> Selected model folder: {model_folder}", log_sink)
    #files = model_finder.discover_model_files(model_folder)
    #emit(f"-> Model name   : {files.model_name}", log_sink)
    #emit(f"-> GGUF model   : {files.gguf}", log_sink)
    #emit(f"-> MMProj       : {files.mmproj if files.mmproj else 'none'}", log_sink)

    if not run_local_only:
        for rpc_server in settings.RPC_SERVERS:
            emit(f"-> Checking remote rpc: {rpc_server.hostname}:{rpc_server.tcpport}", log_sink)
            logger.info(f"DEBUG - {rpc_server}")
            rpc.ensure_remote_rpc(5, rpc_server, log_sink=log_sink)

    all_endpoints = []
    all_endpoints.extend(f"{s.hostname}:{s.tcpport}" for s in settings.RPC_SERVERS)

    if not run_local_only:        
        gpus = devices.list_remote_usable_devices(settings.RPC_SERVERS, log_sink=log_sink)
    else:
        gpus = devices.list_local_usable_devices(settings.LLAMA_SERVER, log_sink=log_sink)

    cmd: list[str] = []

    if not settings.LLAMA_SERVER:
        raise Exception("Passed None settings.LLAMA_SERVER to get_llama_command")

    cmd.extend([
        settings.LLAMA_SERVER.binarypath,
        "--host", settings.LLAMA_SERVER.hostname,
        "--port", settings.LLAMA_SERVER.tcpport,
        "-m", str(files.gguf),
    ])

    if not run_local_only:
        cmd.extend(["--rpc", f"{",".join(all_endpoints)}",
                   "--split-mode", str(settings.DEFAULT_SPLIT_MODE),
                   "--tensor-split", str(tensorsplit),
                   ])

    cmd.extend([
        "--device", gpus,
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

    if files.mmproj and load_mmproj:
        cmd.extend(["--mmproj", str(files.mmproj)])

    return [str(arg) for arg in cmd]