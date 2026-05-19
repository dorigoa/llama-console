from __future__ import annotations
from config_manager import get_settings
from logging_utils import emit, LogSink, setup_console_logging
from object_models import Model
#import devices
#from model_finder import ModelFiles
import rpc

settings = get_settings()

logger = setup_console_logging()

#_____________________________________________________________________________
def get_llama_command(
        M: Model,
        #files: ModelFiles = None,              
        log_sink: LogSink = None, 
        run_local_only: bool = False,
        # tensorsplit: str = "1,1",
        # ctxsize: int = 32768,
        # temperature: float = 0.5,
        # top_p: float = 0.8,
        # top_k: int = 40,
        load_mmproj: bool = False,
        gpus: str = None,
        ) -> list[str]:
    
    #if not files:
    #    raise ValueError("files must not be None")

    if not run_local_only:
        for rpc_server in settings.RPC_SERVERS:
            emit(f"-> Checking remote rpc: {rpc_server.hostname}:{rpc_server.tcpport}", log_sink)
            logger.info(f"DEBUG - {rpc_server}")
            rpc.ensure_remote_rpc(rpc_server, log_sink=log_sink)

    all_endpoints = []
    all_endpoints.extend(f"{s.hostname}:{s.tcpport}" for s in settings.RPC_SERVERS)

    # if not run_local_only:        
    #     gpus = devices.list_remote_usable_devices(settings.RPC_SERVERS, log_sink=log_sink)
    # else:
    #     gpus = devices.list_local_usable_devices(settings.LLAMA_SERVER, log_sink=log_sink)

    cmd: list[str] = []

    if not settings.LLAMA_SERVER:
        raise Exception("Passed None settings.LLAMA_SERVER to get_llama_command")

    cmd.extend([
        settings.LLAMA_SERVER.binarypath,
        "--host", settings.LLAMA_SERVER.bindaddress,
        "--port", settings.LLAMA_SERVER.tcpport,
        "-m", str(M.model_path),
    ])

    if not run_local_only:
        cmd.extend(["--rpc", f"{",".join(all_endpoints)}",
                   "--split-mode", str(settings.DEFAULT_SPLIT_MODE),
                   "--tensor-split", str(M.shard_balance),
                   ])

    cmd.extend([
        "--device", gpus,
        "--jinja",
        "-ngl", str(settings.DEFAULT_NGL),
        "--fit", str(settings.DEFAULT_FIT),
        "-c", str(M.ctxsize),
        "-t", str(settings.DEFAULT_THREADS),
        "-tb", str(settings.DEFAULT_THREAD_BUNCHES),
        "--parallel", str(settings.DEFAULT_PARALLEL),
        "--top-p", M.top_p,
        "--top-k", M.top_k,
    ])

    #if temperature is not None:
    cmd.extend(["--temp", f"{float(M.temperature):.1f}"])

    if M.mmproj and load_mmproj:
        cmd.extend(["--mmproj", str(M.mmproj)])

    return [str(arg) for arg in cmd]