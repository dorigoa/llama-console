from __future__ import annotations
from config_manager import get_settings
from logging_utils import emit, LogSink, setup_console_logging
from object_models import Model

settings = get_settings()

logger = setup_console_logging()

#_____________________________________________________________________________
def get_llama_command(
        M: Model,
        log_sink: LogSink = None, 
        run_local_only: bool = False,
        load_mmproj: bool = False,
        #gpus: str = None,
        ) -> list[str]:

    emit(f"-> Called with Model={M}", log_sink)
    logger.info(f"DEBUG - Called with Model={M}")

    cmd: list[str] = []

    cmd.extend([
        settings.LLAMA_SERVER_BIN,
        "--host", settings.LLAMA_SERVER_BIND,
        "--port", settings.LLAMA_SERVER_PORT,
        "-m", str(M.model_path),
    ])

    if not run_local_only:
        cmd.extend(["--rpc", f"{settings.RPC_SERVERS}",
                   "--split-mode", settings.DEFAULT_SPLIT_MODE,
                   "--tensor-split", M.shard_balance,
                   ])

    if not run_local_only:
        gpus = f"{settings.LOCAL_GPU},{settings.REMOTE_GPUS}"
    else:
        gpus = f"{settings.LOCAL_GPU}"

    cmd.extend([
        "--device", gpus,
        "--jinja",
        "-ngl", str(settings.DEFAULT_NGL),
        "--fit", str(settings.DEFAULT_FIT),
        "-c", str(M.ctxsize),
        "-t", str(settings.DEFAULT_THREADS),
        "-tb", str(settings.DEFAULT_THREAD_BUNCHES),
        "--parallel", str(settings.DEFAULT_PARALLEL),
        "--top-p", f"{float(M.top_p):.2f}", #str(M.top_p),
        "--top-k", str(M.top_k),
        "--seed", "123456789",
    ])

    #if temperature is not None:
    cmd.extend(["--temp", f"{float(M.temperature):.1f}"])

    if M.mmproj_path and load_mmproj:
        cmd.extend(["--mmproj", str(M.mmproj_path)])

    return [str(arg) for arg in cmd]
