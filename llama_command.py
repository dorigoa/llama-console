from __future__ import annotations
from config_manager import get_settings
from object_models import Model
from logzero import logger
from typing import Optional

settings = get_settings()

#_____________________________________________________________________________
def get_llama_command(
        M: Model,
        run_local_only: bool = False,
        load_mmproj: bool = False,
        devices: Optional[list[str]] = None,
        ) -> list[str]:

    logger.debug(f"Called with Model={M}")

    cmd: list[str] = []

    cmd.extend([
        settings.LLAMA_SERVER_BIN,
        "--host", settings.LLAMA_SERVER_BIND,
        "--port", str(settings.LLAMA_SERVER_PORT),
        "-m", str(M.model_path),
    ])

    rpcs=[]
    for k in settings.RPC_SERVERS.keys():
        rpcs.append(f"{k}:{settings.RPC_SERVERS[k]['port']}")

    if not run_local_only:
        cmd.extend(["--rpc", f"{','.join(rpcs)}",
                   "--split-mode", settings.DEFAULT_SPLIT_MODE,
                   "--tensor-split", M.shard_balance,
                   ])

    if devices:
        if run_local_only:
            #gpus = ",".join(d for d in devices if not d.upper().startswith("RPC"))
            gpus = ""
        else:
            gpus = ",".join(devices)
    else: #not run_local_only:
        gpus = f"{settings.GPUS}"
        
    # else:
    #     gpus = None

    if not run_local_only:
        cmd.extend(["--device", gpus])

    cmd.extend([
        #"--device", gpus,
        "--jinja",
        "--metrics",
        "-fa", "auto",
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

    cmd.extend(["--temp", f"{float(M.temperature):.1f}"])

    if M.mmproj_path and load_mmproj:
        cmd.extend(["--mmproj", str(M.mmproj_path)])

    return [str(arg) for arg in cmd]
