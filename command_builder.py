from config_manager import get_settings
from model import Model
import json

settings = get_settings()

#___________________________________________________________________________________
def build_command(binary: str, model: Model, devices: str = "", ctx: int | None = None) -> list[str]:
    cmd = [binary, "-m", str(model.model_path), "-c", str(ctx if ctx is not None else model.ctxsize)]

    if model.fitt:
        cmd += ["-fitt", model.fitt]

    if model.rpcservers:
        rpc_list = ",".join(f"{s.IP}:{s.PORT}" for s in model.rpcservers)
        cmd += ["--rpc", rpc_list]

    if devices:
        cmd += ["--device", devices]

    if model.mmproj_path and str(model.mmproj_path).lower() not in ("none", "null", ""):
        cmd += ["--mmproj", str(model.mmproj_path)]

    data = {"reasoning_effort": model.reasoning}

    cmd += ["--host", settings.ADDRESS_BIND]
    cmd += ["--port", str(settings.PORT_BIND)]
    cmd += ["--split-mode", "layer"]
    cmd += ["--metrics"]
    cmd += ["--jinja"]
    cmd += ["-fa", "on"]
    cmd += ["-fit", "on"] # Using "on" makes the rpc/Vulkan on PC with 2 NVidia cards crash
    cmd += ["-fitc", "8192"]
    cmd += ["-np", "1"] # Required by MTP processing
    cmd += ["--no-warmup"]
    cmd += ["--temp", str(model.temperature)]
    cmd += ["--top-p", str(model.top_p)]
    cmd += ["--top-k", str(model.top_k)]
    cmd += ["--chat-template-kwargs", json.dumps(data)]
    cmd += ["--seed", "123456789"]
    if model.kvquant:
        cmd += ["-ctk", model.kvquant]
        cmd += ["-ctv", model.kvquant]
    cmd += ["--alias", str(model.alias)]
    if model.min_p >= 0:
        cmd += ["--min-p", str(model.min_p)]

    if model.ub:
        cmd += ["-ub", str(model.ub)]
    if model.b:
        cmd += ["-b", str(model.b)]
    if model.mtp:
        cmd += ["--spec-type", "draft-mtp"]
    # Log file for detached execution (UI polls it)
    cmd += ["--log-file", settings.LLAMA_LOG_FILE]

    return cmd