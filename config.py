from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from typing import TypedDict

#_____________________________________________________________________________
class ModelConfig(TypedDict):
    path: str
    ctxsize: int


#_____________________________________________________________________________
def discover_available_models(models_dir: str) -> dict[str, ModelConfig]:
    """Return GGUF models discovered under models_dir.

    Each immediate subdirectory is treated as one model. The model name shown in
    the GUI is the last path component of the directory. The main model path is
    selected as the first non-mmproj *.gguf file found in that directory.
    """
    root = Path(models_dir).expanduser()
    models: dict[str, ModelConfig] = {}

    if not root.is_dir():
        return models

    for model_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
        gguf_files = sorted(model_dir.glob("*.gguf"), key=lambda p: p.name.lower())
        main_candidates = [p for p in gguf_files if "mmproj" not in p.name.lower()]
        if not main_candidates:
            continue

        models[model_dir.name] = {
            "path": str(main_candidates[0]),
            "ctxsize": 0,
        }

    return models

#_____________________________________________________________________________
@dataclass
class Settings:
    UI_TITLE: str = "LLama Launcher by Alvise Dorigo (alvise72@gmail.com)"

    UI_HOST: str = "127.0.0.1"
    UI_PORT: int = 8080
    
    LLAMA_SERVER_HOST: str = "127.0.0.1"
    LLAMA_SERVER_PORT: int = 8088

    RPC_HOST: str = "192.168.20.2"
    RPC_PORT: int = 50000
    RPC_CACHE_PATH: str = "/Volumes/Home/llama.cpp/"
    
    OPENBROWSER: bool = True

    RPC_SERVER_PATH: str = "/usr/local/bin/rpc-server"
    LLAMA_SERVER_PATH: str = "/usr/local/bin/llama-server"

    PERSIST_FILE = "persist.json"

    MODEL_BASE_DIR: str = "/Volumes/Home/gguf_models"
    
    CONTEXT_SIZE_OPTIONS: list[int] = field(default_factory=lambda: [
        0,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
        24576,
        32768,
        49152,
        65536,
        98304,
        131072,
        196608,
        262144,
    ])
    DEFAULT_CONTEXT_SIZE: int = 32768
    DEFAULT_TEMP: float = 0.8

    LLAMA_PARAM: dict = field(default_factory=lambda: {
        "fit": "on",
        "threads": "6",
        "threadsbunch": "8",
        "parallel": "1",
        "ngl": "auto",
        "splitmodes": ["none", "layer", "row", "tensor"],
        "defaultsplitmode": "layer",
        "tensorsplit": "6,12",
        "ctxsize": "0",
        "temperature": 0.3
        #"sslkeyfile": "/Volumes/Home/dorigo_a/llama-server.key",
        #"sslcertfile": "/Volumes/Home/dorigo_a/llama-server.crt",
    })

    AVAILABLE_MODELS: dict[str, ModelConfig] = field(
        default_factory=lambda: discover_available_models(Settings.MODEL_BASE_DIR)
    )
    #DEFAULT_MODEL: str = ""

settings = Settings()
