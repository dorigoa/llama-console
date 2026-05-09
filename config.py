from __future__ import annotations
from dataclasses import dataclass, field
import model_utils

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

    DEFAULT_SHARD_BALANCE: str = "10,10"
    DEFAULT_CONTEXT_SIZE: int = 32768   
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_TOP_K: int   = 40
    DEFAULT_TEMP: float = 0.8

    LLAMA_PARAM: dict = field(default_factory=lambda: {
        "fit": "on",
        "threads": "4",
        "threadsbunch": "4",
        "parallel": "1",
        "ngl": "auto",
        "splitmodes": ["none", "layer", "row", "tensor"],
        "defaultsplitmode": "layer",
        "tensorsplit": "6,12",
        "ctxsize": "0",
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 40,
    })

    AVAILABLE_MODELS: dict[str, model_utils.ModelConfig] = field(
        default_factory=lambda: model_utils.discover_available_models(Settings.MODEL_BASE_DIR)
    )
    
settings = Settings()
