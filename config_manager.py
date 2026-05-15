from dataclasses import dataclass, field
from typing import List, Optional
from object_models import Server
from pathlib import Path



@dataclass
class Settings:
    UI_TITLE: str = "LLama Console by Alvise Dorigo (alvise72@gmail.com)"
    UI_HOST: str = "127.0.0.1"
    UI_PORT: int = 8080
    LLAMA_SERVER_HOST: str = "192.168.1.191"
    LLAMA_SERVER_PORT: int = 8088
    LLAMA_SERVER_BASEURL: str = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
    LLAMA_READY_TIMEOUT_SECONDS: int = 600
    RPC_SERVERS: list[Server] = field(default_factory=lambda: [
        Server(
            hostname="192.168.20.1",
            tcpport=50000,
            platform="Darwin",
            type=ServerType.RPCSERVER,
            cachepath=Path("/Volumes/Home/llama.cpp/"),
            binarypath=Path("/usr/local/bin/rpc-server")
        ),
        Server(
            hostname="192.168.30.2",
            tcpport=50000,
            platform="Windows",
            type=ServerType.RPCSERVER,
            cachepath=Path("/Volumes/Home/llama.cpp/"),
            binarypath=Path(r"C:\llama.cpp\build\bin\Release\rpc-server.exe")
        ),
        # Server(
        #     hostname="192.168.20.2",
        #     tcpport=8088,
        #     platform="Darwin",
        #     type=ServerType.LLAMASERVER,
        #     cachepath=None,
        #     binarypath="/usr/local/bin/llama-server"
        # ),
    ])
    
    LLAMA_SERVER = Server(
            hostname="192.168.20.2",
            tcpport=8088,
            platform="Darwin",
            type=ServerType.LLAMASERVER,
            cachepath=None,
            binarypath="/usr/local/bin/llama-server"
        )
    
    OPENBROWSER: bool = True

    PERSIST_FILE: str = "persist.json"
    MODEL_BASE_DIR: str = "/Volumes/Home/gguf_models"
    CONTEXT_SIZE_OPTIONS: List[int] = field(default_factory=lambda: [
        0, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152,
        65536, 98304, 131072, 196608, 262144
    ])
    DEFAULT_SHARD_BALANCE: str = "1,1"
    DEFAULT_SPLIT_MODE: str = "layer"
    DEFAULT_NGL: str = "all"
    DEFAULT_FIT: str = "off"
    DEFAULT_THREADS: int = 8
    DEFAULT_THREAD_BUNCHES: int = 8
    DEFAULT_PARALLEL: int = 1
    DEFAULT_CONTEXT_SIZE: int = 32768
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_TOP_K: int = 40
    DEFAULT_TEMP: float = 0.8

_settings_instance: Optional[Settings] = None

def get_settings() -> Settings:
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

def reload_settings() -> Settings:
    global _settings_instance
    _settings_instance = Settings()
    return _settings_instance
