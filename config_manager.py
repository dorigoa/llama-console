# config_manager.py
from dataclasses import dataclass, field
from typing import List, Optional
#import os

@dataclass
class RpcServer:
    hostname: str
    tcpport: int
    platform: str

@dataclass
class Settings:
    UI_TITLE: str = "LLama Launcher by Alvise Dorigo (alvise72@gmail.com)"
    UI_HOST: str = "127.0.0.1"
    UI_PORT: int = 8080
    LLAMA_SERVER_HOST: str = "127.0.0.1"
    LLAMA_SERVER_PORT: int = 8088
    LLAMA_SERVER_BASEURL: str = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}"
    #RPC_HOST: str = "192.168.20.2"
    #RPC_PORT: int = 50000
    #RPC_SERVERS: list[RpcServer] = "192.168.20.2:50000"
    RPC_SERVERS: list[RpcServer] = field(default_factory=lambda: [
        RpcServer(
            hostname="192.168.20.2",
            tcpport=50000,
            platform="Darwin",
        ),
        RpcServer(
            hostname="192.168.1.200",
            tcpport=50000,
            platform="Windows",
        ),
    ])
    RPC_CACHE_PATH: str = "/Volumes/Home/llama.cpp/"
    OPENBROWSER: bool = True

    RPC_SERVER_PATH: dict = field(default_factory=lambda:{ 
        "Linux": '/usr/local/bin/rpc-server',
        "Windows": r"C:\llama.cpp\build\bin\Release\rpc-server.exe"
        #"Windows": r"C:\Users\alvis\AppData\Local\Programs\llama.cpp\bin\rpc-server.exe"
        })

    LLAMA_SERVER_PATH: str = "/usr/local/bin/llama-server"
    PERSIST_FILE: str = "persist.json"
    MODEL_BASE_DIR: str = "/Volumes/Home/gguf_models"
    CONTEXT_SIZE_OPTIONS: List[int] = field(default_factory=lambda: [
        0, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152,
        65536, 98304, 131072, 196608, 262144
    ])
    DEFAULT_SHARD_BALANCE: str = "1,1"
    DEFAULT_SPLIT_MODE: str = "layer"
    DEFAULT_NGL: str = "auto"
    DEFAULT_FIT: str = "on"
    DEFAULT_THREADS: int = 4
    DEFAULT_THREAD_BUNCHES: int = 4
    DEFAULT_PARALLEL: int = 1
    DEFAULT_CONTEXT_SIZE: int = 32768
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_TOP_K: int = 40
    DEFAULT_TEMP: float = 0.8

# Singleton instance
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
