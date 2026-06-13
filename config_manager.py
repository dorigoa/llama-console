
from dataclasses import dataclass, field, fields
from typing import List, Optional, Any, Dict
from logzero import logger
from pathlib import Path
import threading
import json
import sys
import os
import re

CONFIG_FILE = Path(os.getenv('LLAMA_CONSOLE_CONFIG_FILE') or str(Path.home() / "llama-console-config.json"))

#_________________________________________________________________________________________
@dataclass
class Settings:
    UI_TITLE: str = "LLama Console by Alvise Dorigo (alvise72@gmail.com)"
    #UI_HOST: str = "127.0.0.1"
    #UI_PORT: int = 8080
    #LLAMA_READY_TIMEOUT_SECONDS: int = 600
    
    #RPC_SERVERS: dict = field(default_factory=lambda: {"192.168.1.191": {"port": 50000, "cachedisk": "/dev/disk4", "cachepath": "/Volumes/Home/llama.cpp", "type": "darwin", "rpcserver": "/usr/local/bin/rpc-server", "remuser": "dorigo_a"}})
    #LOCAL_GPU: str = "MTL0"
    #REMOTE_GPUS: str = "RPC0"
    #GPUS: str = "MTL0"
    #LLAMA_SERVER_HOST: str = "192.168.1.40"
    #LLAMA_SERVER_PORT: int = 8088
    #LLAMA_SERVER_BIND: str = "127.0.0.1"
    #LLAMA_SERVER_BIN: str  = "/usr/local/bin/llama-server"
    
    #OPENBROWSER: bool = True

    

    PERSIST_FILE: str = "/tmp/llama-console-persist.json"
    MODEL_BASE_DIR: str = "/Storage/LLM/gguf_models"
    CONTEXT_SIZE_OPTIONS: List[int] = field(default_factory=lambda: [
        0, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152,
        65536, 98304, 131072, 196608, 262144
    ])
    DEFAULT_SHARD_BALANCE: str = "1"
    DEFAULT_SPLIT_MODE: str = "layer"
    DEFAULT_NGL: str = "999"
    DEFAULT_FIT: str = "off"
    DEFAULT_THREADS: int = 8
    DEFAULT_THREAD_BUNCHES: int = 8
    DEFAULT_PARALLEL: int = 1
    DEFAULT_CONTEXT_SIZE: int = 32768
    DEFAULT_TOP_P: float = 0.9
    DEFAULT_TOP_K: int = 40
    DEFAULT_TEMP: float = 0.8

#_________________________________________________________________________________________
def _load_overrides(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        logger.warning("Config override not found in %s: using defaults.", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {path}: {e}")
        sys.exit(1)
#        return {}
    if not isinstance(data, dict):
        logger.error(f"File {path} must contain a JSON object, found {type(data).__name__}.")
        sys.exit(1)
#        return {}
    logger.info("Config override loaded from %s (%d keys).", path, len(data))
    for k in data:
        logger.debug(f"'{k}': '{data[k]}'")
    return data

#_________________________________________________________________________________________
def _coerce(value: Any, target_type: Any, key: str) -> Any:
    try:
        if target_type is bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if target_type in (int, float, str):
            return target_type(value)
        return value
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Valore non valido per '{key}': atteso {getattr(target_type, '__name__', target_type)}, "
            f"ricevuto {value!r} ({e})."
        ) from e

#_________________________________________________________________________________________
# def _validate_rpc_servers(raw: str) -> None:
#     entries = [e.strip() for e in raw.split(",") if e.strip()]
#     if not entries:
#         raise ValueError(f"RPC_SERVERS='{raw}' is empty")
#     for entry in entries:
#         host, sep, port = entry.rpartition(":")
#         if not (sep and host and port and 1 <= int(port) <= 65535):
#             raise ValueError(f"RPC_SERVERS entry '{entry}' must be 'host:port' with valid port")

#_________________________________________________________________________________________
def _build_settings() -> Settings:
    s = Settings()
    overrides = _load_overrides( CONFIG_FILE )
    if not overrides:
        return s

    type_by_name = {f.name: f.type for f in fields(Settings)}
    unknown = set(overrides) - set(type_by_name)
    if unknown:
        raise ValueError(
            f"Unknown keys in config: {sorted(unknown)}. "
            f"Valid ones are: {sorted(type_by_name)}"
        )

    for k, v in overrides.items():
        setattr(s, k, _coerce(v, type_by_name[k], k))
        # if k == "RPC_SERVERS":
        #     if v:
        #         #logger.debug(f"RPC_SERVERS={v}")
        #         _OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
        #         _IPV4  = rf"(?:{_OCTET}\.){{3}}{_OCTET}"
        #         _PORT  = r"(?:6553[0-5]|655[0-2]\d|65[0-4]\d\d|6[0-4]\d{3}|[1-5]\d{4}|[1-9]\d{0,3})"  # 1-65535
        #         _PAIR  = rf"{_IPV4}:{_PORT}"
        #         RPC_SERVERS_RE = re.compile(rf"^{_PAIR}(?:,{_PAIR})*$")
        #         if not RPC_SERVERS_RE.match():
        #             logger.error(f"RPC_SERVERS={v} is not allowed.")
        #             sys.exit(1)
            
        # if k == "REMOTE_GPUS":
        #     REMOTE_GPUS_RE = re.compile(r"^RPC\d+(?:,RPC\d+)*$")
        #     if not REMOTE_GPUS_RE.match(v):
        #         logger.error(f"REMOTE_GPUS={v} is not allowed.")
        #         sys.exit(1)    
            
        # if k == "DEFAULT_SHARD_BALANCE":
        #     _INT = r"(?:0|[1-9]\d*)"
        #     SHARD_BALANCE_RE = re.compile(rf"^{_INT}(?:,{_INT})*$")
        #     if not SHARD_BALANCE_RE.match(v):
        #         logger.error(f"SHARD_BALANCE_RE={v} is not allowed.")
        #         sys.exit(1)
        #     #num_shards = len(v.split(','))

        # if k == "LOCAL_GPU":
        #     if not v or v=="":
        #         logger.error(f"LOCAL_GPU={v} is not allowed.")
        #         sys.exit(1)

    return s

_settings_lock = threading.Lock()
_settings_instance: Optional[Settings] = None

#_________________________________________________________________________________________
def get_settings() -> Settings:
    global _settings_instance
    with _settings_lock:
        if _settings_instance is None:
            _settings_instance = _build_settings()#Settings()
    return _settings_instance
