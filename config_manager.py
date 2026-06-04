
from dataclasses import dataclass, field, fields
from typing import List, Optional, Any, Dict
from typing import List, Optional
from logzero import logger
from pathlib import Path
import threading
import json
import os

CONFIG_FILE = os.getenv('LLAMA_CONSOLE_CONFIG_FILE') or str(Path.home() / "llama-console-config.json")
#CONFIG_ENV_VAR = "LLAMA_CONSOLE_CONFIG"

#_________________________________________________________________________________________
@dataclass
class Settings:
    UI_TITLE: str = "LLama Console by Alvise Dorigo (alvise72@gmail.com)"
    UI_HOST: str = "127.0.0.1"
    UI_PORT: int = 8080
    LLAMA_READY_TIMEOUT_SECONDS: int = 600
    
    RPC_SERVERS: str = "192.168.20.1:50000"
    LOCAL_GPU: str = "MTL0"
    REMOTE_GPUS: str = "RPC0"
    LLAMA_SERVER_HOST: str = "192.168.20.2"
    LLAMA_SERVER_PORT: int = 8088
    LLAMA_SERVER_BIND: str = "127.0.0.1"
    LLAMA_SERVER_BIN: str  = "/usr/local/bin/llama-server"
    
    OPENBROWSER: bool = True

    PERSIST_FILE: str = "/tmp/llama-console-persist.json"
    MODEL_BASE_DIR: str = "/Storage/LLM/gguf_models"
    CONTEXT_SIZE_OPTIONS: List[int] = field(default_factory=lambda: [
        0, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152,
        65536, 98304, 131072, 196608, 262144
    ])
    #DEFAULT_SHARD_BALANCE: str = "1,1,1,1"
    DEFAULT_SHARD_BALANCE: str = "30,22"
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

#_________________________________________________________________________________________
# def _config_path() -> Path:
#     env = os.environ.get(CONFIG_ENV_VAR)
#     if env:
#         return Path(env).expanduser()
#     return Path.home() / DEFAULT_CONFIG_FILENAME

#_________________________________________________________________________________________
def _load_overrides(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        logger.warn("Config override not found in %s: using defaults.", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        #raise ValueError(f"Invalid JSON in {path}: {e}") from e
        logger.error(f"Invalid JSON in {path}: {e}")
        return {}
    if not isinstance(data, dict):
        #raise ValueError(
        #    
        #)
        logger.error(f"File {path} must contain a JSON object, found {type(data).__name__}.")
        return {}
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
def _build_settings() -> Settings:
    s = Settings()
    overrides = _load_overrides( CONFIG_FILE )
    if not overrides:
        return s

    type_by_name = {f.name: f.type for f in fields(Settings)}
    unknown = set(overrides) - set(type_by_name)
    if unknown:
        raise ValueError(
            f"Chiavi sconosciute nel config: {sorted(unknown)}. "
            f"Valide: {sorted(type_by_name)}"
        )

    for k, v in overrides.items():
        setattr(s, k, _coerce(v, type_by_name[k], k))
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
