
from dataclasses import dataclass, field, fields
from typing import List, Optional, Any, Dict
from logzero import logger
from pathlib import Path
import threading
import json
import sys
import os
import re

_LOCAL_CONFIG = Path(__file__).parent / "config.json"
CONFIG_FILE = Path(
    os.getenv('LLAMA_CONSOLE_CONFIG_FILE') or
    (_LOCAL_CONFIG if _LOCAL_CONFIG.exists() else Path.home() / "llama-console-config.json")
)

#_________________________________________________________________________________________
@dataclass
class Settings:
    ADDRESS_BIND: str = "0.0.0.0"
    PORT_BIND: int = 8088
    PERSIST_FILE: str = "/tmp/llama-console-persist.json"
    UI_TITLE: str = "LLama Console by Alvise Dorigo (alvise72@gmail.com)"
    LLAMA_SERVER_BIN: str = "/usr/local/bin/llama-server"
    LLAMA_SERVER_HOST: str = ""
    LLAMA_SERVER_USER: str = ""
    LLAMA_SERVER_PORT: int = 0
    MODELS_JSON = Path(__file__).parent / "models.json"
    LLAMA_LOG_FILE = "/tmp/llama-server.log"  # shared path for polling the output
    LLAMA_BOOT_LOG = "/tmp/llama-server.boot.log"  # startup stdout/stderr (crash diagnostics)

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
    if not isinstance(data, dict):
        logger.error(f"File {path} must contain a JSON object, found {type(data).__name__}.")
        sys.exit(1)
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
            f"Invalid value for '{key}': expected {getattr(target_type, '__name__', target_type)}, "
            f"got {value!r} ({e})."
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
            f"Unknown keys in config: {sorted(unknown)}. "
            f"Valid ones are: {sorted(type_by_name)}"
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
