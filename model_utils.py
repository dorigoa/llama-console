from typing import Any, Optional
from dataclasses import dataclass, field
from typing import TypedDict
from pathlib import Path

from config import settings

#_____________________________________________________________________________
class ModelConfig(TypedDict):
    path: str
    ctxsize: int
    temperature: float
    top_p: float
    top_k: int
    shard_balance: str

#_____________________________________________________________________________
def _get_configured_value(configured: Any, key: str, cast_type=float, default=0):
    if isinstance(configured, dict):
        value = configured.get(key, default)
        try:
            return cast_type(value)
        except (TypeError, ValueError):
            return default
    return default

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
            "shard_balance": "1,1",
            "top_p": 0.8,
            "top_k": 50
        }

    return models

#_____________________________________________________________________________
def default_context_size_for_model(model_name: Optional[str]) -> int:
    """Return model-specific ctxsize when configured, otherwise global default."""
    if model_name and model_name in AVAILABLE_MODELS:
        model_ctx = _get_configured_value( AVAILABLE_MODELS[model_name], "ctxsize", int, "8192")
        if model_ctx > 0:
            return model_ctx
    return settings.DEFAULT_CONTEXT_SIZE#int(getattr(settings, "DEFAULT_CONTEXT_SIZE", 32768))

#_____________________________________________________________________________
def default_temp_for_model(model_name: Optional[str]) -> float:
    """Return model-specific temperature when configured, otherwise global default."""
    if model_name and model_name in AVAILABLE_MODELS:
        model_temp = _get_configured_value(AVAILABLE_MODELS[model_name], "temperature", float, 0.3)
        if model_temp > 0:
            return model_temp
    return settings.DEFAULT_TEMP

#_____________________________________________________________________________
def default_top_p_for_model(model_name: Optional[str]) -> float:
    """Return model-specific top_p when configured, otherwise global default."""
    if model_name and model_name in AVAILABLE_MODELS:
        model_top_p = _get_configured_value(AVAILABLE_MODELS[model_name], "top_p", float, 0.5)
        if model_top_p > 0:
            return model_top_p
    return settings.DEFAULT_TOP_P

#_____________________________________________________________________________
def default_top_k_for_model(model_name: Optional[str]) -> float:
    """Return model-specific top_k when configured, otherwise global default."""
    if model_name and model_name in AVAILABLE_MODELS:
        model_top_k = _get_configured_value(AVAILABLE_MODELS[model_name], "top_k", int, 30)
        if model_top_k > 0:
            return model_top_k
    return settings.DEFAULT_TOP_K

#_____________________________________________________________________________
def default_shard_balance_for_model(model_name: Optional[str]) -> float:
    """Return shard_balance (which depends on the cluster) when configured, otherwise global default."""
    if model_name and model_name in AVAILABLE_MODELS:
        model_shard_balance = _get_configured_value(AVAILABLE_MODELS[model_name], "shard_balance", str, "10,10")
        if model_shard_balance:
            return model_shard_balance
    return settings.DEFAULT_SHARD_BALANCE

#_____________________________________________________________________________
def configured_model_path(configured: Any) -> str:
    """Extract the main GGUF path/folder from one AVAILABLE_MODELS value.

    Supports both legacy string values and dict values, for example:
        "Model": "/path/to/model.gguf"
        "Model": {"model": "/path/to/model.gguf", "mmproj": "/path/to/mmproj.gguf"}
    """
    if isinstance(configured, (str, Path)):
        return str(configured)

    if isinstance(configured, dict):
        # Prefer explicit main-model keys. Do not pick mmproj unless it is the only usable path.
        preferred_keys = (
            "model",
            "model_path",
            "path",
            "gguf",
            "file",
            "filename",
            "model_file",
            "folder",
            "directory",
            "dir",
        )
        for key in preferred_keys:
            value = configured.get(key)
            if isinstance(value, (str, Path)) and str(value).strip():
                return str(value)

        # Fallback: first string/path value that does not look like a multimodal projector.
        for value in configured.values():
            if isinstance(value, (str, Path)) and str(value).strip():
                candidate = str(value)
                if "mmproj" not in Path(candidate).name.lower():
                    return candidate

        # Last resort: any string/path value, including mmproj.
        for value in configured.values():
            if isinstance(value, (str, Path)) and str(value).strip():
                return str(value)

    raise TypeError(f"Unsupported AVAILABLE_MODELS entry: {configured!r}")

#_____________________________________________________________________________
def path_to_model_folder(path_string: str | Path) -> Path:
    """Accept either a GGUF file path or a model directory path."""
    p = Path(path_string).expanduser()

    # Path.exists() is intentionally not required here because network volumes may be late-mounted.
    # If the configured string ends with .gguf, treat it as a model file and use its parent.
    if p.suffix.lower() == ".gguf":
        return p.parent.resolve()

    return p.resolve()

#_____________________________________________________________________________
def extract_model_from_props(payload: dict[str, Any]) -> Optional[str]:
    for key in ("model_path", "model", "model_name", "model_alias"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

#_____________________________________________________________________________
def extract_model_from_openai_models(payload: dict[str, Any]) -> Optional[str]:
    """Extract model id/name from /v1/models compatible response."""
    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            model_id = first.get("id") or first.get("name")
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()
    return None



AVAILABLE_MODELS: dict[str, ModelConfig] = discover_available_models(settings.MODEL_BASE_DIR)