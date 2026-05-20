from dataclasses import dataclass, field
from pathlib import Path
from object_models import Model
import persist
from config_manager import get_settings

settings = get_settings()
    
_AVAILABLE_MODELS: list[Model] = []

#_____________________________________________________________________________
def get_model_by_name( name: str ) -> Model:
    for model in _AVAILABLE_MODELS:
        if name == model.model_name:
            return model
    return None
    
#_____________________________________________________________________________
def get_last_started_model( ) -> Model:
    last_start_time = 0
    candidate: Model = None
    for m in _AVAILABLE_MODELS:
        if m.last_started > last_start_time:
            last_start_time = m.last_started
            candidate = m
    if candidate:
        return candidate
    
    if len(_AVAILABLE_MODELS):
        return _AVAILABLE_MODELS[0]
    else:
        return None
    
#_____________________________________________________________________________
def get_available_model_names( refresh: bool = False ) -> list[str]:
    return [m.model_name for m in get_available_models(refresh)]

#_____________________________________________________________________________
def get_available_models( refresh: bool = False ) -> list[Model]:
    global _AVAILABLE_MODELS
    if refresh:
        _AVAILABLE_MODELS = _discover_available_models( )
    return _AVAILABLE_MODELS

#_____________________________________________________________________________
def _discover_available_models( ) -> list[Model]:

    root = Path(settings.MODEL_BASE_DIR)
    if not root.is_dir():
        return []
    model_list = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.name.lower()
    )

    models = []

    # READ models extra param from persist
    data = persist.get_params_handler().load_params()


    for m in model_list:
        pmmproj = _find_mmproj_file( m )
    
        if m.name in data:
            M = Model(
                model_name    = m.name,
                model_path    = m / f"{m.name}.gguf",
                mmproj_path   = pmmproj,
                ctxsize       = data[m.name]['context_size'],
                temperature   = data[m.name]['temperature'],
                top_p         = data[m.name]['top_p'],
                top_k         = data[m.name]['top_k'],
                shard_balance = data[m.name]['shard_balance'],
                last_started  = data[m.name]['last_started']
            )
        else:
            M = Model(
                model_name      = m.name,
                model_path      = m / f"{m.name}.gguf",
                mmproj_path     = pmmproj,
                ctxsize         = settings.DEFAULT_CONTEXT_SIZE,
                temperature     = settings.DEFAULT_TEMP,
                top_p           = settings.DEFAULT_TOP_P,
                top_k           = settings.DEFAULT_TOP_K,
                shard_balance   = "1,1",
                last_started    = False
            )
        models.append(M)

    return models

#_____________________________________________________________________________
def _find_mmproj_file(directory: str | Path) -> Path | None:
    base = Path(directory)
    for p in base.iterdir():
        if p.is_file() and p.name.startswith("mmproj"):
            return p
    return None

_AVAILABLE_MODELS: list[Model] = _discover_available_models( )