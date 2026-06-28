import json
from json import JSONDecodeError
from config_manager import get_settings
from pathlib import Path
from typing import Any, Dict

from typing import Optional

JsonDict = Dict[str, Any]

class JsonParams:
    def __init__(self, filename: str | Path):
        self.path = Path(filename)

    #___________________________________________________________________________________
    def load_params(self) -> Dict[str, JsonDict]:
        if not self.path.exists():
            return {}

        with self.path.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except JSONDecodeError as je:
                raise Exception(f"Couldn't decode the JSON: {je}")
            except Exception as e:
                raise Exception(f"Couldn't decode the JSON (generic exception): {e}")

        if not isinstance(data, dict):
            raise ValueError(f"JSON file {self.path} must contain a top-level object")

        return data

    #___________________________________________________________________________________
    def save_param(self, key: str, value: JsonDict) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("key must be a non-empty string")

        if not isinstance(value, dict):
            raise TypeError("value must be a dict")

        # Validate JSON serializability before touching the file.
        try:
            json.dumps(value, ensure_ascii=False)
        except TypeError as exc:
            raise TypeError(f"value for key {key!r} is not JSON-serializable: {exc}") from exc

        data = self.load_params()
        data[key] = value

        self.path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

        tmp_path.replace(self.path)

_jsonparam : Optional[JsonParams] = None

def get_params_handler( ) -> JsonParams:
    global _jsonparam
    if _jsonparam is None:
        _jsonparam = JsonParams( get_settings().PERSIST_FILE )
    return _jsonparam
