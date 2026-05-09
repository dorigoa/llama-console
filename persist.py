import json
from pathlib import Path
from typing import Any, Dict

JsonDict = Dict[str, Any]

class JsonParams:
    def __init__(self, filename: str | Path):
        self.path = Path(filename)

    #___________________________________________________________________________________
    def load_params(self) -> Dict[str, JsonDict]:
        """
        Load all parameters from the JSON file.

        The file must contain a JSON object whose values are JSON objects.
        Returns an empty dict if the file does not exist yet.
        """
        if not self.path.exists():
            return {}

        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"{self.path} does not contain a JSON object")

        return data

    #___________________________________________________________________________________
    def save_param(self, key: str, value: JsonDict) -> None:
        """
        Save or update a single key/dict pair.
        Creates the JSON file if it does not exist.

        The value must be a JSON-serializable dict.
        """
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
