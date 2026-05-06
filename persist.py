import json
from pathlib import Path
from typing import Dict


class JsonParams:
    def __init__(self, filename: str | Path):
        self.path = Path(filename)

    def load_params(self) -> Dict[str, str]:
        """
        Load all parameters from the JSON file.

        Returns an empty dict if the file does not exist yet.
        """
        if not self.path.exists():
            return {}

        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"{self.path} does not contain a JSON object")

        return data

    def save_param(self, key: str, value: str) -> None:
        """
        Save or update a single key/value pair.
        Creates the JSON file if it does not exist.
        """
        data = self.load_params()
        data[key] = value

        self.path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

        tmp_path.replace(self.path)