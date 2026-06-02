from dataclasses import dataclass
from pathlib import Path

#_________________________________________
@dataclass
class Model:
    model_name: str
    model_path: Path
    mmproj_path: Path | None
    ctxsize: int
    temperature: float
    top_p: float
    top_k: int
    shard_balance: str
    last_started: int

