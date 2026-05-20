
#from enum import Enum, unique
from dataclasses import dataclass
from pathlib import Path

#_________________________________________
# @unique
# class ServerType(Enum):
#     RPCSERVER = "rpc"
#     LLAMASERVER = "llama"

#_________________________________________
# @dataclass
# class Server:
#     hostname: str
#     tcpport: int
#     bindaddress: str | None
#     platform: str
#     cachepath: Path | None
#     binarypath: Path
#     type: ServerType

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

