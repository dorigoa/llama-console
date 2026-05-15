
from enum import Enum, auto, StrEnum, unique
from dataclasses import dataclass
from pathlib import Path

@unique
class ServerType(Enum):
    RPCSERVER = auto()
    LLAMASERVER = auto()

@dataclass
class Server:
    hostname: str
    tcpport: int
    platform: str
    #type: ServerType
    cachepath: Path
    binarypath: Path

