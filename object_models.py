
from enum import Enum, auto, unique
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
    cachepath: Path
    binarypath: Path
    type: ServerType

