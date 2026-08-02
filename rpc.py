from dataclasses import dataclass
from logzero import logger
from pathlib import Path
import json
import sys

#___________________________________________________________________________________
@dataclass
class RpcServer:
    IP: str
    PORT: int
    cachepath: str
    bin: str
    remuser: str
    cachedisk: str | None = None

#___________________________________________________________________________________
def load_rpcs(config_path: Path):
    with config_path.open(encoding="utf-8") as f:
        rpcs = json.load(f)

    rpc_section = rpcs.get("RPC_SERVERS", {})
    if not rpc_section:
        logger.error(f"Malformed json file {config_path}: missing section 'RPC_SERVERS'")
        sys.exit(1)

    rpc = {}
    
    for name in rpc_section:
        if not rpc_section[name].get("ip"):
            logger.error(f"Malformed json file {config_path}: missing attribute 'ip' for {name}")
            sys.exit(1)
        if not rpc_section[name].get("ip"):
                    logger.error(f"Malformed json file {config_path}: missing attribute 'ip' for {name}")
                    sys.exit(1)
        if not rpc_section[name].get("port"):
                    logger.error(f"Malformed json file {config_path}: missing attribute 'ip' for {name}")
                    sys.exit(1)
        if not rpc_section[name].get("cachepath"):
                    logger.error(f"Malformed json file {config_path}: missing attribute 'ip' for {name}")
                    sys.exit(1)
        if not rpc_section[name].get("bin"):
                    logger.error(f"Malformed json file {config_path}: missing attribute 'ip' for {name}")
                    sys.exit(1)
        if not rpc_section[name].get("remuser"):
                    logger.error(f"Malformed json file {config_path}: missing attribute 'ip' for {name}")
                    sys.exit(1)
        rpc[name] = RpcServer(
                IP=rpc_section[name]["ip"],
                PORT=int(rpc_section[name]["port"]),
                cachepath=rpc_section[name]["cachepath"],
                bin=rpc_section[name]["bin"],
                remuser=rpc_section[name]["remuser"]
            )
    return rpc

#___________________________________________________________________________________
if __name__ == "__main__":
    import argparse
    from config_manager import get_settings

    parser = argparse.ArgumentParser(description="List RPC servers from rpc config file")

    parser.add_argument(
        "--models-config", default=Path(__file__).parent / "rpc.json",
        help="path of models json description file"
    )
    
    args = parser.parse_args()

    config_path = Path(args.models_config)

    if not config_path.exists():
        logger.error(f"Error: config file '{config_path}' not found")
        sys.exit(1)
    
    settings = get_settings()
    
    rpcs = load_rpcs(config_path=config_path)
    logger.debug(f"{len(rpcs)} RPC servers loaded")
    logger.debug(f"RPC={rpcs}")