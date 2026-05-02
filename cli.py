from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from config import Settings
from logging_utils import emit, LogSink, setup_console_logging

settings = Settings()

#_____________________________________________________________________________


#_____________________________________________________________________________
# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Launch llama-server with automatic GGUF/mmproj discovery and remote RPC startup.",
#         formatter_class=argparse.ArgumentDefaultsHelpFormatter,
#     )
#     parser.add_argument("model", type=Path, help="Model folder containing the GGUF model")
#     parser.add_argument("--rpc-server", dest="rpc_server", default=settings.rpc_host)
#     parser.add_argument("--rpc-port", dest="rpc_port", type=int, default=settings.rpc_port)
#     #parser.add_argument("--rpc-host-ssh", dest="rpc_host_ssh", default=settings.rpc_host_ssh)
#     parser.add_argument("--listen-host", default=settings.llama_server_host)
#     parser.add_argument("--listen-port", type=int, default=settings.llama_server_port)
#     parser.add_argument("--tensor-split", dest="tensorsplit", default=settings.llama_param["tensorsplit"])
#     parser.add_argument("--split-mode", dest="splitmode", default=settings.llama_param["defaultsplitmode"], choices=settings.llama_param["splitmodes"])
#     parser.add_argument("--context-size", dest="ctxsize", default=settings.llama_param["ctxsize"])
#     #parser.add_argument("--ssl-key-file", dest="sslkey", type=Path, default=Path(settings.llama_param["sslkeyfile"]))
#     #parser.add_argument("--ssl-cert-file", dest="sslcert", type=Path, default=Path(settings.llama_param["sslcertfile"]))
#     #parser.add_argument("--no-sudo", action="store_true", required=False, default=True)
#     parser.add_argument("--skip-ssl-check", action="store_true")
#     return parser.parse_args()

#_____________________________________________________________________________
# def main() -> int:
#     setup_console_logging()
#     args = parse_args()
#     cmd = get_llama_command(
#         args.model,
#         rpc_server=args.rpc_server,
#         rpc_port=args.rpc_port,
#         #rpc_host_ssh=args.rpc_host_ssh,
#         listen_host=args.listen_host,
#         listen_port=args.listen_port,
#         tensorsplit=args.tensorsplit,
#         splitmode=args.splitmode,
#         ctxsize=args.ctxsize,
#         skip_ssl_check=args.skip_ssl_check,
#     )
#     subprocess.run(cmd, check=True)
#     return 0

#_____________________________________________________________________________
# if __name__ == "__main__":
#     sys.exit(main())
