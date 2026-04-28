from __future__ import annotations

import sys
import argparse
import subprocess
from pathlib import Path

#________________________________________________________________________________________
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch llama-server with automatic GGUF/mmproj discovery and remote RPC startup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("model_folder", type=Path, help="Folder containing the GGUF model")
    parser.add_argument("--rpc-server", dest="rpc_server", default="192.168.20.2", help="RPC server IP address used by llama-server")
    parser.add_argument("--rpc-port", dest="rpc_port", type=int, default=50000, help="RPC server TCP port")
    parser.add_argument("--rpc-host-ssh", dest="rpc_host_ssh", default="192.168.1.190", help="SSH host used to start/check the remote RPC server")
    parser.add_argument("--listen-host", default="0.0.0.0", help="Address where llama-server listens")
    parser.add_argument("--listen-port", type=int, default=443, help="Port where llama-server listens")
    parser.add_argument("--tensor-split", dest="tensorsplit", default="7,10", help="Tensor split passed to llama-server")
    parser.add_argument("--split-mode", dest="splitmode", default="layer", help="Tensor split passed to llama-server", choices=['none','layer','row','tensor'])
    parser.add_argument("--context-size", dest="ctxsize", default="0", help="Override the default context size embedded in the model", choices=['4096','8192','16384','32768','65536','131072','262144'])
    parser.add_argument("--threads", type=int, default=6, help="Number of CPU threads for llama-server")
    parser.add_argument("--threads-batch", type=int, default=8, help="Number of batch CPU threads for llama-server")
    #parser.add_argument("--parallel", type=int, default=1, help="Number of parallel sequences for llama-server")
    parser.add_argument("--ssl-key-file", dest="sslkey", type=Path, default=Path("/Volumes/Home/dorigo_a/llama-server.key"), help="TLS private key file")
    parser.add_argument("--ssl-cert-file", dest="sslcert", type=Path, default=Path("/Volumes/Home/dorigo_a/llama-server.crt"), help="TLS certificate file")
    parser.add_argument("--open-url", default="https://localhost:443", help="URL to open after llama-server is ready")
    parser.add_argument("--no-sudo", action="store_true", help="Do not prefix llama-server with sudo")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser when the server is ready")
    parser.add_argument("--skip-ssl-check", action="store_true", help="Do not verify that key/cert files exist before launch")

    return parser.parse_args()

#________________________________________________________________________________________
def main() -> int:
    args = parse_args()
    #config = make_config(args)

    # Import local/runtime modules only after argparse has handled --help/-h.
    # This makes help work even if those modules have heavy imports or runtime side effects.
    from logzero import logger
    import model_finder
    import launcher
    import devices
    import rpc

    files = model_finder.discover_model_files(args.model_folder)

    print(f"Model folder : {files.model_dir}")
    print(f"Model name   : {files.model_name}")
    print(f"GGUF model   : {files.gguf}")
    print(f"MMProj       : {files.mmproj if files.mmproj else 'none'}")

    logger.info(f"Model folder : {files.model_dir}")
    logger.info(f"Model name   : {files.model_name}")
    logger.info(f"GGUF model   : {files.gguf}")
    logger.info(f"MMProj       : {files.mmproj if files.mmproj else 'none'}")
    if not args.skip_ssl_check:
        launcher.validate_ssl_files( args.sslkey, args.sslcert )
            
    
    rpc.ensure_remote_rpc(  args.rpc_host_ssh,
                            5,
                            args.rpc_server,
                            args.rpc_port )

    gpus = devices.list_usable_devices( args.rpc_server, args.rpc_port )
    logger.info(f"GPU devices  : {gpus}")

    cmd = launcher.build_llama_command( "/usr/local/bin/llama-server",
                                        args.rpc_server,
                                        args.rpc_port,
                                        str(files.gguf),
                                        str(files.mmproj),
                                        gpus,
                                        str(args.sslkey),
                                        str(args.sslcert),
                                        args.tensorsplit,
                                        args.splitmode )
    launcher.print_command( cmd )
    
    subprocess.run( cmd, check=True )
    return 0

if __name__ == "__main__":
    sys.exit(main())
