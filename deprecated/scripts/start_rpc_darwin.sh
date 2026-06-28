#!/bin/bash
set -euo pipefail

if [[ -z "${1:-}" || -z "${2:-}" || -z "${3:-}" || -z "${4:-}" || -z "${5:-}" ]]; then
    echo "Usage: $0 <HOST> <USER> <DISK> <RPC_SERVER_BIN> <RPC_CACHE_PATH>" >&2
    exit 1
fi

HOST="$1"
USER="$2"
DISK="$3"
RPC_SERVER_BIN="$4"
RPC_CACHE_PATH="$5"

ssh -n "$USER@$HOST" "diskutil mountDisk '$DISK'; nohup env LLAMA_CACHE='$RPC_CACHE_PATH' '$RPC_SERVER_BIN' --host 0.0.0.0 --port 50000 -c >/tmp/rpc-server.log 2>&1 </dev/null &"
