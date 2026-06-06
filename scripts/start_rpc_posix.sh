#!/bin/bash
set -euo pipefail

if [[ -z "${1:-}" || -z "${2:-}" || -z "${3:-}" || -z "${4:-}" ]]; then
    echo "Usage: $0 <HOST> <DISK> <RPC_SERVER_BIN> <RPC_CACHE_PATH>" >&2
    exit 1
fi

HOST="$1"
DISK="$2"
RPC_SERVER_BIN="$3"
RPC_CACHE_PATH="$4"

ssh -n dorigo_a@"$HOST" "diskutil mountDisk '$DISK'; nohup env LLAMA_CACHE='$RPC_CACHE_PATH' '$RPC_SERVER_BIN' --host 0.0.0.0 --port 50000 -c >/tmp/rpc-server.log 2>&1 </dev/null &"
