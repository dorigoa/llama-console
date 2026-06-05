#!/bin/bash
set -euo pipefail

if [[ -z "${1:-}" || -z "${2:-}" ]]; then
    echo "Usage: $0 <HOST> <DISK>" >&2
    exit 1
fi

HOST="$1"
DISK="$2"

ssh -n dorigo_a@"$HOST" "diskutil mountDisk '$DISK'; nohup env LLAMA_CACHE=/Volumes/Home/llama.cpp /usr/local/bin/rpc-server --host 0.0.0.0 --port 50000 -c >/tmp/rpc-server.log 2>&1 </dev/null &"
