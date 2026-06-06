#!/bin/bash
set -euo pipefail

if [[ -z "${1:-}" || -z "${2:-}" || -z "${3:-}" || -z "${4:-}" ]]; then
    echo "Usage: $0 <HOST> <DISK> <RPC_SERVER_BIN> <RPC_CACHE_PATH>" >&2
    exit 1
fi

HOST="$1"
# $2 (DISK) unused on Windows — no disk mount needed
RPC_SERVER_BIN="$3"
RPC_CACHE_PATH="$4"

ssh -n dorigo_a@"$HOST" \
    "powershell.exe -NoProfile -NonInteractive -Command \"\$env:LLAMA_CACHE='$RPC_CACHE_PATH'; Start-Process -FilePath '$RPC_SERVER_BIN' -ArgumentList '--host','0.0.0.0','--port','50000','-c' -WindowStyle Hidden -RedirectStandardOutput 'C:\\Temp\\rpc-server.log' -RedirectStandardError 'C:\\Temp\\rpc-server-err.log'\""
