#!/bin/bash
set -euo pipefail

if [[ -z "${1:-}" || -z "${2:-}" || -z "${3:-}" || -z "${4:-}" || -z "${5:-}" ]]; then
    echo "Usage: $0 <HOST> <USER> <DISK> <RPC_SERVER_BIN> <RPC_CACHE_PATH>" >&2
    exit 1
fi

HOST="$1"
USER="$2"
# $3 (DISK) inutilizzato su Windows — nessun mount necessario
RPC_SERVER_BIN="$4"
RPC_CACHE_PATH="$5"

ssh "${USER}@${HOST}" "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c set LLAMA_CACHE=${RPC_CACHE_PATH}&&${RPC_SERVER_BIN} --host 0.0.0.0 --port 50000 -c'}"