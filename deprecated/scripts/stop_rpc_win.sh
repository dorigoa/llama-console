#!/bin/bash
set -euo pipefail

if [[ -z "${1:-}" || -z "${2:-}" ]]; then
    echo "Usage: $0 <HOST> <USER>" >&2
    exit 1
fi

HOST="$1"
USER="$2"

# Stop-Process -Name matches the process name without extension (rpc-server).
# -ErrorAction SilentlyContinue avoids failure when the process is not running.
ssh -n "$USER@$HOST" \
    "powershell -NoProfile -NonInteractive -Command \"Stop-Process -Name rpc-server -Force -ErrorAction SilentlyContinue\""
