#!/bin/bash
set -euo pipefail

if [[ -z "${1:-}" || -z "${2:-}" ]]; then
    echo "Usage: $0 <HOST> <USER>" >&2
    exit 1
fi

HOST="$1"
USER="$2"

ssh -n "$USER@$HOST" "killall rpc-server"
