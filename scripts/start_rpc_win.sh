# #!/bin/bash
# set -euo pipefail

# if [[ -z "${1:-}" || -z "${2:-}" || -z "${3:-}" || -z "${4:-}" || -z "${5:-}" ]]; then
#     echo "Usage: $0 <HOST> <USER> <DISK> <RPC_SERVER_BIN> <RPC_CACHE_PATH>" >&2
#     exit 1
# fi

# HOST="$1"
# USER="$2"
# # $3 (DISK) unused on Windows — no disk mount needed
# RPC_SERVER_BIN="$4"
# RPC_CACHE_PATH="$5"

# # The Windows SSH server's default shell is PowerShell. Sending "cmd.exe /c
# # ... start /b ..." causes PowerShell to intercept "start" as Start-Process
# # before cmd.exe sees it. Use PowerShell directly instead.
# # -WindowStyle Hidden creates a new hidden console window for rpc-server,
# # detached from the SSH session, so SSH exits immediately without hanging.
# # PowerShell 5.1 Start-Process uses UseShellExecute=$true by default and does
# # NOT inherit $env: changes from the current session. To pass LLAMA_CACHE, we
# # wrap the command in cmd.exe and set the variable inline with "set VAR=...".
# # Start-Process -WindowStyle Hidden gives cmd its own hidden console, detached
# # from the SSH session, so SSH exits immediately without hanging.
# #ssh -n "$USER@$HOST" \
# #    "powershell -NoProfile -NonInteractive -Command \"Start-Process -FilePath 'cmd.exe' -ArgumentList '/c','set LLAMA_CACHE=$RPC_CACHE_PATH&&$RPC_SERVER_BIN --host 0.0.0.0 --port 50000 -c' -WindowStyle Hidden\""

# ssh $USER@$HOST "Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd /c set LLAMA_CACHE=D:\llama-rpc-cache&&C:\llama.cpp\bin\rpc-server.exe --host 0.0.0.0 --port 50000 -c'}"

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