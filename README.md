# start_model.py

Command-line tool to launch **`llama-server`** (from [llama.cpp](https://github.com/ggml-org/llama.cpp)) for a model described in [`models.json`](models.json).

The script takes care of:

- **building the full `llama-server` command line** from the model's parameters (context size, sampling, KV-cache quantization, chat template, etc.);
- **starting the server locally or on a remote host over SSH**, in *detached* mode (the process survives the SSH session closing) and verifying that it actually stayed up;
- **managing distributed RPC servers** (probe, automatic start, GPU device discovery, kill);
- providing **server lifecycle commands**: status, stop, and log tailing.

> `start_model.py` is the command-line "engine" behind the GUI ([`llama-console-gui.py`](llama-console-gui.py)), but it can be used entirely on its own.

---

## Requirements

- **Python 3.10+**.
- The dependencies in [`requirements.txt`](requirements.txt):

  ```bash
  pip install -r requirements.txt
  ```

- The `llama-server` binary (from llama.cpp) installed **locally** or **on the remote host**, at the path given by `LLAMA_SERVER_BIN`.
- For remote/RPC usage: **passwordless SSH** (key-based) to the `LLAMA_SERVER_HOST` host and to the RPC nodes. The script uses `BatchMode=yes`, so any password prompt makes the command fail.

---

## Configuration

### `config.json` — global settings

Settings are loaded by [`config_manager.py`](config_manager.py), which looks, in order, for:

1. the `LLAMA_CONSOLE_CONFIG_FILE` environment variable;
2. `config.json` in the script's directory (if it exists);
3. `~/llama-console-config.json`.

If no file is found start_model.py will not start. **Unknown keys cause an error.**
Defaults are already defined in the `config.json` file located in the same directory as `start_model.py`.
| Key | Default | Description |
|---|---|---|
| `ADDRESS_BIND` | `0.0.0.0` | Address `llama-server` binds to (`--host`). |
| `PORT_BIND` | `8088` | Port `llama-server` binds to (`--port`). This port is also queried by `--server-status` to read back the active model. |
| `LLAMA_SERVER_BIN` | `/opt/llama.cpp/llama-server` | Path to the `llama-server` binary (local or on the remote host). |
| `LLAMA_SERVER_HOST` | `"192.168.1.191"` | SSH host on which to start/manage the server. **Empty = run locally.** |
| `LLAMA_SERVER_USER` | `"alvise"` | SSH user for `LLAMA_SERVER_HOST`. |
| `LLAMA_LOG_FILE` | `"/tmp/llama-server.log"` | Log file where llama-server process will write its output. |
| `LLAMA_BOOT_LOG` | `"/tmp/llama-server.boot.log"` | Log file of the boot process (including log before llama-server is started). |
| `MODELS_JSON` | `"./models.json"` | A json file with the description of all available models and their specs. |
| `UI_TITLE` | *(see default)* | Title shown by the GUI. |

### `models.json` — model catalog

Models are defined in [`models.json`](models.json). Structure:

```json
{
  "MODEL_BASE_DIR": "/Volumes/Home/gguf_models",
  "models": {
    "Model-name": {
      "ALIAS": "alias-name",
      "MMPROJ": null,
      "ctx": 512000,
      "TEMP": 1.0,
      "TOPP": 0.95,
      "TOPK": 64,
      "MINP": -1,
      "REAS": "high",
      "FITT": "6144,12288,2048,1024",
      "KVQUANT": null,
      "UB": null,
      "B": null,
      "MTP": false,
      "RPC_SERVERS": {
        "192.168.20.1": {
          "bin": "/opt/llama.cpp/rpc-server",
          "cachepath": "/Volumes/Home/llama-rpc-cache",
          "port": 50000,
          "remuser": "alvise"
        }
      }
    }
  }
}
```

Notes:

- `MODEL_BASE_DIR` (top level) is **required**: model files are looked up at `MODEL_BASE_DIR/<name>.gguf` (the `.gguf` extension is appended if missing).
- If the model file **does not exist** (checked over SSH too when a remote host is configured), the model is **skipped** with a `[SKIP]` message on stderr.
- `RPC_SERVERS` is optional: without it the model runs only on the host's local devices.
- The name you pass to `start_model.py` is the model's **key** in `models.json` (without `.gguf`).
- Quick JSON validation: `jq . models.json`.

For a detailed dump of the catalog (with file sizes and RPC nodes) you can run the models module directly:

```bash
python model.py            # uses models.json and the host from config.json
python model.py --remote-host 192.168.1.191 --remote-user alvise
```

---

## Usage

General syntax:

```bash
python start_model.py [MODEL_NAME] [OPTIONS]
```

`MODEL_NAME` is the model's key in `models.json`. It is optional because several options (e.g. `--list-models`, `--server-status`, `--tail-log`) do not require it.

### Starting a model

```bash
python start_model.py NVIDIA-Nemotron-3-Super-120B-A12B-UD-IQ4_NL
```

What happens:

1. Loads the settings and the model catalog.
2. Verifies that the `llama-server` binary exists (locally or over SSH).
3. If the model has RPC servers configured, checks that they are reachable and **starts them automatically** if they are down; then discovers the list of available GPU devices.
4. Builds the command line and starts `llama-server` in the background (*detached*).
5. Waits a few seconds and verifies (via `pgrep`) that the process stayed up; on a startup crash it prints the last lines of the *boot log*.

If `LLAMA_SERVER_HOST` is set, all of this happens **on the remote host over SSH**; otherwise it happens locally.

### Listing available models

```bash
python start_model.py --list-models
```

Prints the models (with size in GiB and number of RPC nodes) and exits.

### Managing a running server

```bash
python start_model.py --server-status    # is it running? which model?
python start_model.py --kill-server       # stop llama-server (SIGTERM, then SIGKILL)
python start_model.py --tail-log          # follow the runtime log (Ctrl-C to exit)
python start_model.py --tail-log -n 200   # with 200 initial lines of context
```

### Previewing the command (without starting anything)

```bash
python start_model.py MODEL-NAME --dry-run
```

Prints the exact `llama-server` command line that would be executed, without starting the RPC servers or the model. Useful for debugging.

### Managing the RPC servers

```bash
python start_model.py MODEL-NAME --only-check-rpc     # only check reachability
python start_model.py MODEL-NAME --only-start-rpc     # start the RPC nodes and exit
python start_model.py MODEL-NAME --only-list-devices  # list GPU devices (does NOT start RPC)
python start_model.py MODEL-NAME --kill-rpc-server    # killall rpc-server on every RPC node
```

> RPC operations always originate from `LLAMA_SERVER_HOST` (only that host can reach the RPC network): the script uses nested SSH *laptop → host → RPC node*.

---

## Overriding model parameters

Parameters taken from `models.json` can be overridden on the fly for a single run:

| Option | Type | Overrides |
|---|---|---|
| `--override-temp` | float | `TEMP` (temperature) |
| `--override-top-p` | float | `TOPP` |
| `--override-top-k` | int | `TOPK` |
| `--override-min-p` | float | `MINP` (only passed when ≥ 0) |
| `--override-ctx` | int | `ctx` (context size) |
| `--override-devices` | csv | GPU device list (skips auto-discovery) |
| `--override-fitt` | csv | `FITT` (fill-in-the-tensor split) |

⚠️ `--override-devices` and `--override-fitt` must **always be given together**: specifying only one is an error. Both accept comma-separated alphanumeric tokens (e.g. `CUDA0,CUDA1`).

Example:

```bash
python start_model.py MODEL-NAME \
  --override-temp 0.7 \
  --override-ctx 32768 \
  --override-devices CUDA0,CUDA1 \
  --override-fitt 8192,8192
```

---

## Full list of options

| Option | Description |
|---|---|
| `MODEL_NAME` | Model name as in `models.json` (without `.gguf`). |
| `--dry-run` | Print the command without executing it. |
| `--list-models` | List the available models and exit. |
| `--server-status` | Check whether `llama-server` is running and exit. |
| `--kill-server` | Stop the `llama-server` process and exit. |
| `--tail-log` | Follow (`tail -F`) the runtime log until Ctrl-C. |
| `--tail-lines`, `-n INT` | Initial lines to show with `--tail-log` (default: 50). |
| `--only-start-rpc` | Start only the RPC nodes and exit. |
| `--only-check-rpc` | Only check the RPC nodes' reachability. |
| `--only-list-devices` | List local + RPC GPU devices (does not start the RPC nodes). |
| `--kill-rpc-server` | `killall rpc-server` on every RPC node of the model. |
| `--override-temp FLOAT` | Override the temperature. |
| `--override-top-p FLOAT` | Override top-p. |
| `--override-top-k INT` | Override top-k. |
| `--override-min-p FLOAT` | Override min-p. |
| `--override-ctx INT` | Override the context size. |
| `--override-devices STR` | GPU device list (csv). Also requires `--override-fitt`. |
| `--override-fitt STR` | FITT split (csv). Also requires `--override-devices`. |

---

## Log files

| File | Contents |
|---|---|
| `/tmp/llama-server.log` | **Runtime** log of `llama-server` (via `--log-file`). Followable with `--tail-log`. |
| `/tmp/llama-server.boot.log` | **Startup stdout/stderr**: captures immediate crashes (bad argument, missing library, OOM) that happen before the runtime log is opened. |

With a remote host these paths are the ones **on the remote host** and are read over SSH.

---

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Operation succeeded. |
| `1` | Generic error (model not found, missing binary, server failed to start, RPC unreachable, etc.). |
| `2` | `LLAMA_SERVER_HOST` **unreachable over SSH** (distinct from the server simply not running). |

---

## Quick examples

```bash
# List the models
python start_model.py --list-models

# Start a model (locally or on the remote host, per config.json)
python start_model.py Qwen3-30B-A3B

# See the command that would run, without starting anything
python start_model.py Qwen3-30B-A3B --dry-run

# Check the status and follow the logs
python start_model.py --server-status
python start_model.py --tail-log

# Stop everything
python start_model.py --kill-server
python start_model.py Qwen3-30B-A3B --kill-rpc-server
```
