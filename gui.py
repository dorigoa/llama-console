from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from persist import JsonParams
import re
from nicegui import ui

from launcher import get_llama_command, format_command
from config import settings
from logging_utils import emit, setup_console_logging

logger = setup_console_logging()

LLAMA_READY_LOG_MARKERS = (
    "server is listening on",
    "all slots are idle",
)
LLAMA_READY_TIMEOUT_SECONDS = 300

params = JsonParams( settings.PERSIST_FILE )

#_____________________________________________________________________________
def notify_user(message: str, *, type: str = "info") -> None:
    """Show a persistent, manually dismissible NiceGUI notification."""
    try:
        ui.notify(message, type=type, timeout=0, close_button=True)
    except TypeError:
        # Compatibility fallback for older NiceGUI versions without close_button.
        ui.notify(message, type=type, timeout=0)

#_____________________________________________________________________________
def is_llama_ready_log_line(text: str) -> bool:
    """Return True when llama-server output indicates that serving is actually ready."""
    lowered = text.lower()
    return any(marker in lowered for marker in LLAMA_READY_LOG_MARKERS)

#_____________________________________________________________________________
def ui_log(message: str) -> None:
    """Write to NiceGUI log widget if the browser client still exists."""
    try:
        log_area.push(str(message))
    except RuntimeError as exc:
        if "client this element belongs to has been deleted" in str(exc):
            return
        raise

#_____________________________________________________________________________
def configured_model_path(configured: Any) -> str:
    """Extract the main GGUF path/folder from one settings.AVAILABLE_MODELS value.

    Supports both legacy string values and dict values, for example:
        "Model": "/path/to/model.gguf"
        "Model": {"model": "/path/to/model.gguf", "mmproj": "/path/to/mmproj.gguf"}
    """
    if isinstance(configured, (str, Path)):
        return str(configured)

    if isinstance(configured, dict):
        # Prefer explicit main-model keys. Do not pick mmproj unless it is the only usable path.
        preferred_keys = (
            "model",
            "model_path",
            "path",
            "gguf",
            "file",
            "filename",
            "model_file",
            "folder",
            "directory",
            "dir",
        )
        for key in preferred_keys:
            value = configured.get(key)
            if isinstance(value, (str, Path)) and str(value).strip():
                return str(value)

        # Fallback: first string/path value that does not look like a multimodal projector.
        for value in configured.values():
            if isinstance(value, (str, Path)) and str(value).strip():
                candidate = str(value)
                if "mmproj" not in Path(candidate).name.lower():
                    return candidate

        # Last resort: any string/path value, including mmproj.
        for value in configured.values():
            if isinstance(value, (str, Path)) and str(value).strip():
                return str(value)

    raise TypeError(f"Unsupported settings.AVAILABLE_MODELS entry: {configured!r}")

#_____________________________________________________________________________
def path_to_model_folder(path_string: str | Path) -> Path:
    """Accept either a GGUF file path or a model directory path."""
    p = Path(path_string).expanduser()

    # Path.exists() is intentionally not required here because network volumes may be late-mounted.
    # If the configured string ends with .gguf, treat it as a model file and use its parent.
    if p.suffix.lower() == ".gguf":
        return p.parent.resolve()

    return p.resolve()


#_____________________________________________________________________________
def configured_context_size(configured: Any) -> int:
    """Extract a per-model context size from model config, if present."""
    if isinstance(configured, dict):
        value = configured.get("ctxsize", 0)
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0
    return 0

#_____________________________________________________________________________
def configured_temp(configured: Any) -> float:
    """Extract a per-model temperature from model config, if present."""
    if isinstance(configured, dict):
        value = configured.get("temp", 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0
    return 0

#_____________________________________________________________________________
def configured_top_p(configured: Any) -> float:
    """Extract a per-model top_p from model config, if present."""
    if isinstance(configured, dict):
        value = configured.get("top_p", 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0
    return 0

#_____________________________________________________________________________
def configured_top_k(configured: Any) -> float:
    """Extract a per-model top_p from model config, if present."""
    if isinstance(configured, dict):
        value = configured.get("top_k", 0)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0
    return 0

#_____________________________________________________________________________
def configured_shard_balance(configured: Any) -> float:
    """Extract a per-model shard_balance from model config, if present."""
    if isinstance(configured, dict):
        value = configured.get("shard_balance", 0)
        try:
            return value
        except (TypeError, ValueError):
            return ""
    return ""

#_____________________________________________________________________________
def format_context_size(value: int) -> str:
    """Human-readable label for context size values."""
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}k"
    return str(value)

#_____________________________________________________________________________
def configured_context_options() -> dict[int, str]:
    """Return NiceGUI select options for context sizes.
    NiceGUI expects dict options in the form {value: label}; therefore
    the selected value remains the integer context size, while the UI
    shows the compact label such as "32k".
    """
    values = settings.CONTEXT_SIZE_OPTIONS #getattr(settings, "CONTEXT_SIZE_OPTIONS", [
    
    return {int(v): format_context_size(int(v)) for v in values}

#_____________________________________________________________________________
def available_model_names() -> list[str]:
    """Return model names discovered/configured for the model combo box."""
    return sorted(settings.AVAILABLE_MODELS.keys(), key=str.lower)

#_____________________________________________________________________________
def default_model_name() -> Optional[str]:
    names = available_model_names()
    return names[0] if names else None

#_____________________________________________________________________________
def default_context_size_for_model(model_name: Optional[str]) -> int:
    """Return model-specific ctxsize when configured, otherwise global default."""
    if model_name and model_name in settings.AVAILABLE_MODELS:
        model_ctx = configured_context_size(settings.AVAILABLE_MODELS[model_name])
        if model_ctx > 0:
            return model_ctx
    return settings.DEFAULT_CONTEXT_SIZE#int(getattr(settings, "DEFAULT_CONTEXT_SIZE", 32768))

#_____________________________________________________________________________
def default_temp_for_model(model_name: Optional[str]) -> float:
    """Return model-specific temperature when configured, otherwise global default."""
    if model_name and model_name in settings.AVAILABLE_MODELS:
        model_temp = configured_temp(settings.AVAILABLE_MODELS[model_name])
        if model_temp > 0:
            return model_temp
    return settings.DEFAULT_TEMP

#_____________________________________________________________________________
def default_top_p_for_model(model_name: Optional[str]) -> float:
    """Return model-specific top_p when configured, otherwise global default."""
    if model_name and model_name in settings.AVAILABLE_MODELS:
        model_top_p = configured_temp(settings.AVAILABLE_MODELS[model_name])
        if model_top_p > 0:
            return model_top_p
    return settings.DEFAULT_TOP_P

#_____________________________________________________________________________
def default_top_k_for_model(model_name: Optional[str]) -> float:
    """Return model-specific top_k when configured, otherwise global default."""
    if model_name and model_name in settings.AVAILABLE_MODELS:
        model_top_k = configured_temp(settings.AVAILABLE_MODELS[model_name])
        if model_top_k > 0:
            return model_top_k
    return settings.DEFAULT_TOP_K

#_____________________________________________________________________________
def default_shard_balance_for_model(model_name: Optional[str]) -> float:
    """Return shard_balance (which depends on the cluster) when configured, otherwise global default."""
    if model_name and model_name in settings.AVAILABLE_MODELS:
        model_shard_balance = configured_temp(settings.AVAILABLE_MODELS[model_name])
        if model_shard_balance:
            return model_shard_balance
    return settings.DEFAULT_SHARD_BALANCE

#_____________________________________________________________________________
def persisted_data_for_model(model_name: Optional[str]) -> dict | None: #-> Optional[dict]:
    """Return the context size stored in persist.json for model_name, if present and valid."""
    if not model_name:
        return None

    try:
        persisted = params.load_params()
    except Exception as exc:
        emit(f"Could not load persisted parameters from {settings.PERSIST_FILE}: {exc}", None)
        return None

    if model_name not in persisted:
        return None

    try:
        return persisted[model_name]
    except (TypeError, ValueError):
        emit(f"Ignoring invalid persisted context size for {model_name!r}: {persisted[model_name]!r}", None)
        return None

#_____________________________________________________________________________
def selected_data_for_model(model_name: Optional[str]) -> tuple[int, float, float, int]:
    """Return persisted ctxsize when available, otherwise configured/default ctxsize."""
    persisted_data = persisted_data_for_model(model_name)
    if persisted_data is not None:
        return persisted_data['context_size'], persisted_data['temperature'], persisted_data['top_p'], persisted_data['top_k'], persisted_data['shard_balance']
    return default_context_size_for_model(model_name), default_temp_for_model(model_name), default_top_p_for_model( model_name ), default_top_k_for_model( model_name ), default_shard_balance_for_model( model_name )

#_____________________________________________________________________________
def normalize_context_size_for_select(ctx: int) -> int:
    """Return a context size accepted by the context select widget."""
    valid_values = set(configured_context_options().keys())

    if ctx in valid_values:
        return ctx

    fallback_ctx = int(getattr(settings, "DEFAULT_CONTEXT_SIZE", 32768))
    if fallback_ctx in valid_values:
        return fallback_ctx

    return next(iter(valid_values))

#_____________________________________________________________________________
def update_data_from_model() -> None:
    """Set context combo box from persist.json, then config/default fallback."""
    model_name = str(model_select.value) if model_select.value else None
    ctx, temp, top_p, top_k, shard_balance = selected_data_for_model(model_name)
    ctx = normalize_context_size_for_select(ctx)
    context_select.set_value( ctx )
    temperature_select.set_value(f"{float(temp):.1f}")
    top_p_input.set_value(f"{float(top_p):.1f}")
    top_k_input.set_value(f"{int(top_k)}")
    shard_balance_input.set_value( shard_balance )


#_____________________________________________________________________________
def _local_llama_base_urls() -> list[str]:
    """Return candidate local URLs for probing an already-running llama-server."""
    port = settings.LLAMA_SERVER_PORT
    return [
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
    ]

#_____________________________________________________________________________
def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a small local inspection command without raising on non-zero exit."""
    return subprocess.run(
        args,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )

#_____________________________________________________________________________
def _parse_pid_lines(output: str) -> list[int]:
    """Parse one PID per line, preserving order and removing duplicates."""
    pids: list[int] = []
    seen: set[int] = set()
    for line in output.splitlines():
        try:
            pid = int(line.strip())
        except ValueError:
            continue
        if pid > 0 and pid not in seen:
            pids.append(pid)
            seen.add(pid)
    return pids

#_____________________________________________________________________________
def find_listening_pids_on_port(port: int) -> list[int]:
    """Return local PIDs listening on the given TCP port.

    lsof works on macOS and most Linux systems. ss is used as a Linux fallback.
    """
    try:
        lsof = _run_command(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"])
        pids = _parse_pid_lines(lsof.stdout)
        if pids:
            return pids
    except Exception as exc:
        emit(f"lsof lookup failed: {exc}", ui_log)

    try:
        ss = _run_command(["ss", "-H", "-ltnp", f"sport = :{port}"])
    except Exception as exc:
        emit(f"ss lookup failed: {exc}", ui_log)
        return []

    pids: list[int] = []
    seen: set[int] = set()
    for token in ss.stdout.replace(',', ' ').split():
        if not token.startswith('pid='):
            continue
        try:
            pid = int(token.removeprefix('pid='))
        except ValueError:
            continue
        if pid > 0 and pid not in seen:
            pids.append(pid)
            seen.add(pid)
    return pids

#_____________________________________________________________________________
def kill_pids_sync(pids: list[int], *, terminate_timeout: float = 10.0) -> tuple[list[int], list[str]]:
    """Terminate, then force-kill if required. Returns affected PIDs and error strings."""
    import time

    current_pid = os.getpid()
    targets = [pid for pid in pids if pid != current_pid]
    killed: list[int] = []
    errors: list[str] = []

    if not targets:
        return killed, ["No killable PID found"]

    for pid in targets:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            killed.append(pid)
        except PermissionError as exc:
            errors.append(f"PID {pid}: permission denied while sending SIGTERM: {exc}")
        except OSError as exc:
            errors.append(f"PID {pid}: failed to send SIGTERM: {exc}")

    end_time = time.monotonic() + terminate_timeout
    while time.monotonic() < end_time:
        alive: list[int] = []
        for pid in targets:
            if pid in killed:
                continue
            try:
                os.kill(pid, 0)
                alive.append(pid)
            except ProcessLookupError:
                killed.append(pid)
            except PermissionError:
                alive.append(pid)
        if not alive:
            return killed, errors
        time.sleep(0.2)

    for pid in targets:
        if pid in killed:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            killed.append(pid)
        except PermissionError as exc:
            errors.append(f"PID {pid}: permission denied while sending SIGKILL: {exc}")
        except OSError as exc:
            errors.append(f"PID {pid}: failed to send SIGKILL: {exc}")

    return killed, errors

#_____________________________________________________________________________
def _json_get(url: str, timeout: float = 2.0) -> dict[str, Any]:
    """Small blocking JSON GET helper. Run it via asyncio.to_thread()."""
    req = Request(url, headers={"Accept": "application/json"})

    with urlopen(req, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        if not raw:
            return {}
        return json.loads(raw)

#_____________________________________________________________________________
def _extract_model_from_openai_models(payload: dict[str, Any]) -> Optional[str]:
    """Extract model id/name from /v1/models compatible response."""
    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            model_id = first.get("id") or first.get("name")
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()
    return None

#_____________________________________________________________________________
def _extract_model_from_props(payload: dict[str, Any]) -> Optional[str]:
    """Extract model id/path from llama.cpp /props response, if settings.AVAILAble."""
    for key in ("model_path", "model", "model_name", "model_alias"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None

#_____________________________________________________________________________
def _match_configured_model(detected_model: str) -> str:
    """Try to map llama-server reported model string to one settings.AVAILABLE_MODELS key."""
    detected = detected_model.strip()
    detected_path = Path(detected)
    detected_name = detected_path.name
    detected_stem = detected_path.stem

    for logical_name, configured in settings.AVAILABLE_MODELS.items():
        try:
            main_path = configured_model_path(configured)
        except TypeError as exc:
            emit(f"Skipping invalid settings.AVAILABLE_MODELS entry {logical_name!r}: {exc}", None)
            continue

        configured_path = Path(main_path).expanduser()
        folder = path_to_model_folder(main_path)
        candidates = {
            logical_name,
            str(configured_path),
            configured_path.name,
            configured_path.stem,
            str(folder),
            folder.name,
        }
        if detected in candidates or detected_name in candidates or detected_stem in candidates:
            return logical_name

    return detected

#_____________________________________________________________________________
def probe_existing_llama_server_sync() -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """Probe an already-running llama-server.

    Returns: (running, model, base_url, error)
    """
    last_error: Optional[str] = None

    for base_url in _local_llama_base_urls():
        for endpoint, extractor in (
            ("/v1/models", _extract_model_from_openai_models),
            ("/props", _extract_model_from_props),
        ):
            url = f"{base_url}{endpoint}"
            try:
                payload = _json_get(url)
                model = extractor(payload)
                return True, model, base_url, None
            except (HTTPError, URLError, TimeoutError, ConnectionError, json.JSONDecodeError, OSError) as exc:
                last_error = f"{url}: {exc}"
                continue

    return False, None, None, last_error

#_____________________________________________________________________________
async def detect_existing_llama_server(*, verbose: bool = True) -> bool:
    """Detect a llama-server already running before this GUI launched it.

    This updates permanent UI elements. It intentionally does not use notify_user().
    """
    status_label.set_text("llama-server status: checking...")
    status_detail_label.set_text(f"Probe target: local port {settings.LLAMA_SERVER_PORT}")

    running, detected_model, base_url, error = await asyncio.to_thread(probe_existing_llama_server_sync)

    if running:
        display_model = _match_configured_model(detected_model) if detected_model else "unknown model"
        #global_display_model = display_model
        chat_url = await get_browser_based_llama_url()

        if "127.0.0.1" not in str(chat_url):
            chat_url=chat_url.replace("http","https")
            chat_url=chat_url.replace(":8088","")

        status_label.set_text("llama-server status: already running")
        status_detail_label.set_text(
            f"Detected endpoint: {base_url} | Model: {display_model} "
        )
        set_link_target(status_chat_link, chat_url)
        status_chat_link.visible = True
        status_chat_button.visible = True

        emit(f"Detected already-running llama-server at {base_url}", ui_log)
        emit(f"Detected model: {display_model}", ui_log)
        emit(f"Chat URL: {chat_url}", ui_log)
        return True

    status_label.set_text("llama-server status: not detected")
    status_detail_label.set_text(f"No server answered on local port {settings.LLAMA_SERVER_PORT}")
    status_chat_link.visible = False
    status_chat_button.visible = False
    if verbose:
        emit(f"No existing llama-server detected. Last probe error: {error}", ui_log)
    return False

#_____________________________________________________________________________
async def get_browser_based_llama_url() -> str:
    """Build the llama-server chat URL from the browser-visible GUI URL."""
    port = settings.LLAMA_SERVER_PORT
    js = f"""
        (() => {{
            const hostname = window.location.hostname;
            return `http://${{hostname}}:{port}/`;
        }})()
    """
    return str(await ui.run_javascript(js))

#_____________________________________________________________________________
def set_link_target(link: ui.link, url: str) -> None:
    """Update a NiceGUI link target and visible text."""
    link.set_text(url)
    link.props(f'href="{url}"')

#_____________________________________________________________________________
def open_chat_dialog(model_name: str, chat_url: str) -> None:
    chat_model_label.set_text(f"Model: {model_name}")
    set_link_target(chat_url_link, chat_url)
    chat_dialog.open()

#_____________________________________________________________________________
class LlamaManager:

    #_____________________________________________________________________________________
    def __init__(self) -> None:
        self.process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._ready_event: Optional[asyncio.Event] = None
        self._ready_reason: Optional[str] = None

    #_____________________________________________________________________________________
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    #_____________________________________________________________________________________
    async def start_server(self, 
                           model_name: str, 
                           configured: Any, 
                           context_size: int, 
                           temperature: float, 
                           top_p: float, 
                           top_k: int, 
                           shard_balance: str, 
                           load_mmproj: bool) -> bool:
        if self.is_running():
            msg = "llama-server is already running"
            emit(msg, ui_log)
            notify_user(msg, type="warning")
            return False

        if await detect_existing_llama_server(verbose=False):
            msg = "llama-server is already active on the configured port"
            emit(msg, ui_log)
            notify_user(msg, type="warning")
            return False

        try:
            configured_path = configured_model_path(configured)
            model_folder = path_to_model_folder(configured_path)
        except Exception as exc:
            msg = f"Invalid model configuration for {model_name}: {exc}"
            emit(msg, ui_log)
            status_label.set_text("llama-server status: invalid model configuration")
            status_detail_label.set_text(str(exc))
            notify_user(msg, type="negative")
            return False

        emit("--- Start requested ---", ui_log)
        emit(f"Selected model : {model_name}", ui_log)
        emit(f"Configured path: {configured_path}", ui_log)
        emit(f"Model folder   : {model_folder}", ui_log)
        emit(f"Context size   : {context_size}", ui_log)
        emit(f"Temperature    : {temperature}", ui_log)
        emit(f"Top_p          : {top_p}", ui_log)
        emit(f"Top_k          : {top_k}", ui_log)
        emit(f"Sharding       : {shard_balance}", ui_log)
        emit(f"Load mmproj    : {load_mmproj}", ui_log)

        try:

            cmd = await asyncio.to_thread(
                get_llama_command,
                model_folder,
                ui_log,
                ctxsize=context_size,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                tensorsplit=shard_balance,
                load_mmproj=load_mmproj,
            )

            cmd = [str(arg) for arg in cmd]
            emit(f"Final command: {format_command(cmd)}", ui_log)

            emit("Launching llama-server process...", ui_log)
            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            status_label.set_text("llama-server status: starting")
            status_detail_label.set_text(
                f"Starting model: {model_name}; waiting for llama-server readiness log "
                "('server is listening on ...' or 'all slots are idle')"
            )
            notify_user(f"Starting {model_name}...", type="info")

            self._ready_event = asyncio.Event()
            self._ready_reason = None
            self._reader_task = asyncio.create_task(self._read_process_output(model_name, context_size, temperature, top_p, top_k, shard_balance, self.process))

            await asyncio.sleep(0.5)
            if self.process.returncode is not None:
                emit(f"llama-server exited immediately with return code {self.process.returncode}", ui_log)
                status_label.set_text("llama-server status: failed")
                status_detail_label.set_text(f"Model {model_name} exited immediately")
                notify_user(f"{model_name} exited before becoming ready", type="negative")
                return False

            try:
                assert self._ready_event is not None
                await asyncio.wait_for(self._ready_event.wait(), timeout=LLAMA_READY_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                msg = (
                    f"llama-server did not emit a readiness line within "
                    f"{LLAMA_READY_TIMEOUT_SECONDS} seconds"
                )
                emit(msg, ui_log)
                status_label.set_text("llama-server status: starting, readiness not confirmed")
                status_detail_label.set_text(msg)
                notify_user(msg, type="warning")
                return False

            if self.process.returncode is not None:
                emit(f"llama-server exited before readiness completed with return code {self.process.returncode}", ui_log)
                status_label.set_text("llama-server status: failed")
                status_detail_label.set_text(f"Model {model_name} exited before readiness completed")
                notify_user(f"{model_name} exited before becoming ready", type="negative")
                return False

            chat_url = await get_browser_based_llama_url()
            status_label.set_text("llama-server status: running")
            status_detail_label.set_text(
                f"Started by this GUI | Model: {model_name} | Ready: {self._ready_reason or 'confirmed'}"
            )
            set_link_target(status_chat_link, chat_url)
            status_chat_link.visible = True
            status_chat_button.visible = True
            notify_user(f"{model_name} is ready", type="positive")
            return True

        except Exception as exc:
            self.process = None
            msg = f"Start failed: {exc}"
            emit(msg, ui_log)
            status_label.set_text("llama-server status: start failed")
            status_detail_label.set_text(str(exc))
            notify_user(msg, type="negative")
            return False

    #_____________________________________________________________________________________
    async def _read_process_output(self, model_name: str, 
                                   context_size: int, 
                                   temperature: float,
                                   top_p: float,
                                   top_k: int,
                                   shard_balance: str,
                                   process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None

        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                text = line.decode(errors="replace").rstrip()
                if text:
                    emit(f"[llama-server] {text}", ui_log)
                    if self.process is process and self._ready_event is not None and not self._ready_event.is_set():
                        if is_llama_ready_log_line(text):
                            self._ready_reason = text
                            self._ready_event.set()
                            emit(f"llama-server readiness confirmed by log line: {text}", ui_log)
                            model_persist_data = {
                                "context_size": context_size,
                                "temperature": temperature,
                                "top_p": top_p,
                                "top_k": top_k,
                                "shard_balance": shard_balance
                            }
                            params.save_param(model_name, model_persist_data)

            return_code = await process.wait()
            emit(f"llama-server exited with return code {return_code}", ui_log)
            status_label.set_text("llama-server status: stopped")
            status_detail_label.set_text(f"Last model: {model_name} | Return code: {return_code}")

        except asyncio.CancelledError:
            emit("Log reader task cancelled", ui_log)
            raise
        except Exception as exc:
            emit(f"Error while reading llama-server output: {exc}", ui_log)
            status_label.set_text("llama-server status: log reader error")
            status_detail_label.set_text(str(exc))
        finally:
            if self.process is process:
                self.process = None
                self._ready_event = None
                self._ready_reason = None
            if self._reader_task is asyncio.current_task():
                self._reader_task = None

    #_____________________________________________________________________________________
    async def stop_server(self) -> None:
        if self.is_running():
            assert self.process is not None
            emit("Stopping GUI-started llama-server...", ui_log)
            status_label.set_text("llama-server status: stopping")
            self.process.terminate()
            ui.notify("Initiated Stopping llama-server", type="info", timeout=0, close_button=True)
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
                emit("GUI-started llama-server terminated", ui_log)
            except asyncio.TimeoutError:
                emit("GUI-started llama-server did not terminate; killing it", ui_log)
                self.process.kill()
                await self.process.wait()
                emit("GUI-started llama-server killed", ui_log)

                pids = await asyncio.to_thread(find_listening_pids_on_port, port)
                if pids:
                    emit("Process still running, forcing kill with killall -9", ui_log)
                    #await asyncio.to_thread(os.system, "killall -9 llama-server")
                    try:
                        subprocess.run(["killall", "-9", "llama-server"], check=True, capture_output=True)
                        emit("Force kill executed successfully", ui_log)
                    except subprocess.CalledProcessError as e:
                        emit(f"Failed to force kill: {e}", ui_log)

            status_label.set_text("llama-server status: stopped")
            status_detail_label.set_text("Stopped GUI-started process")
            status_chat_link.visible = False
            status_chat_button.visible = False
            notify_user("Server stopped", type="info")
            return

        port = settings.LLAMA_SERVER_PORT
        emit(f"No GUI-started process handle; looking for external listener on TCP port {port}...", ui_log)
        status_label.set_text("llama-server status: stopping external process")
        status_detail_label.set_text(f"Searching for listener on TCP port {port}")

        pids = await asyncio.to_thread(find_listening_pids_on_port, port)
        if not pids:
            msg = f"No process is listening on TCP port {port}"
            emit(msg, ui_log)
            status_label.set_text("llama-server status: not detected")
            status_detail_label.set_text(msg)
            status_chat_link.visible = False
            status_chat_button.visible = False
            notify_user(msg, type="warning")
            return

        emit(f"Killing external llama-server/listener PIDs on port {port}: {pids}", ui_log)
        killed, errors = await asyncio.to_thread(kill_pids_sync, pids)

        for err in errors:
            emit(err, ui_log)

        still_running, _, _, _ = await asyncio.to_thread(probe_existing_llama_server_sync)
        if still_running:
            msg = "Requested kill, but llama-server is still responding"
            status_label.set_text("llama-server status: still running")
            status_detail_label.set_text(msg)
            notify_user(msg, type="negative")
            return

        status_label.set_text("llama-server status: stopped")
        status_detail_label.set_text(f"Stopped external listener on port {port}; PIDs: {killed or pids}")
        status_chat_link.visible = False
        status_chat_button.visible = False
        notify_user("External server stopped", type="info")



manager = LlamaManager()

ui.colors(primary="#6e93d6")

#_____________________________________________________________________________
with ui.dialog() as chat_dialog, ui.card().classes("w-full max-w-lg"):
    ui.label("llama-server is running").classes("text-h6")
    chat_model_label = ui.label("").classes("text-subtitle2")
    ui.label("Open the chat interface:")
    chat_url_link = ui.link("", target="_blank").classes("text-blue-600 underline break-all")
    with ui.row().classes("w-full justify-end gap-2"):
        ui.button("Close", on_click=chat_dialog.close)
        ui.button(
            "Open chat",
            on_click=lambda: ui.run_javascript(f"window.open('{chat_url_link.text}', '_blank')"),
            icon="open_in_new",
        )

#_____________________________________________________________________________
with ui.header().classes("items-center justify-between"):
    ui.label(settings.UI_TITLE).classes("text-h6")
    ui.button("Stop Model", on_click=manager.stop_server, icon="stop", color="red")

#_____________________________________________________________________________
with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-4"):
    with ui.card().classes("w-full p-4"):
        status_label = ui.label("llama-server status: not checked yet").classes("font-bold")
        status_detail_label = ui.label("Startup detection pending...").classes("text-sm text-gray-600")
        status_chat_link = ui.link("", target="_blank").classes("text-blue-600 underline break-all")
        status_chat_link.visible = False
        with ui.row().classes("gap-2 mt-2"):
            ui.button("Recheck status", on_click=detect_existing_llama_server, icon="refresh")
            status_chat_button = ui.button(
                "Open chat",
                on_click=lambda: ui.run_javascript(f"window.open('{status_chat_link.text}', '_blank')"),
                icon="open_in_new",
            )
            status_chat_button.visible = False

    with ui.card().classes("w-full p-4"):
        ui.label("Select a model").classes("text-subtitle1")

        with ui.row().classes("w-full gap-4 mt-4 items-end"):

            model_select = ui.select(
                options=available_model_names(),
                value=default_model_name(),
                label="Select a model from the list below...",
                on_change=lambda _: update_data_from_model(),
            ).classes("flex-1")

            model_list_refresh = ui.button("Refresh List", on_click=None, icon="refresh").classes("mt-4")

        with ui.row().classes("w-full gap-4 mt-4 items-end"):
            ctx, temp, top_p, top_k, shard_balance = selected_data_for_model(default_model_name())
            context_select = ui.select(
                options=configured_context_options(),
                value=normalize_context_size_for_select( ctx ),
                label="Context size (0 = auto)",
            ).classes("flex-[2]")

            temperature_select = ui.select(
                options=[f"{i / 10:.1f}" for i in range(1, 11)],
                value=f"{float(temp):.1f}",
                label="Temperature",
            ).classes("flex-[1]")

            top_p_input = ui.input(
                value="0.9",
                label="Top_p",
            ).classes("flex-[1]")
            
            top_k_input = ui.input(
                value="40",
                label="Top_k",
            ).classes("flex-[1]")

        shard_balance_input = ui.input(
                value="6,12",
                label="Shard balance",
            ).classes("flex-[1]")
        
        mmproj_select = ui.checkbox('Load MM Projector if available', value=False).classes("flex-[1]")

        async def start_selected_model() -> None:
            if not model_select.value:
                emit("Start ignored: no model selected", ui_log)
                notify_user("Select a model!", type="warning")
                return

            if context_select.value is None:
                emit("Start ignored: no context size selected", ui_log)
                notify_user("Select a context size!", type="warning")
                return
            try:
                context_size = int(context_select.value)
            except (TypeError, ValueError):
                emit(f"Start ignored: invalid context size: {context_select.value!r}", ui_log)
                notify_user("Invalid context size!", type="warning")
                return
            
            if top_p_input.value is None:
                emit("Start ignored: no Top_p selected", ui_log)
                notify_user("Input a Top_p between 0 and 1 (1 decimal digit)", type="warning")

            if top_k_input.value is None:
                emit("Start ignored: no Top_k selected", ui_log)
                notify_user("Input a Top_k integer between 20 and 100", type="warning")

            if shard_balance_input.value is None:
                emit("Start ignored: no Shard Balance selected", ui_log)
                notify_user("Input a Shard balance string (e.g. 5,5)", type="warning")

            try:
                temperature = float(temperature_select.value)
            except (TypeError, ValueError):
                emit(f"Start ignored: invalid temperature: {temperature_select.value!r}", ui_log)
                notify_user("Invalid temperature!", type="warning")
                return
            
            try:
                top_p = float(top_p_input.value)
            except (TypeError, ValueError):
                emit(f"Start ignored: invalid Top_p: {top_p_input.value!r}", ui_log)
                notify_user("Invalid Top_p!", type="warning")
                return
            
            try:
                top_k = int(top_k_input.value)
            except (TypeError, ValueError):
                emit(f"Start ignored: invalid Top_k: {top_k_input.value!r}", ui_log)
                notify_user("Invalid Top_k!", type="warning")
                return
                        
            if not shard_balance_input.value:
                _shard_balance = settings.LLAMA_PARAM['tensorsplit']
            else:
                _shard_balance = shard_balance_input.value

            pattern = r"\d+(?:\.\d+)?(?:,\d+(?:\.\d+)?)+"
            
            if not re.match(pattern, _shard_balance):
                _shard_balance = settings.LLAMA_PARAM['tensorsplit']

            model_name = str(model_select.value)
            configured = settings.AVAILABLE_MODELS[model_name]
            started = await manager.start_server(model_name, configured, context_size, temperature, top_p, top_k, _shard_balance, mmproj_select.value)

            if started:
                chat_url = await get_browser_based_llama_url()
                emit(f"Chat URL: {chat_url}", ui_log)
                open_chat_dialog(model_name, chat_url)

        ui.button("Launch Model", on_click=start_selected_model, icon="play_arrow").classes("mt-4")

    async def clear_log() -> None:
        log_area.clear()

    with ui.row().classes("w-full gap-4 mt-4 items-end"):
        ui.label("Server Logs").classes("flex-1")
        clear_log = ui.button("Clear Logs", on_click=clear_log, icon="delete").classes("mt-4")
    log_area = ui.log().classes("w-full h-96 font-mono text-xs bg-black text-green-400")

ui.timer(0.5, detect_existing_llama_server, once=True)

emit("GUI loaded", None)
emit(f"Models directory: {settings.MODEL_BASE_DIR}", None)
emit(f"Available models: {len(settings.AVAILABLE_MODELS)}", None)
emit(f"NiceGUI listening on http://{settings.UI_HOST}:{settings.UI_PORT}", None)

#_____________________________________________________________________________
ui.run(
    title=settings.UI_TITLE,
    host=settings.UI_HOST,
    port=settings.UI_PORT,
)
