from __future__ import annotations

import re
import json
import time
import shlex
import logzero
import asyncio
import subprocess
from pathlib import Path
from logzero import logger
from nicegui import app, ui
from typing import Any, Optional
from collections.abc import Callable
from gguf import GGUFReader, GGUFValueType
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

import utils
import persist
import model_utils
from object_models import Model
from config_manager import get_settings
from llama_command import get_llama_command

logzero.loglevel(logzero.DEBUG)

settings = get_settings()

LogSink = Optional[Callable[[str], None]]

#_____________________________________________________________________________
LLAMA_READY_LOG_MARKERS = (
    "server is listening on",
    "all slots are idle",
)

#_____________________________________________________________________________
def emit(message: str, sink: LogSink = None) -> None:
    logger.info(message)

    if sink is not None:
        try:
            sink(message)
        except Exception:  # pragma: no cover – defensive, should never crash the app
            logger.error("Unable to write message to UI log sink")

#_____________________________________________________________________________
def notify_user(message: str, *, type: str = "info") -> None:
    logger.info(message)
    ui.notify(message, type=type, timeout=15000, close_button=True)

#_____________________________________________________________________________
def _gguf_value(field) -> Any:
    """Estrae uno scalare/stringa da un ReaderField, robusto rispetto alla
    versione di `gguf` (le versioni recenti espongono .contents())."""
    if field is None:
        return None
    contents = getattr(field, "contents", None)
    if callable(contents):
        try:
            return field.contents()
        except Exception:
            pass
    if not field.parts or not field.data:
        return None
    part = field.parts[field.data[-1]]
    if field.types and field.types[0] == GGUFValueType.STRING:
        return bytes(part).decode("utf-8", errors="replace")
    return part[0] if len(part) else None

#_____________________________________________________________________________
def read_gguf_trained_context_length(model_path: str) -> Optional[int]:
    """Training Context length read from .gguf, or None.
    Key: '<general.architecture>.context_length'."""
    try:
        reader = GGUFReader(model_path, mode="r")
    except Exception as exc:
        emit(f"GGUFReader: cannot open {model_path}: {exc}", ui_log)
        return None

    arch = _gguf_value(reader.fields.get("general.architecture"))
    if not arch:
        emit(f"general.architecture assente in {model_path}", ui_log)
        return None

    ctx = _gguf_value(reader.fields.get(f"{arch}.context_length"))
    if ctx is None:
        emit(f"{arch}.context_length assente in {model_path}", ui_log)
        return None
    try:
        return int(ctx)
    except (TypeError, ValueError):
        emit(f"Unexpected context_length value: {ctx!r}", ui_log)
        return None

#_____________________________________________________________________________
async def update_trained_ctx_label(modelname: Optional[str]) -> None:
    M = model_utils.get_model_by_name(modelname) if modelname else None
    if M is None or not M.model_path:
        trained_ctx_label.set_text("Trained context: —")
        return

    trained_ctx_label.set_text("Trained context: …")  # feedback immediato
    n = await asyncio.to_thread(read_gguf_trained_context_length, str(M.model_path))

    # Applica il risultato solo se nel frattempo la selezione non è cambiata.
    if model_select.value != modelname:
        return

    trained_ctx_label.set_text(
        f"Trained context: {n:,} tokens" if n is not None else "Trained context: unknown"
    )

#_____________________________________________________________________________
def is_llama_ready_log_line(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in LLAMA_READY_LOG_MARKERS)


LLAMA_LOG_NOISE_MARKERS = (
    "update_slots: all slots are idle",
)

#_____________________________________________________________________________
def is_llama_log_noise(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in LLAMA_LOG_NOISE_MARKERS)

#_____________________________________________________________________________
# Shared, UI-agnostic log buffer. The llama-server reader task only APPENDS
# here; rendering to the browser is done by a per-client ui.timer created
# inside the @ui.page('/') function. This is what makes the log survive a
# browser reload: on reload the page is rebuilt and the buffer is replayed.
LOG_BUFFER: list[str] = []
LOG_DROPPED: int = 0            # lines trimmed from the front of LOG_BUFFER
MAX_LOG_LINES: int = 5000       # cap to bound memory

detected_devices: list[str] = []   # populated by "List devices" button

#_____________________________________________________________________________
def ui_log(message: str) -> None:
    """Append a log line to the shared buffer (no direct UI access)."""
    global LOG_DROPPED
    LOG_BUFFER.append(str(message))
    overflow = len(LOG_BUFFER) - MAX_LOG_LINES
    if overflow > 0:
        del LOG_BUFFER[:overflow]
        LOG_DROPPED += overflow

#_____________________________________________________________________________
def update_data_from_modelname( modelname: str ) -> None:
    update_data_from_model( model_utils.get_model_by_name( modelname ) )

#_____________________________________________________________________________
def update_data_from_model( M: Model ) -> None:

    if M is None:
        context_select.set_value(settings.DEFAULT_CONTEXT_SIZE)
        temperature_select.set_value(f"{float(settings.DEFAULT_TEMP):.1f}")
        top_p_input.set_value(f"{float(settings.DEFAULT_TOP_P):.1f}")
        top_k_input.set_value(f"{int(settings.DEFAULT_TOP_K)}")
        return
    try:
        all_persisted_params = persist.get_params_handler().load_params()
        persisted = all_persisted_params.get(M.model_name, {})
    except Exception as exc:
        emit(f"Could not load persisted parameters: {exc}", ui_log)
        persisted = {}
    context_select.set_value(
        persisted.get("context_size", M.ctxsize)
    )
    temperature_select.set_value(
        f"{float(persisted.get('temperature', M.temperature)):.1f}"
    )
    top_p_input.set_value(
        str(persisted.get("top_p", M.top_p))
    )
    top_k_input.set_value(
        str(persisted.get("top_k", M.top_k))
    )
    if persisted:
        emit(f"Loaded persisted parameters for model: {M.model_name}", ui_log)
    else:
        emit(f"No persisted parameters for model: {M.model_name}; using model defaults", ui_log)

#_____________________________________________________________________________
async def refresh_model_list() -> None:
    models = model_utils.get_available_model_names( refresh = True ) or []

    selected_model = model_utils.get_last_started_model()
    selected_name = selected_model.model_name if selected_model else None
    safe_value = (
        selected_name
        if selected_name in models
        else (models[0] if models else None)
    )
    model_select.set_options(models, value=safe_value)
    if safe_value:
        update_data_from_modelname(safe_value)
        await update_trained_ctx_label(safe_value)
    else:
        update_data_from_model(None)
        await update_trained_ctx_label(None)

    emit(f"Model list refreshed: {len(models)} models found", ui_log)
    notify_user(f"Model list refreshed: {len(models)} models found", type="positive")

#_____________________________________________________________________________
def _run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
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
def find_listening_pids_on_port(port: int) -> list[int] | None:
    try:
        lsof = _run_command(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"])
        pids = _parse_pid_lines(lsof.stdout)
        if pids:
            return pids
    except Exception as exc:
        emit(f"lsof lookup failed: {exc}", ui_log)
    return

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
def _http_get_text(url: str, timeout: float = 1.5) -> Optional[str]:
    """Blocking text GET; returns None on any failure (no exceptions raised).
    Run it via asyncio.to_thread()."""
    req = Request(url, headers={"Accept": "text/plain"})
    try:
        with urlopen(req, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, TimeoutError, ConnectionError, OSError):
        return None

#_____________________________________________________________________________
def _extract_n_decoded(slot: dict) -> Optional[int]:
    """Token generati finora dalla richiesta in corso su uno slot.
    Struttura osservata: slot['next_token'][0]['n_decoded']."""
    nt = slot.get("next_token")
    if isinstance(nt, list) and nt:
        nt = nt[0]
    if isinstance(nt, dict) and isinstance(nt.get("n_decoded"), int):
        return nt["n_decoded"]
    # fallback: alcune versioni espongono n_decoded sullo slot
    if isinstance(slot.get("n_decoded"), int):
        return slot["n_decoded"]
    return None

#_____________________________________________________________________________
def fetch_active_slot_progress_sync() -> Optional[tuple[int, int]]:
    """Ritorna (id_task, n_decoded) del primo slot in elaborazione, oppure None
    se nessuno sta generando / server non raggiungibile.
    Bloccante: chiamare via asyncio.to_thread().
    Richiede l'endpoint /slots abilitato in llama-server."""
    url = f"http://{settings.LLAMA_SERVER_BIND}:{settings.LLAMA_SERVER_PORT}/slots"
    text = _http_get_text(url)
    if not text:
        return None
    try:
        slots = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(slots, list):
        return None
    for slot in slots:
        if not isinstance(slot, dict) or not slot.get("is_processing"):
            continue
        n_decoded = _extract_n_decoded(slot)
        id_task = slot.get("id_task")
        if isinstance(n_decoded, int) and isinstance(id_task, int):
            return (id_task, n_decoded)
    return None

#_____________________________________________________________________________
def probe_existing_llama_server_sync() -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
    last_error: Optional[str] = None

    url = f"http://{settings.LLAMA_SERVER_BIND}:{settings.LLAMA_SERVER_PORT}/v1/models"
    try:
        payload = _json_get(url)
        model = (payload['data'][0]['id'])
        return True, model, url, None
    except (HTTPError, URLError, TimeoutError, ConnectionError, json.JSONDecodeError, OSError, KeyError, IndexError) as exc:
        last_error = f"{url}: {exc}"

    return False, None, None, last_error

#_____________________________________________________________________________
async def detect_existing_llama_server(*, verbose: bool = True) -> bool:
    status_label.set_text("llama-server status: checking...")
    status_detail_label.set_text(f"Probe target: local port {settings.LLAMA_SERVER_PORT}")

    running, detected_model, url, error = await asyncio.to_thread(probe_existing_llama_server_sync)

    if running:
        display_model = detected_model if detected_model else "unknown model"

        chat_url = await get_browser_based_llama_url()

        status_label.set_text("llama-server status: already running")
        display_model_ = display_model.replace(".gguf", "")
        #ctx = persist.get_settings()
        all_persisted_params = persist.get_params_handler().load_params()
        persisted = all_persisted_params.get(display_model_)
        c = persisted.get("context_size")
        t = persisted.get("temperature")
        tp= persisted.get("top_p")
        tk= persisted.get("top_k")
        sb= persisted.get("shard_balance")
        status_detail_label.set_text(
            f"Detected endpoint: {chat_url} | Model: {display_model_} | ctx: {c} | temp: {t} | top_p: {tp} | top_k: {tk} | shard_balance: {sb}"
        )
        set_link_target(status_chat_link, chat_url)
        status_chat_link.visible = True
        status_chat_button.visible = True

        emit(f"Detected already-running llama-server at {chat_url}", ui_log)
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
    """Return the appropriate llama-server URL for the browser."""
    port = settings.LLAMA_SERVER_PORT
    js = f"""
        (() => {{
            const hostname = window.location.hostname;
            return `http://${{hostname}}:{port}/`;
        }})()
    """
    url = str(await ui.run_javascript(js))
    try:
        hostname = re.search(r"//([^/:]+)", url).group(1)
    except Exception:
        hostname = ""
    is_local = (
        hostname == "localhost"
        or hostname == "127.0.0.1"
        or re.match(r"^10(?:\.\d{1,3}){3}$", hostname)
        or re.match(r"^192\.168(?:\.\d{1,3}){2}$", hostname)
        or re.match(r"^172\.(?:1[6-9]|2[0-9]|3[0-1])(?:\.\d{1,3}){2}$", hostname)
    )
    if not is_local and settings.LLAMA_SERVER_BIND not in url:
        url = url.replace('http', 'https', 1)
        url = url.replace(f'{settings.LLAMA_SERVER_PORT}', '8443', count=1)
    return url

#_____________________________________________________________________________
def set_link_target(link: ui.link, url: str) -> None:
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
                           M: Model,
                           load_mmproj: bool,
                           run_local_only: bool = False,
                            ) -> bool:
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

        emit("------ Start requested ------", ui_log)
        emit(f"Run local      : {run_local_only}", ui_log)
        if settings.RPC_SERVERS:
            emit(f"RPC server(s)  : {settings.RPC_SERVERS}", ui_log)
        emit(f"Selected model : {M.model_name}", ui_log)
        emit(f"Configured path: {str(M.model_path)}", ui_log)
        emit(f"Context size   : {M.ctxsize}", ui_log)
        emit(f"Temperature    : {M.temperature}", ui_log)
        emit(f"Top_p          : {M.top_p}", ui_log)
        emit(f"Top_k          : {M.top_k}", ui_log)
        emit(f"Sharding       : {M.shard_balance}", ui_log)
        emit(f"Load mmproj    : {load_mmproj}", ui_log)
        if M.mmproj_path and load_mmproj:
            emit(f"MMProj file    : {str(M.mmproj_path)}", ui_log)
        if detected_devices:
            emit(f"Devices        : {', '.join(detected_devices)}", ui_log)
        else:
            emit(f"Devices        : {settings.GPUS} (from config GPUS)", ui_log)
        emit(f"-----------------------------", ui_log)

        try:
            cmd = await asyncio.to_thread(
                get_llama_command,
                M,
                run_local_only=run_local_only,
                load_mmproj=load_mmproj,
                devices=detected_devices if detected_devices else [g.strip() for g in settings.GPUS.split(",") if g.strip()],
            )

            logger.debug(f"Executing command: {cmd}")
            emit(f"-> Launching command: {" ".join(shlex.quote(str(x)) for x in cmd)}", ui_log)
            emit("->", ui_log)
            emit("->", ui_log)
            emit("-> Follows the llama-server stdout ", ui_log)
            emit("->", ui_log)
            emit("->", ui_log)

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            self.process = process

            status_label.set_text("llama-server status: starting")
            status_detail_label.set_text(
                f"Starting model: {M.model_name}; waiting for llama-server readiness log "
                "('server is listening on ...' or 'all slots are idle')"
            )
            notify_user(f"Starting {M.model_name}...", type="info")

            self._ready_event = asyncio.Event()
            self._ready_reason = None
            M = Model(
                model_name=M.model_name,
                model_path=str(M.model_path),
                mmproj_path=(str(M.mmproj_path) if M.mmproj_path else None),
                ctxsize=M.ctxsize,
                temperature=M.temperature,
                top_p=M.top_p,
                top_k=M.top_k,
                shard_balance=M.shard_balance,
                last_started=0,
            )

            self._reader_task = asyncio.create_task(self._read_process_output(M, process))

            await asyncio.sleep(0.5)

            if process.returncode is not None:
                emit(f"llama-server exited immediately with return code {process.returncode}", ui_log)
                status_label.set_text("llama-server status: failed")
                status_detail_label.set_text(f"Model {M.model_name} exited immediately")
                notify_user(f"{M.model_name} exited before becoming ready", type="negative")
                return False

            try:
                assert self._ready_event is not None
                await asyncio.wait_for(self._ready_event.wait(), timeout=settings.LLAMA_READY_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                msg = (
                    f"llama-server did not emit a readiness line within "
                    f"{settings.LLAMA_READY_TIMEOUT_SECONDS} seconds"
                )
                emit(msg, ui_log)
                status_label.set_text("llama-server status: starting, readiness not confirmed")
                status_detail_label.set_text(msg)
                notify_user(msg, type="warning")
                return False

            if process.returncode is not None:
                emit(f"llama-server exited before readiness completed with return code {process.returncode}", ui_log)
                status_label.set_text("llama-server status: failed")
                status_detail_label.set_text(f"Model {M.model_name} exited before readiness completed")
                notify_user(f"{M.model_name} exited before becoming ready", type="negative")
                return False

            chat_url = await get_browser_based_llama_url()
            status_label.set_text("llama-server status: running")
            status_detail_label.set_text(
                f"Started by this GUI | Model: {M.model_name} | Ready: {self._ready_reason or 'confirmed'}"
            )
            set_link_target(status_chat_link, chat_url)
            status_chat_link.visible = True
            status_chat_button.visible = True
            notify_user(f"{M.model_name} is ready", type="positive")
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
    async def _read_process_output(self,
                                   M: Model,
                                   process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None

        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                text = line.decode(errors="replace").rstrip()
                if text:
                    # La rilevazione di readiness deve girare anche sulle righe
                    # che poi sopprimiamo come rumore.
                    is_ready_line = (
                        self.process is process
                        and self._ready_event is not None
                        and not self._ready_event.is_set()
                        and is_llama_ready_log_line(text)
                    )

                    # Sopprime il rumore periodico del nostro polling di /slots
                    # ("update_slots: all slots are idle"), tranne quando quella
                    # riga è ciò che conferma la readiness allo startup.
                    if is_llama_log_noise(text) and not is_ready_line:
                        continue

                    emit(f"[llama-server] {text}", ui_log)

                    if is_ready_line:
                        self._ready_reason = text
                        self._ready_event.set()
                        emit(f"llama-server readiness confirmed by log line: {text}", ui_log)
                        model_persist_data = {
                            "context_size": context_select.value,
                            "temperature": temperature_select.value,
                            "top_p": top_p_input.value,
                            "top_k": top_k_input.value,
                            "shard_balance": M.shard_balance,
                            "last_started": int(time.time()),
                        }
                        persist.get_params_handler().save_param(M.model_name, model_persist_data)


            return_code = await process.wait()
            emit(f"llama-server exited with return code {return_code}", ui_log)
            status_label.set_text("llama-server status: stopped")
            status_detail_label.set_text(f"Last model: {M.model_name} | Return code: {return_code}")

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
        port = settings.LLAMA_SERVER_PORT
        if self.is_running():
            assert self.process is not None
            emit("Stopping GUI-started llama-server...", ui_log)
            status_label.set_text("llama-server status: stopping")
            self.process.terminate()
            notify_user( "Initiated Stopping llama-server", type="info" )
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
        killed, errors = await asyncio.to_thread(utils.kill_pids_sync, pids)

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

#_____________________________________________________________________________
async def ask_shard_balance(default_value: str) -> str | None:
    result: dict[str, str | None] = {"value": None}
    done = asyncio.Event()

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-md"):
        ui.label("Shard balance").classes("text-h6")
        ui.label(f"Insert tensor split values for GPUs: {settings.LOCAL_GPU},{settings.REMOTE_GPUS}").classes("text-sm text-gray-600")
        shard_input = ui.input(
            label="Shard balance",
            value=default_value,
            placeholder=settings.DEFAULT_SHARD_BALANCE,
        ).classes("w-full")

        def confirm() -> None:
            result["value"] = str(shard_input.value or "").strip()
            dialog.close()
            done.set()

        def cancel() -> None:
            result["value"] = None
            dialog.close()
            done.set()

        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=cancel)
            ui.button("OK", on_click=confirm, icon="check")

    dialog.open()
    await done.wait()
    return result["value"]

#_____________________________________________________________________________
# The whole UI lives inside @ui.page('/'): it is rebuilt for every browser
# (re)connection. That is the key fix — a reloaded tab is a NEW client, so the
# UI (including the log) is recreated and the log history is replayed from the
# shared LOG_BUFFER, while a per-client timer keeps tailing live output.
@ui.page('/')
def main_page() -> None:
    global status_label, status_detail_label, status_chat_link, status_chat_button
    global model_select, trained_ctx_label, context_select, temperature_select
    global top_p_input, top_k_input, mmproj_select, run_local_only_checkbox
    global log_area, chat_dialog, chat_model_label, chat_url_link

    ui.colors(primary="#6e93d6")

    ui.add_head_html("""
        <style>
        .custom-log-scrollbar,
        .custom-log-scrollbar * {
            scrollbar-width: thin;
            scrollbar-color: #22c55e #111827;
        }

        .custom-log-scrollbar::-webkit-scrollbar,
        .custom-log-scrollbar *::-webkit-scrollbar {
            width: 12px;
            height: 12px;
        }

        .custom-log-scrollbar::-webkit-scrollbar-track,
        .custom-log-scrollbar *::-webkit-scrollbar-track {
            background: #111827;
        }

        .custom-log-scrollbar::-webkit-scrollbar-thumb,
        .custom-log-scrollbar *::-webkit-scrollbar-thumb {
            background-color: #22c55e;
            border-radius: 8px;
            border: 2px solid #111827;
        }

        .custom-log-scrollbar::-webkit-scrollbar-corner,
        .custom-log-scrollbar *::-webkit-scrollbar-corner {
            background: #111827;
        }
        </style>
        """)

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

        if settings.RPC_SERVERS:
            with ui.card().classes("w-full p-4"):
                ui.label("RPC Servers").classes("text-subtitle1 font-bold")

                rpc_checkboxes: dict[str, ui.checkbox] = {}
                for host, cfg in settings.RPC_SERVERS.items():
                    cb_label = (
                        f"{host}"
                        f"  port={cfg.get('port', '?')}"
                        f"  disk={cfg.get('cachedisk', '?')}"
                        f"  type={cfg.get('type', 'darwin')}"
                    )
                    rpc_checkboxes[host] = ui.checkbox(cb_label, value=True)

                async def launch_rpc_servers() -> None:
                    selected = [h for h, cb in rpc_checkboxes.items() if cb.value]
                    if not selected:
                        notify_user("No RPC server selected", type="warning")
                        return
                    emit("------ Launch RPC servers ------", ui_log)
                    for host in selected:
                        cfg = settings.RPC_SERVERS[host]
                        remuser = cfg.get("remuser", "")
                        cachedisk = cfg.get("cachedisk", "")
                        type = cfg.get("type", "")
                        rpcserver = cfg.get("rpcserver", "")
                        rpccachepath = cfg.get("cachepath", "")
                        script = Path(__file__).parent / "scripts" / f"start_rpc_{type}.sh"
                        
                        rpcserver = rpcserver.replace("\\", "\\\\")
                        rpccachepath = rpccachepath.replace("\\", "\\\\")
                        emit(f"Launching: {script} {host} {remuser} {cachedisk} {rpcserver} {rpccachepath}")
                        try:
                            result = await asyncio.to_thread(
                                subprocess.run,
                                [str(script), host, remuser, cachedisk, rpcserver, rpccachepath],
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                timeout=30,
                            )
                            for line in (result.stdout or "").splitlines():
                                emit(f"[rpc:{host}] {line}", ui_log)
                            if result.returncode == 0:
                                emit(f"RPC server launched on {host} (exit 0)", ui_log)
                                notify_user(f"RPC server launched on {host}", type="positive")
                            else:
                                emit(f"RPC launch failed on {host} (exit {result.returncode})", ui_log)
                                notify_user(f"RPC launch failed on {host} (exit {result.returncode})", type="negative")
                        except Exception as exc:
                            emit(f"RPC launch error on {host}: {exc}", ui_log)
                            notify_user(f"RPC launch error on {host}: {exc}", type="negative")
                    emit("--------------------------------", ui_log)

                async def stop_rpc_servers() -> None:
                    selected = [h for h, cb in rpc_checkboxes.items() if cb.value]
                    if not selected:
                        notify_user("No RPC server selected", type="warning")
                        return
                    emit("------ Stop RPC servers ------", ui_log)
                    for host in selected:
                        cfg = settings.RPC_SERVERS[host]
                        remuser = cfg.get("remuser", "")
                        rpc_type = cfg.get("type", "posix")
                        stop_script = Path(__file__).parent / "scripts" / f"stop_rpc_{rpc_type}.sh"
                        emit(f"Stopping RPC on {host} (user: {remuser}, script: {stop_script.name})...", ui_log)
                        try:
                            result = await asyncio.to_thread(
                                subprocess.run,
                                [str(stop_script), host, remuser],
                                text=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT,
                                timeout=15,
                            )
                            for line in (result.stdout or "").splitlines():
                                emit(f"[rpc-stop:{host}] {line}", ui_log)
                            if result.returncode == 0:
                                emit(f"RPC server stopped on {host}", ui_log)
                                notify_user(f"RPC server stopped on {host}", type="positive")
                            else:
                                emit(f"RPC stop on {host} (exit {result.returncode}) — may not have been running", ui_log)
                                notify_user(f"RPC stop on {host}: exit {result.returncode}", type="warning")
                        except Exception as exc:
                            emit(f"RPC stop error on {host}: {exc}", ui_log)
                            notify_user(f"RPC stop error on {host}: {exc}", type="negative")
                    emit("------------------------------", ui_log)

                devices_label = ui.label("").classes("text-caption text-grey mt-1")

                async def list_devices() -> None:
                    global detected_devices
                    rpc_arg = ",".join(
                        f"{host}:{cfg.get('port', 50000)}"
                        for host, cfg in settings.RPC_SERVERS.items()
                    )
                    cmd = (
                        f"LD_LIBRARY_PATH=/usr/local/lib /usr/local/bin/llama-server"
                        f" --rpc {rpc_arg} --list-devices"
                        f" | grep -v '0 MiB free'"
                    )
                    emit(f"------ List devices ({rpc_arg}) ------", ui_log)
                    try:
                        result = await asyncio.to_thread(
                            subprocess.run,
                            cmd,
                            shell=True,
                            text=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            timeout=30,
                        )
                        lines = (result.stdout or "").splitlines()
                        for line in lines:
                            emit(line, ui_log)
                        found = [line.split(":")[0].strip() for line in lines if ":" in line]
                        detected_devices = found
                        devices_label.set_text(", ".join(found) if found else "no devices found")
                        if found:
                            emit(f"Detected devices: {', '.join(found)}", ui_log)
                    except Exception as exc:
                        emit(f"List devices error: {exc}", ui_log)
                        notify_user(f"List devices error: {exc}", type="negative")
                    emit("--------------------------------------", ui_log)

                with ui.row().classes("mt-4 gap-2"):
                    ui.button("Launch RPC servers", on_click=launch_rpc_servers, icon="dns")
                    ui.button("Stop RPC servers", on_click=stop_rpc_servers, icon="power_off", color="orange")
                    ui.button("List devices", on_click=list_devices, icon="devices")

        with ui.card().classes("w-full p-4"):
            ui.label("Select a model").classes("text-subtitle1 font-bold")

            with ui.row().classes("w-full gap-4 mt-4 items-end"):
                available_models = model_utils.get_available_model_names(refresh=False) or []
                last_started = model_utils.get_last_started_model()
                last_model_name = last_started.model_name if last_started else None
                initial_model_name = (
                    last_model_name
                    if last_model_name in available_models
                    else (available_models[0] if available_models else None)
                )

                async def _on_model_change(e) -> None:
                    update_data_from_modelname(e.value)
                    await update_trained_ctx_label(e.value)

                model_select = ui.select(
                    options=available_models,
                    value=initial_model_name,
                    label="Select a model from the list below..." if available_models else f"No model found in {settings.MODEL_BASE_DIR}",
                    #on_change=lambda e: (update_data_from_modelname(e.value), update_trained_ctx_label(e.value)),
                    on_change=_on_model_change
                ).classes("flex-1")

                trained_ctx_label = ui.label("Trained context: —").classes(
                    "text-base text-gray-600 self-stretch flex items-center whitespace-nowrap"
                ).classes("flex-1")
                #update_trained_ctx_label(initial_model_name)
                ui.timer(0.1, lambda: update_trained_ctx_label(initial_model_name), once=True)

                #model_list_refresh = ui.button("Refresh List", on_click=refresh_model_list, icon="refresh").classes("mt-4")
                model_list_refresh = ui.button("Refresh List", on_click=refresh_model_list, icon="refresh").classes("flex-1")

            with ui.row().classes("w-full gap-4 mt-4 items-end"):
                M = model_utils.get_model_by_name(model_select.value) if model_select.value else None
                context_select = ui.select(
                    options=utils.configured_context_options(),
                    value=M.ctxsize if M else settings.DEFAULT_CONTEXT_SIZE,
                    label="Context size (0 = auto)",
                ).classes("flex-[2]")

                temperature_select = ui.select(
                    options=[f"{i / 10:.1f}" for i in range(1, 11)],
                    value=f"{float(M.temperature if M else settings.DEFAULT_TEMP):.1f}",
                    label="Temperature",
                ).classes("flex-[1]")

                top_p_input = ui.input(
                    value=M.top_p if M else settings.DEFAULT_TOP_P,
                    label="Top_p",
                ).classes("flex-[1]")

                top_k_input = ui.input(
                    value=M.top_k if M else settings.DEFAULT_TOP_K,
                    label="Top_k",
                ).classes("flex-[1]")

            mmproj_select = ui.checkbox('Load MM Projector if available', value=False).classes("flex-[1]")
            label = "Run local only (no --rpc flag)"
                
            run_local_only_checkbox = ui.checkbox(
                label,
                value=False,
            ).classes("flex-[1] mt-2").set_enabled( settings.RPC_SERVERS is not None and settings.RPC_SERVERS!="" )

            if not settings.RPC_SERVERS or settings.RPC_SERVERS == "":
                run_local_only_checkbox.set_value( True )

            async def start_selected_model() -> None:

                if not model_select.value or model_select.value.strip()=="":
                    emit("Start ignored: no model selected", ui_log)
                    notify_user("Select a model!", type="warning")
                    return

                m = model_utils.get_model_by_name(str(model_select.value))
                if not m:
                    emit(f"Start ignored: model_utils.get_model_by_name({str(model_select.value)} returned None)", ui_log)
                    notify_user(f"Start ignored: model_utils.get_model_by_name({str(model_select.value)} returned None)", type="warning")
                    return

                if context_select.value is None:
                    emit("Start ignored: no context size selected", ui_log)
                    notify_user("Select a context size!", type="warning")
                    return
                try:
                    context_size = int(context_select.value)
                    digits = re.sub(r"\D", "", trained_ctx_label.text or "")
                    trained_ctx_size = int(digits) if digits else 0
                    if trained_ctx_size > 0 and context_size > trained_ctx_size:
                        mex = f"Start ignored: invalid context size > trained context size ({trained_ctx_size})"
                        emit(mex, ui_log)
                        notify_user(mex, type="warning")
                        return
                except (TypeError, ValueError):
                    emit(f"Start ignored: invalid context size: {context_select.value!r}", ui_log)
                    notify_user("Invalid context size!", type="warning")
                    return

                if top_p_input.value is None:
                    emit("Start ignored: no Top_p selected", ui_log)
                    notify_user("Input a Top_p between 0 and 1 (1 decimal digit)", type="warning")
                    return

                if top_k_input.value is None:
                    emit("Start ignored: no Top_k selected", ui_log)
                    notify_user("Input a Top_k integer between 20 and 100", type="warning")
                    return

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

                model_name_for_default = str(model_select.value) if model_select.value else None

                try:
                    all_persisted_params = persist.get_params_handler().load_params()
                    persisted = all_persisted_params.get(model_name_for_default, {})
                except Exception as exc:
                    emit(f"Could not load persisted shard balance: {exc}", ui_log)
                    persisted = {}

                default_shard_balance = str(
                    persisted.get("shard_balance")
                    or m.shard_balance
                    or settings.DEFAULT_SHARD_BALANCE
                )
                _shard_balance = default_shard_balance

                if not bool(run_local_only_checkbox.value):
                    requested_shard_balance = await ask_shard_balance(default_shard_balance)
                    if requested_shard_balance is None:
                        emit("Start cancelled: shard balance dialog closed", ui_log)
                        notify_user("Launch cancelled", type="warning")
                        return

                    pattern = r"^\d+(?:,\d+)+$"
                    if not re.match(pattern, requested_shard_balance):
                        emit(f"Invalid shard balance {requested_shard_balance!r}; using default {settings.DEFAULT_SHARD_BALANCE!r}", ui_log)
                        notify_user("Invalid shard balance; using default", type="warning")
                        _shard_balance = settings.DEFAULT_SHARD_BALANCE
                    else:
                        _shard_balance = requested_shard_balance

                effective_model = Model(
                    model_name=m.model_name,
                    model_path=str(m.model_path),
                    mmproj_path=(str(m.mmproj_path) if m.mmproj_path else None),
                    ctxsize=context_size,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    shard_balance=_shard_balance,
                    last_started=0,
                )

                started = await manager.start_server(
                    effective_model,
                    bool(mmproj_select.value),
                    bool(run_local_only_checkbox.value),
                )

                if started:
                    chat_url = await get_browser_based_llama_url()
                    emit(f"->", ui_log)
                    emit(f"->", ui_log)
                    emit(f"-> Chat URL: {chat_url}", ui_log)
                    open_chat_dialog(m.model_name, chat_url)

            ui.button("Launch Model", on_click=start_selected_model, icon="play_arrow").classes("mt-4")

        async def clear_log() -> None:
            global LOG_DROPPED
            LOG_BUFFER.clear()
            LOG_DROPPED = 0
            cursor["next"] = 0
            log_area.clear()

        with ui.row().classes("w-full gap-4 mt-4 items-end"):
            ui.label("Server Logs").classes("flex-1 font-bold")
            # Tokens-per-second label, aggiornata via polling dell'endpoint
            # /metrics di llama-server (vedi _update_tps qui sotto).
            tps_label = ui.label("# t/s: —").classes("flex-none font-mono text-sm")
            clear_log_button = ui.button("Clear Logs", on_click=clear_log, icon="delete").classes("mt-4")

            # Stato per il calcolo del rate live (per-client): t/s come
            # Δn_decoded/Δt tra due poll consecutivi dello stesso task.
            tps_state: dict[str, Any] = {"task": None, "n": None, "t": None, "last": None}

            async def _update_tps() -> None:
                try:
                    prog = await asyncio.to_thread(fetch_active_slot_progress_sync)
                except Exception:
                    prog = None

                now = time.monotonic()

                if prog is None:
                    tps_state["task"] = tps_state["n"] = tps_state["t"] = None
                    last = tps_state["last"]
                    tps_label.set_text("# t/s: —" if last is None else f"# t/s: {last:.1f} (idle)")
                    return

                task, n = prog

                if task != tps_state["task"] or tps_state["n"] is None or tps_state["t"] is None:
                    tps_state["task"], tps_state["n"], tps_state["t"] = task, n, now
                    last = tps_state["last"]
                    tps_label.set_text("# t/s: …" if last is None else f"# t/s: {last:.1f}")
                    return

                dn = n - tps_state["n"]
                dt = now - tps_state["t"]
                tps_state["n"], tps_state["t"] = n, now  # avanza la finestra

                if dt > 0 and dn >= 0:
                    rate = dn / dt
                    tps_state["last"] = rate
                    tps_label.set_text(f"# t/s: {rate:.1f}")

            tps_timer = ui.timer(2.0, _update_tps)

        log_area = (
            ui.log()
            .classes("w-full h-96 font-mono text-xs bg-black text-green-400 custom-log-scrollbar")
            .style("overflow: auto; white-space: pre;")
        )

    # --- Per-client log rendering -------------------------------------------
    # cursor["next"] is the absolute index (counting dropped lines) of the next
    # line this client must display. It starts at 0, so the first timer tick
    # replays the whole buffer (history restored after a reload); subsequent
    # ticks push only the newly appended lines (live tail). The timer is owned
    # by this client and auto-stops when the tab disconnects.
    cursor = {"next": 0}

    def _tail() -> None:
        # This timer belongs to THIS page's client. On reload/close NiceGUI keeps
        # the old client alive for a short grace period before deleting it, and
        # the orphaned timer can still fire once: log_area.push() then creates a
        # child Label on a dead client and raises RuntimeError. Catch it and stop
        # this dead timer; the fresh page has its own (live) timer.
        total = LOG_DROPPED + len(LOG_BUFFER)
        if cursor["next"] < LOG_DROPPED:        # fell behind the trim window
            cursor["next"] = LOG_DROPPED
        try:
            while cursor["next"] < total:
                log_area.push(LOG_BUFFER[cursor["next"] - LOG_DROPPED])
                cursor["next"] += 1             # advance per line -> no dupes on retry
        except RuntimeError:
            log_timer.deactivate()              # client gone: silence the orphan

    log_timer = ui.timer(0.2, _tail)
    ui.timer(0.5, detect_existing_llama_server, once=True)

    # Stop dei timer per-client alla disconnessione del tab (reload/chiusura),
    # PRIMA che NiceGUI distrugga gli slot della pagina. Evita il
    # RuntimeError "The parent slot of the element has been deleted." che un
    # timer orfano solleverebbe al tick successivo: l'eccezione nasce dentro il
    # loop del timer di NiceGUI (nell'ingresso del parent_slot), quindi un
    # try/except dentro la callback non può intercettarla.
    def _stop_client_timers() -> None:
        for t in (tps_timer, log_timer):
            try:
                t.delete()
            except Exception:
                pass

    ui.context.client.on_disconnect(_stop_client_timers)

@app.on_startup
def _log_startup() -> None:
    emit("GUI loaded", None)
    emit(f"Models directory: {settings.MODEL_BASE_DIR}", None)
    emit(f"Available models: {len(model_utils.get_available_model_names())}", None)
    emit(f"NiceGUI listening on http://{settings.UI_HOST}:{settings.UI_PORT}", None)


#_____________________________________________________________________________
ui.run(
    title=settings.UI_TITLE,
    host=settings.UI_HOST,
    port=settings.UI_PORT,
)
