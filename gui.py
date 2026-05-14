from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import re
from nicegui import ui

from launcher import get_llama_command, format_command
from config_manager import get_settings, RpcServer
from logging_utils import emit, setup_console_logging
import model_utils
import utils
from persist import JsonParams
import rpc

settings = get_settings()

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
# def available_model_names() -> list[str]:
#     """Return model names discovered/configured for the model combo box."""
#     return sorted(model_utils.AVAILABLE_MODELS.keys(), key=str.lower)
def available_model_names() -> list[str]:
    model_utils.AVAILABLE_MODELS = model_utils.discover_available_models(settings.MODEL_BASE_DIR)
    return sorted(model_utils.AVAILABLE_MODELS.keys(), key=str.lower)

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
        emit(f"Ignoring invalid persisted data for {model_name!r}: {persisted[model_name]!r}", None)
        return None

#_____________________________________________________________________________
def selected_data_for_model(model_name: Optional[str]) -> tuple[int, float, float, int, str]:
    """Return persisted ctxsize when available, otherwise configured/default ctxsize."""
    persisted_data = persisted_data_for_model(model_name)
    if persisted_data is not None:
        return persisted_data['context_size'], persisted_data['temperature'], persisted_data['top_p'], persisted_data['top_k'], persisted_data['shard_balance']
    return model_utils.default_context_size_for_model(model_name), model_utils.default_temp_for_model(model_name), model_utils.default_top_p_for_model( model_name ), model_utils.default_top_k_for_model( model_name ), model_utils.default_shard_balance_for_model( model_name )



#_____________________________________________________________________________
def update_data_from_model() -> None:
    """Set context combo box from persist.json, then config/default fallback."""
    model_name = str(model_select.value) if model_select.value else None
    ctx, temp, top_p, top_k, _shard_balance = selected_data_for_model(model_name)
    #ctx = utils.normalize_context_size_for_select(ctx)
    context_select.set_value( ctx )
    temperature_select.set_value(f"{float(temp):.1f}")
    top_p_input.set_value(f"{float(top_p):.1f}")
    top_k_input.set_value(f"{int(top_k)}")

#_____________________________________________________________________________
def refresh_model_list() -> None:
    """Reload available models and refresh the model select widget."""
    current_model = str(model_select.value) if model_select.value else None
    model_names = available_model_names()

    if current_model in model_names:
        selected_model = current_model
    else:
        selected_model = next(iter(model_names), None)

    model_select.set_options(model_names, value=selected_model)

    if selected_model:
        update_data_from_model()
        emit(f"Model list refreshed: {len(model_names)} models found", ui_log)
        notify_user(f"Model list refreshed: {len(model_names)} models found", type="positive")
    else:
        emit("Model list refreshed: no models found", ui_log)
        notify_user("Model list refreshed: no models found", type="warning")

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
def _match_configured_model(detected_model: str) -> str:
    """Try to map llama-server reported model string to one model_utils.AVAILABLE_MODELS key."""
    detected = detected_model.strip()
    detected_path = Path(detected)
    detected_name = detected_path.name
    detected_stem = detected_path.stem

    for logical_name, configured in model_utils.AVAILABLE_MODELS.items():
        try:
            main_path = model_utils.configured_model_path(configured)
        except TypeError as exc:
            emit(f"Skipping invalid model_utils.AVAILABLE_MODELS entry {logical_name!r}: {exc}", None)
            continue

        configured_path = Path(main_path).expanduser()
        folder = model_utils.path_to_model_folder(main_path)
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
    last_error: Optional[str] = None

    for endpoint, extractor in (
        ("/v1/models", model_utils.extract_model_from_openai_models),
        ("/props", model_utils.extract_model_from_props),
    ):
        url = f"{settings.LLAMA_SERVER_BASEURL}{endpoint}"
        try:
            payload = _json_get(url)
            model = extractor(payload)
            return True, model, settings.LLAMA_SERVER_BASEURL, None
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
                           load_mmproj: bool,
                           run_local_only: bool = False,) -> bool:
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
            configured_path = model_utils.configured_model_path(configured)
            model_folder = model_utils.path_to_model_folder(configured_path)
        except Exception as exc:
            msg = f"Invalid model configuration for {model_name}: {exc}"
            emit(msg, ui_log)
            status_label.set_text("llama-server status: invalid model configuration")
            status_detail_label.set_text(str(exc))
            notify_user(msg, type="negative")
            return False

        all_endpoints = utils.get_all_rpc_servers()
        all_rpc = ",".join(all_endpoints)

        emit("--- Start requested ---", ui_log)
        emit(f"Run local      : {run_local_only}", ui_log)
        emit(f"RPC server(s)  : {all_rpc}", ui_log)
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
                model_folder,
                ui_log,
                run_local_only=run_local_only,
                tensorsplit=shard_balance,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
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
async def ask_shard_balance(default_value: str) -> str | None:
    """Ask for tensor split/shard balance only when RPC execution is enabled."""
    result: dict[str, str | None] = {"value": None}
    done = asyncio.Event()

    with ui.dialog() as dialog, ui.card().classes("w-full max-w-md"):
        ui.label("Shard balance").classes("text-h6")
        ui.label("Insert tensor split values, for example: 6,12").classes("text-sm text-gray-600")
        shard_input = ui.input(
            label="Shard balance",
            value=default_value,
            placeholder="6,12",
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
                options=available_model_names(),#next(iter(available_model_names()), None),
                value=next(iter(available_model_names()), None),
                label="Select a model from the list below...",
                on_change=lambda _: update_data_from_model(),
            ).classes("flex-1")

            model_list_refresh = ui.button("Refresh List", on_click=refresh_model_list, icon="refresh").classes("mt-4")

        with ui.row().classes("w-full gap-4 mt-4 items-end"):
            ctx, temp, top_p, top_k, shard_balance = selected_data_for_model( next(iter(available_model_names()), None) )
            context_select = ui.select(
                options=utils.configured_context_options(),
                value=ctx,#utils.normalize_context_size_for_select( ctx ),
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

        mmproj_select = ui.checkbox('Load MM Projector if available', value=False).classes("flex-[1]")

        run_local_only_checkbox = ui.checkbox(
            "Run local only (no --rpc flag)",
            value=False,
        ).classes("flex-[1] mt-2")

        # rpc_server_input = ui.input(
        #     label="RPC server(s) – IP:porta (es. 192.168.1.10:12345,10.0.0.5:12345)",
        #     placeholder="192.168.1.10:12345,10.0.0.5:12345",
        #     value="",
        # ).classes("w-1/2")

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
                        
            run_local_only = bool(run_local_only_checkbox.value)

            model_name_for_default = str(model_select.value) if model_select.value else None
            _ctx, _temp, _top_p, _top_k, persisted_shard_balance = selected_data_for_model(model_name_for_default)
            _shard_balance = str(persisted_shard_balance or settings.DEFAULT_SHARD_BALANCE)

            if not run_local_only:
                requested_shard_balance = await ask_shard_balance(_shard_balance)
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

                
            model_name = str(model_select.value)
            configured = model_utils.AVAILABLE_MODELS[model_name]
            started = await manager.start_server(model_name, 
                                                 configured, 
                                                 context_size, 
                                                 temperature, 
                                                 top_p, top_k, 
                                                 _shard_balance, 
                                                 mmproj_select.value,
                                                 run_local_only,)

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
emit(f"Available models: {len(model_utils.AVAILABLE_MODELS)}", None)
emit(f"NiceGUI listening on http://{settings.UI_HOST}:{settings.UI_PORT}", None)

#_____________________________________________________________________________
ui.run(
    title=settings.UI_TITLE,
    host=settings.UI_HOST,
    port=settings.UI_PORT,
)
