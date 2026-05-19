from __future__ import annotations

import shlex
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from object_models import Model
import re
import time
from nicegui import ui

import devices

from llama_command import get_llama_command
from config_manager import get_settings
from logging_utils import emit, setup_console_logging
import model_utils
import utils
import persist

settings = get_settings()

logger = setup_console_logging()

LLAMA_READY_LOG_MARKERS = (
    "server is listening on",
    "all slots are idle",
)

#_____________________________________________________________________________
def notify_user(message: str, *, type: str = "info") -> None:
    try:
        ui.notify(message, type=type, timeout=0, close_button=True)
    except TypeError:
        ui.notify(message, type=type, timeout=0)

#_____________________________________________________________________________
def is_llama_ready_log_line(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in LLAMA_READY_LOG_MARKERS)

#_____________________________________________________________________________
def ui_log(message: str) -> None:
    try:
        log_area.push(str(message))
    except RuntimeError as exc:
        if "client this element belongs to has been deleted" in str(exc):
            return
        raise

def update_data_from_modelname( modelname: str ) -> None:
    update_data_from_model( model_utils.get_model_by_name( modelname ) )

#_____________________________________________________________________________
def update_data_from_model( M: Model ) -> None:
    
    #M = model_utils.get_model_by_name( modelname )

    if M:
        context_select.set_value( M.ctxsize )
        temperature_select.set_value(f"{float(M.temperature):.1f}")
        top_p_input.set_value(f"{float(M.top_p):.1f}")
        top_k_input.set_value(f"{int(M.top_k)}")
    else:
        context_select.set_value( settings.DEFAULT_CONTEXT_SIZE )
        temperature_select.set_value(f"{float(settings.DEFAULT_TEMP):.1f}")
        top_p_input.set_value(f"{float(settings.DEFAULT_TOP_P):.1f}")
        top_k_input.set_value(f"{int(settings.DEFAULT_TOP_K)}")

#_____________________________________________________________________________
def refresh_model_list() -> None:
    models = model_utils.get_available_model_names( refresh = False ) # refresh has been already done in the previous call

    selected_model = model_utils.get_last_started_model()
    model_select.set_options(models, value=selected_model.model_name)

    update_data_from_model( selected_model )
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
def probe_existing_llama_server_sync() -> tuple[bool, Optional[str], Optional[str], Optional[str]]:
    last_error: Optional[str] = None

    url = f"http://{settings.LLAMA_SERVER.bindaddress}:{settings.LLAMA_SERVER.tcpport}/v1/models"
    try:
        payload = _json_get(url)
        model = (payload)
        return True, model, url, None
    except (HTTPError, URLError, TimeoutError, ConnectionError, json.JSONDecodeError, OSError) as exc:
        last_error = f"{url}: {exc}"

    return False, None, None, last_error

#_____________________________________________________________________________
async def detect_existing_llama_server(*, verbose: bool = True) -> bool:
    status_label.set_text("llama-server status: checking...")
    status_detail_label.set_text(f"Probe target: local port {settings.LLAMA_SERVER.tcpport}")

    running, detected_model, url, error = await asyncio.to_thread(probe_existing_llama_server_sync)

    if running:
        display_model = detected_model if detected_model else "unknown model"
        
        chat_url = await get_browser_based_llama_url()

        status_label.set_text("llama-server status: already running")
        status_detail_label.set_text(
            f"Detected endpoint: {chat_url} | Model: {display_model} "
        )
        set_link_target(status_chat_link, chat_url)
        status_chat_link.visible = True
        status_chat_button.visible = True

        emit(f"Detected already-running llama-server at {chat_url}", ui_log)
        emit(f"Detected model: {display_model}", ui_log)
        emit(f"Chat URL: {chat_url}", ui_log)
        return True

    status_label.set_text("llama-server status: not detected")
    status_detail_label.set_text(f"No server answered on local port {settings.LLAMA_SERVER.tcpport}")
    status_chat_link.visible = False
    status_chat_button.visible = False
    if verbose:
        emit(f"No existing llama-server detected. Last probe error: {error}", ui_log)
    return False

#_____________________________________________________________________________
async def get_browser_based_llama_url() -> str:
    port = settings.LLAMA_SERVER.tcpport
    js = f"""
        (() => {{
            const hostname = window.location.hostname;
            return `http://${{hostname}}:{port}/`;
        }})()
    """
    url = str(await ui.run_javascript(js))
    if settings.LLAMA_SERVER.bindaddress not in url:
        url=url.replace('http','https')
        url=url.replace(f':{settings.LLAMA_SERVER.tcpport}','')
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
                           shard_balance: str | None = None ) -> bool:
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
        emit(f"RPC server(s)  : {",".join( utils.get_all_rpc_servers() )}", ui_log)
        emit(f"Selected model : {M.model_name}", ui_log)
        emit(f"Configured path: {str(M.model_path)}", ui_log)
        emit(f"Context size   : {M.ctxsize}", ui_log)
        emit(f"Temperature    : {M.temperature}", ui_log)
        emit(f"Top_p          : {M.top_p}", ui_log)
        emit(f"Top_k          : {M.top_k}", ui_log)
        emit(f"Sharding       : shard_balance", ui_log)
        emit(f"Load mmproj    : {load_mmproj}", ui_log)
        if M.mmproj_path and load_mmproj:
            emit(f"MMProj file    : {str(M.mmproj_path)}", ui_log)
        emit(f"-----------------------------", ui_log)
        
        try:
            if run_local_only:
                gpus = devices.list_local_usable_devices(settings.LLAMA_SERVER, ui_log)
            else:
                gpus = devices.list_remote_usable_devices(settings.RPC_SERVERS, ui_log)
        except Exception as exc:
            msg = f"Device discovery failed: {exc}"
            emit(msg, ui_log)
            notify_user(msg, type="negative")
            status_label.set_text("llama-server status: device discovery failed")
            status_detail_label.set_text(str(exc))
            return False

        try:
            cmd = await asyncio.to_thread(
                get_llama_command,
                M,
                ui_log,
                run_local_only=run_local_only,
                load_mmproj=load_mmproj,
                gpus=gpus,
            )

            cmd = [str(arg) for arg in cmd]
            logger.info(f"DEBUG = cmd={cmd}")
            emit(f"-> Launching command: {" ".join(shlex.quote(str(x)) for x in cmd)}", ui_log)
            emit("->", ui_log)
            emit("->", ui_log)
            emit("-> Follows the llama-server stdout ", ui_log)
            emit("->", ui_log)
            emit("->", ui_log)
            
            #self.process = await asyncio.create_subprocess_exec(
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

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
                mmproj_path=str(M.mmproj_path),
                ctxsize=M.ctxsize,
                temperature=M.temperature,
                top_p=M.top_p,
                top_k=M.top_k,
                shard_balance=shard_balance,
                last_started=0,
            )
            #self._reader_task = asyncio.create_task(self._read_process_output(M, self.process))
            self._reader_task = asyncio.create_task(self._read_process_output(M, process))

            await asyncio.sleep(0.5)
            #if self.process.returncode is not None:
            if process.returncode is not None:
                #emit(f"llama-server exited immediately with return code {self.process.returncode}", ui_log)
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

            if self.process.returncode is not None:
                #emit(f"llama-server exited before readiness completed with return code {self.process.returncode}", ui_log)
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
                    emit(f"[llama-server] {text}", ui_log)
                    if self.process is process and self._ready_event is not None and not self._ready_event.is_set():
                        if is_llama_ready_log_line(text):
                            self._ready_reason = text
                            self._ready_event.set()
                            emit(f"llama-server readiness confirmed by log line: {text}", ui_log)
                            model_persist_data = {
                                "context_size": context_select.value,#M.ctxsize,
                                "temperature": temperature_select.value,#M.temperature,
                                "top_p": top_p_input.value,#M.top_p,
                                "top_k": top_k_input.value,#M.top_k,
                                "shard_balance": M.shard_balance,#ask_shard_balance, #M.shard_balance,
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

        port = settings.LLAMA_SERVER.tcpport
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
async def ping_servers() -> dict[str, float]:
    ping_results = {}
    
    servers_to_ping = []
    
    # Add RPC servers
    for server in settings.RPC_SERVERS + [settings.LLAMA_SERVER]:
        servers_to_ping.append(server.hostname)
    
    # Ping each server
    for server in servers_to_ping:
        try:
            # Use a timeout to avoid hanging
            ping_time = await asyncio.wait_for(
                asyncio.to_thread(utils.ping, server), 
                timeout=5.0
            )
            ping_results[server] = ping_time if ping_time is not None else float('inf')
        except asyncio.TimeoutError:
            ping_results[server] = float('inf')
        except Exception as e:
            logger.error(f"Error pinging {server}: {e}")
            ping_results[server] = float('inf')
    
    return ping_results

#_____________________________________________________________________________
async def update_ping_status() -> None:
    ping_results = await ping_servers()
    
    for label in ping_labels.values():
        label.set_text("Pinging...")
    
    for hostname, ping_time in ping_results.items():
        if hostname in ping_labels:
            if ping_time == float('inf'):
                ping_labels[hostname].set_text("Ping failed")
            else:
                ping_labels[hostname].set_text(f"{ping_time:.2f} ms")

#_____________________________________________________________________________
async def refresh_ping_status() -> None:
    """Refresh ping status for all servers."""
    await update_ping_status()

#_____________________________________________________________________________
ping_labels: dict[str, ui.label] = {}

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

    ####################################
    with ui.card().classes("w-full p-4"):
        ui.label("Server Ping Status").classes("font-bold")

        for server in settings.RPC_SERVERS + [settings.LLAMA_SERVER]:
            with ui.row().classes("items-center gap-2"):
                ui.label(f"{server.hostname} ({server.type.value}):").classes("w-40")
                ping_labels[server.hostname] = ui.label("Pinging...").classes("font-mono")

        def _schedule_ping_refresh() -> None:
            """Schedule a single execution of `refresh_ping_status`."""
            asyncio.create_task(refresh_ping_status())

        ui.timer(0.2, _schedule_ping_refresh, once=True)
        ui.timer(20.0, _schedule_ping_refresh)
    ####################################

    with ui.card().classes("w-full p-4"):
        ui.label("Select a model").classes("text-subtitle1 font-bold")

        with ui.row().classes("w-full gap-4 mt-4 items-end"):

            model_select = ui.select(
                options=model_utils.get_available_model_names( refresh = False ),#available_model_names(),#next(iter(available_model_names()), None),
                value=model_utils.get_last_started_model( ).model_name,
                label="Select a model from the list below...",
                on_change=lambda e: update_data_from_modelname( e.value ),
            ).classes("flex-1")

            model_list_refresh = ui.button("Refresh List", on_click=refresh_model_list, icon="refresh").classes("mt-4")

        with ui.row().classes("w-full gap-4 mt-4 items-end"):
            #ctx, temp, top_p, top_k, shard_balance = selected_data_for_model( next(iter(available_model_names()), None) )
            M = model_utils.get_model_by_name(model_select.value)
            context_select = ui.select(
                options=utils.configured_context_options(),
                value=M.ctxsize,#ctx,#utils.normalize_context_size_for_select( ctx ),
                label="Context size (0 = auto)",
            ).classes("flex-[2]")

            temperature_select = ui.select(
                options=[f"{i / 10:.1f}" for i in range(1, 11)],
                value=f"{float(M.temperature):.1f}",
                label="Temperature",
            ).classes("flex-[1]")

            top_p_input = ui.input(
                value=M.top_p,#"0.9",
                label="Top_p",
            ).classes("flex-[1]")
            
            top_k_input = ui.input(
                value=M.top_k,#"40",
                label="Top_k",
            ).classes("flex-[1]")

        mmproj_select = ui.checkbox('Load MM Projector if available', value=False).classes("flex-[1]")

        run_local_only_checkbox = ui.checkbox(
            "Run local only (no --rpc flag)",
            value=False,
        ).classes("flex-[1] mt-2")

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
                        
            #run_local_only = 

            model_name_for_default = str(model_select.value) if model_select.value else None
            #_ctx, _temp, _top_p, _top_k, persisted_shard_balance = selected_data_for_model(model_name_for_default)
            M = model_utils.get_model_by_name( model_name_for_default )
            _shard_balance = str(M.shard_balance or settings.DEFAULT_SHARD_BALANCE)

            if not bool(run_local_only_checkbox.value):
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

            m = model_utils.get_model_by_name(str(model_select.value))
            started = await manager.start_server(m, bool(mmproj_select.value), bool(run_local_only_checkbox.value), _shard_balance )

            if started:
                chat_url = await get_browser_based_llama_url()
                emit(f"->", ui_log)
                emit(f"->", ui_log)
                emit(f"-> Chat URL: {chat_url}", ui_log)
                open_chat_dialog(m.model_name, chat_url)

        ui.button("Launch Model", on_click=start_selected_model, icon="play_arrow").classes("mt-4")

    async def clear_log() -> None:
        log_area.clear()

    with ui.row().classes("w-full gap-4 mt-4 items-end"):
        ui.label("Server Logs").classes("flex-1 font-bold")
        clear_log = ui.button("Clear Logs", on_click=clear_log, icon="delete").classes("mt-4")
    
    log_area = (
        ui.log()
        .classes("w-full h-96 font-mono text-xs bg-black text-green-400 custom-log-scrollbar")
        .style("overflow: auto; white-space: pre;")
    )

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


ui.timer(0.5, detect_existing_llama_server, once=True)

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
