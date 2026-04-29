from __future__ import annotations

import asyncio
import json
import ssl
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from nicegui import ui

from cli import get_llama_command
from config import Settings
from logging_utils import emit, setup_console_logging

settings = Settings()
logger = setup_console_logging()

def ui_log(message: str) -> None:
    """Write to NiceGUI log widget and to stdout logging."""
    log_area.push(message)


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


def path_to_model_folder(path_string: str | Path) -> Path:
    """Accept either a GGUF file path or a model directory path."""
    p = Path(path_string).expanduser()

    # Path.exists() is intentionally not required here because network volumes may be late-mounted.
    # If the configured string ends with .gguf, treat it as a model file and use its parent.
    if p.suffix.lower() == ".gguf":
        return p.parent.resolve()

    return p.resolve()


def _local_llama_base_urls() -> list[str]:
    """Return candidate local URLs for probing an already-running llama-server."""
    port = settings.llama_server_port
    return [
        f"https://127.0.0.1:{port}",
        f"https://localhost:{port}",
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
    ]


def _json_get(url: str, timeout: float = 2.0) -> dict[str, Any]:
    """Small blocking JSON GET helper. Run it via asyncio.to_thread()."""
    req = Request(url, headers={"Accept": "application/json"})

    # Self-signed certificates are expected in this project; this probe is local only.
    context = ssl._create_unverified_context() if url.startswith("https://") else None

    with urlopen(req, timeout=timeout, context=context) as response:
        raw = response.read().decode("utf-8", errors="replace")
        if not raw:
            return {}
        return json.loads(raw)


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


def _extract_model_from_props(payload: dict[str, Any]) -> Optional[str]:
    """Extract model id/path from llama.cpp /props response, if settings.AVAILAble."""
    for key in ("model_path", "model", "model_name", "model_alias"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


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
            except (HTTPError, URLError, TimeoutError, ConnectionError, json.JSONDecodeError, ssl.SSLError, OSError) as exc:
                last_error = f"{url}: {exc}"
                continue

    return False, None, None, last_error


async def detect_existing_llama_server(*, verbose: bool = True) -> bool:
    """Detect a llama-server already running before this GUI launched it.

    This updates permanent UI elements. It intentionally does not use ui.notify().
    """
    status_label.set_text("llama-server status: checking...")
    status_detail_label.set_text(f"Probe target: local port {settings.llama_server_port}")

    running, detected_model, base_url, error = await asyncio.to_thread(probe_existing_llama_server_sync)

    if running:
        display_model = _match_configured_model(detected_model) if detected_model else "unknown model"
        status_label.set_text("llama-server status: already running")
        status_detail_label.set_text(
            f"Detected endpoint: {base_url} | Model: {display_model} | "
            "Note: this process was not started by this GUI instance."
        )
        emit(f"Detected already-running llama-server at {base_url}", ui_log)
        emit(f"Detected model: {display_model}", ui_log)
        return True

    status_label.set_text("llama-server status: not detected")
    status_detail_label.set_text(f"No server answered on local port {settings.llama_server_port}")
    if verbose:
        emit(f"No existing llama-server detected. Last probe error: {error}", ui_log)
    return False


async def get_browser_based_llama_url() -> str:
    """Build the llama-server chat URL from the browser-visible GUI URL."""
    port = settings.llama_server_port
    js = f"""
        (() => {{
            const protocol = window.location.protocol || 'https:';
            const hostname = window.location.hostname;
            return `${{protocol}}//${{hostname}}:{port}/`;
        }})()
    """
    return str(await ui.run_javascript(js))


def open_chat_dialog(model_name: str, chat_url: str) -> None:
    chat_model_label.set_text(f"Model: {model_name}")
    chat_url_link.set_text(chat_url)
    chat_url_link.props(f'href="{chat_url}"')
    chat_dialog.open()


class LlamaManager:
    def __init__(self) -> None:
        self.process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

    async def start_server(self, model_name: str, configured: Any) -> bool:
        if self.is_running():
            msg = "llama-server is already running"
            emit(msg, ui_log)
            ui.notify(msg, type="warning")
            return False

        # Prevent accidental second instance if llama-server is already running outside this GUI process.
        if await detect_existing_llama_server(verbose=False):
            msg = "llama-server is already active on the configured port"
            emit(msg, ui_log)
            ui.notify(msg, type="warning")
            return False

        try:
            configured_path = configured_model_path(configured)
            model_folder = path_to_model_folder(configured_path)
        except Exception as exc:
            msg = f"Invalid model configuration for {model_name}: {exc}"
            emit(msg, ui_log)
            status_label.set_text("llama-server status: invalid model configuration")
            status_detail_label.set_text(str(exc))
            ui.notify(msg, type="negative")
            return False

        emit("--- Start requested ---", ui_log)
        emit(f"Selected model : {model_name}", ui_log)
        emit(f"Configured path: {configured_path}", ui_log)
        emit(f"Model folder   : {model_folder}", ui_log)

        try:
            cmd = await asyncio.to_thread(get_llama_command, model_folder, ui_log)

            emit("Launching llama-server process...", ui_log)

            self.process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            status_label.set_text("llama-server status: starting")
            status_detail_label.set_text(f"Starting model: {model_name}")
            ui.notify(f"Starting {model_name}...", type="info")
            self._reader_task = asyncio.create_task(self._read_process_output(model_name, self.process))

            await asyncio.sleep(0.5)
            if self.process.returncode is not None:
                emit(f"llama-server exited immediately with return code {self.process.returncode}", ui_log)
                status_label.set_text("llama-server status: failed")
                status_detail_label.set_text(f"Model {model_name} exited immediately")
                return False

            status_label.set_text("llama-server status: running")
            status_detail_label.set_text(f"Started by this GUI | Model: {model_name}")
            return True

        except Exception as exc:
            self.process = None
            msg = f"Start failed: {exc}"
            emit(msg, ui_log)
            status_label.set_text("llama-server status: start failed")
            status_detail_label.set_text(str(exc))
            ui.notify(msg, type="negative")
            return False

    async def _read_process_output(self, model_name: str, process: asyncio.subprocess.Process) -> None:
        assert process.stdout is not None

        try:
            while True:
                line = await process.stdout.readline()
                if not line:
                    break

                text = line.decode(errors="replace").rstrip()
                if text:
                    emit(f"[llama-server] {text}", ui_log)

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
            if self._reader_task is asyncio.current_task():
                self._reader_task = None

    async def stop_server(self) -> None:
        if not self.is_running():
            msg = "No GUI-started server active"
            emit(msg, ui_log)
            status_label.set_text("llama-server status: no GUI-started process")
            status_detail_label.set_text(
                "If llama-server was started outside this GUI, stop it from the shell/process manager."
            )
            ui.notify(msg, type="warning")
            return

        assert self.process is not None
        emit("Stopping llama-server...", ui_log)
        status_label.set_text("llama-server status: stopping")
        self.process.terminate()

        try:
            await asyncio.wait_for(self.process.wait(), timeout=10)
            emit("llama-server terminated", ui_log)
        except asyncio.TimeoutError:
            emit("llama-server did not terminate; killing it", ui_log)
            self.process.kill()
            await self.process.wait()
            emit("llama-server killed", ui_log)

        ui.notify("Server stopped", type="info")


manager = LlamaManager()

ui.colors(primary="#6e93d6")

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

with ui.header().classes("items-center justify-between"):
    ui.label("LLM Server Control Panel").classes("text-h6")
    ui.button("Stop Server", on_click=manager.stop_server, icon="stop", color="red")

with ui.column().classes("w-full max-w-4xl mx-auto p-4 gap-4"):
    with ui.card().classes("w-full p-4"):
        ui.label("llama-server status").classes("text-subtitle1")
        status_label = ui.label("llama-server status: not checked yet").classes("font-bold")
        status_detail_label = ui.label("Startup detection pending...").classes("text-sm text-gray-600")
        ui.button("Recheck status", on_click=detect_existing_llama_server, icon="refresh").classes("mt-2")

    with ui.card().classes("w-full p-4"):
        ui.label("Select a model").classes("text-subtitle1")

        model_select = ui.select(
            options=list(settings.AVAILABLE_MODELS.keys()),
            value=None,
            label="Select a model from the list...",
        ).classes("w-full")

        async def start_selected_model() -> None:
            if not model_select.value:
                emit("Start ignored: no model selected", ui_log)
                ui.notify("Select a model!", type="warning")
                return

            model_name = str(model_select.value)
            configured = settings.AVAILABLE_MODELS[model_name]
            started = await manager.start_server(model_name, configured)

            if started:
                chat_url = await get_browser_based_llama_url()
                emit(f"Chat URL: {chat_url}", ui_log)
                open_chat_dialog(model_name, chat_url)

        ui.button("Start Server", on_click=start_selected_model, icon="play_arrow").classes("mt-4")

    ui.label("Server Logs").classes("text-subtitle2")
    log_area = ui.log().classes("w-full h-96 font-mono text-xs bg-black text-green-400")

# Run detection once when the page is ready. ui.timer keeps the callback attached
# to the page instead of running it as a raw background task with no UI context.
ui.timer(0.5, detect_existing_llama_server, once=True)

emit("GUI loaded", None)
emit(f"settings.AVAILAble models: {len(settings.AVAILABLE_MODELS)}", None)
emit(f"NiceGUI listening on {settings.ui_host}:{settings.ui_port}", None)

ui.run(
    title=settings.UI_TITLE,
    host=settings.ui_host,
    port=settings.ui_port,
    ssl_certfile=settings.ui_ssl_certfile,
    ssl_keyfile=settings.ui_ssl_keyfile,
)
