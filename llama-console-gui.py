"""NiceGUI front-end for start_model.py.

Everything this UI knows about models and about the server comes from
`start_model.py --json`: the CLI is the single source of truth, and the two
communicate over a real data structure instead of the printed text the UI used
to slice by word position.
"""

import asyncio
import contextlib
import json
import os
import shlex
import signal
import sys
from pathlib import Path

from logzero import logger
from nicegui import app, context, ui

from config_manager import get_settings
from model import get_model_byname

settings = get_settings()

# Load RPC server definitions
_rpc_config_path = Path(__file__).parent / "rpc.json"
try:
    with open(_rpc_config_path) as f:
        _rpc_config = json.load(f)
    RPC_SERVERS: dict = _rpc_config.get("RPC_SERVERS", {})
except (FileNotFoundError, json.JSONDecodeError):
    RPC_SERVERS = {}

# Resolve start_model.py next to this file: relying on the process CWD broke as
# soon as the systemd unit was started from anywhere else.
_START_MODEL = str(Path(__file__).parent / "start_model.py")
# sys.executable, not "python3": the child must run in the same interpreter (and
# virtualenv) as the GUI, not whatever "python3" resolves to in PATH.
_PY = sys.executable

_CTX_MIN = 8192
_CTX_STEP = 1024


#___________________________________________________________________________________
async def _capture(argv: list[str]) -> tuple[str, int]:
    """Run argv to completion off the event loop; return (stdout, returncode)."""
    logger.debug(f"Executing {argv}")
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if stderr:
        logger.debug(stderr.decode(errors="replace").rstrip())
    return stdout.decode(errors="replace"), proc.returncode


#___________________________________________________________________________________
async def _run_json(args: list[str]) -> dict | None:
    """Run start_model.py with --json and return the parsed stdout, or None.

    The return code is deliberately ignored: --server-status exits 1 to mean
    'not running', which is a perfectly valid answer, not a failure. What tells
    success from failure here is whether stdout carried parseable JSON.
    """
    out, rc = await _capture([_PY, _START_MODEL, *args, "--json"])
    if not out.strip():
        logger.error(f"start_model.py {' '.join(args)} produced no output (rc={rc})")
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        logger.error(f"start_model.py {' '.join(args)} returned invalid JSON: {e}")
        return None


#___________________________________________________________________________________
# Model-list cache: --list-models is slow (SSH round-trips), so it runs once at
# application boot and, after that, only when the user presses "Recheck models".
# Every page (re)load reads this cache instead of spawning start_model.py again.
_model_specs: dict[str, dict] | None = None
_model_specs_lock = asyncio.Lock()


async def fetch_model_specs(force: bool = False) -> tuple[dict[str, dict] | None, bool]:
    """Return ({name: spec}, ok). Runs --list-models only when the cache is
    empty or force=True; a failed forced refresh keeps the previous list."""
    global _model_specs
    async with _model_specs_lock:
        if _model_specs is None or force:
            data = await _run_json(["--list-models"])
            if data is None:
                return _model_specs, False
            _model_specs = {m["name"]: m for m in data.get("models", [])}
        return _model_specs, True


# Kick the fetch off in the background at boot: pages opened before it finishes
# just await the same in-flight call via the lock instead of starting their own.
app.on_startup(lambda: asyncio.create_task(fetch_model_specs()))


#___________________________________________________________________________________
class LlamaConsoleGUI:
    """One instance per connected browser.

    The UI used to be built once at import time, so every client shared the same
    widget objects; it is now created inside an @ui.page handler (see below),
    which is also what puts UI updates on the right client context.
    """

    def __init__(self):
        self.specs: dict[str, dict] = {}
        self.log_task: asyncio.Task | None = None
        self._status_busy = False
        self._start_busy = False
        self.start_button = None

        self.status_server_label = None
        self.status_model_label = None
        self.status_ctx_label = None
        self.status_temp_label = None
        self.status_quant_label = None
        self.model_dropdown = None
        self.ctx_slider = None
        self.ctx_label = None
        self.temp_slider = None
        self.temp_label = None
        self.log_window = None
        self.stop_log_button = None
        self.server_checkboxes: dict[str, ui.checkbox] = {}

    # ---------------------------------------------------------------- status ---
    def _update_rpc_checkboxes( self ) -> None:
        selected_model = get_model_byname(self.model_dropdown.value, settings.MODELS_JSON, settings.RPC_JSON )
        #logger.debug(f"Selected model: {selected_model.model_name}")
        self._uncheck_rpc_checkboxes()
        
        for cb_name in self.server_checkboxes:
            #logger.debug(f"current cb_name=[{cb_name}]")
            for rpc in selected_model.rpcservers:
                #logger.debug(f"current rpc name=[{rpc.name}]")
                if rpc.name == cb_name:
                    #logger.debug(f"MATCH - settings checkbox to True")
                    self.server_checkboxes[cb_name].value = True

    # ---------------------------------------------------------------- status ---
    def _uncheck_rpc_checkboxes(self):
        for name in self.server_checkboxes:
            self.server_checkboxes[name].value = False

    # ---------------------------------------------------------------- data ---
    async def load_models(self, force: bool = False) -> bool:
        specs, ok = await fetch_model_specs(force)
        if specs is None:
            self.model_dropdown.options = {"": "No models found"}
            self.model_dropdown.update()
            ui.notify("Could not load the model list", type="negative")
            return False
        if not ok:
            ui.notify("Could not reload the model list — keeping the previous one",
                      type="negative")

        self.specs = specs
        if not self.specs:
            self.model_dropdown.options = {"": "No models found"}
            self.model_dropdown.update()
            return ok

        # Dict options: the value stays the bare model name, so nothing
        # downstream has to strip the ' (60 GiB - 2 RPC)' suffix back off.
        self.model_dropdown.options = {
            name: f"{name} ({int(s['size_gib']) if s.get('size_gib') is not None else '?'} GiB"
                  f" - {s['rpc_count']} RPC)"
            for name, s in self.specs.items()
        }
        self.model_dropdown.props(remove='disable')
        self.model_dropdown.props('label="Model selector"')
        # Keep the current selection across a recheck when it still exists.
        previous = self.model_dropdown.value
        selected = previous if previous in self.specs else next(iter(self.specs))
        self.model_dropdown.set_value(selected)
        self.model_dropdown.update()
        self._apply_model_spec(selected)
        self._update_rpc_checkboxes()
        return ok

    async def recheck_models(self) -> None:
        """Re-run --list-models on demand — the only trigger besides app boot."""
        self.model_dropdown.props('disable')
        self.model_dropdown.props('label="Reloading models..."')
        ui.notify("Rechecking models...")
        if await self.load_models(force=True):
            ui.notify("Model list updated", type="positive")

    # ---------------------------------------------------------------- update ---
    async def update_status(self) -> None:
        # Polled every 5 s by a ui.timer: if the previous check is still in
        # flight (slow SSH round-trip), skip this tick instead of stacking up.
        if self._status_busy:
            return
        self._status_busy = True
        try:
            await self._update_status_once()
        finally:
            self._status_busy = False

    async def _update_status_once(self) -> None:
        info = await _run_json(["--server-status"])
        if info is None:
            self.status_server_label.set_text("Server Status: UNKNOWN")
            self.status_server_label.style("color: orange;")
            return

        running = bool(info.get("running"))
        # Starting a second model over a live server is never valid, so START
        # follows the polled status; the click handler re-checks to close the
        # window between two polls.
        if running:
            self.start_button.disable()
        else:
            self.start_button.enable()
        color = "#00ff88" if running else "red"
        self.status_server_label.set_text(
            f"Server Status: {'RUNNING' if running else 'NOT RUNNING'}"
        )
        # for label in (self.status_server_label, self.status_model_label,
        #               self.status_ctx_label, self.status_temp_label):
        #     label.style(f"color: {color};")
        self.status_server_label.style(f"color: {color};")

        if running and info.get("ready"):
            self.status_model_label.set_text(f" - Model   : {info['model'].strip()}")
            c = (str(info['ctx'])).strip()
            self.status_ctx_label.set_text(  f" - Context : {c} tokens")
            # Rounded: llama-server reports the float32 round-trip of 0.6 as
            # 0.6000000238418579.
            self.status_temp_label.set_text( f" - Temp    : {float(info['temperature']):.1f}")
            self.status_quant_label.set_text( f" - Quant   : {info['quant']}")
        elif running:
            self.status_model_label.set_text("Model: (starting up...)")
            self.status_ctx_label.set_text("")
            self.status_temp_label.set_text("")
            self.status_quant_label.set_text("")
        else:
            self.status_model_label.set_text("")
            self.status_ctx_label.set_text("")
            self.status_temp_label.set_text("")
            self.status_quant_label.set_text("")

    async def refresh(self) -> None:
        await self.update_status()
        ui.notify("Status updated")

    # -------------------------------------------------------------- sliders ---
    def _apply_model_spec(self, model_name: str) -> None:
        """Point both sliders at the selected model's bounds and defaults."""
        spec = self.specs.get(model_name)
        if spec is None:
            return

        native_ctx = int(spec["native_ctx"])
        # min() guards a model whose native context is below the usual floor:
        # a slider with min > max cannot be dragged at all.
        ctx_min = min(_CTX_MIN, native_ctx)
        ctx_value = max(ctx_min, min(int(spec["ctx"]), native_ctx))
        # element.props is a public observable dict in NiceGUI 3.x, so assigning
        # to it schedules the update by itself. The value still goes through
        # set_value(): pushing it as a 'value=' prop is what breaks dragging.
        self.ctx_slider.props['min'] = ctx_min
        self.ctx_slider.props['max'] = native_ctx
        self.ctx_slider.set_value(ctx_value)
        self.ctx_label.set_text(f"Context: {ctx_value:,}  (max: {native_ctx:,})")

        # NOTE: unchanged semantics — the model's configured temperature doubles
        # as the slider maximum, so it can only be lowered from here.
        max_temp = float(spec["temperature"])
        self.temp_slider.props['min'] = 0
        self.temp_slider.props['max'] = max_temp
        self.temp_slider.set_value(max_temp)
        self.temp_label.set_text(f"Temperature: {max_temp:.2f}  (max: {max_temp:.2f})")

        self._update_rpc_checkboxes( )
        

    def _on_model_change(self, e) -> None:
        if e.value:
           self._apply_model_spec(e.value)
        
    # ------------------------------------------------------------- commands ---
    async def _stream(self, argv: list[str]) -> int:
        """Run argv and push its output into the log window line by line."""
        # start_new_session gives the child its own process group so that the
        # whole tree can be signalled at once. Terminating just the direct child
        # left start_model.py's own `ssh -t ... tail -F` orphaned onto init,
        # still holding a tail open on the remote host.
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            async for raw in proc.stdout:
                self.log_window.push(raw.decode(errors="replace").rstrip())
            return await proc.wait()
        finally:
            if proc.returncode is None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)

    async def start_selected_model(self) -> None:
        if self._start_busy:
            ui.notify("A model start is already in progress", type="warning")
            return
        model = self.model_dropdown.value
        spec = self.specs.get(model)
        if spec is None:
            ui.notify("Please select a model first", type="warning")
            return

        self._start_busy = True
        try:
            # Fresh check at click time: the polled status (and the START
            # button state it drives) can be up to 5 s stale.
            info = await _run_json(["--server-status"])
            if info is not None and info.get("running"):
                current = (info.get("model") or "").strip()
                suffix = f" ({current})" if current else ""
                ui.notify(f"llama-server is already running{suffix} — "
                          "stop it before starting another model", type="negative")
                await self.update_status()
                return

            args = [model]
            ctx_value = int(self.ctx_slider.value)
            args += ["--override-ctx", str(ctx_value)]

            override_rpc = []
            for cb_name in self.server_checkboxes:
                if self.server_checkboxes[cb_name].value == True:
                    override_rpc.append(cb_name)

            if len(override_rpc):
                args += ["--override-rpc", f"{','.join(override_rpc)}"]

            # Only override the temperature if the user actually moved the slider.
            temp_value = float(self.temp_slider.value)
            if abs(temp_value - float(spec["temperature"])) > 0.001:
                args += ["--override-temp", f"{temp_value:.4f}"]
            args += ["--debug"]

            ui.notify(f"Starting model {model} (ctx={ctx_value})...")
            # -u keeps the child's output unbuffered so lines arrive as they happen.
            cmd = [_PY, "-u", _START_MODEL, *args]
            self.log_window.push(f"--- Starting model: {shlex.join(cmd)} ---")
            try:
                rc = await self._stream(cmd)
            except OSError as e:
                self.log_window.push(f"Launch Error: {e}")
                ui.notify(f"Error starting model: {e}", type="negative")
            else:
                if rc == 0:
                    ui.notify(f"Model {model} started successfully", type="positive")
                else:
                    ui.notify(f"Error starting model (exit {rc})", type="negative")
            await self.update_status()
        finally:
            self._start_busy = False

    async def stop_server(self) -> None:
        ui.notify("Stopping server...")
        out, rc = await _capture([_PY, _START_MODEL, "--kill-server"])
        if rc == 0:
            ui.notify("Server stopped", type="positive")
        else:
            ui.notify(f"Error stopping server: {out.strip()}", type="negative")
        await self.update_status()

    # ------------------------------------------------------------ log tail ---
    async def _stream_logs(self) -> None:
        argv = [_PY, "-u", _START_MODEL, "--tail-log", "-n", "1000"]
        try:
            await self._stream(argv)
        except asyncio.CancelledError:
            self.log_window.push("--- log streaming stopped ---")
            raise
        except OSError as e:
            self.log_window.push(f"Log Error: {e}")
        finally:
            if self.stop_log_button is not None:
                self.stop_log_button.disable()

    def start_log_streaming(self) -> None:
        # Without this guard every click stacked another `tail -F` that nothing
        # could ever stop.
        if self.log_task is not None and not self.log_task.done():
            ui.notify("Log streaming is already running", type="warning")
            return
        self.log_task = asyncio.create_task(self._stream_logs())
        self.stop_log_button.enable()
        ui.notify("Log streaming started")

    def stop_log_streaming(self) -> None:
        if self.log_task is None or self.log_task.done():
            ui.notify("Log streaming is not running", type="warning")
            return
        self.log_task.cancel()
        ui.notify("Log streaming stopped")

    def cleanup(self) -> None:
        """Tear down the tail when the browser tab goes away."""
        if self.log_task is not None and not self.log_task.done():
            self.log_task.cancel()

    # ------------------------------------------------------------------ UI ---
    def build_ui(self) -> None:
        with ui.column().classes('w-full items-center p-8'):
            with ui.column().classes('items-center q-mb-md'):
                ui.label("LLama.cpp Console").classes('text-h5')
                ui.label("by Alvise Dorigo").classes('text-h5')
                ui.link("https://github.com/dorigoa/llama-console",
                        "https://github.com/dorigoa/llama-console").classes('text-caption no-underline')

            with ui.column().classes('w-full max-w-2xl gap-1 q-mb-4 pr-4'):
                self.status_server_label = ui.label("Checking llama-server status...")
                self.status_model_label = ui.label("")
                self.status_quant_label = ui.label("")
                self.status_ctx_label = ui.label("")
                self.status_temp_label = ui.label("")
                for label in (self.status_server_label, self.status_model_label,
                              self.status_ctx_label, self.status_temp_label, self.status_quant_label):
                    label.style('font-size: 0.9rem; font-weight: 600; white-space: nowrap;')
                for label in (self.status_model_label,
                              self.status_ctx_label, self.status_temp_label,self.status_quant_label):
                    label.classes('font-mono').style('font-size: 0.9rem; font-weight: 600; white-space: pre;')
                    # label.style(
                    #     'font-family: "JetBrains Mono", "Fira Code", "DejaVu Sans Mono", Menlo, Consolas, monospace; '
                    #     'font-size: 0.9rem; font-weight: 600; white-space: nowrap;'
                    # )

                ui.button("Refresh", on_click=self.refresh).props('outline small')

            with ui.card().classes('w-full max-w-2xl p-4'):
                ui.label("Model Control").classes('text-h6')

                with ui.row().classes('w-full items-center q-mb-md'):
                    self.model_dropdown = ui.select(
                        options={"": "Loading models..."},
                        value="",
                        label="Loading models...",
                        on_change=self._on_model_change,
                    ).props('disable').classes('flex-grow')

                    self.start_button = ui.button(
                        "START", on_click=self.start_selected_model).props('color=green')
                    ui.button("STOP", on_click=self.stop_server).props('color=red')

                ui.button("Recheck models",
                          on_click=self.recheck_models).props('small outline')

                # RPC server checkboxes — one per server defined in rpc.json

                #selected_model = get_model_byname(self.model_dropdown.value, settings.MODELS_JSON, settings.RPC_JSON )

                # if not selected_model:
                #     selected_model = get_model_byname(self.model_dropdown.value, settings.MODELS_JSON, settings.RPC_JSON )

                if RPC_SERVERS:
                    with ui.row().classes('w-full items-center q-mt-sm gap-3'):
                        ui.label('RPC servers:').classes('text-subtitle1')
                        for name in RPC_SERVERS:
                            self.server_checkboxes[name] = ui.checkbox(name)
                            # if name == selected_model.model_name:
                            #     self.server_checkboxes[name].value = True
                            # else:
                            #     self.server_checkboxes[name].value = False

                with ui.column().classes('w-full q-mt-sm'):
                    self.ctx_label = ui.label("Context: —").classes('text-subtitle1')
                    self.ctx_slider = ui.slider(
                        min=_CTX_MIN, max=262144, value=_CTX_MIN, step=_CTX_STEP,
                        on_change=lambda e: self.ctx_label.set_text(f"Context: {e.value:,}")
                    ).classes('flex-grow').props('color=green')

                with ui.column().classes('w-full q-mt-sm'):
                    self.temp_label = ui.label("Temperature: —").classes('text-subtitle1')
                    self.temp_slider = ui.slider(
                        min=0, max=1.0, value=1.0, step=0.01,
                        on_change=lambda e: self.temp_label.set_text(f"Temperature: {e.value:.2f}")
                    ).classes('flex-grow').props('color=orange')

            ui.label("Server Logs").classes('text-h6 q-mt-lg')
            with ui.row().classes('w-full items-center q-mb-sm'):
                ui.button("Connect to llama-server logs",
                          on_click=self.start_log_streaming).props('small')
                self.stop_log_button = ui.button("Stop log streaming",
                                                 on_click=self.stop_log_streaming).props('small outline')
                self.stop_log_button.disable()
                ui.button("Clear Logs", on_click=lambda: self.log_window.clear()).props('small outline')

            self.log_window = ui.log().classes(
                'w-full h-[600px] bg-black text-green-400 font-mono text-xs custom-log')

        ui.add_head_html('''
<style>
.custom-log {
    scrollbar-width: thin !important;
    scrollbar-color: #4caf50 #1a1a1a !important;
}
.custom-log::-webkit-scrollbar {
    width: 14px !important;
    height: 14px !important;
}
.custom-log::-webkit-scrollbar-track {
    background: #1a1a1a !important;
    border-radius: 7px !important;
}
.custom-log::-webkit-scrollbar-thumb {
    background: #4caf50 !important;
    border-radius: 7px !important;
    border: 3px solid #1a1a1a !important;
}
.custom-log::-webkit-scrollbar-thumb:hover {
    background: #66bb6a !important;
}
</style>
''')

#___________________________________________________________________________________
@ui.page('/')
def index() -> None:
    ui.dark_mode().enable()
    gui = LlamaConsoleGUI()
    gui.build_ui()
    context.client.on_disconnect(gui.cleanup)
    # Models come from the app-level cache filled at boot, so a page (re)load
    # no longer re-runs --list-models; only "Recheck models" forces it.
    ui.timer(0, gui.load_models, once=True)
    # Status polling: fires immediately on connect, then every 5 s, so the
    # user always sees whether (and what) the server is running.
    ui.timer(5.0, gui.update_status)


ui.run(title=settings.UI_TITLE, port=settings.UI_PORT, host="0.0.0.0", reload=False, show=False)
