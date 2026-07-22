import subprocess
import threading
import logzero
from logzero import logger
import re
from nicegui import ui
from config_manager import get_settings

settings = get_settings()


def run_command(args):
    """Helper to run a shell command and return output."""
    try:
        full_command = ["python3", "start_model.py"] + args
        logger.debug(f"Executing Command from array {full_command}")
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            check=False
        )
        output = result.stdout.strip()
        logger.debug(f"output={output}")
        return output, result.returncode
    except Exception as e:
        return str(e), 1


def get_server_status():
    output, rc = run_command(["--server-status"])
    is_running = ('is RUNNING' in output)
    status_text = "RUNNING" if is_running else "NOT RUNNING"
    if is_running:
        if "Running model:" in output:
            # here we assume that the model name doesn't contain space(s)
            # we also assume that the output format is not gonna change...
            pieces = output.split(' ')
            l = len(pieces)
            model = pieces[l-5]
            ctx = pieces[l-1]
            status_text += f" | {model} | ctx {ctx}"
    color = "green" if is_running else "red"
    return status_text, color#, output


def get_available_models():
    output, rc = run_command(["--list-models"])
    if rc != 0:
        return []
    # The output format is:
    # Available models:
    #   model1 (10 GiB - 0 RPC)
    #   model2 (5 GiB - 2 RPC)
    models = []
    lines = output.splitlines()
    for line in lines:
        line = line.strip()
        if line and not line.startswith("Available models:"):
            # Extract the model name (everything before the first '(')
            #match = re.match(r"^([^(]+)\\(.*\\)$", line)
            #if match:
            #    models.append(match.group(1).strip())
            #else:
            models.append(line)
    return models


class LlamaConsoleGUI:
    def __init__(self):
        self.status_label = None
        self.model_dropdown = None
        self.log_window = None
        self.log_thread = None
        self.stop_log_event = threading.Event()

    def _spawn_loader(self):
        """Called on event loop thread: spawn background thread for blocking I/O."""
        threading.Thread(target=self._load_data_async, daemon=True).start()

    def _load_data_async(self):
        """Run in a background thread: fetch models and status, then update UI."""
        models = get_available_models()
        text, color = get_server_status()
        self._apply_data(models, text, color)

    def _apply_data(self, models, text, color):
        """Called on the main event loop thread to update UI widgets."""
        if self.model_dropdown is not None:
            self.model_dropdown.options = models if models else ["No models found"]
            try:
                self.model_dropdown.props(remove='disable')
            except:
                pass
            self.model_dropdown.update()

        if self.status_label is not None:
            self.status_label.set_text(f"Server Status: {text}")
            self.status_label.style(f"color: {color}; font-weight: bold")
            ui.notify(f"Status updated: {text}")

    def _refresh_status_thread(self):
        """Background thread for status refresh."""
        text, color = get_server_status()
        self._apply_data(None, text, color)

    def update_status(self):
        """Refresh status asynchronously (non-blocking)."""
        threading.Thread(target=self._refresh_status_thread, daemon=True).start()

    def start_selected_model(self):
        model = self.model_dropdown.value
        if not model:
            ui.notify("Please select a model first", type="warning")
            return

        ui.notify(f"Starting model {model.split(' ')[0]}...")
        def task():
            output, rc = run_command([model.split(' ')[0]])
            if rc == 0:
                ui.notify(f"Model {model} started successfully", type="positive")
            else:
                ui.notify(f"Error starting model: {output}", type="negative")
            self.update_status()

        threading.Thread(target=task, daemon=True).start()

    def stop_server(self):
        ui.notify("Stopping server...")
        def task():
            output, rc = run_command(["--kill-server"])
            if rc == 0:
                ui.notify("Server stopped", type="positive")
            else:
                ui.notify(f"Error stopping server: {output}", type="negative")
            self.update_status()

        threading.Thread(target=task, daemon=True).start()

    def stream_logs(self):
        process = None
        try:
            process = subprocess.Popen(
                ["python3", "start_model.py", "--tail-log", "-n", "1000"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            while not self.stop_log_event.is_set():
                line = process.stdout.readline()
                if not line:
                    break
                self.log_window.push(line.strip())

        except Exception as e:
            self.log_window.push(f"Log Error: {str(e)}")
        finally:
            if 'process' in locals():
                if process:
                    process.terminate()

    def start_log_streaming(self):
        self.stop_log_event.clear()
        self.log_thread = threading.Thread(target=self.stream_logs, daemon=True)
        self.log_thread.start()
        ui.notify("Log streaming started")

    def build_ui(self):
        ui.dark_mode().enable()

        with ui.column().classes('w-full items-center p-8'):
            ui.label(settings.UI_TITLE).classes('text-h4 q-mb-md')

            with ui.row().classes('items-center q-mb-md'):
                self.status_label = ui.label("Checking llama-server status...")
                ui.button("Refresh", on_click=self.update_status).props('small outline')

            with ui.card().classes('w-full max-w-2xl p-4'):
                ui.label("Model Control").classes('text-h6')

                with ui.row().classes('w-full items-center q-mb-md'):
                    self.model_dropdown = ui.select(
                        options=["Loading models..."],
                        label="Select Model",
                        on_change=lambda e: ui.notify(f"Selected: {e.value}")
                    ).props('disable').classes('flex-grow')

                    ui.button("START", on_click=self.start_selected_model).props('color=green')
                    ui.button("STOP", on_click=self.stop_server).props('color=red')

            ui.label("Server Logs").classes('text-h6 q-mt-lg')
            with ui.row().classes('w-full items-center q-mb-sm'):
                ui.button("Connect to Logs", on_click=self.start_log_streaming).props('small')
                ui.button("Clear Logs", on_click=lambda: self.log_window.clear()).props('small outline')

            self.log_window = ui.log().classes('w-full h-96 bg-black text-green-400 font-mono text-xs')

        # Use ui.timer to defer data loading until the event loop is running.
        # This ensures the page renders immediately without blocking.
        ui.timer(0, self._spawn_loader, once=True)


gui_app = LlamaConsoleGUI()
gui_app.build_ui()

ui.run(title=settings.UI_TITLE, port=8501, host="0.0.0.0", reload=False, show=False)
