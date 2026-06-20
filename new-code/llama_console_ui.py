#!/usr/bin/env python3
"""Streamlit GUI for managing llama-server models via start_model.py."""

import html as _html
import os
import sys
import json
import pty
import re
import select
import queue
import threading
import subprocess
import time
import signal
import urllib.request
#import urllib.error
from datetime import datetime, timezone
import streamlit as st
from pathlib import Path
from config_manager import get_settings

# Path to persistent log file (shared across page reloads)
_LOG_FILE_PATH = Path(get_settings().PERSIST_FILE).with_name("llama-console-logs.txt")

MODELS_JSON = Path(__file__).parent / "models.json"
START_SCRIPT = Path(__file__).parent / "start_model.py"

# Strip ANSI colour/cursor codes and bare carriage returns written by llama-server
# when it detects a TTY (which is exactly what we give it via the PTY).
_ANSI_RE = re.compile(rb"\x1b\[[0-9;]*[a-zA-Z]|\x1b[()][AB012]|\r")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _save_persist(model_name: str, alias: str, pid: int, port: int, started_at: str, ready_at: str, ctx: int = 0) -> None:
    settings = get_settings()
    data = {
        "model_name": model_name,
        "alias": alias,
        "pid": pid,
        "port": port,
        "api_url": f"http://127.0.0.1:{port}/v1",
        "started_at": started_at,
        "ready_at": ready_at,
        "ctx": ctx,
    }
    Path(settings.PERSIST_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")

#___________________________________________________________________________________
def _clear_persist() -> None:
    settings = get_settings()
    p = Path(settings.PERSIST_FILE)
    if p.exists():
        p.unlink()

#___________________________________________________________________________________
class _RecoveredProcess:
    """Minimal Popen-compatible wrapper for a PID we didn't spawn ourselves."""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> int | None:
        try:
            os.kill(self.pid, 0)
            return None  # still alive
        except (ProcessLookupError, PermissionError):
            return 1

    def terminate(self) -> None:
        try:
            os.kill(self.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

#___________________________________________________________________________________
def _try_recover_from_persist() -> None:
    """Called once per session: restore running state from the persist file if valid."""
    if st.session_state.get("_recovery_done"):
        return
    st.session_state._recovery_done = True

    settings = get_settings()
    persist_path = Path(settings.PERSIST_FILE)
    if not persist_path.exists():
        return

    try:
        data = json.loads(persist_path.read_text(encoding="utf-8"))
    except Exception:
        return

    pid: int | None = data.get("pid")
    port: int | None = data.get("port")
    model_name: str | None = data.get("model_name")
    alias: str = data.get("alias", model_name or "")

    if not (pid and port and model_name):
        return

    recovered = _RecoveredProcess(pid)
    if recovered.poll() is not None:
        _clear_persist()
        return

    # Fast API check (refused immediately if not up yet, so 2 s timeout is safe)
    api_ready = False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as resp:
            api_ready = resp.status == 200
    except Exception:
        pass

    st.session_state.process = recovered
    st.session_state.running_model = model_name
    st.session_state.running_ctx = data.get("ctx")

    if api_ready:
        st.session_state.server_ready = True
    else:
        # Model still loading — start a probe thread to catch when it's ready
        st.session_state.server_ready = False
        rq: queue.Queue = queue.Queue()
        st.session_state.ready_queue = rq
        threading.Thread(
            target=_probe_server,
            args=(recovered, port, model_name, alias, rq, st.session_state.running_ctx or 0),
            daemon=True,
        ).start()

#___________________________________________________________________________________
def _probe_server(
    proc: subprocess.Popen,
    port: int,
    model_name: str,
    alias: str,
    ready_q: queue.Queue,
    timeout: int = 420,
    ctx: int = 0,
) -> None:
    """Poll /v1/models until the server is ready or timeout/crash."""
    url = f"http://127.0.0.1:{port}/v1/models"
    started_at = datetime.now(timezone.utc).isoformat()
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if proc.poll() is not None:
            ready_q.put(False)
            return
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    ready_at = datetime.now(timezone.utc).isoformat()
                    _save_persist(model_name, alias, proc.pid, port, started_at, ready_at, ctx)
                    ready_q.put(True)
                    return
        except Exception:
            pass
        time.sleep(3)

    ready_q.put(False)

#___________________________________________________________________________________
def _fmt_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    b = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"

#___________________________________________________________________________________
def _fmt_ctx(size: int | None) -> str:
    """Format context size as human-readable string (e.g. 131072 -> '128K')."""
    if not size:
        return ""
    if size >= 1024:
        k = size / 1024
        # Show integer if it's a clean number
        if k == int(k):
            return f"{int(k)}K"
        return f"{k:.1f}K"
    return str(size)

#___________________________________________________________________________________
def _load_entries() -> list[dict]:
    """All models from models.json, including ones whose files are missing."""
    raw = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    base = Path(raw["MODEL_BASE_DIR"])
    entries = []
    for name, spec in raw.get("models", {}).items():
        fname = name if name.endswith(".gguf") else f"{name}.gguf"
        path = base / fname
        rpc_servers = spec.get("RPC_SERVERS", {})
        entries.append({
            "name": name,
            "alias": str(spec.get("ALIAS", name)),
            "path": path,
            "size": _fmt_size(path),
            "exists": path.exists(),
            "ctx": int(spec.get("ctx", 0)),
            "temp":  float(spec["TEMP"]),
            "top_p": float(spec["TOPP"]),
            "top_k": int(spec["TOPK"]),
            "min_p": float(spec["MINP"]),
            "rpc_servers": rpc_servers,
            "rpc_count": len(rpc_servers),
        })
    entries.sort(key=lambda e: e["name"])
    return entries

#___________________________________________________________________________________
def _get_llama_url(port: int) -> str:
    """Return the llama-server URL using the browser's current scheme and hostname."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(st.context.url)
        scheme = parsed.scheme
        if scheme == "https":
            port = 8443
        return f"{scheme}://{parsed.hostname}:{port}"
    except Exception:
        pass
    try:
        headers = st.context.headers
        host = headers.get("host", "localhost").split(":")[0]
        scheme = headers.get("x-forwarded-proto", "http")
        if scheme == "https":
            port = 8443
        return f"{scheme}://{host}:{port}"
    except Exception:
        return f"http://localhost:{port}"

#___________________________________________________________________________________
def _is_running() -> bool:
    p = st.session_state.get("process")
    return p is not None and p.poll() is None

#___________________________________________________________________________________
@st.cache_data(ttl=1, show_spinner=False)
def _check_server_ready(url: str) -> bool:
    """Probe the llama-server API. Cached for 1s so it doesn't block on every render."""
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False

#___________________________________________________________________________________
def _drain_queue() -> None:
    q: queue.Queue = st.session_state.log_queue
    lines: list[str] = st.session_state.log_lines
    while True:
        try:
            lines.append(q.get_nowait().rstrip())
        except queue.Empty:
            break

#___________________________________________________________________________________
def _pty_reader(master_fd: int, q: queue.Queue, log_path: Path | None = None) -> None:
    """Read bytes from the PTY master, strip ANSI codes, split on newlines."""
    buf = b""
    while True:
        try:
            r, _, _ = select.select([master_fd], [], [], 0.5)
            if not r:
                continue
            chunk = os.read(master_fd, 4096)
            if not chunk:
                # macOS may return b"" instead of raising OSError; keep going
                continue
            chunk = _ANSI_RE.sub(b"", chunk)
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                decoded_line = line.decode("utf-8", errors="replace")
                q.put(decoded_line)
                if log_path is not None:
                    try:
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(decoded_line + "\n")
                    except Exception:
                        pass
        except OSError:
            # EIO when slave side is fully closed (process exited)
            break
    if buf:
        decoded_buf = buf.decode("utf-8", errors="replace")
        q.put(decoded_buf)
        if log_path is not None:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(decoded_buf + "\n")
            except Exception:
                pass
    try:
        os.close(master_fd)
    except OSError:
        pass


# ─── auto-refreshing log pane ─────────────────────────────────────────────────

@st.fragment(run_every=1)
def _log_pane() -> None:
    _drain_queue()

    # Drain probe result from background thread (fallback — inline probe in col_info is primary)
    ready_q: queue.Queue = st.session_state.get("ready_queue", queue.Queue())
    try:
        while True:
            result = ready_q.get_nowait()
            if result and not st.session_state.server_ready:
                st.session_state.server_ready = True
                st.rerun()
    except queue.Empty:
        pass

    proc = st.session_state.get("process")
    lines: list[str] = st.session_state.get("log_lines", [])

    if proc is not None and proc.poll() is not None and lines:
        code = proc.poll()
        if code == 0:
            st.success(f"Process exited cleanly (code {code})")
        else:
            st.error(f"Process exited with code {code}")

    tail = "\n".join(lines[-1000:])
    st.markdown(
        '<div style="height:480px;overflow-y:scroll;background:#0e1117;color:#d1d5db;'
        'font-family:monospace;font-size:13px;padding:8px 12px;border-radius:4px;'
        f'border:1px solid #333;white-space:pre-wrap">{_html.escape(tail)}</div>',
        unsafe_allow_html=True,
    )

    # Clear logs button moved to top of log section


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="LLama Console", layout="wide", page_icon="🦙")

    settings = get_settings()

    # session state defaults
    for k, v in {
        "process": None,
        "log_lines": [],
        "log_queue": queue.Queue(),
        "ready_queue": queue.Queue(),
        "running_model": None,
        "running_ctx": None,
        "server_ready": None,  # None=idle, False=loading, True=ready
        "_recovery_done": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # Persistent log file path
    st.session_state.log_file_path = _LOG_FILE_PATH
    # Load any existing persisted logs
    if _LOG_FILE_PATH.is_file():
        try:
            persisted = _LOG_FILE_PATH.read_text(encoding="utf-8").splitlines()
            st.session_state.log_lines = persisted
        except Exception:
            # If reading fails, start with empty logs
            st.session_state.log_lines = []
    _try_recover_from_persist()

    st.title(settings.UI_TITLE)

    st.markdown(
        "<style>[data-testid='stSelectbox'] input { pointer-events: none !important; }</style>",
        unsafe_allow_html=True,
    )

    entries = _load_entries()

    # ── model selector ────────────────────────────────────────────────────────
    def _label(e: dict) -> str:
        flag = "" if e["exists"] else "  ⚠️ file missing"
        rpc = f"  #RPC_srv: {e['rpc_count']}" if e["rpc_count"] else ""
        return f"{e['name']}   [{e['size']}]{rpc}{flag}"

    # Determine default selection based on running model
    default_index = 0
    running_name = st.session_state.get("running_model")
    if running_name:
        for i, e in enumerate(entries):
            if e["name"] == running_name:
                default_index = i
                break
    idx = st.selectbox(
        "Model",
        range(len(entries)),
        index=default_index,
        format_func=lambda i: _label(entries[i]),
        disabled=_is_running(),
    )

    entry = entries[idx]

    # ── parameter overrides ───────────────────────────────────────────────────
    with st.expander("Parameter overrides (leave blank to use model defaults)", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            temp_str = st.text_input(
                "temperature",
                placeholder=f"default: {entry['temp']}",
                key="ov_temp",
            )
        with c2:
            topp_str = st.text_input(
                "top_p",
                placeholder=f"default: {entry['top_p']}",
                key="ov_topp",
            )
        with c3:
            topk_str = st.text_input(
                "top_k",
                placeholder=f"default: {entry['top_k']}",
                key="ov_topk",
            )
        with c4:
            minp_str = st.text_input(
                "min_p",
                placeholder=f"default: {entry['min_p']}",
                key="ov_minp",
            )

    # ── start / stop ──────────────────────────────────────────────────────────
    col_start, col_stop, col_info = st.columns([1, 1, 5], vertical_alignment="center")

    with col_start:
        start_clicked = st.button(
            "▶  Start",
            disabled=_is_running() or not entry["exists"],
            use_container_width=True,
            type="primary",
        )

    with col_stop:
        stop_clicked = st.button(
            "⏹  Stop",
            disabled=not _is_running(),
            use_container_width=True,
        )

    with col_info:
        if _is_running():
            ctx_label = f"{_fmt_ctx(st.session_state.running_ctx)}" if st.session_state.running_ctx else ""
            # Inline probe: if still marking as "loading", check if server just became ready
            if not st.session_state.server_ready:
                is_ready = _check_server_ready(f"http://127.0.0.1:{settings.PORT_BIND}/v1/models")
                if is_ready:
                    st.session_state.server_ready = True
                    st.session_state.ready_queue = queue.Queue()  # discard old queue
                    st.rerun()  # rerun to show "ready" immediately
            if st.session_state.server_ready:
                api_url = _get_llama_url(settings.PORT_BIND)
                st.markdown(
                    f'<span style="color:#22c55e;font-size:14px;font-weight:600">'
                    f'▶ ready — {st.session_state.running_model} |  ctx {ctx_label}  | </span>'
                    f'&nbsp;&nbsp;<a href="{api_url}" target="_blank" '
                    f'style="font-size:13px;color:#60a5fa;text-decoration:none">{api_url}</a>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span style="color:#f59e0b;font-size:14px;font-weight:600">'
                    f'⏳ loading — {st.session_state.running_model}{ctx_label}</span>',
                    unsafe_allow_html=True,
                )
        elif st.session_state.process is not None:
            st.markdown(
                '<span style="color:#6b7280;font-size:14px;font-weight:600">⏹ stopped</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span style="color:#6b7280;font-size:14px;font-weight:600">● idle</span>',
                unsafe_allow_html=True,
            )

    if start_clicked:
        cmd = [sys.executable, "-u", str(START_SCRIPT), entry["name"]]
        if temp_str.strip():
            cmd += ["--override-temp",  temp_str.strip()]
        if topp_str.strip():
            cmd += ["--override-top-p", topp_str.strip()]
        if topk_str.strip():
            cmd += ["--override-top-k", topk_str.strip()]
        if minp_str.strip():
            cmd += ["--override-min-p", minp_str.strip()]

        # Reset in-memory logs and queues
        st.session_state.log_lines = []
        st.session_state.log_queue = queue.Queue()
        st.session_state.ready_queue = queue.Queue()
        st.session_state.server_ready = False
        st.session_state.rpc_kill_results = {}

        # Truncate persistent log file for a fresh start
        try:
            _LOG_FILE_PATH.write_text("", encoding="utf-8")
        except Exception:
            # If writing fails, ignore – logs will still be in-memory for this session
            pass

        master_fd, slave_fd = pty.openpty()
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=slave_fd,
            stderr=slave_fd,
            close_fds=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        os.close(slave_fd)  # parent does not need the slave end

        st.session_state.process = proc
        st.session_state.running_model = entry["name"]
        st.session_state.running_ctx = entry["ctx"]
        # Persist the launched model early so it survives page reloads even before the server is ready.
        started_at = datetime.now(timezone.utc).isoformat()
        _save_persist(entry["name"], entry["alias"], proc.pid, settings.PORT_BIND, started_at, "", entry["ctx"])

        threading.Thread(
            target=_pty_reader,
            args=(master_fd, st.session_state.log_queue, _LOG_FILE_PATH),
            daemon=True,
        ).start()

        threading.Thread(
            target=_probe_server,
            args=(proc, settings.PORT_BIND, entry["name"], entry["alias"], st.session_state.ready_queue, entry["ctx"]),
            daemon=True,
        ).start()

        st.rerun()

    if stop_clicked and st.session_state.process is not None:
        st.session_state.process.terminate()
        st.session_state.process = None
        st.session_state.running_model = None
        st.session_state.running_ctx = None
        st.session_state.server_ready = None
        _clear_persist()
        st.rerun()

    # ── kill RPC servers ──────────────────────────────────────────────────────
    col_kill, col_kill_status = st.columns([1, 5], vertical_alignment="center")
    with col_kill:
        kill_rpc_clicked = st.button(
            "☠  Kill RPC servers",
            disabled=(entry["rpc_count"] == 0),
            use_container_width=True,
        )
    with col_kill_status:
        results = st.session_state.get("rpc_kill_results", {})
        if results:
            parts = []
            for host, (ok, msg) in results.items():
                color = "#22c55e" if ok else "#ef4444"
                parts.append(f'<span style="color:{color}">{host}: {msg}</span>')
            st.markdown(
                '<span style="font-size:13px;font-family:monospace">'
                + "&nbsp;&nbsp;|&nbsp;&nbsp;".join(parts)
                + "</span>",
                unsafe_allow_html=True,
            )

    if kill_rpc_clicked:
        kill_results: dict[str, tuple[bool, str]] = {}
        for host, srv in entry["rpc_servers"].items():
            user = srv.get("remuser", "root")
            binary = os.path.basename(srv.get("bin", "rpc-server"))
            ssh_cmd = [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=5",
                "-o", "BatchMode=yes",
                f"{user}@{host}",
                f"killall {binary}",
            ]
            try:
                r = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=10)
                if r.returncode == 0:
                    kill_results[host] = (True, "killed")
                else:
                    err = r.stderr.strip() or f"exit {r.returncode}"
                    kill_results[host] = (False, err)
            except subprocess.TimeoutExpired:
                kill_results[host] = (False, "timeout")
            except Exception as exc:
                kill_results[host] = (False, str(exc))
        st.session_state.rpc_kill_results = kill_results
        st.rerun()

    # ── log window ────────────────────────────────────────────────────────────
    col_label, col_button = st.columns([1, 1], vertical_alignment="center")
    with col_label:
        st.subheader("Logs")
    with col_button:
        if st.button("Clear logs", key="clear_btn"):
            st.session_state.log_lines = []
            # Truncate the persistent log file as well
            try:
                _LOG_FILE_PATH.write_text("", encoding="utf-8")
            except Exception:
                pass
            st.rerun()
    _log_pane()

#___________________________________________________________________________________
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        _in_streamlit = get_script_run_ctx() is not None
    except Exception:
        _in_streamlit = False

    if _in_streamlit:
        main()
    else:
        subprocess.run(["streamlit", "run", __file__] + sys.argv[1:])
