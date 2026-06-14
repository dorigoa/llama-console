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
import urllib.error
from datetime import datetime, timezone
import streamlit as st
from pathlib import Path

from config_manager import get_settings

MODELS_JSON = Path(__file__).parent / "models.json"
START_SCRIPT = Path(__file__).parent / "start_model.py"

# Strip ANSI colour/cursor codes and bare carriage returns written by llama-server
# when it detects a TTY (which is exactly what we give it via the PTY).
_ANSI_RE = re.compile(rb"\x1b\[[0-9;]*[a-zA-Z]|\x1b[()][AB012]|\r")


# ─── helpers ──────────────────────────────────────────────────────────────────

def _save_persist(model_name: str, alias: str, pid: int, port: int, started_at: str, ready_at: str) -> None:
    settings = get_settings()
    data = {
        "model_name": model_name,
        "alias": alias,
        "pid": pid,
        "port": port,
        "api_url": f"http://127.0.0.1:{port}/v1",
        "started_at": started_at,
        "ready_at": ready_at,
    }
    Path(settings.PERSIST_FILE).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _clear_persist() -> None:
    settings = get_settings()
    p = Path(settings.PERSIST_FILE)
    if p.exists():
        p.unlink()


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

    if api_ready:
        st.session_state.server_ready = True
    else:
        # Model still loading — start a probe thread to catch when it's ready
        st.session_state.server_ready = False
        rq: queue.Queue = queue.Queue()
        st.session_state.ready_queue = rq
        threading.Thread(
            target=_probe_server,
            args=(recovered, port, model_name, alias, rq),
            daemon=True,
        ).start()


def _probe_server(
    proc: subprocess.Popen,
    port: int,
    model_name: str,
    alias: str,
    ready_q: queue.Queue,
    timeout: int = 300,
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
                    _save_persist(model_name, alias, proc.pid, port, started_at, ready_at)
                    ready_q.put(True)
                    return
        except Exception:
            pass
        time.sleep(3)

    ready_q.put(False)


def _fmt_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    b = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def _load_entries() -> list[dict]:
    """All models from models.json, including ones whose files are missing."""
    raw = json.loads(MODELS_JSON.read_text(encoding="utf-8"))
    base = Path(raw["MODEL_BASE_DIR"])
    entries = []
    for name, spec in raw.get("models", {}).items():
        fname = name if name.endswith(".gguf") else f"{name}.gguf"
        path = base / fname
        entries.append({
            "name": name,
            "alias": str(spec.get("ALIAS", name)),
            "path": path,
            "size": _fmt_size(path),
            "exists": path.exists(),
            "temp":  float(spec["TEMP"]),
            "top_p": float(spec["TOPP"]),
            "top_k": int(spec["TOPK"]),
            "min_p": float(spec["MINP"]),
        })
    return entries


def _is_running() -> bool:
    p = st.session_state.get("process")
    return p is not None and p.poll() is None


def _drain_queue() -> None:
    q: queue.Queue = st.session_state.log_queue
    lines: list[str] = st.session_state.log_lines
    while True:
        try:
            lines.append(q.get_nowait().rstrip())
        except queue.Empty:
            break


def _pty_reader(master_fd: int, q: queue.Queue) -> None:
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
                q.put(line.decode("utf-8", errors="replace"))
        except OSError:
            # EIO when slave side is fully closed (process exited)
            break
    if buf:
        q.put(buf.decode("utf-8", errors="replace"))
    try:
        os.close(master_fd)
    except OSError:
        pass


# ─── auto-refreshing log pane ─────────────────────────────────────────────────

@st.fragment(run_every=1)
def _log_pane() -> None:
    _drain_queue()

    # Drain probe result — triggers full page rerun on state change
    ready_q: queue.Queue = st.session_state.get("ready_queue", queue.Queue())
    try:
        result = ready_q.get_nowait()
        if st.session_state.get("server_ready") != result:
            st.session_state.server_ready = result
            st.rerun(scope="app")
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

    col_clear, _ = st.columns([1, 5])
    with col_clear:
        if st.button("Clear logs", key="clear_btn"):
            st.session_state.log_lines = []
            st.rerun(scope="fragment")


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
        "server_ready": None,  # None=idle, False=loading, True=ready
        "_recovery_done": False,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

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
        return f"{e['name']}   [{e['size']}]{flag}"

    idx = st.selectbox(
        "Model",
        range(len(entries)),
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
            if st.session_state.server_ready:
                badge_color, badge_text = "#22c55e", f"▶ ready — {st.session_state.running_model}"
            else:
                badge_color, badge_text = "#f59e0b", f"⏳ loading — {st.session_state.running_model}"
        elif st.session_state.process is not None:
            badge_color, badge_text = "#6b7280", "⏹ stopped"
        else:
            badge_color, badge_text = "#6b7280", "● idle"
        st.markdown(
            f'<span style="color:{badge_color};font-size:14px;font-weight:600">{badge_text}</span>',
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

        st.session_state.log_lines = []
        st.session_state.log_queue = queue.Queue()
        st.session_state.ready_queue = queue.Queue()
        st.session_state.server_ready = False

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

        threading.Thread(
            target=_pty_reader,
            args=(master_fd, st.session_state.log_queue),
            daemon=True,
        ).start()

        threading.Thread(
            target=_probe_server,
            args=(proc, settings.PORT_BIND, entry["name"], entry["alias"], st.session_state.ready_queue),
            daemon=True,
        ).start()

        st.rerun()

    if stop_clicked and st.session_state.process is not None:
        st.session_state.process.terminate()
        st.session_state.process = None
        st.session_state.running_model = None
        st.session_state.server_ready = None
        _clear_persist()
        st.rerun()

    # ── log window ────────────────────────────────────────────────────────────
    st.subheader("Logs")
    _log_pane()


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
