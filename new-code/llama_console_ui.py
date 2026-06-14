#!/usr/bin/env python3
"""Streamlit GUI for managing llama-server models via start_model.py."""

import os
import sys
import json
import pty
import re
import select
import queue
import threading
import subprocess
import streamlit as st
from pathlib import Path

from config_manager import get_settings

MODELS_JSON = Path(__file__).parent / "models.json"
START_SCRIPT = Path(__file__).parent / "start_model.py"

# Strip ANSI colour/cursor codes and bare carriage returns written by llama-server
# when it detects a TTY (which is exactly what we give it via the PTY).
_ANSI_RE = re.compile(rb"\x1b\[[0-9;]*[a-zA-Z]|\x1b[()][AB012]|\r")


# ─── helpers ──────────────────────────────────────────────────────────────────

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
            r, _, _ = select.select([master_fd], [], [], 0.2)
            if not r:
                continue
            chunk = os.read(master_fd, 4096)
            if not chunk:
                break
            chunk = _ANSI_RE.sub(b"", chunk)
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                q.put(line.decode("utf-8", errors="replace"))
        except OSError:
            # Slave side closed (process exited) → EIO on macOS/Linux
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
    proc = st.session_state.get("process")
    lines: list[str] = st.session_state.get("log_lines", [])

    if proc is not None and proc.poll() is not None and lines:
        code = proc.poll()
        if code == 0:
            st.success(f"Process exited cleanly (code {code})")
        else:
            st.error(f"Process exited with code {code}")

    tail = "\n".join(lines[-1000:])
    st.text_area(
        "log_area",
        value=tail,
        height=480,
        label_visibility="collapsed",
        key="log_textarea",
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
        "running_model": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

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

    col_drop, col_status = st.columns([5, 1])
    with col_drop:
        idx = st.selectbox(
            "Model",
            range(len(entries)),
            format_func=lambda i: _label(entries[i]),
            disabled=_is_running(),
        )
    with col_status:
        st.write("")
        st.write("")
        if _is_running():
            st.success(f"▶ running")
        elif st.session_state.process is not None:
            st.info("⏹ stopped")
        else:
            st.info("idle")

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
    col_start, col_stop, col_info = st.columns([1, 1, 5])

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
            st.write(f"**Model:** {st.session_state.running_model}")

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

        t = threading.Thread(
            target=_pty_reader,
            args=(master_fd, st.session_state.log_queue),
            daemon=True,
        )
        t.start()
        st.rerun()

    if stop_clicked and st.session_state.process is not None:
        st.session_state.process.terminate()
        st.session_state.running_model = None
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
