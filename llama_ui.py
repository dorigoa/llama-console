#!/usr/bin/env python3
"""Streamlit web UI for llama-console / start_model."""

import json
import select
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from config_manager import get_settings
from model import Model, load_models
from start_model import (
    MODELS_JSON,
    ServerHostUnreachable,
    _server_pids,
    stop_server,
)

st.set_page_config(
    page_title="LLama Console",
    page_icon="🦙",
    layout="wide",
)

settings = get_settings()

st.markdown("""
<style>
[data-testid="stSelectbox"] * {
    font-size: 1.5rem !important;
}
[data-baseweb="popover"] * {
    font-size: 1.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ──────────────────────────────────────────────────────
_DEFAULTS: dict = {
    "models": None,       # list[Model] | None  (None = not yet loaded)
    "load_error": None,   # str | None
    "server_proc": None,  # subprocess.Popen | None
    "log_lines": [],      # list[str]
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load_models_now() -> None:
    try:
        models = load_models(
            MODELS_JSON,
            remote_host=settings.LLAMA_SERVER_HOST,
            remote_user=settings.LLAMA_SERVER_USER,
        )
        st.session_state.models = models
        st.session_state.load_error = None
    except Exception as exc:
        st.session_state.models = []
        st.session_state.load_error = str(exc)


@st.cache_data(show_spinner=False)
def _read_models_json() -> dict:
    """Return the raw parsed models.json (cached for the server lifetime)."""
    with open(MODELS_JSON, encoding="utf-8") as f:
        return json.load(f)


def _native_ctx(model_name: str) -> int:
    raw = _read_models_json()
    spec = raw.get("models", {}).get(model_name, {})
    return int(spec.get("native_ctx", spec.get("ctx", 0)))


def _get_server_status() -> tuple[bool | None, list[str]]:
    """Return (running, pids). running=None means host unreachable."""
    try:
        pids = _server_pids()
        return bool(pids), pids
    except ServerHostUnreachable:
        return None, []
    except Exception:
        return None, []


def _drain_proc() -> None:
    """Non-blocking read of available output from the server subprocess."""
    proc: subprocess.Popen | None = st.session_state.server_proc
    if proc is None or proc.stdout is None:
        return
    try:
        while True:
            ready, _, _ = select.select([proc.stdout], [], [], 0)
            if not ready:
                break
            line: str = proc.stdout.readline()
            if not line:
                break
            st.session_state.log_lines.append(line.rstrip())
    except Exception:
        pass
    rc = proc.poll()
    if rc is not None:
        st.session_state.log_lines.append(f"[process exited, rc={rc}]")
        st.session_state.server_proc = None


def _append_log(msg: str) -> None:
    st.session_state.log_lines.append(msg)


# ── Load models on first render ─────────────────────────────────────────────
if st.session_state.models is None:
    with st.spinner(f"Loading model list from {settings.LLAMA_SERVER_HOST}…"):
        _load_models_now()

# ── Title ───────────────────────────────────────────────────────────────────
st.title("🦙 LLama Console")
st.caption(settings.UI_TITLE)
st.divider()

# ── Status bar ──────────────────────────────────────────────────────────────
status_col, refresh_col = st.columns([6, 1])
with refresh_col:
    if st.button("↻ Refresh\nmodels", use_container_width=True):
        st.session_state.models = None
        _read_models_json.clear()
        st.rerun()
with status_col:
    is_running, pids = _get_server_status()
    if is_running is None:
        st.warning(f"⚠️ Host **{settings.LLAMA_SERVER_HOST}** unreachable")
    elif is_running:
        st.success(f"✅ llama-server **running** — PID(s): {', '.join(pids)}")
    else:
        st.info("⭕ llama-server **not running**")

if st.session_state.load_error:
    st.error(f"Model load error: {st.session_state.load_error}")

# ── Model list guard ────────────────────────────────────────────────────────
models: list[Model] = st.session_state.models or []
if not models:
    st.warning(
        "No models available. "
        "Check that the remote host is reachable and model files exist."
    )
    st.stop()

# ── Model selector ──────────────────────────────────────────────────────────
model_names = [m.model_name for m in models]
selected_name: str = st.selectbox(
    "Select model",
    model_names,
    key="model_select",
)
model: Model = next(m for m in models if m.model_name == selected_name)
native = _native_ctx(selected_name)

# ── Model details expander ───────────────────────────────────────────────────
with st.expander("Model details", expanded=True):
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Size", f"{model.size_gib:.1f} GiB" if model.size_gib else "—")
    mc2.metric("Configured ctx", f"{model.ctxsize:,}")
    mc3.metric("Native ctx (max)", f"{native:,}")
    mc4.metric("RPC servers", len(model.rpcservers))

    notes = []
    if model.rpcservers:
        rpc_str = "  •  ".join(f"{s.IP}:{s.PORT}" for s in model.rpcservers)
        notes.append(f"**RPC:** {rpc_str}")
    if model.fitt:
        notes.append(f"**fitt:** {model.fitt}")
    if model.kvquant:
        notes.append(f"**kvquant:** {model.kvquant}")
    if model.ub:
        notes.append(f"**ub:** {model.ub}")
    if model.b:
        notes.append(f"**b:** {model.b}")
    if notes:
        st.caption("   |   ".join(notes))

# ── Launch parameters ────────────────────────────────────────────────────────
st.subheader("Launch parameters")

# Context slider — capped at native_ctx; step must divide evenly into (value-min)
# All ctx values in models.json are multiples of 512, so step=512 works cleanly.
_CTX_STEP = 512
_ctx_max = max(_CTX_STEP, (native // _CTX_STEP) * _CTX_STEP)
_ctx_default = max(_CTX_STEP, (min(model.ctxsize, native) // _CTX_STEP) * _CTX_STEP)

ctx: int = st.slider(
    "Context size (tokens)",
    min_value=_CTX_STEP,
    max_value=_ctx_max,
    value=_ctx_default,
    step=_CTX_STEP,
    key=f"ctx_{selected_name}",
)

p1, p2, p3, p4 = st.columns(4)
temp  = p1.number_input("Temperature", 0.0,  2.0, value=model.temperature,    step=0.05, format="%.2f", key=f"temp_{selected_name}")
top_p = p2.number_input("Top-P",       0.0,  1.0, value=model.top_p,          step=0.01, format="%.2f", key=f"topp_{selected_name}")
top_k = p3.number_input("Top-K",         0,  500, value=model.top_k,          step=1,                   key=f"topk_{selected_name}")
min_p = p4.number_input("Min-P",      -1.0,  1.0, value=float(model.min_p),   step=0.01, format="%.2f", key=f"minp_{selected_name}")

# ── Action buttons ───────────────────────────────────────────────────────────
btn_l, btn_s = st.columns(2)

with btn_l:
    if st.button("🚀 Launch model", type="primary", use_container_width=True):
        cmd = [sys.executable, str(_HERE / "start_model.py"), selected_name,
               "--override-ctx", str(ctx)]
        if abs(temp  - model.temperature)  > 1e-9: cmd += ["--override-temp",  str(temp)]
        if abs(top_p - model.top_p)        > 1e-9: cmd += ["--override-top-p", str(top_p)]
        if int(top_k) != model.top_k:              cmd += ["--override-top-k", str(int(top_k))]
        if abs(min_p - model.min_p)        > 1e-9: cmd += ["--override-min-p", str(min_p)]

        st.session_state.log_lines = [f"$ {' '.join(cmd)}", ""]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        st.session_state.server_proc = proc
        st.rerun()

with btn_s:
    if st.button("⛔ Stop server", use_container_width=True):
        proc = st.session_state.server_proc
        if proc and proc.poll() is None:
            proc.terminate()
            st.session_state.server_proc = None
            _append_log("[Terminated by UI]")
        else:
            with st.spinner("Stopping remote server…"):
                try:
                    ok = stop_server()
                    _append_log("[Server stopped]" if ok else "[Could not stop server]")
                except ServerHostUnreachable as exc:
                    _append_log(f"[Host unreachable: {exc}]")
                except Exception as exc:
                    _append_log(f"[Stop error: {exc}]")
        st.rerun()

# ── Log area ─────────────────────────────────────────────────────────────────
st.subheader("Server output")

_drain_proc()

log_hdr, clr_btn = st.columns([6, 1])
with clr_btn:
    if st.button("Clear log", use_container_width=True):
        st.session_state.log_lines = []
        st.rerun()

log_text = "\n".join(st.session_state.log_lines[-400:])
st.text_area(
    "log",
    value=log_text,
    height=420,
    label_visibility="collapsed",
)

# Auto-rerun every second while the server subprocess is alive so that
# new log lines are streamed to the browser without manual refresh.
if st.session_state.server_proc is not None:
    time.sleep(1)
    st.rerun()
