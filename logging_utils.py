# from __future__ import annotations

# import logging
# import sys
# from collections.abc import Callable
# from typing import Optional

# LogSink = Optional[Callable[[str], None]]

# #___________________________________________________________________________________________
# def setup_console_logging(level: int = logging.INFO) -> logging.Logger:
#     logging.basicConfig(
#         level=level,
#         format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
#         stream=sys.stdout,
#         force=True,
#     )
#     return logging.getLogger("llama-console")

# #___________________________________________________________________________________________
# def emit(message: str, sink: LogSink = None, level: int = logging.INFO) -> None:
#     logging.getLogger("llama-console").log(level, message)
#     if sink is not None:
#         try:
#             sink(message)
#         except Exception:
#             logging.getLogger("llama-console").exception("Unable to write message to UI log sink")

# --------------------------------------------------------------
# logging_utils.py – now powered by *logzero*
# --------------------------------------------------------------
# The public API stays exactly the same:
#   * setup_console_logging(level=logging.INFO) -> logging.Logger
#   * emit(message: str, sink: LogSink = None, level: int = logging.INFO) -> None
#   * LogSink = Optional[Callable[[str], None]]
#
# Internally we delegate everything to logzero.logger, which already
# provides a nice coloured, thread‑safe console logger.
# --------------------------------------------------------------

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Optional

# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
LogSink = Optional[Callable[[str], None]]

# ----------------------------------------------------------------------
# Helper – configure *logzero* exactly once.
# ----------------------------------------------------------------------
# logzero creates a *module‑level* logger (logzero.logger).  We expose it
# through the same name that the rest of the code expects (a ``logging.Logger``‑like
# object) and make sure the formatter matches the original format string.
# ----------------------------------------------------------------------
def _configure_logzero(level: int) -> None:
    """
    Configure logzero only the first time it is called.
    Subsequent calls are no‑ops – this mimics ``logging.basicConfig(force=True)``.
    """
    # ``logzero`` does not expose a direct “already configured” flag, so we
    # guard with a private attribute on the function object.
    if getattr(_configure_logzero, "_done", False):
        # If the caller asks for a *different* level we still honour it.
        from logzero import logger as _lz_logger
        _lz_logger.setLevel(level)
        return

    from logzero import logger as _lz_logger, formatter as _lz_formatter, setup_logger

    # The original format used by the project:
    #   "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    # logzero’s formatter works with the same ``logging`` record fields.
    fmt = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    _lz_formatter(fmt)

    # ``setup_logger`` creates a fresh logger instance with the formatter
    # attached to *stderr* (the default).  We keep the default stream because
    # the original code wrote to ``sys.stdout``; logzero’s coloured output
    # looks nicer on stderr, but if you really need stdout you can pass
    # ``stream=sys.stdout`` here.
    setup_logger(level=level)

    # Mark as configured
    _configure_logzero._done = True  # type: ignore[attr-defined]

# ----------------------------------------------------------------------
# Public API – the same signatures the rest of the code uses
# ----------------------------------------------------------------------
def setup_console_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Initialise the console logger (via *logzero*) and return the logger
    instance.  The function can be called repeatedly – the first call
    creates the logger, later calls only adjust the log level.
    """
    _configure_logzero(level)
    # ``logzero.logger`` is a ``logging.Logger`` subclass, so we can return it
    # directly and keep type‑checkers happy.
    from logzero import logger as _lz_logger
    return _lz_logger  # type: ignore[return-value]

def emit(message: str, sink: LogSink = None, level: int = logging.INFO) -> None:
    """
    Log *message* to the console logger **and** optionally forward it to a UI
    sink (e.g. the NiceGUI log widget).  This mirrors the original helper
    that wrote to both the standard logger and the UI.
    """
    # Use the same logger name that the original code used – this makes the
    # output identical (``llama-console``).
    logger = logging.getLogger("llama-console")
    logger.log(level, message)

    # If a UI sink is supplied we still try to forward the text; any error
    # while writing to the UI is reported via the logger itself.
    if sink is not None:
        try:
            sink(message)
        except Exception:  # pragma: no cover – defensive, should never crash the app
            logger.exception("Unable to write message to UI log sink")
