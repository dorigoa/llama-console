from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Optional

LogSink = Optional[Callable[[str], None]]


def setup_console_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging so messages are visible when running `python gui.py`."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("llama-orc")


def emit(message: str, sink: LogSink = None, level: int = logging.INFO) -> None:
    """Log to console and, optionally, to a UI sink such as NiceGUI's ui.log().push."""
    logging.getLogger("llama-orc").log(level, message)
    if sink is not None:
        try:
            sink(message)
        except Exception:
            logging.getLogger("llama-orc").exception("Unable to write message to UI log sink")
