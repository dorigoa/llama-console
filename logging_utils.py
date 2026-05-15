from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from typing import Optional

LogSink = Optional[Callable[[str], None]]

#___________________________________________________________________________________________
def setup_console_logging(level: int = logging.INFO) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("llama-console")

#___________________________________________________________________________________________
def emit(message: str, sink: LogSink = None, level: int = logging.INFO) -> None:
    logging.getLogger("llama-console").log(level, message)
    if sink is not None:
        try:
            sink(message)
        except Exception:
            logging.getLogger("llama-console").exception("Unable to write message to UI log sink")
