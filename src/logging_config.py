"""Project-wide logging configuration.

Call :func:`setup_logging` exactly once at process start (entry points do
this automatically). Modules should then obtain their own logger with
``logging.getLogger(__name__)`` — never use bare ``print``.
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party libraries that are too chatty at INFO/DEBUG level.
_NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "asyncio")

_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger with a consistent stdout handler.

    Idempotent: repeated calls are ignored so libraries/entry points can
    call it defensively without duplicating handlers.
    """
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))

    root = logging.getLogger()
    root.setLevel(level.upper())
    root.addHandler(handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).debug("Logging configured at level %s", level.upper())
