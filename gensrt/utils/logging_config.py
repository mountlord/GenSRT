"""Logging configuration for GenSRT.

Uses Rich for coloured, readable console output.
Call :func:`setup_logging` once at startup before any log messages are emitted.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a Rich console handler.

    Args:
        level: Log level string — ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.
               Case-insensitive.
    """
    global _CONFIGURED

    numeric = getattr(logging, level.upper(), logging.INFO)

    try:
        from rich.logging import RichHandler

        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
    except ImportError:
        handler = logging.StreamHandler(sys.stderr)  # type: ignore[assignment]
        handler.setFormatter(
            logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s: %(message)s")
        )

    root = logging.getLogger()

    if _CONFIGURED:
        # Update level only on re-calls (e.g. --verbose overrides early setup)
        root.setLevel(numeric)
        for h in root.handlers:
            h.setLevel(numeric)
        return

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric)
    handler.setLevel(numeric)

    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "filelock", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
