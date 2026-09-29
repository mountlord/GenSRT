"""Single-operation gate shared by the job routes.

One transcription / extraction / burn at a time; a second POST gets HTTP 409
with the name of the file already in flight.  Progress and status callbacks
write into ``_active_operation`` under ``_operation_state_lock`` and
``/api/operation_status`` reads a snapshot of it from another thread.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

from gensrt.constants import PIPELINE_PHASES

logger = logging.getLogger(__name__)


# ── Single-operation gate ─────────────────────────────────────────────────

_operation_lock = threading.Lock()
_operation_state_lock = threading.Lock()
_active_operation: dict[str, Any] | None = None


class OperationBusyError(RuntimeError):
    """Raised when a transcription operation is already in progress."""


def _format_busy_message(active: dict[str, Any] | None) -> str:
    if not active:
        return "A transcription operation is already in progress. Please wait."
    name = active.get("filename") or "unknown file"
    elapsed = max(0, int(time.time() - (active.get("started_at") or time.time())))
    return f"Already transcribing {name!r} ({elapsed}s elapsed). Please wait."


def _begin_long_operation(filename: str) -> None:
    global _active_operation

    if not _operation_lock.acquire(blocking=False):
        with _operation_state_lock:
            active = dict(_active_operation) if _active_operation else None
        raise OperationBusyError(_format_busy_message(active))

    now = time.time()
    with _operation_state_lock:
        _active_operation = {
            "filename": filename,
            "started_at": now,
            "updated_at": now,
            "message": "Starting…",
            "current": 0,
            "total": PIPELINE_PHASES,
            "percent": 0.0,
        }


def _end_long_operation() -> None:
    global _active_operation

    with _operation_state_lock:
        _active_operation = None

    if _operation_lock.locked():
        try:
            _operation_lock.release()
        except RuntimeError:
            pass


def _update_active_operation(
    *,
    message: str | None = None,
    current: int | None = None,
    total: int | None = None,
) -> None:
    with _operation_state_lock:
        if _active_operation is None:
            return
        _active_operation["updated_at"] = time.time()
        if message is not None:
            _active_operation["message"] = str(message)
        if current is not None:
            _active_operation["current"] = max(0, int(current))
        if total is not None:
            _active_operation["total"] = max(0, int(total))

        cur = int(_active_operation.get("current") or 0)
        tot = int(_active_operation.get("total") or 0)
        pct = (cur / tot * 100.0) if tot > 0 else 0.0
        _active_operation["percent"] = max(0.0, min(100.0, pct))


def _snapshot_active_operation() -> dict[str, Any] | None:
    with _operation_state_lock:
        if _active_operation is None:
            return None
        snap = dict(_active_operation)
    snap["elapsed_s"] = max(0.0, time.time() - (snap.get("started_at") or time.time()))
    return snap


def _make_progress_cb() -> Callable[[int, int], None]:
    def _cb(current: int, total: int) -> None:
        _update_active_operation(current=current, total=total)
    return _cb


def _make_status_cb() -> Callable[[str], None]:
    def _cb(message: str) -> None:
        _update_active_operation(message=message)
    return _cb
