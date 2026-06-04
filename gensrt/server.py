"""GenSRT Web Server — thin HTTP adapter over operations.py

Architecture:
- Flask bound to 127.0.0.1, embedded in a pyWebView desktop window.
- Synchronous API calls for transcription operations.
- Single active operation enforced with a threading.Lock (HTTP 409 when busy).
- Progress polling via GET /api/operation_status (poll noise suppressed).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from flask import Flask, Response, jsonify, request, send_from_directory
from werkzeug.serving import WSGIRequestHandler

from gensrt.operations import (
    build_transcription_config,
    read_config_file,
    resolve_output_path,
    run_transcription,
)
from gensrt.constants import PIPELINE_PHASES, SERVER_HOST, SERVER_PORT_RANGE

logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder=None)

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


# ── Quiet poll handler ────────────────────────────────────────────────────

class _QuietPollHandler(WSGIRequestHandler):
    """Suppress log noise from /api/operation_status polling."""

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        line = getattr(self, "requestline", "") or ""
        if "GET /api/operation_status " in line:
            return
        super().log_request(code, size)


# ── Static files ──────────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).parent / "static"


@app.route("/")
def index():
    return send_from_directory(str(_STATIC_DIR), "index.html")


@app.route("/static/<path:filename>")
def static_files(filename: str):
    return send_from_directory(str(_STATIC_DIR), filename)


# ── API endpoints ─────────────────────────────────────────────────────────

@app.route("/api/status")
def api_status():
    """Health-check and version endpoint."""
    from gensrt import __version__
    return jsonify({"status": "ok", "version": __version__})


@app.route("/api/operation_status")
def api_operation_status():
    """Poll endpoint for active operation progress."""
    snap = _snapshot_active_operation()
    if snap is None:
        return jsonify({"active": False})
    return jsonify({"active": True, **snap})


@app.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    """Start a transcription job.

    Expected JSON body:
        {
            "input_path":         "/path/to/media.mkv",   // required
            "output_dir":         "/path/to/output/",     // optional
            "output_filename":    "custom.srt",           // optional
            "translation_engine": "google",               // optional
            "source_language":    "auto",                 // optional
            "no_translate":       false,                  // optional
            "no_vad":             false,                  // optional
            "model":              "large-v3-turbo",       // optional
        }
    """
    body: dict[str, Any] = request.get_json(silent=True) or {}

    input_path_str = body.get("input_path", "").strip()
    if not input_path_str:
        return jsonify({"status": "error", "message": "input_path is required"}), 400

    input_path = Path(input_path_str).expanduser().resolve()
    if not input_path.exists() or not input_path.is_file():
        return jsonify({"status": "error", "message": f"File not found: {input_path}"}), 400

    output_dir_str = body.get("output_dir") or None
    output_dir = Path(output_dir_str).expanduser().resolve() if output_dir_str else None
    output_filename = body.get("output_filename") or None
    output_path = resolve_output_path(input_path, output_dir, output_filename)

    # Build config from defaults + request overrides
    try:
        file_cfg = read_config_file(default_if_missing=True)
    except Exception:
        file_cfg = {}

    overrides: dict[str, Any] = {}
    if "translation_engine" in body:
        overrides["translation_engine"] = body["translation_engine"]
    if "source_language" in body:
        overrides["source_language"] = body["source_language"]
    if body.get("no_translate"):
        overrides["translate"] = False
    if body.get("no_vad"):
        overrides["vad_enabled"] = False
    if "model" in body:
        overrides["model"] = body["model"]

    merged = {**file_cfg, **overrides}
    config = build_transcription_config(merged, auto_detect_backend=True)

    # Gate: only one job at a time
    try:
        _begin_long_operation(input_path.name)
    except OperationBusyError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 409

    def _run() -> None:
        try:
            run_transcription(
                input_path=input_path,
                output_path=output_path,
                config=config,
                progress=_make_progress_cb(),
                status=_make_status_cb(),
            )
            _update_active_operation(
                message=f"Complete — {output_path.name}",
                current=PIPELINE_PHASES,
                total=PIPELINE_PHASES,
            )
        except Exception as exc:
            logger.exception("Transcription failed: %s", exc)
            _update_active_operation(message=f"Error: {exc}")
        finally:
            _end_long_operation()

    thread = threading.Thread(target=_run, daemon=True, name="gensrt-transcribe")
    thread.start()

    return jsonify({
        "status": "started",
        "input": str(input_path),
        "output": str(output_path),
    })


@app.route("/api/config", methods=["GET"])
def api_get_config():
    """Return the current resolved configuration."""
    try:
        file_cfg = read_config_file(default_if_missing=True)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

    from gensrt.config import BUILTIN_DEFAULTS
    merged = {**BUILTIN_DEFAULTS, **file_cfg}
    return jsonify(merged)


@app.route("/api/engines")
def api_engines():
    """Return available translation engines."""
    from gensrt.translation.factory import available_engines
    return jsonify({"engines": available_engines()})


# ── Port discovery ────────────────────────────────────────────────────────

def _find_free_port(host: str = SERVER_HOST) -> int:
    start, end = SERVER_PORT_RANGE
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"No free port found in range {start}–{end}")


# ── Launch ────────────────────────────────────────────────────────────────

def launch_server(
    open_path: str | None = None,
    *,
    console: bool = False,
) -> None:
    """Start the Flask server and open the pyWebView desktop window.

    Args:
        open_path: Optional media file path to pre-load in the UI.
        console:   If ``True``, open the pyWebView DevTools console.
    """
    try:
        import webview
    except ImportError:
        logger.error(
            "pywebview is not installed. Run: pip install pywebview==5.2\n"
            "Or run headless with --input FILE."
        )
        return

    port = _find_free_port()
    url = f"http://{SERVER_HOST}:{port}/"

    # Start Flask in a background daemon thread
    flask_thread = threading.Thread(
        target=lambda: app.run(
            host=SERVER_HOST,
            port=port,
            debug=False,
            use_reloader=False,
            request_handler=_QuietPollHandler,
        ),
        daemon=True,
        name="gensrt-flask",
    )
    flask_thread.start()

    # Brief pause so Flask is ready before pyWebView opens
    time.sleep(0.5)

    logger.info("Flask server: %s", url)

    # ── pyWebView Api class ───────────────────────────────────────────────

    class Api:
        """Methods exposed to the frontend via pywebview.window.pywebviewApi.*"""

        _window: Any = None

        @staticmethod
        def _pwv_pick_one_file(window, file_types: tuple) -> str | None:
            try:
                result = window.create_file_dialog(
                    webview.OPEN_DIALOG,
                    allow_multiple=False,
                    file_types=file_types,
                )
                if result:
                    return result[0]
            except Exception:
                logger.exception("File dialog failed")
            return None

        @staticmethod
        def _pwv_folder_dialog_constant():
            try:
                return webview.FOLDER_DIALOG
            except AttributeError:
                return None

        def select_file(self) -> str | None:
            """Open a native file picker for media files."""
            if not getattr(self, "_window", None):
                return None
            file_types = (
                ("Media files",
                 "*.mp4;*.mkv;*.avi;*.mov;*.webm;*.ts;*.m2ts;*.mp3;*.wav;*.flac;*.aac;*.m4a"),
                ("All files", "*.*"),
            )
            return Api._pwv_pick_one_file(self._window, file_types)

        def select_folder(self) -> str | None:
            """Open a native folder picker."""
            if not getattr(self, "_window", None):
                return None
            dlg = Api._pwv_folder_dialog_constant()
            if dlg is None:
                return None
            try:
                result = self._window.create_file_dialog(dlg)
            except Exception:
                return None
            if not result:
                return None
            return result[0] if isinstance(result, (list, tuple)) else str(result)

        def select_output_folder(self) -> str | None:
            """Alias for select_folder — used by the output directory picker."""
            return self.select_folder()

        def get_open_path(self) -> str | None:
            """Return any pre-loaded path (passed from CLI)."""
            return open_path

    api = Api()

    window = webview.create_window(
        "GenSRT",
        url,
        width=1100,
        height=720,
        js_api=api,
        min_size=(800, 500),
    )
    api._window = window

    # Drag-and-drop handler
    def _on_drop(evt: dict) -> None:
        try:
            files = evt.get("dataTransfer", {}).get("files", [])
            if not files:
                return
            paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
            if paths:
                window.evaluate_js(
                    f"window.gensrtSetInput && window.gensrtSetInput({json.dumps(paths[0])});"
                )
        except Exception:
            logger.exception("Drop handler failed")

    def _on_drag_over(_evt: dict) -> None:
        pass

    def _attach_dom_handlers() -> None:
        try:
            from pywebview.dom import DOMEventHandler
            window.dom.document.events.dragover += DOMEventHandler(
                _on_drag_over,
                prevent_default=True,
                stop_propagation=True,
            )
            window.dom.document.events.drop += DOMEventHandler(
                _on_drop,
                prevent_default=True,
                stop_propagation=True,
            )
            logger.debug("pyWebView drag-and-drop handlers attached")
        except Exception:
            logger.debug("Could not attach drag-and-drop handlers (pywebview version mismatch?)")

    window.events.loaded += _attach_dom_handlers

    webview.start(debug=console)


if __name__ == "__main__":
    if os.environ.get("GENSRT_SERVER_MODE") != "1":
        print(
            "ERROR: server.py should be launched via the gensrt CLI, not directly.",
            file=os.sys.stderr,
        )
        raise SystemExit(1)

    from gensrt.utils.logging_config import setup_logging
    setup_logging("INFO")
    launch_server()
