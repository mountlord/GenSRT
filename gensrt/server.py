"""GenSRT Web Server — Flask app, error handling and the desktop launcher.

Architecture:
- Flask bound to 127.0.0.1, embedded in a pyWebView desktop window.
- Routes live in ``gensrt.api`` as one blueprint per concern (jobs, config,
  media, ocr); this module builds the app, registers them, and owns the
  JSON error handler, port discovery and ``launch_server``.
- Single active operation enforced with a threading.Lock (HTTP 409 when
  busy) — see ``gensrt.api._state``.
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
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.serving import WSGIRequestHandler

from gensrt.api import register as _register_routes
from gensrt.constants import SERVER_HOST, SERVER_PORT_RANGE

logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    static_folder=None,
    template_folder=str(Path(__file__).parent / "templates"),
)
_register_routes(app)

# Names that pre-date the split and are still imported from here (cli.py,
# tests, the packager's import check).  New code should import from the
# owning module in ``gensrt.api``; these stay so the split changes no caller.
from gensrt.api._state import OperationBusyError  # noqa: E402,F401
from gensrt.api.config import (  # noqa: E402,F401
    _CONFIG_VALIDATORS,
    _ct2_format_problem,
    _validate_config_patch,
    api_validate_model,
)
from gensrt.api.jobs import (  # noqa: E402,F401
    _BURN_STAGE_PREFIX,
    _stage_srt_for_burn,
    resolve_drop_targets,
)
from gensrt.api.ocr import (  # noqa: E402,F401
    _get_ocr_translator,
    _ocr_translator_cache,
    _resolve_ocr_engine_key,
)

# ── Quiet poll handler ────────────────────────────────────────────────────

class _QuietPollHandler(WSGIRequestHandler):
    """Suppress log noise from /api/operation_status polling."""

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        line = getattr(self, "requestline", "") or ""
        if "GET /api/operation_status " in line:
            return
        super().log_request(code, size)


@app.errorhandler(Exception)
def _api_errors_stay_json(exc):
    """Return JSON for /api/ failures instead of Flask's HTML error page.

    Every client here parses JSON. An HTML error page produces
    "Unexpected token '<'", which says nothing about what actually broke —
    so API routes report the status and the message in the shape the caller
    already expects.
    """
    from werkzeug.exceptions import HTTPException

    status = exc.code if isinstance(exc, HTTPException) else 500
    if not request.path.startswith("/api/"):
        # Not ours — let Werkzeug render its normal page. RETURNING the
        # HTTPException is how Flask is told "handle this the usual way";
        # re-raising re-enters this handler via handle_exception and turns
        # a plain 404 (/favicon.ico) into a 500 with a traceback.
        if isinstance(exc, HTTPException):
            return exc
        logger.exception("Unhandled error on %s", request.path)
        return "Internal Server Error", 500
    if status >= 500:
        logger.exception("Unhandled error on %s", request.path)
    return jsonify({
        "error": getattr(exc, "description", None) or str(exc) or "Server error",
        "path": request.path,
        "status": status,
    }), status


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
            "pywebview is not installed. Run: pip install -r requirements.txt\n"
            "Or run headless with --input FILE."
        )
        return

    # pywebview 5.4+ deprecated webview.OPEN_DIALOG / SAVE_DIALOG / FOLDER_DIALOG
    # in favour of the FileDialog enum.  Resolve the right symbols once so the
    # Api class can use stable names regardless of the installed version.
    try:
        _DLG_OPEN   = webview.FileDialog.OPEN
        _DLG_SAVE   = webview.FileDialog.SAVE
        _DLG_FOLDER = webview.FileDialog.FOLDER
    except AttributeError:
        _DLG_OPEN   = webview.OPEN_DIALOG    # type: ignore[attr-defined]
        _DLG_SAVE   = webview.SAVE_DIALOG    # type: ignore[attr-defined]
        _DLG_FOLDER = webview.FOLDER_DIALOG  # type: ignore[attr-defined]

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
                    _DLG_OPEN,
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
                return _DLG_FOLDER
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

        def select_video(self) -> str | None:
            """Open a native file picker restricted to video files.

            Called by the new UI's Load button and the click-on-video-area
            handler.  Returns the full filesystem path or ``None`` if the
            user cancelled.
            """
            if not getattr(self, "_window", None):
                return None
            file_types = (
                "Video files (*.mp4;*.mkv;*.webm;*.avi;*.mov;*.ts;*.m2ts;*.m4v)",
                "All files (*.*)",
            )
            try:
                result = self._window.create_file_dialog(
                    _DLG_OPEN,
                    allow_multiple=False,
                    file_types=file_types,
                )
                if result:
                    return result[0] if isinstance(result, (list, tuple)) else str(result)
            except Exception:
                logger.exception("select_video dialog failed")
            return None

        def select_srt(self) -> str | None:
            """Open a native file picker restricted to SRT subtitle files.

            Parallel to :meth:`select_video`, used by the click-on-SRT-area
            handler in player.js.  Native dialogs return the absolute path
            directly, which the HTML ``<input type="file">`` fallback can't
            do reliably in pywebview (File objects don't expose
            ``pywebviewFullPath``), so we need this dedicated entry point to
            make sibling-video discovery work for the file-picker path.

            Returns the full filesystem path or ``None`` if cancelled.
            """
            if not getattr(self, "_window", None):
                return None
            file_types = (
                "Subtitle files (*.srt)",
                "All files (*.*)",
            )
            try:
                result = self._window.create_file_dialog(
                    _DLG_OPEN,
                    allow_multiple=False,
                    file_types=file_types,
                )
                if result:
                    return result[0] if isinstance(result, (list, tuple)) else str(result)
            except Exception:
                logger.exception("select_srt dialog failed")
            return None

        def open_url(self, url: str) -> None:
            """Open a URL in the system's default browser.

            Called from the About panel / external links in the new UI.
            """
            import webbrowser
            try:
                webbrowser.open(str(url))
            except Exception:
                logger.exception("open_url failed: %s", url)

        def toggle_fullscreen(self) -> dict:
            """Toggle native window fullscreen.

            Called from the new UI's fullscreen control.  Returns
            ``{"ok": True}`` on success.
            """
            win = getattr(self, "_window", None)
            if not win:
                return {"ok": False, "error": "window_not_ready"}
            try:
                fn = getattr(win, "toggle_fullscreen", None)
                if callable(fn):
                    fn()
                    return {"ok": True}
                # Fallback for older pywebview that exposes .fullscreen instead.
                cur    = getattr(win, "fullscreen", False)
                set_fn = getattr(win, "set_fullscreen", None)
                if callable(set_fn):
                    set_fn(not cur)
                    return {"ok": True}
                setattr(win, "fullscreen", not cur)
                return {"ok": True}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        def save_srt_as(self, default_filename: str = "", initial_dir: str = "") -> str | None:
            """Native Save dialog for an SRT path (the Save As button).
            Returns the chosen full path or ``None`` if the user cancelled."""
            return self._save_dialog(("SubRip subtitles (*.srt)", "All files (*.*)"),
                                     default_filename or "subtitles.srt", initial_dir)

        def save_mp4_as(self, default_filename: str = "", initial_dir: str = "") -> str | None:
            """Native Save dialog for the MP4 rewrap of a transport stream."""
            # pywebview's filter parser allows only [\w ] in the description
            # (webview/util.py parse_file_type): no hyphen, no punctuation.
            return self._save_dialog(("MP4 video (*.mp4)", "All files (*.*)"),
                                     default_filename or "video.mp4", initial_dir)

        def _save_dialog(self, file_types: tuple, default_filename: str,
                         initial_dir: str = "") -> str | None:
            win = getattr(self, "_window", None)
            if not win:
                return None
            # Try the rich signature first, then degrade gracefully — pywebview's
            # create_file_dialog signature has drifted across versions.
            kwargs: dict = {"file_types": file_types, "save_filename": default_filename}
            if initial_dir:
                kwargs["directory"] = initial_dir
            for call_kwargs in (kwargs, {"file_types": file_types, "save_filename": kwargs["save_filename"]}, {"file_types": file_types}):
                try:
                    result = win.create_file_dialog(_DLG_SAVE, **call_kwargs)
                    break
                except TypeError:
                    continue
                except Exception:
                    # Surface it to the page: a swallowed failure here looks
                    # to the user like a button that does nothing.
                    logger.exception("save dialog failed")
                    raise
            else:
                return None
            if not result:
                return None
            return result[0] if isinstance(result, (list, tuple)) else str(result)

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

    # Create <app dir>/models if it is missing, so the convention is visible
    # in the install folder rather than only mentioned in an error message.
    # Belt and braces with the installer doing the same: this also covers a
    # user who copied the folder somewhere else, or an archive extractor that
    # dropped an empty directory.
    from gensrt.model_paths import ensure_models_dir

    ensure_models_dir()

    api = Api()

    window = webview.create_window(
        "GenSRT",
        url,
        width=1100,
        height=720,
        js_api=api,
        min_size=(800, 500),
        # pywebview disables text selection by default (text_select=False),
        # at the window level — below CSS, so no `user-select: text` rule can
        # override it. That made every error message unselectable, which is
        # why reporting one meant taking a screenshot.
        #
        # PROJECT CONVENTION: any message a user might need to report must be
        # selectable and copyable. The player still feels native because the
        # CSS sets user-select: none on the controls that need it; this only
        # removes the blanket ban.
        text_select=True,
    )
    api._window = window

    # Drag-and-drop handler — captures the full filesystem path of dropped
    # files (via pywebview's ``pywebviewFullPath`` File-object extension) and
    # routes by file type:
    #   • Video → window.tilesterSetVideoPath(path)
    #   • .srt  → window.gensrtLoadSrtFromPath(path)
    # Browser-mode drops (ObjectURL playback / FileReader) happen in player.js
    # and project.js — they don't reach this handler.
    def _on_drop(evt: dict) -> None:
        try:
            files = evt.get("dataTransfer", {}).get("files", [])
            if not files:
                return
            paths = [f.get("pywebviewFullPath") for f in files if f.get("pywebviewFullPath")]
            if not paths:
                return

            video_path, srt_path, srt_is_explicit = resolve_drop_targets(paths)

            if video_path:
                # skipSidecar when we already have the user's SRT in hand.
                opts = ", { skipSidecar: true }" if srt_is_explicit else ""
                window.evaluate_js(
                    f"window.tilesterSetVideoPath && "
                    f"window.tilesterSetVideoPath({json.dumps(video_path)}{opts});"
                )
            if srt_path:
                window.evaluate_js(
                    f"window.gensrtLoadSrtFromPath && "
                    f"window.gensrtLoadSrtFromPath({json.dumps(srt_path)}, "
                    f"{{ skipSiblingVideo: true }});"
                )
        except Exception:
            logger.exception("Drop handler failed")

    def _on_drag_over(_evt: dict) -> None:
        pass

    def _attach_dom_handlers() -> None:
        try:
            # Note: the package is named "pywebview" on PyPI but exports as the
            # ``webview`` Python module — hence the import path below.
            from webview.dom import DOMEventHandler
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

    # The window is closed.  A transport-stream rewrap still running would
    # outlive the process as an orphaned ffmpeg writing into the cache.
    from gensrt.remux import manager as _remux_manager
    _remux_manager.cancel_all()


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
