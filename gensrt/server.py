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
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from werkzeug.serving import WSGIRequestHandler

from gensrt.operations import (
    build_transcription_config,
    read_config_file,
    resolve_output_path,
    run_transcription,
    write_config_file,
)
from gensrt.constants import PIPELINE_PHASES, SERVER_HOST, SERVER_PORT_RANGE

logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    static_folder=None,
    template_folder=str(Path(__file__).parent / "templates"),
)

# ── Single-operation gate ─────────────────────────────────────────────────

_operation_lock = threading.Lock()
_operation_state_lock = threading.Lock()
_active_operation: dict[str, Any] | None = None

# Serialises POST /api/config writes (a fast Save spam would otherwise race).
_config_write_lock = threading.Lock()


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


# ── Media path validation & metadata helpers ──────────────────────────────
#
# Used by /api/media (serves video bytes with HTTP Range support) and
# /api/video_info (returns fps/duration via ffprobe).  All caller-supplied
# paths go through _validate_readable_path before any disk access.

_VIDEO_EXTS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".webm", ".avi", ".mov", ".ts", ".m2ts", ".mts", ".m4v"}
)
_READABLE_EXTS: frozenset[str] = _VIDEO_EXTS | frozenset({".srt"})


def _find_sibling_video(srt_path: Path) -> Path | None:
    """Return the first sibling video file next to *srt_path*, or None.

    Looks for ``<basename>.{mp4,mkv,ts,...}`` in the same directory.  Used by
    the drop handler and the GET /api/srt endpoint to keep the SRT-load and
    video-load UI flows symmetric.
    """
    for ext in _VIDEO_EXTS:
        candidate = srt_path.with_suffix(ext)
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _validate_readable_path(path_str: str) -> tuple[Path, str | None]:
    """Resolve and validate a caller-supplied absolute path for read-only endpoints.

    Returns ``(resolved_path, error_msg_or_None)``.  Only files with allowed
    extensions that exist on disk are accepted.
    """
    if not path_str or not isinstance(path_str, str):
        return Path(), "path must be a non-empty string"
    try:
        p = Path(path_str).expanduser().resolve()
    except Exception as exc:
        return Path(path_str), f"Invalid path: {exc}"
    if p.suffix.lower() not in _READABLE_EXTS:
        return p, (
            f"Unsupported file type '{p.suffix}'. "
            f"Accepted: {', '.join(sorted(_READABLE_EXTS))}"
        )
    if not p.exists() or not p.is_file():
        return p, f"File not found: {p}"
    return p, None


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".mp4", ".m4v"}:
        return "video/mp4"
    if ext == ".webm":
        return "video/webm"
    if ext == ".mkv":
        return "video/x-matroska"
    if ext == ".ts":
        return "video/mp2t"
    if ext == ".mov":
        return "video/quicktime"
    if ext == ".srt":
        return "application/x-subrip"
    return "application/octet-stream"


def _parse_rate_to_float(rate: str) -> Optional[float]:
    """Parse ffprobe frame-rate fields like ``'60000/1001'`` into a float."""
    if not rate:
        return None
    s = str(rate).strip()
    if not s:
        return None
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            num = float(a.strip())
            den = float(b.strip())
            return num / den if den else None
        return float(s)
    except Exception:
        return None


# ── SRT file helpers ──────────────────────────────────────────────────────


def _find_sidecar_srt(video_path: Path) -> Optional[Path]:
    """Look for ``<basename>.srt`` next to a video file.

    Returns the SRT path if it exists, else ``None``.
    """
    candidate = video_path.with_suffix(".srt")
    return candidate if candidate.exists() and candidate.is_file() else None


def _validate_srt_save_path(path_str: str) -> tuple[Path, str | None]:
    """Resolve a caller-supplied destination path for SRT writes.

    The file may not exist yet (that's the common Save case), but the parent
    directory must, and the suffix must be ``.srt``.
    """
    if not path_str or not isinstance(path_str, str):
        return Path(), "path must be a non-empty string"
    try:
        p = Path(path_str).expanduser().resolve()
    except Exception as exc:
        return Path(path_str), f"Invalid path: {exc}"
    if p.suffix.lower() != ".srt":
        return p, "Destination path must end with .srt"
    if not p.parent.exists() or not p.parent.is_dir():
        return p, f"Parent directory does not exist: {p.parent}"
    return p, None


# Lock for serializing SRT writes (matches the _config_write_lock pattern).
_srt_write_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("review.html")


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
    """Poll endpoint for active operation progress.

    Response shape (matches what the right-pane polling client expects):
        idle:   {"status": "idle"}
        active: {"status": "active",
                 "operation": {"kind": "transcribe",
                               "message": "...", "current": N, "total": M,
                               "percent": <0..100>}}

    Percent is derived from current / total — the existing
    _update_active_operation only tracks the raw counters, so the JSON
    response is the right place to compute it for the UI.
    """
    snap = _snapshot_active_operation()
    if snap is None:
        return jsonify({"status": "idle"})

    total   = snap.get("total")   or 0
    current = snap.get("current") or 0
    percent = (100.0 * current / total) if total > 0 else 0.0

    return jsonify({
        "status": "active",
        "operation": {
            "kind":    "transcribe",   # only one job type today
            "message": snap.get("message", "Working..."),
            "current": current,
            "total":   total,
            "percent": percent,
        },
    })


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

    # Run synchronously on the request thread.  Progress polling continues
    # to work because Flask serves /api/operation_status on different threads
    # — the pipeline's progress callback updates shared state read by the
    # polling endpoint.  The HTTP response only returns when transcription
    # is actually complete (or has failed).
    response_data: dict[str, Any]
    status_code = 200
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
        response_data = {
            "status": "ok",
            "input":  str(input_path),
            "output": str(output_path),
        }
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        _update_active_operation(message=f"Error: {exc}")
        response_data = {"status": "error", "message": str(exc)}
        status_code = 500
    finally:
        _end_long_operation()

    return jsonify(response_data), status_code


# ── Config persistence (POST /api/config) ─────────────────────────────────
#
# Validation schema for keys accepted by POST /api/config.  Unknown keys are
# rejected.  Bounds are deliberately wider than the UI's HTML5 ranges so the
# UI can be tightened without backend changes, but tight enough that obviously
# broken values can't be written to gensrt-config.json.

_MODEL_CHOICES = {
    "tiny", "base", "small", "medium", "large",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo",
}
_COMPUTE_CHOICES = {"float32", "float16", "int8_float16", "int8"}
_DEVICE_CHOICES = {"cuda", "cpu", "auto"}
_BACKEND_CHOICES = {"cuda", "rocm", "xpu", "cpu"}
_ENGINE_CHOICES = {"google", "nllb", "marian", "none"}
_LOG_LEVEL_CHOICES = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

_INT_KEYS = {
    "gpu_id", "vad_min_speech_ms", "vad_min_silence_ms", "vad_speech_pad_ms",
}


def _v_str(x):
    return (True, "") if isinstance(x, str) and x else (False, "must be a non-empty string")

def _v_str_in(allowed):
    def inner(x):
        if not isinstance(x, str):
            return False, "must be a string"
        if x not in allowed:
            return False, f"must be one of {sorted(allowed)}"
        return True, ""
    return inner

def _v_bool(x):
    return (True, "") if isinstance(x, bool) else (False, "must be true or false")

def _v_num_range(lo, hi, *, integer=False):
    def inner(x):
        # bool is a subclass of int — reject explicitly to avoid True == 1 surprises.
        if isinstance(x, bool):
            return False, "must be numeric, not boolean"
        if not isinstance(x, (int, float)):
            return False, "must be numeric"
        if not (lo <= x <= hi):
            return False, f"must be between {lo} and {hi}"
        if integer and isinstance(x, float) and not x.is_integer():
            return False, "must be an integer"
        return True, ""
    return inner

def _v_str_or_null(x):
    if x is None or isinstance(x, str):
        return True, ""
    return False, "must be string or null"


_CONFIG_VALIDATORS = {
    # Transcription
    "model":                   _v_str_in(_MODEL_CHOICES),
    "device":                  _v_str_in(_DEVICE_CHOICES),
    "compute_type":            _v_str_in(_COMPUTE_CHOICES),
    "backend":                 _v_str_in(_BACKEND_CHOICES),
    "gpu_id":                  _v_num_range(0, 7, integer=True),
    "source_language":         _v_str,
    "vad_enabled":             _v_bool,
    "vad_threshold":           _v_num_range(0.0, 1.0),
    "vad_min_speech_ms":       _v_num_range(50, 10000, integer=True),
    "vad_min_silence_ms":      _v_num_range(100, 10000, integer=True),
    "vad_speech_pad_ms":       _v_num_range(0, 2000, integer=True),
    "max_subtitle_duration_s": _v_num_range(0.0, 60.0),
    "min_subtitle_duration_s": _v_num_range(0.0, 60.0),
    "translation_engine":      _v_str_in(_ENGINE_CHOICES),
    "translate":               _v_bool,
    # Non-transcription (preserved-through, not currently surfaced in UI)
    "output":                  _v_str_or_null,
    "output_filename":         _v_str_or_null,
    "recurse":                 _v_bool,
    "log_level":               _v_str_in(_LOG_LEVEL_CHOICES),
}


def _resolve_config_save_path() -> Path:
    """Return the path POST /api/config should write to.

    Mirrors the auto-discovery used by GET /api/config so saves land where the
    next read will find them.  When no file exists yet, falls back to the CWD.
    """
    from gensrt.config import _find_config_file
    existing = _find_config_file()
    if existing is not None:
        return existing
    return Path.cwd() / "gensrt-config.json"


def _validate_config_patch(patch: Any) -> tuple[dict, dict]:
    """Validate a partial config update.

    Returns ``(sanitized, errors)``.  ``errors`` is empty when all keys pass.
    Int-typed numeric values are coerced to ``int`` for clean JSON output.
    """
    if not isinstance(patch, dict):
        return {}, {"_root": "request body must be a JSON object"}

    sanitized: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for key, value in patch.items():
        validator = _CONFIG_VALIDATORS.get(key)
        if validator is None:
            errors[key] = "unknown configuration key"
            continue
        ok, msg = validator(value)
        if not ok:
            errors[key] = msg
            continue
        sanitized[key] = int(value) if key in _INT_KEYS else value

    return sanitized, errors


@app.route("/api/config", methods=["POST"])
def api_save_config():
    """Persist a partial config update to ``gensrt-config.json``.

    Body: JSON object with any subset of allowed keys.
    Behaviour:
      1. Validate every supplied key; reject the whole request on any error.
      2. Acquire the config-write lock to serialise concurrent saves.
      3. If the destination file exists, write ``<name>.bak`` first.  If the
         backup fails, abort the save (no overwrite without a recovery copy).
      4. Merge the patch on top of the existing file contents (preserves keys
         the UI doesn't surface, e.g. ``output``, ``recurse``).
      5. Write the merged dict to disk.

    Returns:
      200 ``{status: "success", saved: {...}, path: "..."}`` on success.
      400 ``{status: "error", errors: {...}}`` on validation failure.
      400 ``{status: "error", message: "..."}`` on malformed JSON.
      500 ``{status: "error", message: "..."}`` on backup or write failure.
    """
    patch = request.get_json(silent=True)
    if patch is None:
        return jsonify({
            "status": "error",
            "message": "Request body must be a JSON object.",
        }), 400

    sanitized, errors = _validate_config_patch(patch)
    if errors:
        return jsonify({"status": "error", "errors": errors}), 400

    if not sanitized:
        return jsonify({"status": "success", "message": "Nothing to save.", "saved": {}})

    with _config_write_lock:
        save_path = _resolve_config_save_path()

        # Backup before overwrite — abort if we can't, to prevent data loss.
        if save_path.exists():
            backup_path = save_path.parent / (save_path.name + ".bak")
            try:
                backup_path.write_bytes(save_path.read_bytes())
                logger.info("Config backup written: %s", backup_path)
            except OSError as exc:
                logger.error("Backup failed for %s: %s", save_path, exc)
                return jsonify({
                    "status": "error",
                    "message": (
                        f"Could not write backup file ({exc}). "
                        "Save aborted to prevent data loss."
                    ),
                }), 500

        # Merge with existing contents so we don't lose keys the UI doesn't expose.
        try:
            existing = read_config_file(default_if_missing=True) or {}
        except Exception as exc:
            logger.error("Could not read existing config before merge: %s", exc)
            return jsonify({
                "status": "error",
                "message": f"Existing config could not be read: {exc}",
            }), 500

        merged = {**existing, **sanitized}

        try:
            write_config_file(save_path, merged)
        except Exception as exc:
            logger.exception("Config save failed")
            return jsonify({"status": "error", "message": str(exc)}), 500

    return jsonify({
        "status": "success",
        "message": f"Saved {len(sanitized)} field(s) to {save_path.name}.",
        "saved": sanitized,
        "path": str(save_path),
    })


@app.route("/api/config", methods=["GET"])
def api_get_config():
    """Return the current resolved configuration.

    Response shape:
        {"status": "success", "config": {<merged defaults + file overrides>}}
    """
    try:
        file_cfg = read_config_file(default_if_missing=True)
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

    from gensrt.config import BUILTIN_DEFAULTS
    merged = {**BUILTIN_DEFAULTS, **file_cfg}
    return jsonify({"status": "success", "config": merged})


@app.route("/api/engines")
def api_engines():
    """Return available translation engines."""
    from gensrt.translation.factory import available_engines
    return jsonify({"engines": available_engines()})


# ── Drop history (no stubs remain) ────────────────────────────────────────
#
# Real endpoints implemented over the course of the drops:
#   /api/media           — Drop G — serves video bytes with HTTP Range support
#   /api/video_info      — Drop G — returns ffprobe metadata
#   /api/srt             — Drop H — read SRT next to video / write segments back
#
# Removed during cleanup:
#   /api/extract, /api/extract_merge      — Drop D
#   /api/project/save, /api/project/save_as — Drop H
#   /api/detect                            — Drop I polish (no callers in new UI)

@app.route("/api/srt", methods=["GET"])
def api_srt_get():
    """Read an SRT file from disk and return its segments as JSON.

    Query parameters (provide one):
      ?path=<srt>     Explicit SRT path.
      ?video=<vid>    Locate ``<basename>.srt`` next to the video.

    Returns:
      200 ``{"path": "...", "segments": [{"index":1, "start_time":1.23,
            "end_time":4.56, "text":"..."}, ...]}``
      404 when no SRT is found (sidecar mode only).
      400 on validation failures.
    """
    srt_path_str = request.args.get("path", "").strip()
    video_path_str = request.args.get("video", "").strip()

    if not srt_path_str and not video_path_str:
        return jsonify({"error": "Provide either ?path= or ?video="}), 400

    if srt_path_str:
        srt_path, err = _validate_readable_path(srt_path_str)
        if err:
            # "File not found" gets 404; malformed input gets 400.
            status = 404 if err.startswith("File not found:") else 400
            return jsonify({"error": err}), status
        if srt_path.suffix.lower() != ".srt":
            return jsonify({"error": "path must point to a .srt file"}), 400
    else:
        video_path, err = _validate_readable_path(video_path_str)
        if err:
            return jsonify({"error": err}), 400
        sidecar = _find_sidecar_srt(video_path)
        if sidecar is None:
            return jsonify({"error": "No sidecar SRT", "path": str(video_path.with_suffix('.srt'))}), 404
        srt_path = sidecar

    try:
        import srt as srt_lib
    except ImportError:
        return jsonify({"error": "Server missing 'srt' package"}), 500

    try:
        # Tolerate BOM + a couple of common encodings before giving up.
        try:
            text = srt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = srt_path.read_text(encoding="utf-8-sig", errors="replace")
        subs = list(srt_lib.parse(text))
    except Exception as exc:
        logger.exception("Failed to parse SRT: %s", srt_path)
        return jsonify({"error": f"Could not parse {srt_path.name}: {exc}"}), 400

    segments = [
        {
            "index":      i + 1,                       # canonical 1-based, reindex on read
            "start_time": s.start.total_seconds(),
            "end_time":   s.end.total_seconds(),
            "text":       s.content,
        }
        for i, s in enumerate(subs)
    ]

    sibling = _find_sibling_video(srt_path)
    return jsonify({
        "path":           str(srt_path),
        "segments":       segments,
        "sibling_video":  str(sibling) if sibling is not None else None,
    })


@app.route("/api/srt", methods=["POST"])
def api_srt_save():
    """Write segments to an SRT file on disk, with backup-on-overwrite.

    Body:
      {
        "path":     "/full/path/to/output.srt",   // required
        "segments": [
          {"start_time": 1.23, "end_time": 4.56, "text": "..."},
          ...
        ]
      }

    Behaviour:
      1. Validate path (must end .srt, parent dir must exist).
      2. If the destination already exists, copy it to <path>.bak first
         (abort the save on backup failure — same safety pattern as
         POST /api/config).
      3. Re-index segments from 1 before serialization (so saves are always
         canonically numbered regardless of what the frontend sent).
    """
    body: dict[str, Any] = request.get_json(silent=True) or {}

    path_str = body.get("path", "")
    segments = body.get("segments")

    if not path_str or not isinstance(path_str, str):
        return jsonify({"status": "error", "message": "path is required"}), 400
    if not isinstance(segments, list):
        return jsonify({"status": "error", "message": "segments must be a list"}), 400

    dest_path, err = _validate_srt_save_path(path_str)
    if err:
        return jsonify({"status": "error", "message": err}), 400

    try:
        import srt as srt_lib
    except ImportError:
        return jsonify({"status": "error", "message": "Server missing 'srt' package"}), 500

    # Build srt.Subtitle objects with strict validation.
    from datetime import timedelta
    subs: list = []
    for i, seg in enumerate(segments):
        try:
            start = float(seg.get("start_time"))
            end   = float(seg.get("end_time"))
        except (TypeError, ValueError):
            return jsonify({
                "status": "error",
                "message": f"Segment {i + 1}: start_time/end_time must be numeric",
            }), 400
        if not (end > start):
            return jsonify({
                "status": "error",
                "message": f"Segment {i + 1}: end_time must be after start_time",
            }), 400
        text = str(seg.get("text") or "")
        subs.append(srt_lib.Subtitle(
            index=i + 1,
            start=timedelta(seconds=start),
            end=timedelta(seconds=end),
            content=text,
        ))

    with _srt_write_lock:
        # Backup before overwrite — abort if backup fails.
        if dest_path.exists():
            backup_path = dest_path.parent / (dest_path.name + ".bak")
            try:
                backup_path.write_bytes(dest_path.read_bytes())
                logger.info("SRT backup written: %s", backup_path)
            except OSError as exc:
                logger.error("Backup failed for %s: %s", dest_path, exc)
                return jsonify({
                    "status": "error",
                    "message": f"Could not write backup file ({exc}). Save aborted.",
                }), 500

        try:
            composed = srt_lib.compose(subs, reindex=True, start_index=1)
            dest_path.write_text(composed, encoding="utf-8")
        except Exception as exc:
            logger.exception("SRT save failed")
            return jsonify({"status": "error", "message": str(exc)}), 500

    return jsonify({
        "status":   "ok",
        "path":     str(dest_path),
        "count":    len(subs),
        "message":  f"Wrote {len(subs)} segment(s) to {dest_path.name}.",
    })


@app.route("/api/video_info")
def api_video_info():
    """Return basic video metadata via ffprobe (fps / duration / frame count).

    The new UI's player calls this on every video load to populate the FPS
    display in the footer and enable frame-accurate scrubbing.
    """
    path_str = request.args.get("path", "")
    if not path_str:
        return jsonify({"error": "path required"}), 400

    video_path, err = _validate_readable_path(path_str)
    if err:
        return jsonify({"error": err}), 400

    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of", "default=nw=1", str(video_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return jsonify({"error": "ffprobe not found on PATH"}), 500
    if proc.returncode != 0:
        return jsonify({"error": "ffprobe failed", "stderr": (proc.stderr or "").strip()}), 500

    fields: dict[str, str] = {}
    for line in (proc.stdout or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fields[k.strip()] = v.strip()

    r_rate   = fields.get("r_frame_rate")
    avg_rate = fields.get("avg_frame_rate")
    nb_s     = fields.get("nb_frames")
    dur_s    = fields.get("duration")

    nb_frames: Optional[int] = None
    if nb_s and nb_s.upper() != "N/A":
        try: nb_frames = int(nb_s)
        except Exception: nb_frames = None

    duration: Optional[float] = None
    if dur_s and dur_s.upper() != "N/A":
        try: duration = float(dur_s)
        except Exception: duration = None

    return jsonify({
        "path":           str(video_path),
        "r_frame_rate":   r_rate,
        "avg_frame_rate": avg_rate,
        "r_fps":          _parse_rate_to_float(r_rate or ""),
        "avg_fps":        _parse_rate_to_float(avg_rate or ""),
        "nb_frames":      nb_frames,
        "duration_s":     duration,
    })


@app.route("/api/media")
def api_media():
    """Serve local media files by absolute path with HTTP Range support.

    The Range header is required for seeking to work in the embedded video
    element.  We honour it but fall back to whole-file delivery for clients
    that don't send one.
    """
    from flask import send_file

    path_str = request.args.get("path", "")
    if not path_str:
        return "path required", 400

    media_path, err = _validate_readable_path(path_str)
    if err:
        return err, 400

    file_size    = media_path.stat().st_size
    range_header = request.headers.get("Range", "")
    mime         = _guess_mime(media_path)

    # No Range header → send the whole file.
    if not range_header:
        return send_file(str(media_path), mimetype=mime, conditional=True)

    # Parse "bytes=START-END".
    try:
        units, _, rng = range_header.partition("=")
        if units.strip().lower() != "bytes":
            raise ValueError("only byte ranges supported")
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        end   = int(end_s)   if end_s   else file_size - 1
        start = max(0, min(start, file_size - 1))
        end   = max(start, min(end, file_size - 1))
    except Exception:
        logger.warning("Bad Range header: %s", range_header)
        return ("Bad Range", 416)

    length = end - start + 1

    def _stream():
        with open(media_path, "rb") as f:
            f.seek(start)
            remaining = length
            chunk = 1024 * 1024
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    rv = Response(_stream(), 206, mimetype=mime, direct_passthrough=True)
    rv.headers.add("Content-Range",  f"bytes {start}-{end}/{file_size}")
    rv.headers.add("Accept-Ranges",  "bytes")
    rv.headers.add("Content-Length", str(length))
    rv.headers.add("Cache-Control",  "no-cache")
    return rv


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
            """Open the native Save dialog for choosing an SRT output path.

            Called by the Save As button in the new UI.  Returns the chosen
            full path or ``None`` if the user cancelled.
            """
            win = getattr(self, "_window", None)
            if not win:
                return None
            file_types = (
                "SubRip subtitles (*.srt)",
                "All files (*.*)",
            )
            # Try the rich signature first, then degrade gracefully — pywebview's
            # create_file_dialog signature has drifted across versions.
            kwargs: dict = {"file_types": file_types, "save_filename": default_filename or "subtitles.srt"}
            if initial_dir:
                kwargs["directory"] = initial_dir
            for call_kwargs in (kwargs, {"file_types": file_types, "save_filename": kwargs["save_filename"]}, {"file_types": file_types}):
                try:
                    result = win.create_file_dialog(_DLG_SAVE, **call_kwargs)
                    break
                except TypeError:
                    continue
                except Exception:
                    logger.exception("save_srt_as dialog failed")
                    return None
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

            video_path = next((p for p in paths if Path(p).suffix.lower() in _VIDEO_EXTS), None)
            srt_path   = next((p for p in paths if Path(p).suffix.lower() == ".srt"), None)

            # If user dropped an SRT alone, try to find a sibling video next
            # to it.  This is the symmetrical counterpart of the sidecar-SRT
            # auto-discovery that fires when a video is dropped.  When a
            # sibling video is found, we route through the video-load path —
            # the sidecar hook in project.js will then re-discover and load
            # this same SRT, so we don't issue a separate gensrtLoadSrtFromPath.
            if srt_path and not video_path:
                sibling = _find_sibling_video(Path(srt_path))
                if sibling is not None:
                    video_path = str(sibling)
                    srt_path = None

            if video_path:
                window.evaluate_js(
                    f"window.tilesterSetVideoPath && window.tilesterSetVideoPath({json.dumps(video_path)});"
                )
            if srt_path:
                window.evaluate_js(
                    f"window.gensrtLoadSrtFromPath && window.gensrtLoadSrtFromPath({json.dumps(srt_path)});"
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
