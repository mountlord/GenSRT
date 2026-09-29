"""Pages, static files and local-media access for the review window.

Routes (URLs unchanged from the pre-split server.py):
  GET  /                      GET  /static/<path>
  GET  /api/status            GET  /favicon.ico
  GET  /api/srt               POST /api/srt
  GET  /api/video_info        GET  /api/media
  POST /api/media/prepare     GET  /api/media/status
  POST /api/media/export
"""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

from flask import Blueprint, Response, jsonify, render_template, request, send_from_directory

from gensrt import remux
from gensrt.api._paths import (
    _find_sibling_video,
    _find_sidecar_srt,
    _guess_mime,
    _parse_rate_to_float,
    _validate_readable_path,
    _validate_save_path,
    _validate_srt_save_path,
)

logger = logging.getLogger(__name__)

bp = Blueprint("media", __name__)

# Lock for serializing SRT writes (matches _config_write_lock in api.config).
_srt_write_lock = threading.Lock()


# ── Static files ──────────────────────────────────────────────────────────

_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@bp.route("/")
def index():
    return render_template("review.html")


@bp.route("/static/<path:filename>")
def static_files(filename: str):
    return send_from_directory(str(_STATIC_DIR), filename)


# ── API endpoints ─────────────────────────────────────────────────────────

@bp.route("/api/status")
def api_status():
    """Health-check and version endpoint."""
    from gensrt import __version__
    return jsonify({"status": "ok", "version": __version__})


@bp.route("/favicon.ico")
def _favicon():
    """No favicon is shipped; answer 204 so every page load stops logging a
    404 for it."""
    return ("", 204)


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

@bp.route("/api/srt", methods=["GET"])
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


@bp.route("/api/srt", methods=["POST"])
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

        # WebVTT companion — same cues, written next to the SRT.  Non-fatal:
        # the SRT is the user's primary artifact; failing the whole save
        # because of a derived-file glitch would be hostile.
        vtt_path = dest_path.with_suffix(".vtt")
        vtt_ok = False
        try:
            from gensrt.srt.builder import write_vtt
            write_vtt(subs, vtt_path)
            vtt_ok = True
        except Exception as exc:
            logger.warning("VTT companion write failed (%s) — SRT saved OK.", exc)

    return jsonify({
        "status":   "ok",
        "path":     str(dest_path),
        "vtt_path": str(vtt_path) if vtt_ok else None,
        "count":    len(subs),
        "message":  (
            f"Wrote {len(subs)} segment(s) to {dest_path.name}"
            + (f" + {vtt_path.name}." if vtt_ok else ".")
        ),
    })


@bp.route("/api/video_info")
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

    from gensrt.ffmpeg_util import get_ffprobe_exe, get_subprocess_creationflags

    cmd = [
        get_ffprobe_exe(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,duration",
        "-of", "default=nw=1", str(video_path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            creationflags=get_subprocess_creationflags(),
        )
    except FileNotFoundError:
        return jsonify({"error": "ffprobe not found (bundled binary missing and not on PATH)"}), 500
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


@bp.route("/api/media/prepare", methods=["POST"])
def api_media_prepare():
    """Start (or report) the MP4 rewrap of a transport-stream recording.

    Body ``{"path": "<absolute path>"}``.  Returns
    ``{"status": "ready"|"preparing"|"error", "progress": 0..1|null,
    "error": str|null, "remux": bool}``.  For a container the element plays
    natively (mp4, mkv, webm) the answer is ``ready`` with ``remux: false``
    and nothing is started.  Poll ``GET /api/media/status`` afterwards.
    """
    body = request.get_json(silent=True) or {}
    media_path, err = _validate_readable_path(str(body.get("path") or ""))
    if err:
        return jsonify({"error": err}), 400
    if not remux.needs_remux(media_path):
        return jsonify({"status": "ready", "progress": 1.0, "error": None, "remux": False})
    st = remux.manager.prepare(media_path)
    return jsonify({**st.as_dict(), "remux": True})


@bp.route("/api/media/status")
def api_media_status():
    """Progress of a rewrap started by /api/media/prepare (``?path=``)."""
    media_path, err = _validate_readable_path(request.args.get("path", ""))
    if err:
        return jsonify({"error": err}), 400
    if not remux.needs_remux(media_path):
        return jsonify({"status": "ready", "progress": 1.0, "error": None, "remux": False})
    return jsonify({**remux.manager.status(media_path).as_dict(), "remux": True})


@bp.route("/api/media/export", methods=["POST"])
def api_media_export():
    """Copy the MP4 rewrap of a transport stream to a path of the user's
    choosing.  Body ``{"path": <source .ts>, "dest": <target .mp4>}``.
    The rewrap must be ready (the player only offers Save once it is)."""
    import shutil

    body = request.get_json(silent=True) or {}
    media_path, err = _validate_readable_path(str(body.get("path") or ""))
    if err:
        return jsonify({"error": err}), 400
    if not remux.needs_remux(media_path):
        return jsonify({"error": "Only transport-stream recordings have a rewrap to save."}), 400
    dest, err = _validate_save_path(str(body.get("dest") or ""), ".mp4")
    if err:
        return jsonify({"error": err}), 400
    st = remux.manager.status(media_path)
    if st.state != "ready" or st.output is None:
        return jsonify({"error": "The rewrap is not ready yet.", **st.as_dict()}), 409
    if dest.resolve() == st.output.resolve():
        return jsonify({"error": "That is the cache file itself."}), 400
    try:
        shutil.copyfile(st.output, dest)
    except OSError as exc:
        logger.error("Saving rewrap of %s to %s failed: %s", media_path.name, dest, exc)
        return jsonify({"error": f"Could not write {dest.name}: {exc}"}), 500
    logger.info("Saved rewrap of %s to %s", media_path.name, dest)
    return jsonify({"status": "ok", "dest": str(dest), "bytes": dest.stat().st_size})


@bp.route("/api/media")
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

    mime = _guess_mime(media_path)
    if remux.needs_remux(media_path):
        # The embedded browser cannot open a transport stream (see
        # gensrt/remux.py).  Serve the cached MP4 rewrap instead; the
        # player asks /api/media/prepare first, so an unprepared file here
        # is a direct URL, and 409 with the status tells it what to do.
        st = remux.manager.status(media_path)
        if st.state != "ready" or st.output is None:
            return jsonify({"error": "This file is being prepared for playback.",
                            **st.as_dict()}), 409
        media_path = st.output
        mime = "video/mp4"

    file_size    = media_path.stat().st_size
    range_header = request.headers.get("Range", "")

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
