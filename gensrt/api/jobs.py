"""Long-running jobs: transcribe, burn-in and their progress poll.

Routes (URLs unchanged from the pre-split server.py):
  GET  /api/operation_status
  POST /api/transcribe
  POST /api/burn
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

from gensrt.api._paths import _VIDEO_EXTS, _find_sibling_video, _validate_readable_path
from gensrt.api._state import (
    OperationBusyError,
    _begin_long_operation,
    _end_long_operation,
    _make_progress_cb,
    _make_status_cb,
    _snapshot_active_operation,
    _update_active_operation,
)
from gensrt.constants import PIPELINE_PHASES
from gensrt.operations import (
    build_transcription_config,
    read_config_file,
    resolve_output_path,
    run_transcription,
)

logger = logging.getLogger(__name__)

bp = Blueprint("jobs", __name__)


# Prefix for the temp SRT copies staged for ffmpeg burn-in.  Kept as a module
# constant because both the staging and the housekeeping sweep match on it.
_BURN_STAGE_PREFIX = "gensrt_burn_"
# Staged copies older than this are swept on the next burn.
_BURN_STAGE_MAX_AGE_S = 24 * 60 * 60


def resolve_drop_targets(paths: list[str]) -> tuple[str | None, str | None, bool]:
    """Decide what a set of dropped file paths means.

    Returns ``(video_path, srt_path, srt_is_explicit)``.

    The rules, in order:

    * A dropped video is loaded.
    * A dropped SRT is loaded, and is *explicit* — the user named that exact
      file, so sidecar discovery must not later substitute a different one.
    * An SRT dropped alone also loads its sibling video, so the user gets a
      working player rather than cues with nothing to play against.

    That last rule used to discard the dropped path and let sidecar discovery
    re-find it after the video loaded.  Discovery prefers ``<basename>.srt``,
    so dropping ``clip.ml.srt`` loaded the video and then silently swapped in
    ``clip.srt``.  Discovery is for when the user gave us no SRT; it must
    never override one they did give us.
    """
    video_path = next(
        (p for p in paths if Path(p).suffix.lower() in _VIDEO_EXTS), None
    )
    srt_path = next(
        (p for p in paths if Path(p).suffix.lower() == ".srt"), None
    )
    srt_is_explicit = srt_path is not None

    if srt_path and not video_path:
        sibling = _find_sibling_video(Path(srt_path))
        if sibling is not None:
            video_path = str(sibling)

    return video_path, srt_path, srt_is_explicit


def _stage_srt_for_burn(srt_path: Path) -> Path:
    """Copy *srt_path* to a temp file with a filtergraph-safe ASCII name.

    See the comment in :func:`api_burn` for why this exists rather than an
    escaping routine.  The staged name is ``gensrt_burn_<8 hex>.srt`` — only
    ``[A-Za-z0-9_]`` plus the extension, so no layer of ffmpeg's argument
    parsing has anything to chew on.

    Bytes are copied verbatim; encoding is untouched, so a UTF-8 Malayalam
    SRT reaches libass exactly as written.

    Cleanup: the burn is deliberately detached and survives app close, so we
    cannot delete the staged file when ffmpeg finishes without holding a
    watcher process open.  Instead each call sweeps stale copies from previous
    burns.  Files are a few KB, and the sweep is bounded by the temp dir
    listing, so this stays cheap.

    Returns:
        Path to the staged copy.

    Raises:
        OSError: If the temp copy cannot be written.
    """
    import shutil
    import tempfile
    import time as _time
    import uuid

    tmp_dir = Path(tempfile.gettempdir())

    # Housekeeping: drop staged copies from earlier runs.  Best-effort — a
    # file we cannot remove (still open, permissions) is skipped silently.
    cutoff = _time.time() - _BURN_STAGE_MAX_AGE_S
    try:
        for stale in tmp_dir.glob(f"{_BURN_STAGE_PREFIX}*.srt"):
            try:
                if stale.stat().st_mtime < cutoff:
                    stale.unlink()
                    logger.debug("Swept stale burn staging file: %s", stale.name)
            except OSError:
                continue
    except OSError:
        pass

    staged = tmp_dir / f"{_BURN_STAGE_PREFIX}{uuid.uuid4().hex[:8]}.srt"
    shutil.copyfile(srt_path, staged)
    logger.debug("Staged %s → %s for burn-in", srt_path.name, staged.name)
    return staged


@bp.route("/api/operation_status")
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


@bp.route("/api/transcribe", methods=["POST"])
def api_transcribe():
    """Start a transcription job.

    Expected JSON body:
        {
            "input_path":         "/path/to/media.mkv",   // required
            "output_dir":         "/path/to/output/",     // optional
            "output_filename":    "custom.srt",           // optional
            "translation_engine": "nllb",                 // optional
            "source_language":    "auto",                 // optional
            "target_language":    "en",                   // optional
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
    # NOTE: output_path is resolved AFTER the config is built so the
    # language-suffix naming convention (movie.ml.srt, etc.) can pick up
    # the actual target_language for this job.

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
    if "target_language" in body:
        overrides["target_language"] = body["target_language"]
    if body.get("no_translate"):
        overrides["translate"] = False
    if body.get("no_vad"):
        overrides["vad_enabled"] = False
    if "model" in body:
        overrides["model"] = body["model"]

    merged = {**file_cfg, **overrides}
    config = build_transcription_config(merged, auto_detect_backend=True)

    # Resolved here (after config) so the language suffix uses the real target.
    # When translate=False the subtitles are in the *source* language, so we
    # use that for the suffix.  If source_language is "auto", Whisper will
    # detect it later — we can't pre-compute a suffix, so we fall back to
    # unsuffixed (matches the pre-Drop-I.7 behaviour for that edge case).
    if config.translate:
        effective_lang = config.target_language
    else:
        effective_lang = (
            config.source_language
            if config.source_language and config.source_language.lower() != "auto"
            else "en"  # "en" -> no suffix in resolve_output_path
        )
    output_path = resolve_output_path(
        input_path, output_dir, output_filename, effective_lang
    )

    # Engine policy: reject mismatched engine + target before any expensive
    # work (audio extract, model load) runs.  The same gate fires in
    # pipeline.run_pipeline; doing it here too means /api/transcribe can
    # return 400 with the user-facing message instead of 500 from the
    # exception path.
    from gensrt.exceptions import ConfigError
    from gensrt.pipeline import validate_translation_config
    try:
        validate_translation_config(config)
    except ConfigError as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400

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
        result = run_transcription(
            input_path=input_path,
            output_path=output_path,
            config=config,
            progress=_make_progress_cb(),
            status=_make_status_cb(),
        )
        translation_error = getattr(result, "translation_error", None)
        _update_active_operation(
            message=(f"Complete — {output_path.name}" if not translation_error
                     else f"Complete but NOT TRANSLATED — {output_path.name}"),
            current=PIPELINE_PHASES,
            total=PIPELINE_PHASES,
        )
        response_data = {
            "status": "ok",
            "input":  str(input_path),
            "output": str(output_path),
            "translation_error": translation_error,
        }
    except Exception as exc:
        logger.exception("Transcription failed: %s", exc)
        _update_active_operation(message=f"Error: {exc}")
        response_data = {"status": "error", "message": str(exc)}
        status_code = 500
    finally:
        _end_long_operation()

    return jsonify(response_data), status_code


@bp.route("/api/burn", methods=["POST"])
def api_burn():
    """Burn an SRT into a copy of the video using ffmpeg.

    Fire-and-forget: spawns ffmpeg via subprocess.Popen and returns
    immediately with the output path.  The user can keep working in
    the app — start another transcribe, load a new video, etc. — while
    ffmpeg runs in the background.  Closing the app does NOT stop the
    burn (the child process is detached on Windows via
    CREATE_NEW_PROCESS_GROUP, and inherits no useful handles on
    POSIX).

    Body:
      {
        "video_path": "/full/path/to/video.mp4",   // required
        "srt_path":   "/full/path/to/video.ml.srt" // required
      }

    Output:
      Always writes to ``<video_stem>_subbed.mp4`` in the same directory
      as the source video, regardless of the source container.  This
      gives one consistent, broadly-compatible output format and avoids
      libavcodec gotchas like "VP9 doesn't fit in mp4" or "libx264
      doesn't fit in webm".  Existing _subbed.mp4 is overwritten.

    ffmpeg command shape:
      ffmpeg -y -i <video> -vf subtitles=<srt_basename> \
             -c:v libx264 -crf 18 -preset medium \
             -c:a copy <output>

      The ``subtitles`` libavfilter has notoriously brittle path
      handling on Windows (drive-letter colons, backslashes).  We
      sidestep by setting cwd to the SRT's parent directory and
      passing just the basename to the filter.  Absolute paths are
      fine for ``-i`` and the output argument.

    Returns:
      200 ``{status: "ok", output_path: "...", pid: N, message: "..."}``
          on successful spawn (does NOT mean ffmpeg succeeded; just
          that it started).
      400 ``{status: "error", message: "..."}`` on bad input.
      500 ``{status: "error", message: "..."}`` if ffmpeg can't spawn.
    """
    body: dict[str, Any] = request.get_json(silent=True) or {}

    video_str = body.get("video_path") or ""
    srt_str   = body.get("srt_path") or ""
    if not isinstance(video_str, str) or not video_str.strip():
        return jsonify({"status": "error", "message": "video_path is required"}), 400
    if not isinstance(srt_str, str) or not srt_str.strip():
        return jsonify({"status": "error", "message": "srt_path is required"}), 400

    video_path, err = _validate_readable_path(video_str)
    if err:
        return jsonify({"status": "error", "message": f"video_path: {err}"}), 400

    srt_path, err = _validate_readable_path(srt_str)
    if err:
        return jsonify({"status": "error", "message": f"srt_path: {err}"}), 400
    if srt_path.suffix.lower() != ".srt":
        return jsonify({"status": "error", "message": "srt_path must end in .srt"}), 400

    # Output filename: <video_stem>_subbed.mp4 next to the source.  If that
    # name is taken (earlier burn, accidental double-click, etc.) auto-version
    # to <video_stem>_subbed_1.mp4, _subbed_2.mp4, ... so concurrent or
    # subsequent burns produce distinct files instead of racing on one
    # output path or silently clobbering the previous burn.
    base_name = f"{video_path.stem}_subbed"
    output_path = video_path.with_name(f"{base_name}.mp4")
    n = 1
    while output_path.exists():
        output_path = video_path.with_name(f"{base_name}_{n}.mp4")
        n += 1
        if n > 999:  # paranoia bound — never going to hit this realistically
            return jsonify({
                "status":  "error",
                "message": "Too many existing _subbed files; clean up the folder.",
            }), 500

    from gensrt.ffmpeg_util import get_ffmpeg_exe

    # The SRT is handed to the `subtitles` libavfilter, whose argument goes
    # through THREE layers of parsing (filtergraph → filter options → the
    # filename itself).  Characters that are perfectly ordinary in a media
    # filename — [ ] , ; ' = : — are metacharacters at one or more of those
    # layers.  "Movie [1080p].srt" breaks the filtergraph outright, and
    # scene-release names like that are the norm in this user population.
    #
    # Escaping across three layers correctly is possible but genuinely
    # error-prone, and gets harder with non-ASCII names — exactly the case
    # GenSRT cares most about.  So instead of escaping we sidestep: copy the
    # SRT to a temp file whose name is guaranteed-safe ASCII, and point the
    # filter at that.  Nothing to escape, nothing to get subtly wrong, and it
    # works for any source filename in any script.
    try:
        safe_srt = _stage_srt_for_burn(srt_path)
    except OSError as exc:
        return jsonify({
            "status": "error",
            "message": f"Could not stage subtitle file for burn-in: {exc}",
        }), 500

    cmd = [
        get_ffmpeg_exe(), "-y",
        "-i", str(video_path),
        "-vf", f"subtitles={safe_srt.name}",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-c:a", "copy",
        str(output_path),
    ]

    # Spawn flags so the child survives app close and doesn't pop a
    # console window on Windows.  No effect on POSIX.
    popen_kwargs: dict[str, Any] = {
        "cwd":    str(safe_srt.parent),
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "stdin":  subprocess.DEVNULL,
    }
    if sys.platform.startswith("win"):
        # CREATE_NEW_PROCESS_GROUP: detach from parent so closing the
        # app doesn't terminate ffmpeg.  CREATE_NO_WINDOW: don't pop a
        # console for headless background work.
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        )
    else:
        popen_kwargs["start_new_session"] = True  # POSIX: setsid()

    try:
        proc = subprocess.Popen(cmd, **popen_kwargs)
    except FileNotFoundError:
        return jsonify({
            "status": "error",
            "message": "ffmpeg not found on PATH. Install ffmpeg and try again.",
        }), 500
    except OSError as exc:
        return jsonify({
            "status": "error",
            "message": f"Could not start ffmpeg: {exc}",
        }), 500

    logger.info(
        "Burn started (pid=%s): %s + %s → %s",
        proc.pid, video_path.name, srt_path.name, output_path.name,
    )

    return jsonify({
        "status":      "ok",
        "output_path": str(output_path),
        "pid":         proc.pid,
        "message":     (
            f"Burning subtitles... output will appear at {output_path.name} "
            f"in {output_path.parent}. Closing the app won't stop the burn."
        ),
    })
