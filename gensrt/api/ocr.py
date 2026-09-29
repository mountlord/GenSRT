"""OCR: single-frame reads, whole-video subtitle extraction, and the
picker's cached translation engine.

Routes (URLs unchanged from the pre-split server.py):
  POST /api/ocr               GET  /api/ocr/languages
  POST /api/ocr/extract       POST /api/ocr/extract/cancel
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request

from gensrt.api._paths import _validate_readable_path
from gensrt.api._state import (
    OperationBusyError,
    _begin_long_operation,
    _end_long_operation,
    _update_active_operation,
)
from gensrt.operations import (
    read_config_file,
)

logger = logging.getLogger(__name__)

bp = Blueprint("ocr", __name__)


# ── OCR translation: cached engine, interactive-latency policy ────────────
#
# Two things separate OCR translation from pipeline translation, and both
# argue against reusing the pipeline's engine choice verbatim:
#
# 1. SOMEONE IS WAITING.  The pipeline translates unattended, so v1.2.7's
#    long 429 backoff (2s then 8s) is a good trade: wait out a transient
#    throttle rather than degrade quality. In the picker that same ladder is
#    ~12 seconds of spinner before a fallback that was going to run anyway.
#
# 2. IT IS A HANDFUL OF SHORT STRINGS.  Six subtitle fragments do not need
#    the best available translator badly enough to pay a round trip for.
#
# So "auto" follows translation_engine (both remaining engines are offline);
# setting ocr_translation_engine explicitly overrides it.
_ocr_translator_cache: dict = {}
_ocr_translator_lock = threading.Lock()


def _resolve_ocr_engine_key(cfg: dict) -> str:
    """Which translation engine the picker should use."""
    key = str(cfg.get("ocr_translation_engine") or "auto").strip().lower()
    if key and key != "auto":
        return key

    return str(cfg.get("translation_engine") or "nllb").strip().lower()


def _get_ocr_translator(cfg: dict, key: str):
    """Build the picker's translation engine once and keep it.

    Without this the NLLB model was pushed to the GPU on every single
    request — 2-3 seconds each time, for a model that was already resident
    a moment earlier. Keyed on the settings that shape the engine, so a
    config change produces a new entry rather than a stale one.
    """
    from gensrt.models import TranscriptionConfig
    from gensrt.translation.factory import get_engine

    cache_key = (
        key,
        str(cfg.get("translation_model") or ""),
        str(cfg.get("madlad_model") or ""),
        str(cfg.get("device") or ""),
    )
    with _ocr_translator_lock:
        engine = _ocr_translator_cache.get(cache_key)
        if engine is not None:
            return engine

    tconf = TranscriptionConfig(**{
        k: v for k, v in cfg.items()
        if k in TranscriptionConfig.__dataclass_fields__
    })
    engine = get_engine(key, tconf)
    with _ocr_translator_lock:
        _ocr_translator_cache[cache_key] = engine
    return engine


# ── Subtitle extraction (OCR over a whole video) ──────────────────────────
#
# Runs SYNCHRONOUSLY on the request thread, like transcription does: Flask
# serves /api/operation_status from other threads, so the client posts once
# and polls for progress. Adding a second, different job mechanism for one
# feature would be worse than a long-lived request on localhost.
#
# A feature film at 2 fps is tens of thousands of samples, so cancellation
# matters and returns the cues found so far rather than discarding them.
_ocr_extract_cancel = threading.Event()


def _extract_cfg() -> dict:
    """Effective config for the OCR routes (defaults + the config file)."""
    from gensrt.config import BUILTIN_DEFAULTS

    try:
        return {**BUILTIN_DEFAULTS, **read_config_file(default_if_missing=True)}
    except Exception:
        return dict(BUILTIN_DEFAULTS)


@bp.route("/api/ocr/extract/cancel", methods=["POST"])
def api_ocr_extract_cancel():
    """Ask a running extraction to stop and keep what it has."""
    _ocr_extract_cancel.set()
    return jsonify({"status": "cancelling"})


@bp.route("/api/ocr/extract", methods=["POST"])
def api_ocr_extract():
    """Extract burned-in subtitles from a video.

    Body (JSON) — the contract the Extract Subtitles modal assembles:
      ``video_path``       required
      ``region``           {x, y, w, h} in VIDEO pixels; omit for whole frame
      ``language``         ISO 639-1 for the OCR model
      ``translate``        bool; ``target_language`` for the target
      ``sample_fps``       frames inspected per second
      ``start_time``       seconds; ``end_time`` seconds or null
      ``min_duration_s``   drop shorter cues
      ``similarity``       0-1, "same subtitle" threshold
      ``existing``         "append" | "replace" (echoed back; the client
                           applies it, since it owns the cue list)

    Returns:
      200 ``{"cues": [{index, start, end, text}], "cancelled": bool,
             "existing": "append"}``
      400 bad body, 404 video missing, 409 another operation is running,
      422 unknown language, 500 extraction failed.
    """
    from gensrt.exceptions import ConfigError

    body = request.get_json(silent=True) or {}

    raw_path = (body.get("video_path") or "").strip()
    if not raw_path:
        return jsonify({"error": "video_path is required"}), 400
    # _validate_readable_path RETURNS (path, error) — it does not raise. Every
    # other route unpacks it; this one did not, so the tuple went straight
    # into Path() and produced a TypeError instead of a clean 404.
    video, err = _validate_readable_path(raw_path)
    if err:
        return jsonify({"error": err}), 404

    region = body.get("region") or None
    if region:
        try:
            region = (int(region["x"]), int(region["y"]),
                      int(region["w"]), int(region["h"]))
        except (KeyError, TypeError, ValueError):
            return jsonify({"error": "region must be {x, y, w, h}"}), 400

    # Guarded: gensrt.ocr is imported only from inside functions, which is
    # exactly the pattern PyInstaller's analysis can miss. Unguarded, a
    # packaging gap surfaced as a Flask HTML error page and the client died
    # on "Unexpected token '<'" — true, and useless.
    try:
        from gensrt.ocr.extract import ExtractSettings, extract_subtitles
        from gensrt.ocr.ppocr_onnx import OCRError
    except Exception as exc:
        logger.exception("OCR extraction module unavailable")
        return jsonify({
            "error": f"The subtitle extraction module could not be loaded: "
                     f"{exc}. In a packaged build this usually means "
                     f"gensrt.ocr was not collected — check _internal for "
                     f"gensrt/ocr/extract.pyc and for the rapidocr models."
        }), 500

    try:
        settings = ExtractSettings(
            region=region,
            language=(body.get("language") or "ja").strip().lower(),
            sample_fps=float(body.get("sample_fps") or 2.0),
            start_time=float(body.get("start_time") or 0.0),
            end_time=(float(body["end_time"])
                      if body.get("end_time") not in (None, "") else None),
            min_duration_s=float(body.get("min_duration_s") or 0.0),
            similarity=float(body.get("similarity") or 0.85),
            translate=bool(body.get("translate")),
            target_language=(body.get("target_language") or "en").strip().lower(),
            det_limit_side_len=int(
                body.get("det_limit_side_len")
                or _extract_cfg().get("ocr_det_limit_side_len", 1280) or 1280
            ),
        )
        settings.validate()
    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid settings: {exc}"}), 400
    except ConfigError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        _begin_long_operation(Path(video).name)
    except OperationBusyError as exc:
        return jsonify({"error": str(exc)}), 409

    _ocr_extract_cancel.clear()

    def _progress(done, total, position, found):
        _update_active_operation(
            current=done, total=(total or 0),
            message=(f"Reading frames — {position / 60:.1f} min, "
                     f"{found} cue(s) found"),
        )

    try:
        _update_active_operation(message="Starting extraction…", current=0, total=0)
        cues = extract_subtitles(
            video, settings,
            progress=_progress,
            should_cancel=_ocr_extract_cancel.is_set,
        )
    except (ConfigError, OCRError) as exc:
        # Both carry an actionable, user-facing message — a range past the end
        # of the video, a region larger than the frame, a language with no
        # model. Those are the caller's input to fix, not a server fault, and
        # reporting them as 500 (with a traceback in the console) was wrong.
        logger.warning("Extraction rejected: %s", exc)
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        logger.exception("Subtitle extraction failed")
        return jsonify({"error": str(exc)}), 500
    finally:
        _end_long_operation()

    return jsonify({
        "cues": [
            {"index": c.index, "start": c.start, "end": c.end, "text": c.text}
            for c in cues
        ],
        "cancelled": _ocr_extract_cancel.is_set(),
        "existing": (body.get("existing") or "append"),
        "language": settings.language,
    })


@bp.route("/api/ocr/languages")
def api_ocr_languages():
    """Registry of OCR languages, with which models are already on disk.

    Returns:
      200 ``{"languages": [{"code","label","size_mb","present","note"}, ...],
             "default": "ja"}``
    """
    try:
        from gensrt.ocr.factory import DEFAULT_OCR_LANGUAGE, available_languages

        return jsonify({
            "languages": available_languages(),
            "default": DEFAULT_OCR_LANGUAGE,
        })
    except Exception as exc:      # OCR deps absent in a stripped build
        return jsonify({"error": str(exc), "languages": []}), 500


@bp.route("/api/ocr", methods=["POST"])
def api_ocr():
    """Read on-screen text from a single frame.

    The frame arrives as a data URL captured from the paused ``<video>``
    element by the client. Sending pixels rather than a timestamp is
    deliberate: the browser already holds the decoded frame, so this needs
    no re-seek and no second decode, and what gets read is exactly what the
    user is looking at.

    Body (JSON):
      ``image``     data URL ("data:image/png;base64,...") — required.
      ``language``  ISO 639-1; defaults to the configured ocr_language.
      ``translate`` bool; also translate each reading to target_language.

    Returns:
      200 ``{"language": "ja", "regions": [{index, quad, bbox, text,
            confidence, crop, translation?}, ...]}``
      400 on a malformed body, 422 when the language has no model,
      500 on a missing dependency or a failed model download.
    """
    import base64

    body = request.get_json(silent=True) or {}
    data_url = (body.get("image") or "").strip()
    if not data_url:
        return jsonify({"error": "image (data URL) required"}), 400

    # Accept a bare base64 payload too, so a non-browser caller need not
    # synthesise the data-URL prefix.
    payload = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception:
        return jsonify({"error": "image is not valid base64"}), 400
    if not raw:
        return jsonify({"error": "image is empty"}), 400

    from gensrt.config import BUILTIN_DEFAULTS
    try:
        cfg = {**BUILTIN_DEFAULTS, **read_config_file(default_if_missing=True)}
    except Exception:
        cfg = dict(BUILTIN_DEFAULTS)
    language = (body.get("language") or cfg.get("ocr_language") or "ja").strip().lower()

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        return jsonify({
            "error": f"OCR needs opencv and numpy, which are not installed: {exc}"
        }), 500

    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode the supplied image"}), 400

    from gensrt.exceptions import ConfigError

    try:
        from gensrt.ocr import read_frame

        regions = read_frame(
            frame,
            language,
            max_regions=int(cfg.get("ocr_max_regions", 40) or 40),
            min_confidence=float(cfg.get("ocr_min_confidence", 0.0) or 0.0),
            det_limit_side_len=int(cfg.get("ocr_det_limit_side_len", 1280) or 1280),
        )
    except ConfigError as exc:
        # Unknown language is the caller's mistake, not a server fault.
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:
        logger.exception("OCR failed")
        return jsonify({"error": str(exc)}), 500

    payload_regions = [r.to_dict() for r in regions]

    # Optional translation, reusing whatever engine the run is configured
    # for. Failure here must not lose the OCR result: the source text is
    # still useful, and the picker shows both.
    if body.get("translate") and payload_regions:
        target = (body.get("target_language")
                  or cfg.get("target_language") or "en").strip().lower()
        try:
            engine_key = _resolve_ocr_engine_key(cfg)
            engine = _get_ocr_translator(cfg, engine_key)
            texts = [r["text"] for r in payload_regions]
            for region, translated in zip(
                payload_regions, engine.translate_batch(texts, language, target)
            ):
                region["translation"] = translated
        except Exception as exc:
            logger.warning("OCR translation failed: %s", exc)
            for region in payload_regions:
                region["translation_error"] = str(exc)

    return jsonify({"language": language, "regions": payload_regions})
