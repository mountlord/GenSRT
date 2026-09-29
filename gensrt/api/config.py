"""Configuration persistence, engine roster and model validation.

Routes (URLs unchanged from the pre-split server.py):
  GET  /api/config            POST /api/config
  GET  /api/engines           GET  /api/known_models
  POST /api/validate_model    POST /api/add_known_model
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from flask import Blueprint, jsonify, request

from gensrt.operations import (
    read_config_file,
    write_config_file,
)

# Derived from the translation factory rather than hand-copied: this table
# went stale the moment NLLB shipped, and every engine and config field added
# since v1.2.7 was silently unsaveable from the GUI ("unknown configuration
# key") because the dataclass grew and this did not. Importing the key tuples
# is cheap — factory.py imports the engines themselves lazily.
from gensrt.translation.factory import ENGINE_KEYS as _TR_ENGINE_KEYS

logger = logging.getLogger(__name__)

bp = Blueprint("config", __name__)


# Serialises POST /api/config writes (a fast Save spam would otherwise race).
_config_write_lock = threading.Lock()


# ── Config persistence (POST /api/config) ─────────────────────────────────
#
# Validation schema for keys accepted by POST /api/config.  Unknown keys are
# rejected.  Bounds are deliberately wider than the UI's HTML5 ranges so the
# UI can be tightened without backend changes, but tight enough that obviously
# broken values can't be written to gensrt-config.json.

_MODEL_CHOICES = {
    "tiny", "base", "small", "medium", "large",
    "large-v1", "large-v2", "large-v3", "large-v3-turbo",
}  # kept for legacy reference; runtime validation is now permissive
   # (see _v_str on the "model" field).  The dropdown options are sourced
   # from /api/known_models, which merges these built-ins with the user's
   # gensrt-known-models.json side file.
# "auto" defers to gpu_probe.default_compute_type_for(), which asks
# CTranslate2 what the resolved device actually supports.
_COMPUTE_CHOICES = {"auto", "float32", "float16", "int8_float16", "int8"}
_ASR_ENGINE_CHOICES = {"auto", "chunked", "longform"}
_DEVICE_CHOICES = {"cuda", "cpu", "auto"}
_BACKEND_CHOICES = {"cuda", "rocm", "xpu", "cpu"}
_ENGINE_CHOICES = set(_TR_ENGINE_KEYS)
#: "auto" lets the OCR path pick an offline engine for interactive latency.
_OCR_ENGINE_CHOICES = {"auto", *_TR_ENGINE_KEYS}
_LOG_LEVEL_CHOICES = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}

_INT_KEYS = {
    "gpu_id", "vad_min_speech_ms", "vad_min_silence_ms", "vad_speech_pad_ms",
    "max_line_chars", "max_lines",
    "ocr_max_regions", "ocr_det_limit_side_len",
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

def _v_model_ref(x):
    """A model reference, or "" meaning "use the engine's default".

    These fields were validated as non-empty strings, so a config that had
    never set them rendered an empty box in the GUI and then refused to save
    it — leaving no way to set them except by hand-editing the JSON. Empty
    is a legitimate value: the engines fall back to their own DEFAULT_MODEL.
    """
    if not isinstance(x, str):
        return False, "must be a string"
    return True, ""


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
    "model":                   _v_str,
    "asr_engine":              _v_str_in(_ASR_ENGINE_CHOICES),
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
    "max_line_chars":          _v_num_range(0, 200, integer=True),
    "max_lines":               _v_num_range(1, 10, integer=True),
    "translation_engine":      _v_str_in(_ENGINE_CHOICES),
    "translation_model":       _v_model_ref,
    "madlad_model":            _v_model_ref,
    "translate":               _v_bool,
    "target_language":         _v_str,
    # Chunked inference (v1.2.7)
    "chunk_mode":              _v_str_in({"vad", "fixed"}),
    "snap_onsets":             _v_bool,
    "max_chunk_s":             _v_num_range(1.0, 60.0),
    "min_chunk_s":             _v_num_range(0.0, 30.0),
    # On-screen text recognition
    "ocr_language":            _v_str,
    "ocr_translation_engine":  _v_str_in(_OCR_ENGINE_CHOICES),
    "ocr_min_confidence":      _v_num_range(0.0, 1.0),
    "ocr_max_regions":         _v_num_range(1, 200, integer=True),
    "ocr_det_limit_side_len":  _v_num_range(320, 4096, integer=True),
    # Non-transcription (preserved-through, not currently surfaced in UI)
    "output":                  _v_str_or_null,
    "output_filename":         _v_str_or_null,
    "recurse":                 _v_bool,
    "log_level":               _v_str_in(_LOG_LEVEL_CHOICES),
    # Diagnostics — a path, or "" to disable.  Deliberately not surfaced in
    # the config UI: it is an investigation tool, not a user setting.
    "debug_chunk_dir":         _v_str,
    "dump_segments_dir":       _v_str,
    "heuristics_report_dir":   _v_str,
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


@bp.route("/api/config", methods=["POST"])
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


@bp.route("/api/config", methods=["GET"])
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


@bp.route("/api/engines")
def api_engines():
    """Return available translation engines."""
    from gensrt.translation.factory import available_engines
    return jsonify({"engines": available_engines()})


# ── Custom Whisper models (Drop I.14.7) ───────────────────────────────────
#
# The user can run any faster-whisper-compatible Whisper model — built-in
# names ("large-v3-turbo"), HuggingFace repo IDs ("smcproject/vegam-..."),
# or local paths.  The footer model dropdown is populated from:
#
#   built-in recommended (gensrt.known_models.BUILTIN_RECOMMENDED)
#       + user-added (gensrt-known-models.json side file)
#
# A user can add new models via the "New…" sentinel in the dropdown or
# from the Config modal.  The /api/validate_model endpoint does a light
# HuggingFace metadata check (no weights download) so typos surface
# before the first transcribe call.

@bp.route("/api/known_models", methods=["GET"])
def api_known_models():
    """Return the merged list of selectable Whisper model names.

    Built-in recommended models first, then user-added models from
    ``gensrt-known-models.json``.  Always returns at least the built-ins.
    """
    from gensrt.known_models import get_combined_models
    try:
        models = get_combined_models()
    except Exception as exc:
        logger.warning("known_models load failed: %s", exc)
        from gensrt.known_models import BUILTIN_RECOMMENDED
        models = list(BUILTIN_RECOMMENDED)
    return jsonify({"models": models})


@bp.route("/api/validate_model", methods=["POST"])
def api_validate_model():
    """Light validation of a Whisper model name.

    Performs a metadata-only HuggingFace API check — does NOT download
    weights.  Catches typos and access errors cheaply (1-2 seconds).
    Does NOT guarantee the model is faster-whisper compatible; the real
    test only happens at transcribe time.

    Request body: ``{"model": "org/repo-name"}``
    Response: ``{"status": "ok"}`` on success,
              ``{"status": "error", "message": "..."}`` otherwise.
    """
    from gensrt.known_models import BUILTIN_RECOMMENDED
    body = request.get_json(silent=True) or {}
    from gensrt.model_paths import normalize_model_ref

    name = normalize_model_ref(body.get("model"))

    if not name:
        return jsonify({"status": "error",
                        "message": "Model name is empty."}), 400

    # Built-in short names skip the network round-trip.
    if name in BUILTIN_RECOMMENDED:
        return jsonify({"status": "ok",
                        "message": f"'{name}' is a built-in Whisper model."})

    # Local models skip the network round-trip. resolve_model handles both an
    # explicit path and a bare name under <app dir>/models.
    from gensrt.model_paths import describe_model_locations, resolve_model

    resolved = Path(resolve_model(name))
    if resolved.exists() and resolved.is_dir():
        if (resolved / "model.bin").is_file():
            return jsonify({"status": "ok",
                            "message": f"Local model found: {resolved}"})
        return jsonify({"status": "error", "message": (
            f"'{resolved}' exists but does not contain a CTranslate2 "
            f"'model.bin'. If you converted this model yourself, check the "
            f"conversion completed and that you are pointing at the output "
            f"directory itself rather than its parent."
        )}), 400

    # HuggingFace metadata check: GET /api/models/<repo>.  This returns
    # JSON metadata if the repo exists and is accessible to the current
    # auth token (or public).  We deliberately do NOT load weights.
    #
    # `requests` rather than `urllib`, deliberately.  The two use different
    # TLS trust stores on Windows: urllib goes through Python's ssl module to
    # the Windows certificate store, while requests uses the certifi bundle —
    # which is also what huggingface_hub uses to DOWNLOAD the model.  With
    # urllib, validation could fail on a machine where the download would
    # have worked perfectly (observed on a fresh Windows install whose root
    # store was incomplete).  A validate button that rejects models the app
    # can actually use is worse than no validate button.
    import requests

    url = f"https://huggingface.co/api/models/{name}"
    try:
        resp = requests.get(url, headers={"User-Agent": "GenSRT/1.x"}, timeout=10)

        if resp.status_code in (401, 403):
            # HuggingFace deliberately returns 401 for BOTH a private/gated
            # repo and one that does not exist, so that unauthenticated
            # callers cannot enumerate private repo names.  Saying "exists but
            # requires authentication" therefore claims more than the API
            # told us, and sends someone off to request access to something
            # that may simply not be there.
            suggestion = ""
            if name.lower().startswith(tuple(
                f"{org}/ct2-" for org in (name.split("/", 1)[0],)
            )):
                org, repo = name.split("/", 1)
                suggestion = (
                    f"\n\nIf you added the 'ct2-' prefix yourself, that "
                    f"converted variant may not have been published. Check "
                    f"the original at "
                    f"https://huggingface.co/{org}/{repo[4:]} — if it exists "
                    f"but is a PyTorch model, you can convert it yourself."
                )
            return jsonify({"status": "error", "message": (
                f"'{name}' is not accessible. HuggingFace does not "
                f"distinguish between a repository that is private, one that "
                f"is gated, and one that does not exist.\n\n"
                f"Check the URL in a browser: "
                f"https://huggingface.co/{name}\n"
                f"If the page loads and asks you to request access, run "
                f"`hf auth login` and request it there. If you get a 404, the "
                f"repository name is wrong or the model has not been "
                f"published.{suggestion}\n\n"
                f"If you meant a model you converted yourself, no local model "
                f"of that name was found either.\n\n"
                f"{describe_model_locations()}"
            )}), 400
        if resp.status_code == 404:
            return jsonify({"status": "error", "message": (
                f"'{name}' was not found on HuggingFace, and no local model "
                f"of that name exists.\n\n{describe_model_locations()}"
            )}), 400
        if resp.status_code != 200:
            return jsonify({"status": "error",
                            "message": f"HuggingFace returned HTTP {resp.status_code}."}), 400

        payload = resp.json()

        # Sanity: a real model card has a "modelId" or "id" field.
        if not isinstance(payload, dict) or not (payload.get("modelId") or payload.get("id")):
            return jsonify({"status": "error",
                            "message": "Response did not look like a model card."}), 400

        # The metadata already lists the repo's files, so check the format
        # while we are here.  "Found on HuggingFace" reads as approval, and a
        # user who gets it then waits through a model download only to have
        # transcription fail with "Unable to open file 'model.bin'" has been
        # misled by us.  Costs no extra request.
        problem = _ct2_format_problem(name, payload)
        if problem:
            return jsonify({"status": "error", "message": problem}), 400

        return jsonify({"status": "ok",
                        "message": f"Found '{name}' on HuggingFace."})

    except requests.exceptions.SSLError as exc:
        # Certificate verification failure.  The bare OpenSSL text
        # ("CERTIFICATE_VERIFY_FAILED ... unable to get local issuer
        # certificate") is accurate and useless to anyone who does not
        # already know what a CA bundle is, so say what usually causes it and
        # what to do.  Seen on a freshly installed Windows whose root
        # certificate store had not been populated yet.
        return jsonify({"status": "error", "message": (
            "Could not verify HuggingFace's security certificate. This is "
            "usually a certificate problem on this computer rather than with "
            "the model.\n\n"
            "Try, in order:\n"
            "1. Open https://huggingface.co in your browser once, then retry "
            "— Windows fetches missing certificates on demand.\n"
            "2. Check this computer's date and time are correct.\n"
            "3. If you are on a corporate or school network, antivirus or a "
            "network filter may be intercepting secure connections; your IT "
            "team can advise.\n\n"
            f"Technical detail: {exc}"
        )}), 400
    except OSError as exc:
        # requests raises OSError (not SSLError) when REQUESTS_CA_BUNDLE,
        # CURL_CA_BUNDLE or SSL_CERT_FILE points somewhere unusable. Worth
        # naming, because the cause is an environment variable the user may
        # have set long ago and forgotten — often while working around an
        # earlier certificate problem.
        if "CA certificate" in str(exc) or "certificate bundle" in str(exc):
            import os as _os

            culprits = [
                f"{v}={_os.environ[v]}"
                for v in ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE")
                if _os.environ.get(v)
            ]
            detail = (
                "\n\nThese environment variables are currently set:\n  "
                + "\n  ".join(culprits)
                if culprits else
                "\n\nNo CA-bundle environment variables appear to be set, so "
                "the installation itself may be incomplete."
            )
            return jsonify({"status": "error", "message": (
                "The TLS certificate bundle this computer is configured to use "
                "could not be read.\n\n"
                "This is a configuration problem on this machine, not a problem "
                "with the model. It is usually caused by REQUESTS_CA_BUNDLE, "
                "CURL_CA_BUNDLE or SSL_CERT_FILE pointing at a file that has "
                "been moved or deleted. Clearing those variables restores the "
                "default behaviour."
                f"{detail}\n\nTechnical detail: {exc}"
            )}), 400
        return jsonify({"status": "error",
                        "message": f"Validation failed: {exc}"}), 400
    except requests.exceptions.Timeout:
        return jsonify({"status": "error", "message": (
            "Timed out contacting HuggingFace. Check your internet connection "
            "and try again."
        )}), 400
    except requests.exceptions.RequestException as exc:
        return jsonify({"status": "error",
                        "message": f"Network error contacting HuggingFace: {exc}"}), 400
    except Exception as exc:
        return jsonify({"status": "error",
                        "message": f"Validation failed: {exc}"}), 400


# Files that identify a CTranslate2-converted model, and files that identify
# the unconverted PyTorch original. GenSRT runs on CTranslate2 via
# faster-whisper and cannot load the latter.
_CT2_MARKER = "model.bin"
_TRANSFORMERS_MARKERS = ("pytorch_model.bin", "model.safetensors",
                         "flax_model.msgpack", "tf_model.h5")


def _ct2_format_problem(name: str, payload: dict) -> str | None:
    from gensrt.model_paths import conversion_command, suggested_output_dir

    """Return a user-facing message if *payload* is not a CTranslate2 model.

    ``None`` means "looks loadable, or we could not tell".  Being unsure is
    deliberately treated as fine: the file list is advisory, and blocking a
    model that would actually have worked is worse than letting the load-time
    error speak.
    """
    siblings = payload.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        return None   # no file list available — do not guess

    files = {
        s.get("rfilename", "") for s in siblings if isinstance(s, dict)
    }
    if any(f == _CT2_MARKER or f.endswith("/" + _CT2_MARKER) for f in files):
        return None

    if any(m in files for m in _TRANSFORMERS_MARKERS):
        suggestion = ""
        # Many publishers ship the converted variant under a ct2- prefix.
        if "/" in name:
            org, repo = name.split("/", 1)
            suggestion = f" Look for '{org}/ct2-{repo}' or a similar variant."
        return (
            f"'{name}' is a PyTorch/transformers model. GenSRT runs on "
            f"CTranslate2 and needs a converted model.{suggestion} "
            f"Alternatively convert it yourself. In a separate Python "
            f"environment with ctranslate2, transformers and torch "
            f"installed:\n\n"
            f"{conversion_command(name)}\n\n"
            f"Then enter just the folder name: "
            f"{suggested_output_dir(name).name}"
        )

    return (
        f"'{name}' does not contain a CTranslate2 '{_CT2_MARKER}'. GenSRT "
        f"may not be able to load it."
    )


@bp.route("/api/add_known_model", methods=["POST"])
def api_add_known_model():
    """Append a model name to the user's side file.

    Idempotent — adding an existing name is a no-op.  Built-in names are
    silently dropped (they're always offered regardless).

    Request body: ``{"model": "org/repo-name"}``
    Response: ``{"status": "ok", "models": [...combined list...]}``
    """
    from gensrt.known_models import add_known_model, get_combined_models

    body = request.get_json(silent=True) or {}
    from gensrt.model_paths import normalize_model_ref

    name = normalize_model_ref(body.get("model"))

    if not name:
        return jsonify({"status": "error",
                        "message": "Model name is empty."}), 400

    try:
        add_known_model(name)
        return jsonify({"status": "ok", "models": get_combined_models()})
    except OSError as exc:
        return jsonify({"status": "error",
                        "message": f"Could not write known-models file: {exc}"}), 500
