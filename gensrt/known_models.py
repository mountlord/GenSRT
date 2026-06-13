"""User-known Whisper models side file.

A small JSON file living next to ``gensrt-config.json`` that remembers
which custom Whisper models the user has added via the "New…" affordance
in the GUI footer or Config modal.

Design notes:
    * Built-in recommended models are listed in :data:`BUILTIN_RECOMMENDED`.
      They are always offered in the dropdown — even with an empty side file
      the user has the standard Whisper sizes to pick from.
    * User-added models accumulate in ``gensrt-known-models.json`` so the
      first-time download cost is paid once per model, not per session.
    * The "current model" itself lives in ``gensrt-config.json`` under the
      ``model`` key.  This side file is purely the *menu* of remembered
      models, not the active selection.

Schema::

    {
      "models": ["smcproject/vegam-whisper-medium-ml-int8_float16", ...]
    }

The file is auto-created on first add.  Missing or corrupt files degrade
gracefully to an empty user list.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

KNOWN_MODELS_FILENAME = "gensrt-known-models.json"

# Built-in recommended Whisper models.  These mirror the historical
# ``_MODEL_CHOICES`` list in :mod:`gensrt.server` — by centralising here we
# keep the dropdown contents and the backend's understanding of "known
# safe defaults" in one place.
BUILTIN_RECOMMENDED: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
)


def _resolve_known_models_path() -> Path:
    """Pick the same parent directory used for ``gensrt-config.json``.

    Mirrors the auto-discovery order from :mod:`gensrt.config` so the side
    file lives next to the config the user actually edits.
    """
    import sys

    candidates = [
        Path(sys.argv[0]).resolve().parent / KNOWN_MODELS_FILENAME,
        Path.cwd() / KNOWN_MODELS_FILENAME,
    ]
    # Prefer the one next to a config file already; otherwise CWD wins.
    for p in candidates:
        if p.is_file():
            return p
    return candidates[-1]


def load_known_models() -> list[str]:
    """Read the user-added model list.  Returns ``[]`` on missing/corrupt file."""
    path = _resolve_known_models_path()
    if not path.is_file():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Cannot parse %s (%s); ignoring user-added models.",
            path, exc,
        )
        return []

    if not isinstance(data, dict):
        logger.warning("%s must contain a JSON object; ignoring.", path)
        return []

    models = data.get("models", [])
    if not isinstance(models, list):
        logger.warning("%s 'models' key is not a list; ignoring.", path)
        return []

    # Filter to strings only and de-dup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for item in models:
        if isinstance(item, str) and item.strip() and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def save_known_models(models: Sequence[str]) -> Path:
    """Write the user-added model list atomically.

    Args:
        models: Iterable of HF repo IDs or local paths to remember.

    Returns:
        The path that was written.
    """
    path = _resolve_known_models_path()
    # De-dup while preserving order.
    seen: set[str] = set()
    cleaned: list[str] = []
    for m in models:
        if isinstance(m, str) and m.strip() and m not in seen:
            seen.add(m)
            cleaned.append(m)

    payload = {"models": cleaned}

    # Atomic write: temp file + rename.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)
    logger.debug("Wrote %d known model(s) to %s", len(cleaned), path)
    return path


def add_known_model(name: str) -> list[str]:
    """Add a model to the side file (idempotent).

    Built-in models are not added — they're always offered regardless.

    Args:
        name: HF repo ID (e.g. ``smcproject/vegam-whisper-medium-ml-int8_float16``)
              or local path.

    Returns:
        The full updated list of user-added models (excluding built-ins).
    """
    name = (name or "").strip()
    if not name:
        return load_known_models()
    if name in BUILTIN_RECOMMENDED:
        # Don't pollute the side file with built-ins.
        return load_known_models()

    current = load_known_models()
    if name in current:
        return current
    current.append(name)
    save_known_models(current)
    return current


def get_combined_models() -> list[str]:
    """Return built-in recommended + user-added, deduplicated.

    Built-ins come first (alphabetical "rough size" order matches their
    natural meaning), then user-added models in the order they were saved.
    """
    out: list[str] = list(BUILTIN_RECOMMENDED)
    seen: set[str] = set(out)
    for m in load_known_models():
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out
