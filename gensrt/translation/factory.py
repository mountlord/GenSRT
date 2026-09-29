"""Translation engine factory.

The pipeline always calls :func:`get_engine` rather than importing a
concrete engine class directly.  This keeps the pipeline decoupled from
any specific translation implementation.

Engines
-------
``nllb``
    NLLB-200 on CTranslate2 (v1.2.7).  Fully offline after a one-time
    ~650 MB model download, any of the mapped languages in either
    direction, GPU-accelerated.  The *weights* are CC-BY-NC-4.0
    (non-commercial only) — see :mod:`gensrt.translation.nllb_ct2`.
``madlad``
    MADLAD-400 on CTranslate2.  Also fully offline, also GPU-accelerated,
    and **Apache-2.0** — so it carries no commercial restriction.  The
    trade is size and speed: ~2.9 GB against NLLB's 650 MB, and roughly
    twice the per-cue cost.  Which one reads better is a judgement about
    the user's own material, so GenSRT offers both rather than choosing.
    See :mod:`gensrt.translation.madlad_ct2`.
``none``
    Transcribe without translating.

Removed engines
---------------
``google`` (the unofficial GTX endpoint, v1.0–v1.2.7) rate-limits by IP,
and a subtitling workload — thousands of cues per file, nightly — is
exactly what gets an IP blocked.  The block was observed to outlive an IP
change and persist for months, at which point the engine, its MyMemory
per-cue fallback and the whole ``translation_fallback`` mechanism were
dead weight.  All of it was removed in v1.3.0; both remaining engines run
offline.  ``marian`` was removed in v1.2.5.  A leftover key from an older
config produces an explanation rather than a bare "unknown engine" error.
"""

from __future__ import annotations

import logging
import threading

from gensrt.exceptions import ConfigError
from gensrt.translation.base import PassthroughEngine, TranslationEngine

logger = logging.getLogger(__name__)

#: Valid values for ``translation_engine``.
ENGINE_KEYS = ("nllb", "madlad", "none")

#: Engine keys that used to be valid, with the message a leftover config
#: value produces.
_REMOVED_ENGINES = {
    "marian": (
        "The 'marian' translation engine was removed in v1.2.5. It never "
        "worked reliably and required a ~2.5 GB PyTorch dependency."
    ),
    "google": (
        "The 'google' translation engine (Google GTX) was removed in v1.3.0: "
        "the endpoint blocks IPs that translate at subtitle volumes, and the "
        "block persists for months. GenSRT now translates offline only."
    ),
}
_REMOVED_ADVICE = (
    ' Set "translation_engine" to "nllb" (offline, non-commercial license), '
    '"madlad" (offline, Apache-2.0) or "none" (transcribe only) in '
    "gensrt-config.json or the Configuration editor."
)


def get_engine(key: str, config=None) -> TranslationEngine:
    """Return a :class:`TranslationEngine` instance for *key*.

    Args:
        key:    One of :data:`ENGINE_KEYS`.
        config: Optional :class:`~gensrt.models.TranscriptionConfig`.
                Supplies ``translation_model`` / ``madlad_model`` /
                ``device``.  Engines work with sensible defaults when it is
                omitted, which keeps existing call sites and tests valid.

    Returns:
        A fresh engine instance.

    Raises:
        ConfigError: If *key* is not valid.
    """
    k = (key or "").lower()

    if k == "none":
        engine: TranslationEngine = PassthroughEngine()

    elif k == "nllb":
        from gensrt.translation.nllb_ct2 import NLLBCT2Engine

        engine = NLLBCT2Engine(config)

    elif k == "madlad":
        from gensrt.translation.madlad_ct2 import MADLADCT2Engine

        engine = MADLADCT2Engine(config)

    elif k in _REMOVED_ENGINES:
        raise ConfigError(_REMOVED_ENGINES[k] + _REMOVED_ADVICE)

    else:
        raise ConfigError(
            f"Unknown translation engine: {key!r}. "
            f"Valid choices: {list(ENGINE_KEYS)}"
        )

    logger.debug("Translation engine: %s", engine.name)
    return engine


# One engine per (key, model, device) for the life of the process.  Without
# this a batch run loaded MADLAD (~3 GB of VRAM) twice per file — once to
# translate, once for the heuristics report — and never released the
# previous file's copies before the next Whisper load.  On an 8 GB card
# that is the difference between GPU and CPU for every file after the first.
_shared: dict[tuple, TranslationEngine] = {}
_shared_lock = threading.Lock()


def _shared_key(key: str, config) -> tuple:
    g = lambda name: str(getattr(config, name, "") or "")  # noqa: E731
    return ((key or "").lower(), g("translation_model"), g("madlad_model"), g("device"))


def get_shared_engine(key: str, config=None) -> TranslationEngine:
    """Like :func:`get_engine`, but the same instance is returned for the same
    engine/model/device for the rest of the process."""
    k = _shared_key(key, config)
    with _shared_lock:
        engine = _shared.get(k)
        if engine is None:
            engine = get_engine(key, config)
            _shared[k] = engine
        return engine


def clear_shared_engines() -> None:
    """Drop the cached engines (tests, or a device change)."""
    with _shared_lock:
        _shared.clear()


def available_engines() -> list[str]:
    """Return the list of valid engine keys."""
    return list(ENGINE_KEYS)
