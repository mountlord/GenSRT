"""Translation engine factory.

The pipeline always calls :func:`get_engine` rather than importing a
concrete engine class directly.  This keeps the pipeline decoupled from
any specific translation implementation.

Engines
-------
``google``
    The Google GTX HTTP endpoint.  Any target language, needs a network
    connection, and is rate-limited by IP — see the batch-failure fallback
    below.
``nllb``
    NLLB-200 on CTranslate2 (v1.2.7).  Fully offline after a one-time
    ~650 MB model download, any of the mapped languages in either
    direction, GPU-accelerated.  The *weights* are CC-BY-NC-4.0
    (non-commercial only) — see :mod:`gensrt.translation.nllb_ct2`.
``none``
    Transcribe without translating.

The fallback
------------
``google`` additionally takes a *fallback* — what to do when a GTX batch
fails outright (typically HTTP 429 once an IP is throttled):

    ``nllb``      translate the failed batch offline        (default)
    ``mymemory``  the pre-v1.2.7 behaviour, kept for compatibility
    ``none``      keep the source text for the failed batch

The fallback comes from ``translation_fallback`` in the config, which is
why :func:`get_engine` accepts the config: the factory is the one place
that knows how to wire an engine into another engine's failure path.  The
NLLB fallback is constructed lazily — a run where Google succeeds never
touches the NLLB model at all.
"""

from __future__ import annotations

import logging

from gensrt.exceptions import ConfigError
from gensrt.translation.base import PassthroughEngine, TranslationEngine

logger = logging.getLogger(__name__)

#: Valid values for ``translation_engine``.
ENGINE_KEYS = ("google", "nllb", "none")

#: Valid values for ``translation_fallback`` (Google batch-failure handling).
FALLBACK_KEYS = ("nllb", "mymemory", "none")

# Engines that used to exist. Recognised solely so that a config file left
# over from an earlier version produces an explanation rather than a bare
# "unknown engine" error.  NLLB left this set in v1.2.7: it is back, on
# CTranslate2 this time, with no torch anywhere near it.
_REMOVED_ENGINES = frozenset({"marian"})


def get_engine(key: str, config=None) -> TranslationEngine:
    """Return a :class:`TranslationEngine` instance for *key*.

    Args:
        key:    One of :data:`ENGINE_KEYS`.
        config: Optional :class:`~gensrt.models.TranscriptionConfig`.
                Supplies ``translation_fallback`` (for ``google``) and
                ``translation_model`` / ``device`` (for ``nllb``).  Engines
                work with sensible defaults when it is omitted, which keeps
                existing call sites and tests valid.

    Returns:
        A fresh engine instance.

    Raises:
        ConfigError: If *key* (or the configured fallback) is not valid.
    """
    k = (key or "").lower()

    if k == "none":
        engine: TranslationEngine = PassthroughEngine()

    elif k == "nllb":
        from gensrt.translation.nllb_ct2 import NLLBCT2Engine

        engine = NLLBCT2Engine(config)

    elif k == "google":
        from gensrt.translation.google_gtx import GoogleGTXEngine

        fallback = (
            getattr(config, "translation_fallback", None) or "mymemory"
        ).lower()
        if fallback not in FALLBACK_KEYS:
            raise ConfigError(
                f"Unknown translation_fallback: {fallback!r}. "
                f"Valid choices: {list(FALLBACK_KEYS)}"
            )

        fallback_factory = None
        if fallback == "nllb":
            from gensrt.translation.nllb_ct2 import NLLBCT2Engine

            # Lazy: constructed (and the model loaded) only if a Google
            # batch actually fails.
            def fallback_factory() -> TranslationEngine:  # noqa: E306
                return NLLBCT2Engine(config)

        engine = GoogleGTXEngine(
            fallback=fallback, fallback_engine_factory=fallback_factory
        )

    elif k in _REMOVED_ENGINES:
        raise ConfigError(
            f"The {key!r} translation engine was removed in v1.2.5. It "
            f"never worked reliably and required a ~2.5 GB PyTorch "
            f"dependency for a feature that could only ever produce "
            f"English. Set \"translation_engine\" to \"google\" (any "
            f"target language, needs a network connection), \"nllb\" "
            f"(offline, non-commercial license) or \"none\" (transcribe "
            f"only) in gensrt-config.json."
        )

    else:
        raise ConfigError(
            f"Unknown translation engine: {key!r}. "
            f"Valid choices: {list(ENGINE_KEYS)}"
        )

    logger.debug("Translation engine: %s", engine.name)
    return engine


def available_engines() -> list[str]:
    """Return the list of valid engine keys."""
    return list(ENGINE_KEYS)
