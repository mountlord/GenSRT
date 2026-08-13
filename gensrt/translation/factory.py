"""Translation engine factory.

The pipeline always calls :func:`get_engine` rather than importing a
concrete engine class directly.  This keeps the pipeline decoupled from
any specific translation implementation.
"""

from __future__ import annotations

import logging

from gensrt.exceptions import ConfigError
from gensrt.translation.base import PassthroughEngine, TranslationEngine

logger = logging.getLogger(__name__)

# Populated on first import so engines are not imported unless needed.
_ENGINE_REGISTRY: dict[str, type[TranslationEngine]] = {}

# Engines that used to exist. Recognised solely so that a config file left
# over from an earlier version produces an explanation rather than a bare
# "unknown engine" error.
_REMOVED_ENGINES = frozenset({"nllb", "marian"})


def _registry() -> dict[str, type[TranslationEngine]]:
    global _ENGINE_REGISTRY
    if not _ENGINE_REGISTRY:
        from gensrt.translation.google_gtx import GoogleGTXEngine

        _ENGINE_REGISTRY = {
            "google": GoogleGTXEngine,
            "none": PassthroughEngine,
        }
    return _ENGINE_REGISTRY


def get_engine(key: str) -> TranslationEngine:
    """Return a :class:`TranslationEngine` instance for *key*.

    Args:
        key: One of ``"google"`` or ``"none"``.

    Returns:
        A fresh engine instance.

    Raises:
        ConfigError: If *key* is not a known engine.
    """
    reg = _registry()
    cls = reg.get(key.lower())
    if cls is None:
        if key.lower() in _REMOVED_ENGINES:
            raise ConfigError(
                f"The {key!r} translation engine was removed in v1.2.5. It "
                f"never worked reliably and required a ~2.5 GB PyTorch "
                f"dependency for a feature that could only ever produce "
                f"English. Set \"translation_engine\" to \"google\" "
                f"(any target language, needs a network connection) or "
                f"\"none\" (transcribe only) in gensrt-config.json."
            )
        raise ConfigError(
            f"Unknown translation engine: {key!r}. "
            f"Valid choices: {list(reg.keys())}"
        )

    engine = cls()
    logger.debug("Translation engine: %s", engine.name)
    return engine


def available_engines() -> list[str]:
    """Return the list of valid engine keys."""
    return list(_registry().keys())
