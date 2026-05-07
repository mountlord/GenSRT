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


def _registry() -> dict[str, type[TranslationEngine]]:
    global _ENGINE_REGISTRY
    if not _ENGINE_REGISTRY:
        from gensrt.translation.google_gtx import GoogleGTXEngine
        from gensrt.translation.nllb import NLLBEngine
        from gensrt.translation.marian import MarianEngine

        _ENGINE_REGISTRY = {
            "google": GoogleGTXEngine,
            "nllb": NLLBEngine,
            "marian": MarianEngine,
            "none": PassthroughEngine,
        }
    return _ENGINE_REGISTRY


def get_engine(key: str) -> TranslationEngine:
    """Return a :class:`TranslationEngine` instance for *key*.

    Args:
        key: One of ``"google"``, ``"nllb"``, ``"marian"``, ``"none"``.

    Returns:
        A fresh engine instance.

    Raises:
        ConfigError: If *key* is not a known engine.
    """
    reg = _registry()
    cls = reg.get(key.lower())
    if cls is None:
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
