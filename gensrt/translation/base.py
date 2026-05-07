"""Abstract translation engine interface.

All concrete engines implement :class:`TranslationEngine`.
The pipeline never imports a concrete engine directly — it always
goes through :func:`gensrt.translation.factory.get_engine`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class TranslationEngine(ABC):
    """Abstract base for all translation engines."""

    @abstractmethod
    def translate(self, text: str, source_language: str) -> str:
        """Translate *text* from *source_language* to English.

        Args:
            text:            Text to translate.
            source_language: ISO 639-1 language code (e.g. ``"ja"``, ``"ko"``).

        Returns:
            Translated English text.

        Raises:
            TranslationError: On engine failure.
        """

    def translate_batch(self, texts: list[str], source_language: str) -> list[str]:
        """Translate a list of texts, batching where the engine supports it.

        Default implementation calls :meth:`translate` per item with per-item
        error handling so one bad segment never aborts the whole batch.
        Engines with native batch support (e.g. Google GTX glue-string,
        MarianMT tokeniser) should override this.

        Args:
            texts:           List of texts to translate.
            source_language: ISO 639-1 language code.

        Returns:
            List of translated strings in the same order as *texts*.
        """
        results: list[str] = []
        for text in texts:
            try:
                results.append(self.translate(text, source_language))
            except Exception as exc:
                logger.warning(
                    "[%s] translate failed (%s) — keeping original.", self.name, exc
                )
                results.append(text)
        return results

    @abstractmethod
    def is_available(self) -> bool:
        """Return ``True`` if this engine's dependencies are installed and reachable."""

    @property
    def name(self) -> str:
        """Human-readable engine name for logging."""
        return type(self).__name__


class PassthroughEngine(TranslationEngine):
    """No-op engine — returns input text unchanged.

    Used when ``--translation-engine none`` or source language is English.
    """

    def translate(self, text: str, source_language: str) -> str:
        return text

    def translate_batch(self, texts: list[str], source_language: str) -> list[str]:
        return list(texts)

    def is_available(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return "passthrough"
