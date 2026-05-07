"""Meta NLLB-200 offline translation engine.

Uses the ``facebook/nllb-200-distilled-600M`` model via HuggingFace
``transformers``.  Fully offline after the initial model download.
A single model handles all supported languages.
"""

from __future__ import annotations

import logging

from gensrt.constants import NLLB_LANGUAGE_CODES, NLLB_MODEL_ID
from gensrt.exceptions import TranslationEngineUnavailableError, TranslationError
from gensrt.translation.base import TranslationEngine

logger = logging.getLogger(__name__)

_TARGET_LANG = "eng_Latn"


class NLLBEngine(TranslationEngine):
    """Offline translation via Meta's NLLB-200 model."""

    def __init__(self, model_id: str = NLLB_MODEL_ID) -> None:
        self._model_id = model_id
        self._pipeline = None

    def _load(self) -> None:
        """Lazy-load the model pipeline on first use."""
        if self._pipeline is not None:
            return

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise TranslationEngineUnavailableError(
                "nllb",
                "transformers is not installed. Run: pip install transformers accelerate",
            ) from exc

        logger.info("Loading NLLB-200 model: %s", self._model_id)
        try:
            self._pipeline = pipeline(
                "translation",
                model=self._model_id,
                device_map="auto",
            )
            logger.info("NLLB-200 model loaded.")
        except Exception as exc:
            raise TranslationError("nllb", f"Failed to load model: {exc}") from exc

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return "nllb-200"

    def translate(self, text: str, source_language: str) -> str:
        """Translate *text* to English using NLLB-200.

        Args:
            text:            Text to translate.
            source_language: ISO 639-1 code (e.g. ``"ja"``, ``"ko"``, ``"ml"``).

        Returns:
            Translated English text.

        Raises:
            TranslationError: If the language is unsupported or inference fails.
        """
        results = self.translate_batch([text], source_language)
        return results[0]

    def translate_batch(self, texts: list[str], source_language: str) -> list[str]:
        """Translate a batch of texts via NLLB-200 in a single pipeline call."""
        if not texts:
            return []

        self._load()

        src_flores = NLLB_LANGUAGE_CODES.get(source_language)
        if src_flores is None:
            logger.warning(
                "[nllb] Unsupported language %r — returning originals.", source_language
            )
            return list(texts)

        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        results = list(texts)

        try:
            batch_texts = [t for _, t in non_empty]
            outputs = self._pipeline(  # type: ignore[misc]
                batch_texts,
                src_lang=src_flores,
                tgt_lang=_TARGET_LANG,
                max_length=512,
            )
            for (i, _), out in zip(non_empty, outputs):
                results[i] = out["translation_text"]
        except Exception as exc:
            raise TranslationError("nllb", str(exc)) from exc

        return results
