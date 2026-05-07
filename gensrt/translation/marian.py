"""MarianMT offline translation engine.

Uses Helsinki-NLP ``opus-mt-*-en`` models via HuggingFace ``transformers``.
Fully offline after the initial model download.  One model per language —
they are lazy-loaded and cached in memory.
"""

from __future__ import annotations

import logging

from gensrt.constants import MARIAN_MODELS
from gensrt.exceptions import TranslationEngineUnavailableError, TranslationError
from gensrt.translation.base import TranslationEngine

logger = logging.getLogger(__name__)


class MarianEngine(TranslationEngine):
    """Offline translation via Helsinki-NLP MarianMT models."""

    def __init__(self) -> None:
        # model_id → (tokenizer, model) pairs — lazy-loaded per language
        self._cache: dict[str, tuple] = {}

    def _load_for_language(self, source_language: str) -> tuple:
        """Load (tokenizer, model) for *source_language*, cached after first call."""
        model_id = MARIAN_MODELS.get(source_language)
        if model_id is None:
            raise TranslationError(
                "marian",
                f"No MarianMT model configured for language {source_language!r}. "
                f"Supported: {list(MARIAN_MODELS.keys())}",
            )

        if model_id in self._cache:
            return self._cache[model_id]

        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError as exc:
            raise TranslationEngineUnavailableError(
                "marian",
                "transformers is not installed. Run: pip install transformers accelerate",
            ) from exc

        logger.info("Loading MarianMT model: %s", model_id)
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_id)
            model = MarianMTModel.from_pretrained(model_id)
            self._cache[model_id] = (tokenizer, model)
            logger.info("MarianMT model loaded: %s", model_id)
            return tokenizer, model
        except Exception as exc:
            raise TranslationError("marian", f"Failed to load model {model_id!r}: {exc}") from exc

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    @property
    def name(self) -> str:
        return "marian-mt"

    def translate(self, text: str, source_language: str) -> str:
        results = self.translate_batch([text], source_language)
        return results[0]

    def translate_batch(self, texts: list[str], source_language: str) -> list[str]:
        """Translate a batch via MarianMT tokeniser in a single forward pass."""
        if not texts:
            return []

        tokenizer, model = self._load_for_language(source_language)

        non_empty = [(i, t) for i, t in enumerate(texts) if t.strip()]
        results = list(texts)

        try:
            batch_texts = [t for _, t in non_empty]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            translated_tokens = model.generate(**inputs)
            for (i, _), tokens in zip(non_empty, translated_tokens):
                results[i] = tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as exc:
            raise TranslationError("marian", str(exc)) from exc

        logger.debug("[marian] batch %d → %d items", len(texts), len(non_empty))
        return results
