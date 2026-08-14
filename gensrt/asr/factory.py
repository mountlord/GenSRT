"""ASR engine factory.

The pipeline calls :func:`get_engine_for_model` rather than importing
a concrete engine class directly.  This keeps the pipeline decoupled
from any specific ASR implementation and lets us route between engines
based on the model the user has selected.

Routing rules:
    * Models in ``BUILTIN_RECOMMENDED`` (large-v3-turbo, medium, etc.)
      → :class:`MultilingualWhisperEngine` (current GenSRT behavior;
      identical to v1.1).
    * Models matching a known "always-chunked" pattern (vegam variants)
      → :class:`MonolingualWhisperEngine` (chunked inference, always).
    * Any other custom user-added Whisper model
      → :class:`MonolingualWhisperEngine` by default.  Most community
      fine-tunes are Common-Voice-trained and exhibit the same
      phrase-shaped output as vegam, so chunked inference is the safer
      default.

Mirrors the pattern established by :mod:`gensrt.translation.factory`.
"""

from __future__ import annotations

import logging

from gensrt.asr.base import ASREngine
from gensrt.known_models import BUILTIN_RECOMMENDED

logger = logging.getLogger(__name__)


# Registry of known monolingual fine-tuned Whisper models.
#
# Each entry: (HF repo prefix, ISO 639-1 training language).
#
# Models in this registry behave specially in two ways:
#   1. They route to MonolingualWhisperEngine unconditionally (chunked
#      inference).  See get_engine_for_model() below.
#   2. The MonolingualWhisperEngine SKIPS per-chunk language detection
#      and uses the registered training language directly.  This is
#      essential for vegam, whose language-detection head produces
#      garbage after fine-tuning ('ta', 'ba', 'en', 'sv' detected on
#      short Malayalam chunks).  See get_known_language_for_model().
#
# Prefix-matched so different quantization variants of the same upstream
# model (-int8_float16, -fp16, etc.) all route the same way.
#
# To add a new entry: include the canonical HF repo path (or a path
# prefix that covers all quantization variants) and the ISO 639-1 code
# of the model's training language.  If the model is trained on a single
# language, this list is where it goes.
KNOWN_MONOLINGUAL_MODELS: tuple[tuple[str, str], ...] = (
    # SMC's official Malayalam fine-tune of Whisper-medium.
    ("smcproject/vegam-whisper-medium-ml", "ml"),
    # Kurian Benoy's variants (different quantizations, same base model).
    ("kurianbenoy/vegam-whisper-medium-ml", "ml"),
    # Adalat AI's R-MFT Malayalam fine-tune of Whisper-medium.  Vividh-ASR
    # benchmark reports substantial WER improvements over vegam on broadcast
    # (~43% relative) and global (~26% relative) test splits.  Training data
    # is ~894h across studio/broadcast/spontaneous tiers; technique
    # introduced in arxiv 2605.13087 (Juvekar, Manohar, et al., 2026).
    # CT2 conversion published by Adalat AI directly; license Apache-2.0.
    ("adalat-ai/ct2-whisper-medium-ml-rmft", "ml"),
)

# Derived view: just the prefix strings, for the always-chunked check.
# Kept as a module-level constant so external callers (tests, future
# debug commands) have a stable name to import.
ALWAYS_CHUNKED_MODEL_PREFIXES: tuple[str, ...] = tuple(
    prefix for prefix, _lang in KNOWN_MONOLINGUAL_MODELS
)


def get_engine_for_model(model_name: str, override: str | None = None) -> ASREngine:
    """Return the ASR engine appropriate for *model_name*.

    Args:
        model_name: Whisper model identifier — either a built-in name
                    (``"large-v3-turbo"``), a HuggingFace repo ID
                    (``"smcproject/vegam-whisper-medium-ml-int8_float16"``),
                    or a local directory path.
        override:   ``"chunked"`` or ``"longform"`` to bypass the routing
                    rules entirely; ``"auto"``, ``None`` or ``""`` to apply
                    them.  The rules below are a heuristic about how a model
                    was trained, and there is no way to introspect that from
                    a checkpoint — so the heuristic can be wrong, and a user
                    who knows their material better than we do should be able
                    to say so.

    Returns:
        An :class:`ASREngine` instance ready to call ``transcribe()`` on.

    Notes:
        * The engine instance is freshly constructed — engines are
          stateless and cheap to create; the expensive model load happens
          inside ``transcribe()``.
        * Local-path models always route to the monolingual engine.  We
          can't introspect a local checkpoint to determine its training
          distribution, and the safer default for user-supplied weights
          is chunked inference.
    """
    name = (model_name or "").strip()

    choice = (override or "auto").strip().lower()
    if choice == "chunked":
        engine = _make_monolingual()
        logger.info("ASR engine for %r: %s (forced by asr_engine)", name, engine.name)
        return engine
    if choice == "longform":
        engine = _make_multilingual()
        logger.info("ASR engine for %r: %s (forced by asr_engine)", name, engine.name)
        return engine
    if choice not in ("auto", ""):
        logger.warning("Unrecognised asr_engine %r — using automatic routing.", override)

    # Built-in: always multilingual (current behavior, validated long-form).
    if name in BUILTIN_RECOMMENDED:
        engine = _make_multilingual()
        logger.debug("ASR engine for %r: %s (built-in)", name, engine.name)
        return engine

    # Known always-chunked: route to monolingual unconditionally.
    if _is_always_chunked(name):
        engine = _make_monolingual()
        logger.debug("ASR engine for %r: %s (always-chunked pattern)", name, engine.name)
        return engine

    # Any other custom model: chunked by default.  See module docstring.
    engine = _make_monolingual()
    logger.debug("ASR engine for %r: %s (custom default)", name, engine.name)
    return engine


def _is_always_chunked(model_name: str) -> bool:
    """Test whether *model_name* matches a known monolingual prefix."""
    if not model_name:
        return False
    return any(
        model_name.startswith(prefix) for prefix, _lang in KNOWN_MONOLINGUAL_MODELS
    )


def get_known_language_for_model(model_name: str) -> str | None:
    """Return the registered training language for a known monolingual model.

    Args:
        model_name: HF repo ID, local path, or built-in name.

    Returns:
        ISO 639-1 language code (e.g. ``"ml"`` for vegam) if *model_name*
        starts with a prefix in :data:`KNOWN_MONOLINGUAL_MODELS`.
        ``None`` otherwise — including for any built-in Whisper model.

    Use case:
        :class:`MonolingualWhisperEngine` calls this when the user has
        ``source_language="auto"`` to avoid per-chunk language detection
        on a model whose detection head we know is unreliable.

    Returning ``None`` for unknown custom models is deliberate: the
    engine falls back to first-chunk-detect-and-cache for those,
    which is the right behavior when we don't know the model's
    training distribution a priori.
    """
    if not model_name:
        return None
    for prefix, lang in KNOWN_MONOLINGUAL_MODELS:
        if model_name.startswith(prefix):
            return lang
    return None


def _make_multilingual() -> ASREngine:
    # Local import keeps faster-whisper out of the import graph until
    # actually needed (matches the translation factory's lazy approach).
    from gensrt.asr.multilingual_whisper import MultilingualWhisperEngine
    return MultilingualWhisperEngine()


def _make_monolingual() -> ASREngine:
    from gensrt.asr.monolingual_whisper import MonolingualWhisperEngine
    return MonolingualWhisperEngine()
