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


# Models that MUST use chunked inference.  Detected by prefix match so that
# different quantization variants of the same upstream model all route the
# same way.
#
# When adding entries here: include the canonical HF repo path (or path
# prefix) of any fine-tuned Whisper model known to drop content on
# long-form audio.  These are the entries we *guarantee* are routed
# correctly even if the user hasn't manually opted into chunked inference.
ALWAYS_CHUNKED_MODEL_PREFIXES: tuple[str, ...] = (
    # SMC's official Malayalam fine-tune of Whisper-medium.
    "smcproject/vegam-whisper-medium-ml",
    # Kurian Benoy's variants (different quantizations, same base model).
    "kurianbenoy/vegam-whisper-medium-ml",
)


def get_engine_for_model(model_name: str) -> ASREngine:
    """Return the ASR engine appropriate for *model_name*.

    Args:
        model_name: Whisper model identifier — either a built-in name
                    (``"large-v3-turbo"``), a HuggingFace repo ID
                    (``"smcproject/vegam-whisper-medium-ml-int8_float16"``),
                    or a local directory path.

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
    """Test whether *model_name* matches a known always-chunked pattern."""
    return any(
        model_name.startswith(prefix) for prefix in ALWAYS_CHUNKED_MODEL_PREFIXES
    )


def _make_multilingual() -> ASREngine:
    # Local import keeps faster-whisper out of the import graph until
    # actually needed (matches the translation factory's lazy approach).
    from gensrt.asr.multilingual_whisper import MultilingualWhisperEngine
    return MultilingualWhisperEngine()


def _make_monolingual() -> ASREngine:
    from gensrt.asr.monolingual_whisper import MonolingualWhisperEngine
    return MonolingualWhisperEngine()
