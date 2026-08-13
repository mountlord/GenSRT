"""Abstract ASR engine interface.

All concrete engines implement :class:`ASREngine`.  The pipeline never
imports a concrete engine directly — it always goes through
:func:`gensrt.asr.factory.get_engine_for_model`.

This mirrors the pattern established by :mod:`gensrt.translation`.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from gensrt.models import SRTSegment, TranscriptionConfig

logger = logging.getLogger(__name__)


class ASREngine(ABC):
    """Abstract base for all ASR engines.

    An engine is responsible for converting an audio WAV file into a list
    of :class:`SRTSegment` objects plus a detected language code.  All
    model loading, VAD, chunking, and timestamp handling is encapsulated
    inside the engine — callers see only the segments.
    """

    @abstractmethod
    def transcribe(
        self,
        wav_path: Path,
        config: TranscriptionConfig,
        *,
        status=None,
    ) -> tuple[list[SRTSegment], str]:
        """Transcribe *wav_path* and return SRT segments + detected language.

        Args:
            wav_path: Path to a 16 kHz mono PCM WAV file (as produced by
                      :func:`gensrt.audio.extractor.extract_audio`).
            config:   Fully resolved :class:`TranscriptionConfig`.  The
                      engine reads model name, device, compute type, VAD
                      parameters, and source language from this object.
            status:   Optional ``(str) -> None`` callback for human-readable
                      progress messages.  Engines use it to surface events the
                      user needs to know about mid-run — most importantly a
                      GPU-to-CPU fallback, which changes a 7-minute job into a
                      25-minute one and must not be log-only.

        Returns:
            Tuple of:
                * ``list[SRTSegment]`` with ``index`` starting at 1 and
                  monotonically increasing timestamps.
                * Detected language string (ISO 639-1 code, or ``"unknown"``
                  if detection failed but transcription succeeded).

        Raises:
            TranscriptionError: On engine failure (model load, decode error,
                                etc.).
        """

    @property
    def name(self) -> str:
        """Human-readable engine name for logging."""
        return type(self).__name__
