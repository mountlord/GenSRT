"""Multilingual Whisper ASR engine.

Wraps the existing GenSRT behavior — load a Whisper model via
faster-whisper, call ``model.transcribe()`` with the user's VAD
parameters, return SRT segments.  Suitable for the built-in Whisper
models (``large-v3-turbo``, ``medium``, etc.) which handle long-form
audio natively via their internal 30-second windowing.

This engine is *not* suitable for fine-tuned Whisper models trained on
short-phrase corpora like Common Voice (e.g. vegam).  Those drop content
mid-window because the model has learned a phrase-shaped output
distribution.  Route such models through
:mod:`gensrt.asr.monolingual_whisper` instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

from gensrt.asr.base import ASREngine
from gensrt.exceptions import TranscriptionError
from gensrt.models import SRTSegment, TranscriptionConfig
from gensrt.srt.builder import segments_from_whisper

logger = logging.getLogger(__name__)


class MultilingualWhisperEngine(ASREngine):
    """faster-whisper engine for multilingual / long-form Whisper models.

    Behavior matches GenSRT 1.1 exactly — this class is a refactoring
    seam, not a behavior change.  Existing users on built-in Whisper
    models see identical output before and after Drop I.15.
    """

    def transcribe(
        self,
        wav_path: Path,
        config: TranscriptionConfig,
    ) -> tuple[list[SRTSegment], str]:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError(
                str(wav_path),
                "faster-whisper is not installed. Run: pip install faster-whisper",
            ) from exc

        model = self._load_model(wav_path, config, WhisperModel)

        transcribe_kwargs = self._build_transcribe_kwargs(config)
        self._log_vad_config(config)

        try:
            segments_gen, info = model.transcribe(str(wav_path), **transcribe_kwargs)
            # Consume the generator so the caller can safely delete the WAV.
            raw_segments = list(segments_gen)
        except Exception as exc:
            raise TranscriptionError(str(wav_path), str(exc)) from exc

        detected_language = info.language or "unknown"
        logger.info(
            "[%s] %d raw segments, lang=%s (prob=%.2f)",
            self.name,
            len(raw_segments),
            detected_language,
            info.language_probability,
        )

        srt_segments = segments_from_whisper(raw_segments)
        return srt_segments, detected_language

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_model(
        wav_path: Path,
        config: TranscriptionConfig,
        WhisperModel,  # noqa: N803 — capitalised to mirror upstream class
    ):
        """Load the Whisper model with compute-type fallbacks.

        Some GPUs / CTranslate2 builds reject ``float16`` and require
        falling back to ``int8_float16`` or ``int8``.  We try the user's
        requested type first, then graceful fallbacks.
        """
        compute_type = config.compute_type
        fallbacks: list[str] = [compute_type]
        if compute_type == "float16":
            fallbacks += ["int8_float16", "int8"]
        elif compute_type == "int8_float16":
            fallbacks += ["int8"]

        logger.info(
            "Loading Whisper model: %s  (device=%s, compute=%s)",
            config.model, config.device, compute_type,
        )

        last_exc: Exception | None = None
        for ct in fallbacks:
            try:
                model = WhisperModel(config.model, device=config.device, compute_type=ct)
                if ct != compute_type:
                    logger.warning(
                        "compute_type=%r unsupported — fell back to %r.",
                        compute_type, ct,
                    )
                return model
            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "WhisperModel load failed with compute_type=%r: %s", ct, exc
                )

        raise TranscriptionError(
            str(wav_path), f"Failed to load Whisper model: {last_exc}"
        ) from last_exc

    @staticmethod
    def _build_transcribe_kwargs(config: TranscriptionConfig) -> dict:
        """Translate :class:`TranscriptionConfig` into transcribe() kwargs.

        Honors the user's VAD parameters when ``vad_enabled`` is True.
        """
        source_lang = None if config.source_language == "auto" else config.source_language
        kwargs: dict = dict(
            language=source_lang,
            word_timestamps=True,
            beam_size=5,
        )

        if config.vad_enabled:
            kwargs["vad_filter"] = True
            kwargs["vad_parameters"] = {
                "threshold":                config.vad_threshold,
                "min_speech_duration_ms":   config.vad_min_speech_ms,
                "min_silence_duration_ms":  config.vad_min_silence_ms,
                "speech_pad_ms":            config.vad_speech_pad_ms,
            }

        return kwargs

    @staticmethod
    def _log_vad_config(config: TranscriptionConfig) -> None:
        if config.vad_enabled:
            logger.info(
                "VAD enabled (threshold=%.2f, min_speech=%dms, "
                "min_silence=%dms, speech_pad=%dms)",
                config.vad_threshold, config.vad_min_speech_ms,
                config.vad_min_silence_ms, config.vad_speech_pad_ms,
            )
        else:
            logger.info("VAD disabled — full audio passed to Whisper.")
