"""Monolingual Whisper ASR engine with silent-boundary chunking.

For fine-tuned Whisper models that exhibit phrase-shaped output behavior
(produce ~6-8 seconds of transcription per inference call regardless of
input length).  Vegam is the canonical example.

Approach:
    1. Run silero-VAD with user's parameters to find speech regions.
    2. For each region longer than ``max_chunk_s``, run progressive VAD
       (multiple threshold + min_silence combinations) to find inner
       silence midpoints.  See
       :mod:`gensrt.asr._silence_chunking` for the algorithm.
    3. Slice the audio per chunk and call ``model.transcribe()`` with
       ``vad_filter=False`` on each chunk.  Each chunk fits the model's
       natural output distribution, eliminating the dropped-speech
       problem at the source.
    4. Offset chunk-local timestamps by chunk start time and assemble
       into a single segment list.

Why VAD parameters are honored only for the outer pass:
    The user's ``vad_threshold`` etc. configure speech-vs-silence
    classification for the obvious-speech-detection job (finding
    talking-versus-silent regions).  The progressive inner sweep is
    algorithm-controlled — it sweeps threshold and min_silence to find
    inter-word gaps the outer pass doesn't surface.  Exposing the inner
    parameters would let users break the chunking quality without
    realising it.
"""

from __future__ import annotations

import logging
from pathlib import Path

from gensrt.asr._silence_chunking import (
    DEFAULT_MAX_CHUNK_S,
    DEFAULT_MIN_CHUNK_S,
    plan_chunks,
    summarize_chunk_plan,
)
from gensrt.asr.base import ASREngine
from gensrt.exceptions import TranscriptionError
from gensrt.models import SRTSegment, TranscriptionConfig

logger = logging.getLogger(__name__)


class MonolingualWhisperEngine(ASREngine):
    """Chunked-inference engine for fine-tuned / phrase-shaped Whisper models.

    Routes audio through silent-boundary chunking before per-chunk
    Whisper inference.  Suitable for Whisper models fine-tuned on
    short-phrase corpora (Common Voice, etc.) where calling
    ``model.transcribe()`` on long-form audio results in dropped speech.
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

        # ── Step 1: Load audio into memory ───────────────────────────────
        # We need numpy access for VAD slicing and energy-min detection.
        from gensrt.audio.loader import load_wav_float32
        audio, sr = load_wav_float32(wav_path)
        audio_duration_s = len(audio) / sr

        # ── Step 2: Outer VAD — find speech regions ──────────────────────
        regions = self._outer_vad(audio, sr, config)
        logger.info(
            "[%s] Outer VAD: %d speech region(s) in %.1fs of audio",
            self.name, len(regions), audio_duration_s,
        )

        if not regions:
            logger.warning(
                "[%s] No speech regions detected.  Returning empty transcription.",
                self.name,
            )
            return [], "unknown"

        # ── Step 3: Plan chunks ──────────────────────────────────────────
        chunks = plan_chunks(
            audio, sr, regions,
            max_chunk_s=DEFAULT_MAX_CHUNK_S,
            min_chunk_s=DEFAULT_MIN_CHUNK_S,
        )
        if not chunks:
            logger.warning(
                "[%s] Chunk planner emitted no chunks.  All speech regions "
                "were shorter than min_chunk_s.",
                self.name,
            )
            return [], "unknown"

        stats = summarize_chunk_plan(chunks)
        logger.info(
            "[%s] Chunk plan: %d chunks (%.1f%% silence cuts, %.1f%% energy-min)",
            self.name, stats["n_chunks"], stats["pct_silence"], stats["pct_energy_min"],
        )

        # ── Step 4: Load model + per-chunk inference ─────────────────────
        model = self._load_model(wav_path, config, WhisperModel)

        # Decide the language we'll pass to model.transcribe() per chunk.
        #
        # Priority:
        #   1. User explicit (config.source_language != "auto") → use it
        #      as-is.  The user asked, we comply.
        #   2. Auto + model is in the known-monolingual registry → use
        #      the registered training language.  Fine-tuned models like
        #      vegam have an unreliable language-detection head; running
        #      detection per chunk produces garbage results ('ta', 'ba',
        #      'en' on Malayalam audio).  Skipping it is faster AND
        #      gives the model a consistent language token across chunks.
        #   3. Auto + unknown model → leave as None.  The chunked
        #      transcription loop will detect on the first chunk, then
        #      reuse that result for every subsequent chunk (see
        #      _transcribe_chunks).
        if config.source_language == "auto":
            from gensrt.asr.factory import get_known_language_for_model
            known_lang = get_known_language_for_model(config.model)
            if known_lang is not None:
                source_lang = known_lang
                logger.info(
                    "[%s] Auto-detect + known monolingual model: using "
                    "registered training language %r (skips per-chunk "
                    "detection)",
                    self.name, source_lang,
                )
            else:
                source_lang = None
        else:
            source_lang = config.source_language

        srt_segments, detected_language = self._transcribe_chunks(
            model, audio, sr, chunks, source_lang, wav_path,
        )

        logger.info(
            "[%s] %d segments emitted, lang=%s",
            self.name, len(srt_segments), detected_language,
        )
        return srt_segments, detected_language

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _outer_vad(audio, sr: int, config: TranscriptionConfig) -> list[tuple[float, float]]:
        """Run silero-VAD with user-configured parameters.

        Returns speech regions as ``(start_s, end_s)`` tuples on the audio
        timeline.  When ``config.vad_enabled`` is False, returns a single
        region covering the entire audio (so the chunker still gets a
        chance to break up long audio).
        """
        if not config.vad_enabled:
            duration_s = len(audio) / sr
            return [(0.0, duration_s)]

        from faster_whisper.vad import get_speech_timestamps, VadOptions
        opts = VadOptions(
            threshold=config.vad_threshold,
            min_speech_duration_ms=config.vad_min_speech_ms,
            min_silence_duration_ms=config.vad_min_silence_ms,
            speech_pad_ms=config.vad_speech_pad_ms,
        )
        raw = get_speech_timestamps(audio, opts)
        return [(r["start"] / sr, r["end"] / sr) for r in raw]

    @staticmethod
    def _load_model(
        wav_path: Path,
        config: TranscriptionConfig,
        WhisperModel,  # noqa: N803
    ):
        """Load the Whisper model with compute-type fallbacks.

        Mirrors the loading logic in
        :class:`gensrt.asr.multilingual_whisper.MultilingualWhisperEngine`
        — kept duplicated rather than abstracted because the engines may
        diverge in the future (e.g. monolingual could grow per-chunk
        warm-up logic).
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

    def _transcribe_chunks(
        self,
        model,
        audio,
        sr: int,
        chunks: list[dict],
        source_lang: str | None,
        wav_path: Path,
    ) -> tuple[list[SRTSegment], str]:
        """Run per-chunk inference and assemble segments.

        Each chunk's audio is sliced from the in-memory float32 array and
        written to a per-chunk temp WAV (because faster-whisper's
        transcribe() expects a path or array; passing a path is the
        simpler integration that matches how the model was tested).

        We could pass the numpy slice directly via ``WhisperModel.transcribe(audio=...)``
        — but the test scripts wrote WAVs and we want production to match
        what we validated.  Performance impact is negligible (~100ms per
        chunk for the WAV write).
        """
        import numpy as np
        import tempfile
        import wave

        all_cues: list[tuple[float, float, str]] = []
        chunk_index_for_logging = 0

        # `effective_lang` is what we pass to model.transcribe() at each
        # chunk.  It starts as the caller-supplied source_lang — which may
        # already be set from the user's explicit choice OR from the
        # known-monolingual registry — and stays constant for the run.
        # If source_lang is None (user picked auto AND model isn't in the
        # registry), effective_lang is None on the first chunk only.
        # As soon as the first successful chunk returns a detected
        # language, we lock it in for every remaining chunk.  This avoids
        # the per-chunk detection cost AND prevents inconsistent language
        # conditioning across chunks of the same audio.
        effective_lang: str | None = source_lang
        detected_language: str | None = source_lang  # for the return value

        # Single temp dir for all chunks of this job — auto-cleaned via with.
        with tempfile.TemporaryDirectory(prefix="gensrt_chunks_") as tmp_dir:
            tmp_path = Path(tmp_dir)

            for chunk in chunks:
                chunk_index_for_logging += 1
                c_start = chunk["start_s"]
                c_end = chunk["end_s"]
                s0 = int(c_start * sr)
                s1 = int(c_end * sr)
                samples = audio[s0:s1]

                chunk_wav = tmp_path / f"chunk_{chunk_index_for_logging:04d}.wav"
                _write_chunk_wav(samples, sr, chunk_wav)

                try:
                    segments_gen, info = model.transcribe(
                        str(chunk_wav),
                        language=effective_lang,
                        word_timestamps=True,
                        beam_size=5,
                        # vad_filter=False: we've pre-chunked using our own
                        # progressive VAD; running faster-whisper's VAD
                        # again would be redundant and could trim chunk edges.
                        vad_filter=False,
                    )
                    chunk_segments = list(segments_gen)
                except Exception as exc:
                    # One bad chunk shouldn't abort the whole transcription.
                    # Log it and skip — the user sees a small gap rather
                    # than a complete failure.
                    logger.warning(
                        "[%s] Chunk %d transcribe failed (%s) — skipping.",
                        self.name, chunk_index_for_logging, exc,
                    )
                    continue

                # Lock the language on first successful detection.  This
                # both captures the return value and stops per-chunk
                # detection in faster-whisper for the rest of the run.
                if effective_lang is None and info.language:
                    effective_lang = info.language
                    detected_language = info.language
                    logger.info(
                        "[%s] Detected language %r on first chunk — "
                        "reusing for remaining %d chunk(s)",
                        self.name, effective_lang, len(chunks) - chunk_index_for_logging,
                    )

                # Offset chunk-local timestamps by chunk start.
                for seg in chunk_segments:
                    abs_start = c_start + float(seg.start)
                    abs_end = min(c_start + float(seg.end), c_end)
                    text = seg.text.strip()
                    if text:
                        all_cues.append((abs_start, abs_end, text))

        if not all_cues:
            return [], detected_language or "unknown"

        # Sort by start time (chunks themselves are sorted, but defensive).
        all_cues.sort(key=lambda c: c[0])

        # Assemble SRTSegments with monotonic 1-based indices.
        srt_segments = [
            SRTSegment(index=i, start=start, end=end, text=text)
            for i, (start, end, text) in enumerate(all_cues, start=1)
        ]
        return srt_segments, detected_language or "unknown"


# ──────────────────────────────────────────────────────────────────────────
# Module-private helpers
# ──────────────────────────────────────────────────────────────────────────


def _write_chunk_wav(samples, sr: int, out_path: Path) -> None:
    """Write float32 samples in ``[-1, 1]`` to a 16-bit PCM WAV file."""
    import numpy as np
    import wave

    int16 = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(out_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16.tobytes())
