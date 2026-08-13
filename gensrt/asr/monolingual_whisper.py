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
from gensrt.asr._chunk_debug import (
    ChunkRecord,
    ChunkRecorder,
    NullChunkRecorder,
    log_timing_summary,
)
from dataclasses import replace

from gensrt.asr._model_loader import is_environment_error, load_whisper_model
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
        *,
        status=None,
    ) -> tuple[list[SRTSegment], str]:
        # Fail fast with a clear message if the dependency is absent, rather
        # than after the audio load and chunk planning have already run.
        _whisper_model_class(wav_path)

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

        srt_segments, detected_language = self._transcribe_with_cpu_retry(
            audio, sr, chunks, source_lang, wav_path, config, status
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

    def _transcribe_with_cpu_retry(
        self,
        audio,
        sr: int,
        chunks: list[dict],
        source_lang: str | None,
        wav_path: Path,
        config,
        status=None,
    ) -> tuple[list[SRTSegment], str]:
        """Load the model and run inference, falling back to CPU if the GPU fails.

        This is a *second* fallback layer, distinct from the one in
        :func:`gensrt.asr._model_loader.load_whisper_model`, and it exists
        because the two failures happen at different times.

        CTranslate2 resolves cuBLAS and cuDNN lazily — at the first inference
        call, not when the model is constructed.  So a machine with a working
        NVIDIA driver but missing CUDA runtime libraries loads the model
        happily and then fails on every chunk.  The load-time fallback cannot
        see that; only this can.

        Retrying the whole file on CPU is the right trade: the alternative is
        failing a job that may be one of fifty in a batch.  The warning is
        loud because on a packaged build this nearly always means the CUDA
        libraries were not bundled properly, which is a bug to fix rather than
        a condition to accept.
        """
        WhisperModel = _whisper_model_class(wav_path)
        model = load_whisper_model(wav_path, config, WhisperModel, status=status)

        try:
            return self._transcribe_chunks(
                model, audio, sr, chunks, source_lang, wav_path,
                config=config,
                status=status,
                debug_chunk_dir=getattr(config, "debug_chunk_dir", ""),
            )
        except TranscriptionError:
            if getattr(config, "device", "cuda") == "cpu":
                raise

            logger.warning(
                "[%s] GPU inference failed — retrying the whole file on CPU. "
                "This will be substantially slower. If you expected GPU "
                "acceleration, the CUDA runtime libraries are missing or "
                "unloadable; see the error above.",
                self.name,
            )
            if callable(status):
                status("GPU inference failed — retrying on CPU (much slower)…")

            cpu_config = replace(config, device="cpu", compute_type="int8")
            model = load_whisper_model(
                wav_path, cpu_config, WhisperModel, status=status
            )
            return self._transcribe_chunks(
                model, audio, sr, chunks, source_lang, wav_path,
                config=cpu_config,
                status=status,
                debug_chunk_dir=getattr(cpu_config, "debug_chunk_dir", ""),
            )

    def _transcribe_chunks(
        self,
        model,
        audio,
        sr: int,
        chunks: list[dict],
        source_lang: str | None,
        wav_path: Path,
        config=None,
        status=None,
        debug_chunk_dir: str = "",
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
        import tempfile
        import time

        all_cues: list[tuple[float, float, str]] = []
        chunk_index_for_logging = 0

        # Per-chunk decode telemetry.  Timing is collected unconditionally —
        # it costs one perf_counter call per chunk and drives the outlier
        # summary that runs on every job.  Audio export only happens when a
        # debug directory was configured.
        records: list[ChunkRecord] = []
        if debug_chunk_dir:
            # Use the directory verbatim. It used to be derived from
            # wav_path.stem, but wav_path is the *temp* extracted audio
            # (gensrt_audio_73g7xhsu.wav), so every run produced a fresh
            # randomly-named folder with no way to tell which source video it
            # came from. The caller knows the source name; the engine does not,
            # so the caller now resolves the path.
            recorder = ChunkRecorder(Path(debug_chunk_dir))
            logger.info(
                "[%s] Chunk diagnostics enabled → %s",
                self.name, Path(debug_chunk_dir),
            )
        else:
            recorder = NullChunkRecorder()

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

        # Set when a chunk fails for an environment reason (missing CUDA
        # library, exhausted GPU).  Such a failure will repeat identically on
        # every remaining chunk, so we abandon the pass immediately rather
        # than skipping 43 chunks in a row and returning an empty subtitle
        # file — which is what GenSRT did before this guard existed.
        env_failure: Exception | None = None

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

                rec = ChunkRecord(chunk_index_for_logging, c_start, c_end)
                records.append(rec)
                # Time the decode only — not the WAV write, which is fixed
                # overhead and would blur the signal we are looking for.
                t0 = time.perf_counter()

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
                    rec.wall_s = time.perf_counter() - t0
                    rec.observe_segments(chunk_segments)
                    recorder.record(rec, chunk_wav)
                except Exception as exc:
                    # One bad chunk shouldn't abort the whole transcription.
                    # Log it and skip — the user sees a small gap rather
                    # than a complete failure.
                    rec.wall_s = time.perf_counter() - t0
                    rec.failed = True

                    if is_environment_error(exc):
                        # Not a bad chunk — a broken environment.  Stop now;
                        # the caller decides whether to retry on CPU or fail.
                        env_failure = exc
                        logger.error(
                            "[%s] Chunk %d failed for an environment reason: %s",
                            self.name, chunk_index_for_logging, exc,
                        )
                        break

                    logger.warning(
                        "[%s] Chunk %d transcribe failed (%s) — skipping. "
                        "%.2fs of audio at %.1fs will have no subtitles.",
                        self.name, chunk_index_for_logging, exc,
                        c_end - c_start, c_start,
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

                # Offset chunk-local timestamps by chunk start, and carry the
                # decoder's own metrics plus this segment's position in the
                # chunk forward.  Both are discarded at this boundary in a
                # plain faster-whisper pipeline; keeping them is what makes
                # confidence-based and chunk-tail-based analysis possible
                # downstream (see SRTSegment).
                kept = [seg for seg in chunk_segments if seg.text.strip()]
                for position, seg in enumerate(kept, start=1):
                    abs_start = c_start + float(seg.start)
                    abs_end = min(c_start + float(seg.end), c_end)
                    all_cues.append((
                        abs_start,
                        abs_end,
                        seg.text.strip(),
                        _segment_diagnostics(
                            seg,
                            chunk_index=chunk_index_for_logging,
                            chunk_position=position,
                            chunk_n_segments=len(kept),
                        ),
                    ))

        if env_failure is not None:
            raise _environment_failure_error(wav_path, env_failure, config)

        log_timing_summary(records, self.name)
        recorder.finalize(wav_path.name)

        n_failed = sum(1 for r in records if r.failed)
        if n_failed:
            logger.warning(
                "[%s] %d of %d chunk(s) failed to decode — that audio is absent "
                "from the output.",
                self.name, n_failed, len(records),
            )
            # Skipping the odd bad chunk is the intended resilience.  Losing
            # most of the file is not: a subtitle file covering a third of the
            # audio looks finished and is worse than an honest failure.
            if n_failed * 2 > len(records):
                raise TranscriptionError(
                    str(wav_path),
                    f"{n_failed} of {len(records)} chunks failed to decode. "
                    f"Refusing to write a subtitle file that would be mostly "
                    f"empty. Run with --log-level DEBUG for the per-chunk errors.",
                )

        if not all_cues:
            return [], detected_language or "unknown"

        # Sort by start time (chunks themselves are sorted, but defensive).
        all_cues.sort(key=lambda c: c[0])

        # Assemble SRTSegments with monotonic 1-based indices.
        srt_segments = [
            SRTSegment(index=i, start=start, end=end, text=text, **diag)
            for i, (start, end, text, diag) in enumerate(all_cues, start=1)
        ]
        return srt_segments, detected_language or "unknown"


# ──────────────────────────────────────────────────────────────────────────
# Module-private helpers
# ──────────────────────────────────────────────────────────────────────────


def _segment_diagnostics(
    seg,
    *,
    chunk_index: int | None = None,
    chunk_position: int | None = None,
    chunk_n_segments: int | None = None,
) -> dict:
    """Extract decoder metrics from a faster-whisper Segment.

    Every field is read with getattr and a None default.  These attributes
    exist in faster-whisper 1.x but are not covered by any stability
    guarantee, and ``temperature`` is Optional even when present — a future
    release dropping one should leave the diagnostics blank, not fail a
    transcription.
    """
    def _f(name):
        value = getattr(seg, name, None)
        return None if value is None else float(value)

    return {
        "avg_logprob": _f("avg_logprob"),
        "compression_ratio": _f("compression_ratio"),
        "no_speech_prob": _f("no_speech_prob"),
        "temperature": _f("temperature"),
        "chunk_index": chunk_index,
        "chunk_position": chunk_position,
        "chunk_n_segments": chunk_n_segments,
    }


def _whisper_model_class(wav_path):
    """Import and return ``faster_whisper.WhisperModel``.

    Kept as a function so faster-whisper stays out of the module import graph
    (it pulls in CTranslate2, ONNX Runtime and tokenizers, which matters for
    CLI startup time) and so tests can substitute it.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            str(wav_path),
            "faster-whisper is not installed. Run: pip install faster-whisper",
        ) from exc
    return WhisperModel


def _environment_failure_error(wav_path, exc, config) -> TranscriptionError:
    """Turn a CUDA/library failure into an error a user can act on.

    A bare "Library cublas64_12.dll is not found or cannot be loaded" tells
    someone who did not build this nothing at all.  It is worth spending a few
    lines to say what is missing and what to do about it, because this
    particular failure has an easy fix and an easy workaround.
    """
    device = getattr(config, "device", "cuda") if config else "cuda"
    text = str(exc)

    if device == "cpu":
        return TranscriptionError(
            str(wav_path),
            f"Inference failed on CPU: {text}",
        )

    from gensrt._cuda_dlls import find_cuda_libraries, registered_directories

    missing = [name for name, path in find_cuda_libraries().items() if path is None]
    dirs = registered_directories()

    detail = f"GPU inference failed: {text}\n"
    if missing:
        detail += f"  Missing CUDA librar(ies): {', '.join(missing)}\n"
    if dirs:
        detail += f"  Searched {len(dirs)} registered CUDA director(ies).\n"
    else:
        detail += "  No CUDA library directories were registered.\n"
    detail += (
        "  This usually means the CUDA runtime libraries are not installed "
        "alongside GenSRT.\n"
        "  Workaround: set \"device\": \"cpu\" in gensrt-config.json (slower, "
        "but works anywhere).\n"
        "  From source: pip install -r requirements-cuda.txt"
    )
    return TranscriptionError(str(wav_path), detail)


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
