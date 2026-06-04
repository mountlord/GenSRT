"""Pipeline orchestrator — ties all stages together.

:func:`run_pipeline` is the single entry point used by both the CLI
(headless) and the GUI (via :mod:`gensrt.operations`).  It processes one
media file through the complete flow::

    audio extract → VAD → whisper → translate → SRT write

Progress and status are surfaced via optional callbacks so the same
function works in tqdm-driven CLI mode and polling-driven GUI mode.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from gensrt.constants import PIPELINE_PHASES
from gensrt.models import SRTSegment, TranscriptionConfig, TranscriptionResult

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]   # (current, total)
StatusCallback = Callable[[str], None]           # human-readable phase message


def _noop_status(_: str) -> None:
    pass


def _noop_progress(_c: int, _t: int) -> None:
    pass


def run_pipeline(
    input_path: Path,
    output_path: Path,
    config: TranscriptionConfig,
    *,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> TranscriptionResult:
    """Run the full transcription pipeline on a single media file.

    Args:
        input_path:   Path to the input media file (any FFmpeg-supported format).
        output_path:  Path where the ``.srt`` file will be written.
        config:       Fully resolved :class:`TranscriptionConfig`.
        progress:     Optional ``(current, total)`` callback.
        status:       Optional human-readable phase message callback.

    Returns:
        A :class:`TranscriptionResult` describing the completed job.

    Raises:
        AudioExtractionError: If FFmpeg cannot extract audio.
        TranscriptionError:   If Whisper fails.
        TranslationError:     If the translation engine fails (non-fatal if
                              engine is ``none``).
        OutputError:          If the ``.srt`` file cannot be written.
    """
    if progress is None:
        progress = _noop_progress
    if status is None:
        status = _noop_status

    input_path = Path(input_path).resolve()
    output_path = Path(output_path)

    logger.info("=" * 60)
    logger.info("Processing: %s", input_path.name)
    logger.info("=" * 60)

    t0 = time.perf_counter()
    wav_path: Path | None = None

    try:
        # ── Phase 1: Audio extraction ─────────────────────────────────────
        status("Extracting audio…")
        progress(0, PIPELINE_PHASES)

        from gensrt.audio.extractor import extract_audio
        wav_path = extract_audio(input_path)

        # ── Phase 2: Transcription (VAD runs inside faster-whisper) ───────
        if config.vad_enabled:
            status("Transcribing with VAD…")
        else:
            status("Transcribing…")
        progress(1, PIPELINE_PHASES)

        raw_segments, detected_language = _run_whisper(
            wav_path=wav_path,
            config=config,
        )

        logger.info("Detected language: %s", detected_language)

        # ── Phase 3: Translation ──────────────────────────────────────────
        should_translate = (
            config.translate
            and config.translation_engine != "none"
            and detected_language not in ("en", "english")
        )

        if should_translate:
            status(
                f"Translating ({detected_language} → en) "
                f"via {config.translation_engine}…"
            )
        progress(2, PIPELINE_PHASES)

        srt_segments = _build_segments(
            raw_segments=raw_segments,
            detected_language=detected_language,
            config=config,
            should_translate=should_translate,
        )

        # ── Phase 4: Write SRT ────────────────────────────────────────────
        status("Writing SRT…")
        progress(3, PIPELINE_PHASES)

        from gensrt.srt.builder import build_srt, write_srt
        subtitles = build_srt(srt_segments, max_duration_s=config.max_subtitle_duration_s, min_duration_s=config.min_subtitle_duration_s)
        write_srt(subtitles, output_path)

        # Done — bar to 100%
        progress(PIPELINE_PHASES, PIPELINE_PHASES)

    finally:
        # Always clean up the temp WAV
        if wav_path is not None:
            wav_path.unlink(missing_ok=True)
            logger.debug("Temp WAV removed: %s", wav_path.name)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Done: %s  →  %s  (%.1fs, %d segments, lang=%s)",
        input_path.name,
        output_path.name,
        elapsed,
        len(srt_segments),
        detected_language,
    )
    status(f"Done ({elapsed:.1f}s) — {len(srt_segments)} subtitles written.")

    return TranscriptionResult(
        input_path=input_path,
        output_path=output_path,
        detected_language=detected_language,
        segments=srt_segments,
        config=config,
        elapsed_s=elapsed,
    )


def _run_whisper(
    wav_path: Path,
    config: TranscriptionConfig,
) -> tuple[list, str]:
    """Run faster-whisper and return (raw_segments_list, detected_language).

    VAD is handled entirely inside faster-whisper on the same device as
    the model — no separate Silero pass needed.
    """
    from gensrt.exceptions import TranscriptionError

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriptionError(
            str(wav_path),
            "faster-whisper is not installed. Run: pip install faster-whisper",
        ) from exc

    device = config.device
    compute_type = config.compute_type

    logger.info(
        "Loading Whisper model: %s  (device=%s, compute=%s)",
        config.model, device, compute_type,
    )

    compute_fallbacks = [compute_type]
    if compute_type == "float16":
        compute_fallbacks += ["int8_float16", "int8"]
    elif compute_type == "int8_float16":
        compute_fallbacks += ["int8"]

    model = None
    last_exc: Exception | None = None
    for ct in compute_fallbacks:
        try:
            model = WhisperModel(config.model, device=device, compute_type=ct)
            if ct != compute_type:
                logger.warning(
                    "compute_type=%r unsupported — fell back to %r.", compute_type, ct
                )
            break
        except Exception as exc:
            last_exc = exc
            logger.debug("WhisperModel load failed with compute_type=%r: %s", ct, exc)

    if model is None:
        raise TranscriptionError(
            str(wav_path), f"Failed to load Whisper model: {last_exc}"
        ) from last_exc

    source_lang = None if config.source_language == "auto" else config.source_language

    transcribe_kwargs: dict = dict(
        language=source_lang,
        word_timestamps=True,
        beam_size=5,
    )

    if config.vad_enabled:
        transcribe_kwargs["vad_filter"] = True
        transcribe_kwargs["vad_parameters"] = {
            "threshold": config.vad_threshold,
            "min_speech_duration_ms": config.vad_min_speech_ms,
            "min_silence_duration_ms": config.vad_min_silence_ms,
            "speech_pad_ms": config.vad_speech_pad_ms,
        }
        logger.info(
            "VAD enabled (threshold=%.2f, min_speech=%dms, min_silence=%dms, speech_pad=%dms)",
            config.vad_threshold, config.vad_min_speech_ms, config.vad_min_silence_ms,
            config.vad_speech_pad_ms,
        )
    else:
        logger.info("VAD disabled — full audio passed to Whisper.")

    try:
        segments_gen, info = model.transcribe(str(wav_path), **transcribe_kwargs)
        # Consume generator to a list (so temp WAV can be safely deleted)
        raw_segments = list(segments_gen)
    except Exception as exc:
        raise TranscriptionError(str(wav_path), str(exc)) from exc

    detected_language = info.language or "unknown"
    logger.info(
        "Whisper: %d segments, lang=%s (prob=%.2f)",
        len(raw_segments),
        detected_language,
        info.language_probability,
    )
    return raw_segments, detected_language


def _build_segments(
    raw_segments: list,
    detected_language: str,
    config: TranscriptionConfig,
    should_translate: bool,
) -> list[SRTSegment]:
    """Convert raw Whisper segments to translated :class:`SRTSegment` objects.

    Args:
        raw_segments:     List of ``faster_whisper.transcribe.Segment`` objects.
        detected_language: Detected language code.
        config:           Transcription config (translation engine key).
        should_translate: Whether to run translation.

    Returns:
        List of :class:`SRTSegment` objects.
    """
    from gensrt.srt.builder import segments_from_whisper

    srt_segments = segments_from_whisper(raw_segments)

    if not should_translate:
        return srt_segments

    from gensrt.translation.factory import get_engine
    engine = get_engine(config.translation_engine)

    texts = [seg.text for seg in srt_segments]
    try:
        translated_texts = engine.translate_batch(texts, detected_language)
    except Exception as exc:
        logger.warning(
            "translate_batch failed (%s) — keeping all originals.", exc
        )
        translated_texts = texts

    return [
        SRTSegment(index=seg.index, start=seg.start, end=seg.end, text=en)
        for seg, en in zip(srt_segments, translated_texts)
    ]
