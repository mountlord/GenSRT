"""Pipeline orchestrator — ties all stages together.

:func:`run_pipeline` is the single entry point used by both the CLI
(headless) and the GUI (via :mod:`gensrt.operations`).  It processes one
media file through the complete flow::

    audio extract → ASR engine → translate → SRT write

The ASR stage is dispatched through :func:`gensrt.asr.get_engine_for_model`,
which selects either the multilingual or monolingual engine based on the
configured model.  See :mod:`gensrt.asr.factory` for routing rules.

Progress and status are surfaced via optional callbacks so the same
function works in tqdm-driven CLI mode and polling-driven GUI mode.
"""

from __future__ import annotations

import logging
from dataclasses import replace
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


def validate_translation_config(config: TranscriptionConfig) -> None:
    """Reject translation configs that cannot be honoured.

    Retained as the single place engine/fallback/target compatibility is
    checked, and called before any expensive work runs — an invalid key
    should surface here, not after the audio extract and model load.  Both
    translating engines (Google GTX and NLLB) handle any mapped target
    language.
    """
    if not config.translate:
        return
    if config.translation_engine.lower() == "none":
        return

    # Resolving the engine validates the engine key AND the fallback key
    # (the factory wires translation_fallback into the Google engine);
    # get_engine raises ConfigError with an actionable message for removed
    # or unknown values.
    from gensrt.translation.factory import get_engine

    get_engine(config.translation_engine, config)


def _needs_nllb(config: TranscriptionConfig) -> bool:
    """Whether this run could call the NLLB engine.

    True when NLLB is the primary engine, or when Google is primary with
    NLLB as its batch-failure fallback.
    """
    if not config.translate:
        return False
    engine = config.translation_engine.lower()
    if engine == "nllb":
        return True
    return (
        engine == "google"
        and (config.translation_fallback or "").lower() == "nllb"
    )


def ensure_translation_model(
    config: TranscriptionConfig, *, status=None
) -> TranscriptionConfig:
    """Fetch the NLLB model up front if this run might need it.

    Runs before any transcription work, so the one-time ~650 MB download
    happens in the same run — the same interactive moment — as a first-time
    Whisper model download, and never lazily in the middle of an unattended
    job (where a stalled fetch or a flaky connection would fail the file
    *after* transcription had already spent its time).

    Degradation is deliberately asymmetric:

    * NLLB as the *primary* engine and unavailable → raise.  The user asked
      for offline translation; silently doing something else would be worse
      than stopping.
    * NLLB as the *fallback* and unavailable → warn once and continue with
      ``translation_fallback="none"``.  The run can still succeed entirely
      via Google; the fallback quality degrades to keeping originals.

    Returns:
        *config*, possibly with the fallback downgraded (the dataclass is
        frozen, so degradation produces a new instance).
    """
    if not _needs_nllb(config):
        return config

    from gensrt.translation.nllb_ct2 import ensure_model

    try:
        ensure_model(config.translation_model, status=status)
        return config
    except Exception as exc:
        if config.translation_engine.lower() == "nllb":
            raise
        logger.warning(
            "NLLB fallback model unavailable (%s) — failed Google batches "
            "will keep their source text for this run.", exc,
        )
        return replace(config, translation_fallback="none")


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
        TranscriptionError:   If the ASR engine fails.
        TranslationError:     If the translation engine fails (non-fatal if
                              engine is ``none``).
        OutputError:          If the ``.srt`` file cannot be written.
    """
    if progress is None:
        progress = _noop_progress
    if status is None:
        status = _noop_status

    # Reject engine + target_language combinations the chosen engine can't
    # honour, before any expensive work (audio extract / model load) runs.
    validate_translation_config(config)

    # Fetch the offline translation model up front (one-time), alongside —
    # not instead of — whatever Whisper model download the run may trigger.
    config = ensure_translation_model(config, status=status)

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

        # ── Phase 2: Transcription (engine selected by model name) ────────
        if config.vad_enabled:
            status("Transcribing with VAD…")
        else:
            status("Transcribing…")
        progress(1, PIPELINE_PHASES)

        # Resolve the chunk-diagnostics directory here, where the source
        # filename is known. The engine only sees the temp extracted audio.
        asr_config = config
        if config.debug_chunk_dir:
            asr_config = replace(
                config,
                debug_chunk_dir=str(Path(config.debug_chunk_dir) / input_path.stem),
            )

        srt_segments, detected_language = _run_asr(
            wav_path=wav_path,
            config=asr_config,
            status=status,
        )

        logger.info("Using language: %s", detected_language)

        # Diagnostics dump, deliberately placed here: after ASR so the
        # decoder metrics are present, before translation so the text is the
        # model's own, and before build_srt so the timings are the model's own
        # too.  Any later and two of those three are gone.
        if config.dump_segments_dir:
            from gensrt.segment_dump import write_segment_dump

            write_segment_dump(
                srt_segments,
                Path(config.dump_segments_dir) / f"{input_path.stem}.segments.csv",
            )

        # ── Phase 3: Translation ──────────────────────────────────────────
        # Normalize "english" → "en" so faster-whisper's occasional name-form
        # output compares correctly to ISO codes in the target.
        det_norm = "en" if detected_language.lower() in ("english", "en") else detected_language.lower()
        tgt_norm = config.target_language.lower()
        should_translate = (
            config.translate
            and config.translation_engine != "none"
            and det_norm != tgt_norm
        )

        if should_translate:
            status(
                f"Translating ({detected_language} → {config.target_language}) "
                f"via {config.translation_engine}…"
            )
        progress(2, PIPELINE_PHASES)

        srt_segments = _maybe_translate(
            segments=srt_segments,
            detected_language=detected_language,
            config=config,
            should_translate=should_translate,
        )

        # ── Phase 4: Write SRT (+ VTT companion) ──────────────────────────
        status("Writing SRT…")
        progress(3, PIPELINE_PHASES)

        from gensrt.srt.builder import (
            build_srt,
            summarize_segment_durations,
            write_srt,
            write_vtt,
        )

        # Log the model's OWN duration distribution before any capping,
        # flooring, or overlap clamping touches it.  Once build_srt has run,
        # the true sub-floor durations are gone — so if it is not recorded
        # here it cannot be recovered from the output.  Cheap, and it makes
        # every run self-documenting for investigation purposes.
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Raw ASR cue durations (pre-post-processing): %s",
                summarize_segment_durations(srt_segments),
            )

        subtitles = build_srt(
            srt_segments,
            max_duration_s=config.max_subtitle_duration_s,
            min_duration_s=config.min_subtitle_duration_s,
            max_line_chars=config.max_line_chars,
            max_lines=config.max_lines,
        )
        write_srt(subtitles, output_path)

        # WebVTT companion — same cues, lands next to the SRT (movie.srt
        # → movie.vtt, movie.ml.srt → movie.ml.vtt).  Non-fatal on failure:
        # the SRT is what the user asked for, the VTT is a bonus that
        # makes HTML5 / Jellyfin / browser playback work without a polyfill.
        try:
            write_vtt(subtitles, output_path.with_suffix(".vtt"))
        except Exception as exc:
            logger.warning("VTT companion write failed (%s) — SRT saved OK.", exc)

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


def _run_asr(
    wav_path: Path,
    config: TranscriptionConfig,
    status: StatusCallback | None = None,
) -> tuple[list[SRTSegment], str]:
    """Dispatch the ASR stage through the engine factory.

    Returns ``(segments, detected_language)`` — the engine produces
    :class:`SRTSegment` objects directly, so no further conversion is
    needed before translation.

    *status* is forwarded to the engine so it can surface mid-run events the
    user must see — chiefly a GPU-to-CPU fallback at model load.
    """
    from gensrt.asr import get_engine_for_model
    from gensrt.asr.factory import get_known_language_for_model

    engine = get_engine_for_model(config.model, getattr(config, "asr_engine", "auto"))
    logger.info("ASR engine: %s (model=%s)", engine.name, config.model)

    # Chunking a multilingual model with automatic language detection means
    # each chunk is detected independently, so the language can flip part-way
    # through a file on ambiguous audio. The registered monolingual models
    # avoid this by using their known training language; a general model has
    # no such fallback, so say so rather than let it surprise someone.
    if (
        engine.name == "MonolingualWhisperEngine"
        and config.source_language in ("auto", "", None)
        and get_known_language_for_model(config.model) is None
    ):
        logger.warning(
            "Chunked inference with source_language='auto': each chunk is "
            "language-detected on its own, so the language can change part-way "
            "through the file. Set the source language explicitly (e.g. "
            "--source-language ja) for consistent results."
        )
    return engine.transcribe(wav_path, config, status=status)


def _maybe_translate(
    segments: list[SRTSegment],
    detected_language: str,
    config: TranscriptionConfig,
    should_translate: bool,
) -> list[SRTSegment]:
    """Translate *segments* via the configured engine, if appropriate.

    When ``should_translate`` is False, returns *segments* unchanged.
    Translation failures are logged at WARNING and fall back to the
    untranslated source text — one failed segment never aborts the
    whole batch.

    Args:
        segments:           Source-language segments from the ASR engine.
        detected_language:  ISO code detected by the engine.
        config:             Translation engine + target language come
                            from here.
        should_translate:   Pipeline-level gate from
                            :func:`run_pipeline`.
    """
    if not should_translate:
        return segments

    from gensrt.translation.factory import get_engine
    engine = get_engine(config.translation_engine, config)

    texts = [seg.text for seg in segments]
    try:
        translated_texts = engine.translate_batch(
            texts, detected_language, config.target_language
        )
    except Exception as exc:
        logger.warning(
            "translate_batch failed (%s) — keeping all originals.", exc
        )
        translated_texts = texts

    return [
        # replace() rather than a fresh SRTSegment: translation changes only
        # the text, and rebuilding by hand silently dropped every diagnostic
        # field the engines had just populated.
        replace(seg, text=tr)
        for seg, tr in zip(segments, translated_texts)
    ]
