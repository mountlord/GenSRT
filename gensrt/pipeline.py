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
    """Reject translation configs that the chosen engine can't honour.

    Engine policy:
        Only the ``google`` engine supports non-English target languages.
        Marian and NLLB are X→English only.

    Raises:
        ConfigError: When the user wants a non-English target with an
                     engine other than ``google``.

    No-ops when ``config.translate`` is False or when the target is
    English.
    """
    from gensrt.exceptions import ConfigError
    if not config.translate:
        return
    if config.target_language.lower() == "en":
        return
    if config.translation_engine.lower() == "google":
        return
    raise ConfigError(
        f"Target language {config.target_language!r} is only supported by the "
        f"'google' translation engine.  Current engine: "
        f"{config.translation_engine!r}.  Switch the engine to Google, set "
        f"the target language to English, or disable translation."
    )


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

        srt_segments, detected_language = _run_asr(
            wav_path=wav_path,
            config=config,
        )

        logger.info("Using language: %s", detected_language)

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

        from gensrt.srt.builder import build_srt, write_srt, write_vtt
        subtitles = build_srt(srt_segments, max_duration_s=config.max_subtitle_duration_s, min_duration_s=config.min_subtitle_duration_s)
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
) -> tuple[list[SRTSegment], str]:
    """Dispatch the ASR stage through the engine factory.

    Returns ``(segments, detected_language)`` — the engine produces
    :class:`SRTSegment` objects directly, so no further conversion is
    needed before translation.
    """
    from gensrt.asr import get_engine_for_model

    engine = get_engine_for_model(config.model)
    logger.info("ASR engine: %s (model=%s)", engine.name, config.model)
    return engine.transcribe(wav_path, config)


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
    engine = get_engine(config.translation_engine)

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
        SRTSegment(index=seg.index, start=seg.start, end=seg.end, text=tr)
        for seg, tr in zip(segments, translated_texts)
    ]
