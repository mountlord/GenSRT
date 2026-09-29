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
from gensrt.exceptions import ConfigError
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
    translating engines (NLLB and MADLAD) handle any mapped target
    language.
    """
    if not config.translate:
        return
    if config.translation_engine.lower() == "none":
        return

    # Resolving the engine validates the key; get_engine raises ConfigError
    # with an actionable message for removed or unknown values.
    from gensrt.translation.factory import get_engine

    get_engine(config.translation_engine, config)


def validate_model_language(config: TranscriptionConfig) -> None:
    """Refuse a source language a registered monolingual model cannot produce.

    A Japanese-only fine-tune handed Korean audio does not fail: it emits
    plausible Japanese, the translator renders that into fluent English, and
    four broadcasts' worth of subtitles are fiction before anyone notices.
    The registry knows the model's language; when the user names a
    different one, stop before the audio extract.  ``auto`` is fine — the
    engine substitutes the registered language itself.
    """
    from gensrt.asr.factory import get_known_language_for_model

    known = get_known_language_for_model(config.model)
    requested = (config.source_language or "auto").strip().lower()
    if known is None or requested in ("", "auto"):
        return
    if requested.split("-")[0] != known:
        raise ConfigError(
            f"Model {config.model!r} is a {known!r}-only fine-tune and cannot "
            f"transcribe source language {requested!r}: it would emit {known!r} "
            f"text for whatever it hears, and the translator would translate "
            f"that. Use a multilingual model (e.g. large-v3-turbo) for "
            f"{requested!r}, or set the source language to {known!r}."
        )


def validate_chunking_config(config: TranscriptionConfig) -> None:
    """Reject chunk-size settings that cannot work, before any audio work.

    ``min_chunk_s`` must be positive and strictly below ``max_chunk_s``
    (the cut-placement window is ``[start+min, start+max]`` — an empty
    window would make subdivision impossible), and ``max_chunk_s`` may not
    exceed Whisper's 30-second receptive window, beyond which the tail of
    every chunk would be silently ignored by the model.
    """
    mn, mx = config.min_chunk_s, config.max_chunk_s
    if not (mn > 0):
        raise ConfigError(f"min_chunk_s must be > 0 (got {mn}).")
    if not (mx > mn):
        raise ConfigError(
            f"max_chunk_s ({mx}) must be greater than min_chunk_s ({mn})."
        )
    if mx > 30.0:
        raise ConfigError(
            f"max_chunk_s ({mx}) exceeds Whisper's 30s window; audio past "
            f"30s in a chunk would be silently ignored."
        )


def _offline_engine_needed(config: TranscriptionConfig) -> str | None:
    """Which offline translation engine this run could call, if any.

    Returns ``"nllb"``, ``"madlad"`` or ``None``.
    """
    if not config.translate:
        return None
    engine = (config.translation_engine or "").lower()
    if engine in ("nllb", "madlad"):
        return engine
    return None


def _needs_nllb(config: TranscriptionConfig) -> bool:
    """Whether this run could call the NLLB engine specifically.

    Kept as a thin wrapper over :func:`_offline_engine_needed` because it
    is part of this module's tested surface.
    """
    return _offline_engine_needed(config) == "nllb"


def ensure_translation_model(
    config: TranscriptionConfig, *, status=None
) -> TranscriptionConfig:
    """Fetch the NLLB model up front if this run might need it.

    Runs before any transcription work, so the one-time ~650 MB download
    happens in the same run — the same interactive moment — as a first-time
    Whisper model download, and never lazily in the middle of an unattended
    job (where a stalled fetch or a flaky connection would fail the file
    *after* transcription had already spent its time).

    An unavailable model raises: the user asked for offline translation by
    name, and silently doing something else would be worse than stopping.

    Returns:
        *config*, unchanged (kept as a return value for the call sites).
    """
    which = _offline_engine_needed(config)
    if which is None:
        return config

    if which == "madlad":
        from gensrt.translation.madlad_ct2 import ensure_model
        model_ref = config.madlad_model
    else:
        from gensrt.translation.nllb_ct2 import ensure_model
        model_ref = config.translation_model

    ensure_model(model_ref, status=status)
    return config


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
    validate_chunking_config(config)
    validate_model_language(config)

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

        # ── Phase 2b: Post-ASR heuristics ────────────────────────────────
        # After the dump (so it records the model's raw output) and before
        # translation (so a run of 221 identical "ああ" cues is not sent
        # through the translator 221 times).
        from gensrt.heuristics import load_heuristics, run_heuristics, strip_subjects

        heuristics = load_heuristics()
        raw_segments = list(srt_segments)          # for the run report
        srt_segments, heuristics_stats = run_heuristics(
            srt_segments, heuristics, detected_language
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

        source_texts = [seg.text for seg in srt_segments]   # positional pairing
        _TRANSLATION_FAILURE.clear()
        srt_segments = _maybe_translate(
            segments=srt_segments,
            detected_language=detected_language,
            config=config,
            should_translate=should_translate,
        )

        # ── Phase 3b: post-translation heuristics ────────────────────────
        # Pro-drop source + English target: strip the subject the translator
        # invented, when the source line names none.  Needs the source text
        # beside the translation, so it lives here and not in Phase 2b.
        translation_error = _TRANSLATION_FAILURE[0] if _TRANSLATION_FAILURE else None
        if translation_error:
            status("TRANSLATION FAILED — subtitles written in the source language.")
        if should_translate and not translation_error:
            subject_rules = heuristics.subject_for(detected_language, config.target_language)
            srt_segments, n_subj, subj_examples = strip_subjects(
                source_texts, srt_segments, subject_rules
            )
            heuristics_stats.subject_stripped = n_subj
            heuristics_stats.subject_examples = subj_examples
            if n_subj:
                logger.info(
                    "Heuristics [subject/%s]: stripped the leading subject pronoun "
                    "from %d line(s) whose source names no subject",
                    subject_rules.language, n_subj,
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

    # The run report pairs the model's raw short strings with their
    # translations so a user who cannot read the source script can still
    # decide what to drop.  Strings the rules dropped never reached the
    # translator, so the report fetches those few itself when an engine is
    # configured.  Never allowed to fail the run.
    report = None
    try:
        from gensrt.heuristics import build_report

        translate_fn = None
        if should_translate:
            def translate_fn(texts, _det=detected_language):
                from gensrt.translation.factory import get_shared_engine

                # Same instance the translation used: no second model load.
                engine = get_shared_engine(config.translation_engine, config)
                return engine.translate_batch(list(texts), _det, config.target_language)

        report = build_report(
            raw_segments, srt_segments, heuristics.for_language(detected_language),
            translate=translate_fn, source_path=heuristics.source_path,
            stats=heuristics_stats,
        )
        if report is not None and getattr(config, "heuristics_report_dir", ""):
            txt, _js = report.write(Path(config.heuristics_report_dir), input_path.stem)
            logger.info("Heuristics report written: %s", txt)
    except Exception as exc:
        logger.debug("Heuristics report not built: %s", exc)

    return TranscriptionResult(
        input_path=input_path,
        output_path=output_path,
        detected_language=detected_language,
        segments=srt_segments,
        config=config,
        elapsed_s=elapsed,
        heuristics_report=report,
        translation_error=translation_error,
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


# Set by _maybe_translate when a file goes out untranslated; read and cleared
# by run_pipeline so the failure lands on the TranscriptionResult.
_TRANSLATION_FAILURE: list[str] = []


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

    from gensrt.translation.factory import get_shared_engine
    engine = get_shared_engine(config.translation_engine, config)

    texts = [seg.text for seg in segments]
    try:
        translated_texts = engine.translate_batch(
            texts, detected_language, config.target_language
        )
    except Exception as exc:
        # The engines already walked their own fallbacks (smaller batches,
        # CPU).  Reaching here means the file is going out UNTRANSLATED, and
        # that must not hide behind a warning: log at ERROR and record it on
        # the result so the CLI summary and the GUI can say so.
        logger.error(
            "TRANSLATION FAILED for this file (%s) — subtitles are being "
            "written in the SOURCE language.", exc,
        )
        _TRANSLATION_FAILURE.append(str(exc))
        translated_texts = texts

    return [
        # replace() rather than a fresh SRTSegment: translation changes only
        # the text, and rebuilding by hand silently dropped every diagnostic
        # field the engines had just populated.
        replace(seg, text=tr)
        for seg, tr in zip(segments, translated_texts)
    ]
