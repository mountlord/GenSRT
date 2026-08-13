"""SRT file construction.

Converts faster-whisper segment output into standard ``.srt`` subtitle files
using the ``srt`` library.

Usage::

    from gensrt.srt.builder import build_srt, write_srt
    subtitles = build_srt(segments)
    write_srt(subtitles, Path("output.srt"))

Timing policy (applied in this order by :func:`build_srt`):

    1. **Cap**  — ``max_duration_s`` shortens cues that hang on screen.
    2. **Floor** — ``min_duration_s`` lengthens cues that flash past.
    3. **Overlap clamp** — a cue's end is pulled back so it never runs past
       the next cue's start.  This runs *last* and can partially undo the
       floor; that is intentional, since two cues on screen at once is worse
       than one cue that is briefly short.

Text policy:

    :func:`_wrap_text` wraps into ``max_lines`` lines of ``max_line_chars``.
    **It never discards text.**  If the cue does not fit, extra lines are
    emitted and a DEBUG line is logged.  Words are never split mid-word,
    which also means Indic grapheme clusters and CJK runs are never broken.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path

from gensrt.exceptions import OutputError
from gensrt.models import SRTSegment

logger = logging.getLogger(__name__)

# Maximum characters per subtitle line before we wrap.
#
# 42 is the long-standing broadcast convention (BBC/Netflix style guides sit
# between 37 and 42).  GenSRT used 84 through v1.2.1, which is roughly two
# conventional lines crammed into one and is not comfortably readable at
# playback speed.  Both values are overridable per-run via
# TranscriptionConfig.max_line_chars for users who preferred the old shape.
DEFAULT_MAX_LINE_CHARS: int = 42
# Preferred maximum lines per subtitle block.  This is a *target*, not a
# truncation point — see _wrap_text.
DEFAULT_MAX_LINES: int = 2

# Backwards-compatible aliases (some external scripts imported these).
_MAX_LINE_CHARS: int = DEFAULT_MAX_LINE_CHARS
_MAX_LINES: int = DEFAULT_MAX_LINES


def _wrap_text(
    text: str,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> str:
    """Wrap *text* into lines of at most *max_line_chars* characters.

    **No text is ever discarded.**  *max_lines* is a soft target: when the
    content genuinely needs more lines than that, the extra lines are kept
    and a DEBUG message records it.  Losing the tail of a subtitle silently
    is far worse than showing a three-line cue.

    Words are never split.  A single token longer than *max_line_chars*
    (a long compound, a URL, or an entire space-free CJK/Thai run) is placed
    on its own line and allowed to overflow, because character-level
    splitting would break Indic grapheme clusters and CJK word boundaries.

    Args:
        text:           Cue text.
        max_line_chars: Soft character budget per line.  ``<= 0`` disables
                        wrapping entirely.
        max_lines:      Preferred line count.  ``<= 0`` means "no preference".

    Returns:
        The wrapped text with ``\\n`` line separators.  Always contains every
        word from *text*.
    """
    text = text.strip()
    if not text or max_line_chars <= 0:
        return text
    if len(text) <= max_line_chars:
        return text

    words = text.split()
    if not words:
        return text

    lines: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join(current + [word])
        if current and len(candidate) > max_line_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)

    if current:
        lines.append(" ".join(current))

    if max_lines > 0 and len(lines) > max_lines:
        logger.debug(
            "Cue needed %d lines at %d chars (target %d); keeping all text: %.40s…",
            len(lines), max_line_chars, max_lines, text,
        )

    return "\n".join(lines)


def build_srt(
    segments: list[SRTSegment],
    max_duration_s: float = 0.0,
    min_duration_s: float = 0.0,
    max_line_chars: int = DEFAULT_MAX_LINE_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
) -> list:
    """Convert :class:`SRTSegment` objects to ``srt.Subtitle`` objects.

    Args:
        segments:       Ordered list of subtitle segments.
        max_duration_s: If > 0, cap each subtitle's display duration to this
                        many seconds.  Fixes subtitles that hang on screen for
                        minutes when Whisper assigns a very long end timestamp.
        min_duration_s: If > 0, ensure each subtitle displays for at least this
                        many seconds.  Fixes subtitles that disappear too fast.
                        **Set to 0 when measuring the model's own timestamp
                        behaviour** — see the note below.
        max_line_chars: Soft per-line character budget for text wrapping.
        max_lines:      Preferred line count per cue.

    Returns:
        List of ``srt.Subtitle`` objects ready for serialisation.

    Note on measurement:
        ``min_duration_s`` rewrites the *end* timestamp of every cue shorter
        than the floor, collapsing the true sub-floor duration distribution
        to a single value.  Any analysis of model timestamp behaviour must be
        done on the incoming :class:`SRTSegment` list (or with
        ``min_duration_s=0``), never on the composed SRT.  This function logs
        a one-line summary at INFO whenever the floor fires, so the effect is
        visible in the run log rather than silent.
    """
    try:
        import srt
    except ImportError as exc:
        raise OutputError(
            "srt library is not installed. Run: pip install srt"
        ) from exc

    subtitles = []
    n_capped = 0
    n_floored = 0
    raw_durations_below_floor: list[float] = []

    for seg in segments:
        if not seg.text.strip():
            continue

        start = datetime.timedelta(seconds=seg.start)
        end = datetime.timedelta(seconds=seg.end)

        # Guard: end must be strictly after start
        if end <= start:
            end = start + datetime.timedelta(milliseconds=100)

        # Cap maximum display duration
        if max_duration_s > 0:
            max_end = start + datetime.timedelta(seconds=max_duration_s)
            if end > max_end:
                n_capped += 1
                logger.debug(
                    "Capping subtitle %d: %.1fs → %.1fs (was %.1fs)",
                    seg.index,
                    seg.start,
                    seg.start + max_duration_s,
                    (end - start).total_seconds(),
                )
                end = max_end

        # Floor minimum display duration
        if min_duration_s > 0:
            min_end = start + datetime.timedelta(seconds=min_duration_s)
            if end < min_end:
                n_floored += 1
                raw_durations_below_floor.append((end - start).total_seconds())
                logger.debug(
                    "Flooring subtitle %d: %.1fs → %.1fs (was %.3fs)",
                    seg.index,
                    seg.start,
                    seg.start + min_duration_s,
                    (end - start).total_seconds(),
                )
                end = min_end

        content = _wrap_text(seg.text, max_line_chars, max_lines)
        subtitles.append(srt.Subtitle(index=seg.index, start=start, end=end, content=content))

    n_clamped = _clamp_overlaps(subtitles)

    _log_timing_summary(
        n_total=len(subtitles),
        n_capped=n_capped,
        n_floored=n_floored,
        n_clamped=n_clamped,
        min_duration_s=min_duration_s,
        raw_durations_below_floor=raw_durations_below_floor,
    )

    return subtitles


def _clamp_overlaps(subtitles: list) -> int:
    """Pull each cue's end back so it never runs into the next cue's start.

    Mutates *subtitles* in place.  Cues are assumed to be sorted by start
    time (both ASR engines emit them that way).

    Overlapping cues are a real defect: players either stack them, flicker
    between them, or drop one, and the behaviour differs per player.  The
    duration floor in :func:`build_srt` is the most common source — flooring
    a 0.4s cue to 1.0s will happily run it over the top of its neighbour.

    When two cues share a start time we cannot fix the overlap by shortening
    (there is no room), so the cue is left alone and a warning is logged —
    that case indicates a genuine problem upstream in the ASR engine, not
    something to paper over here.

    Returns:
        The number of cues whose end timestamp was shortened.
    """
    n_clamped = 0
    for i in range(len(subtitles) - 1):
        cur = subtitles[i]
        nxt = subtitles[i + 1]
        if cur.end <= nxt.start:
            continue
        if nxt.start <= cur.start:
            logger.warning(
                "Cue %d starts at or before cue %d (%.3fs vs %.3fs) — "
                "cannot clamp overlap without reordering; leaving as-is.",
                i + 2, i + 1,
                nxt.start.total_seconds(), cur.start.total_seconds(),
            )
            continue
        logger.debug(
            "Clamping cue %d end %.3fs → %.3fs (overlapped next cue)",
            i + 1, cur.end.total_seconds(), nxt.start.total_seconds(),
        )
        cur.end = nxt.start
        n_clamped += 1
    return n_clamped


def _log_timing_summary(
    *,
    n_total: int,
    n_capped: int,
    n_floored: int,
    n_clamped: int,
    min_duration_s: float,
    raw_durations_below_floor: list[float],
) -> None:
    """Emit a one-line INFO summary of the timing adjustments that fired.

    This exists so that post-processing effects on cue timings are visible in
    the run log.  In particular, when the duration floor rewrites a large
    fraction of cues, the log says so explicitly and reports what the *true*
    durations were — which is the number that matters when characterising a
    model's timestamp behaviour.
    """
    if not (n_capped or n_floored or n_clamped):
        return

    parts: list[str] = []
    if n_capped:
        parts.append(f"{n_capped} capped")
    if n_floored:
        parts.append(f"{n_floored} floored")
    if n_clamped:
        parts.append(f"{n_clamped} overlap-clamped")

    logger.info(
        "Timing adjustments on %d cue(s): %s.", n_total, ", ".join(parts)
    )

    if n_floored and raw_durations_below_floor:
        vals = sorted(raw_durations_below_floor)
        pct = 100.0 * n_floored / n_total if n_total else 0.0
        logger.info(
            "Duration floor (%.2fs) rewrote %d/%d cues (%.1f%%). "
            "TRUE pre-floor durations: min %.3fs, median %.3fs, max %.3fs. "
            "Cue durations in the written SRT are NOT the model's own "
            "timestamps for these cues — measure on the pre-build_srt "
            "segments or re-run with min_subtitle_duration_s=0.",
            min_duration_s, n_floored, n_total, pct,
            vals[0], vals[len(vals) // 2], vals[-1],
        )


def write_srt(subtitles: list, output_path: Path) -> None:
    """Serialise *subtitles* to a ``.srt`` file.

    Args:
        subtitles:   List of ``srt.Subtitle`` objects.
        output_path: Destination path.

    Raises:
        OutputError: If the file cannot be written.
    """
    try:
        import srt
    except ImportError as exc:
        raise OutputError("srt library is not installed.") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        content = srt.compose(subtitles)
        output_path.write_text(content, encoding="utf-8")
        logger.info("SRT written: %s  (%d subtitles)", output_path, len(subtitles))
    except OSError as exc:
        raise OutputError(f"Cannot write SRT file {output_path}: {exc}") from exc


def _fmt_vtt_time(td: datetime.timedelta) -> str:
    """Format a timedelta as a WebVTT timestamp (``HH:MM:SS.mmm``).

    WebVTT uses ``.`` as the millisecond separator, where SRT uses ``,``.
    """
    total = max(0.0, td.total_seconds())
    h = int(total // 3600)
    m = int((total % 3600) // 60)
    s = total - (h * 3600) - (m * 60)
    # `06.3f` pads to width 6 with leading zero: 5.5 → "05.500".
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def write_vtt(subtitles: list, output_path: Path) -> None:
    """Serialise *subtitles* to a WebVTT (``.vtt``) file.

    WebVTT is the format HTML5 ``<video>`` elements support natively as a
    ``<track>`` source, so producing one alongside the SRT lets browsers,
    Jellyfin, and most modern players use the subtitles without any
    polyfill.  The cue text and timing are identical to the SRT — only the
    header (``WEBVTT``) and the timestamp separator (``.`` instead of
    ``,``) differ.  Cue identifiers (the index numbers in SRT) are
    omitted; they're optional in WebVTT and most players don't display
    them.

    Args:
        subtitles:   List of ``srt.Subtitle`` objects (same input as
                     :func:`write_srt`).
        output_path: Destination path (usually the SRT path with the
                     suffix replaced — ``.srt`` → ``.vtt``).

    Raises:
        OutputError: If the file cannot be written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        lines = ["WEBVTT", ""]
        for sub in subtitles:
            lines.append(f"{_fmt_vtt_time(sub.start)} --> {_fmt_vtt_time(sub.end)}")
            lines.append(sub.content)
            lines.append("")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("VTT written: %s  (%d cues)", output_path, len(subtitles))
    except OSError as exc:
        raise OutputError(f"Cannot write VTT file {output_path}: {exc}") from exc


def segments_from_whisper(raw_segments, index_offset: int = 0) -> list[SRTSegment]:
    """Convert raw faster-whisper segment objects to :class:`SRTSegment` list.

    Args:
        raw_segments:  Iterable of ``faster_whisper.transcribe.Segment`` objects.
        index_offset:  Starting index for subtitle numbering (default: 0 → 1-based).

    Returns:
        List of :class:`SRTSegment` objects.
    """
    def _f(seg, name):
        value = getattr(seg, name, None)
        return None if value is None else float(value)

    results: list[SRTSegment] = []
    for i, seg in enumerate(raw_segments, start=1):
        text = seg.text.strip()
        if not text:
            continue
        results.append(SRTSegment(
            index=i + index_offset,
            start=float(seg.start),
            end=float(seg.end),
            text=text,
            # Decoder metrics carried through for downstream analysis.  The
            # chunk_* fields stay None here: the long-form engine hands the
            # whole file to Whisper in one call, so there is no chunk plan and
            # no such thing as a chunk tail.
            avg_logprob=_f(seg, "avg_logprob"),
            compression_ratio=_f(seg, "compression_ratio"),
            no_speech_prob=_f(seg, "no_speech_prob"),
            temperature=_f(seg, "temperature"),
        ))
    return results


def summarize_segment_durations(segments: list[SRTSegment]) -> dict:
    """Return the raw duration distribution of ASR output, pre-post-processing.

    Intended for investigation work: this reads the segments *before*
    :func:`build_srt` applies any cap, floor, or overlap clamp, so the numbers
    reflect what the model actually emitted.

    Returns a dict with ``n``, ``min_s``, ``median_s``, ``max_s``, and
    ``histogram`` — counts bucketed at 0.25s intervals up to 3s, plus a
    ``3.00+`` bucket.  Empty input returns ``{"n": 0}``.
    """
    if not segments:
        return {"n": 0}

    durations = sorted(max(0.0, s.end - s.start) for s in segments)
    buckets: dict[str, int] = {}
    for d in durations:
        if d >= 3.0:
            key = "3.00+"
        else:
            lo = int(d / 0.25) * 0.25
            key = f"{lo:.2f}-{lo + 0.25:.2f}"
        buckets[key] = buckets.get(key, 0) + 1

    return {
        "n": len(durations),
        "min_s": round(durations[0], 3),
        "median_s": round(durations[len(durations) // 2], 3),
        "max_s": round(durations[-1], 3),
        "histogram": buckets,
    }
