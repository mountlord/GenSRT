"""SRT file construction.

Converts faster-whisper segment output into standard ``.srt`` subtitle files
using the ``srt`` library.

Usage::

    from gensrt.srt.builder import build_srt, write_srt
    subtitles = build_srt(segments)
    write_srt(subtitles, Path("output.srt"))
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from gensrt.exceptions import OutputError
from gensrt.models import SRTSegment

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Maximum characters per subtitle line before we wrap.
_MAX_LINE_CHARS: int = 84
# Maximum lines per subtitle block.
_MAX_LINES: int = 2


def _wrap_text(text: str) -> str:
    """Wrap *text* into at most ``_MAX_LINES`` lines of ``_MAX_LINE_CHARS``."""
    text = text.strip()
    if len(text) <= _MAX_LINE_CHARS:
        return text

    words = text.split()
    lines: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join(current + [word])
        if len(candidate) > _MAX_LINE_CHARS and current:
            lines.append(" ".join(current))
            current = [word]
            if len(lines) >= _MAX_LINES:
                break
        else:
            current.append(word)

    if current:
        lines.append(" ".join(current))

    return "\n".join(lines[:_MAX_LINES])


def build_srt(segments: list[SRTSegment], max_duration_s: float = 0.0) -> list:
    """Convert :class:`SRTSegment` objects to ``srt.Subtitle`` objects.

    Args:
        segments:       Ordered list of subtitle segments.
        max_duration_s: If > 0, cap each subtitle's display duration to this
                        many seconds.  Fixes subtitles that hang on screen for
                        minutes when Whisper assigns a very long end timestamp.

    Returns:
        List of ``srt.Subtitle`` objects ready for serialisation.
    """
    try:
        import srt
    except ImportError as exc:
        raise OutputError(
            "srt library is not installed. Run: pip install srt"
        ) from exc

    subtitles = []
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
                logger.debug(
                    "Capping subtitle %d: %.1fs → %.1fs (was %.1fs)",
                    seg.index,
                    seg.start,
                    seg.start + max_duration_s,
                    (end - start).total_seconds(),
                )
                end = max_end

        content = _wrap_text(seg.text)
        subtitles.append(srt.Subtitle(index=seg.index, start=start, end=end, content=content))

    return subtitles


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


def segments_from_whisper(raw_segments, index_offset: int = 0) -> list[SRTSegment]:
    """Convert raw faster-whisper segment objects to :class:`SRTSegment` list.

    Args:
        raw_segments:  Iterable of ``faster_whisper.transcribe.Segment`` objects.
        index_offset:  Starting index for subtitle numbering (default: 0 → 1-based).

    Returns:
        List of :class:`SRTSegment` objects.
    """
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
        ))
    return results
