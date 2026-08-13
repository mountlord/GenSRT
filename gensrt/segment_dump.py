"""Write the raw ASR segment table to CSV for offline analysis.

This is the measurement surface for cleanup work.  Deciding which cues are
spurious means comparing candidate rules against cues a human has judged, and
that comparison needs the model's own numbers next to each cue — which are
otherwise discarded the moment the segments are formatted into an SRT.

Two properties matter and are easy to get wrong:

**Written before post-processing.**  ``build_srt`` caps, floors and
overlap-clamps cue timings.  A duration read from the finished SRT is
GenSRT's arithmetic, not the model's.  The dump is taken directly from the
engine output so the ``duration_s`` column is what Whisper actually emitted.

**Written before translation.**  The ``text`` column is the model's own
output in the source language.  Translated text would be useless for judging
whether the model hallucinated.

The ``index`` column matches the cue numbers in the written ``.srt``, so a
list of hand-flagged cue numbers can be joined straight onto this table.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from gensrt.models import SRTSegment

logger = logging.getLogger(__name__)

# Column order is deliberate: identity, then timing, then what the decoder
# said about itself, then where the segment sat in the chunk plan, then the
# text last because it is the widest field and variable-width columns at the
# end keep the file readable in a terminal.
COLUMNS = [
    "index",
    "start_s",
    "end_s",
    "duration_s",
    "avg_logprob",
    "compression_ratio",
    "no_speech_prob",
    "temperature",
    "chunk_index",
    "chunk_position",
    "chunk_n_segments",
    "is_chunk_tail",
    "is_chunk_sole",
    "chars",
    "text",
]


def _row(seg: SRTSegment) -> dict:
    def r(value, places=4):
        return "" if value is None else round(value, places)

    return {
        "index": seg.index,
        "start_s": r(seg.start, 3),
        "end_s": r(seg.end, 3),
        "duration_s": r(seg.duration, 3),
        "avg_logprob": r(seg.avg_logprob),
        "compression_ratio": r(seg.compression_ratio),
        "no_speech_prob": r(seg.no_speech_prob),
        "temperature": r(seg.temperature, 2),
        "chunk_index": "" if seg.chunk_index is None else seg.chunk_index,
        "chunk_position": "" if seg.chunk_position is None else seg.chunk_position,
        "chunk_n_segments": (
            "" if seg.chunk_n_segments is None else seg.chunk_n_segments
        ),
        "is_chunk_tail": int(seg.is_chunk_tail),
        "is_chunk_sole": int(seg.is_chunk_sole),
        "chars": len(seg.text),
        "text": seg.text,
    }


def write_segment_dump(segments: list[SRTSegment], path: Path) -> None:
    """Write *segments* to *path* as UTF-8 CSV with a BOM.

    The BOM is there so Excel opens Malayalam text correctly on Windows
    without an import wizard.  It is invisible to ``csv`` readers and to
    pandas, so it costs nothing on the analysis side.

    Never raises: a diagnostics dump failing must not fail a transcription
    that otherwise succeeded.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=COLUMNS)
            writer.writeheader()
            for seg in segments:
                writer.writerow(_row(seg))
    except Exception as exc:
        # Deliberately broad. This is a diagnostics side-effect; a
        # transcription that otherwise succeeded must not fail because a
        # debug CSV could not be written. Beyond OSError, a malformed path
        # raises ValueError on some platforms.
        logger.warning("Could not write segment dump %s: %s", path, exc)
        return

    n_with_logprob = sum(1 for s in segments if s.avg_logprob is not None)
    logger.info(
        "Segment dump written: %s  (%d segment(s), %d with decoder metrics)",
        path, len(segments), n_with_logprob,
    )
    if segments and n_with_logprob == 0:
        logger.warning(
            "No decoder metrics were captured — confidence-based analysis will "
            "not be possible from this dump. This usually means the installed "
            "faster-whisper no longer exposes them on Segment objects."
        )
