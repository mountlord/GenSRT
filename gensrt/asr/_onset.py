"""Snap chunk-local segment timestamps to the audible speech onset.

The problem, measured
---------------------
In fixed-window mode a chunk is 5-8 s of audio with no guarantee that speech
starts at its first sample.  Whisper's timestamp tokens are good when a
chunk holds continuous dialogue, but for a lone short utterance the model
emits its habitual ``0.00 → 2.00``: start at the chunk's first sample, two
seconds long, wherever the word actually sits.  On one 152-minute file 560
of 1,293 cues were exactly that — every one of them displayed 2-4 s before
the speaker opened their mouth, and a viewer counting "1, 2, 3, 4" before
each line is right.

The fix
-------
Run silero-VAD over the chunk at a LOW threshold (``0.15``, not the 0.5 the
outer VAD uses) to find where voiced sound actually begins — low, because
the whole point of fixed-window mode is material the outer VAD calls
non-speech.  Then, for each segment whose model start does not fall inside
a detected speech region, move it to the first onset in its span and carry
its end along so the duration is preserved.  A segment whose start already
sits inside speech is left alone: the model's stamp is plausible and moving
it would be guessing.

The core, :func:`snap_to_onsets`, is a pure function over ``(start, end)``
pairs and speech regions so it can be tested without a VAD model; the
engine wraps it with :func:`speech_regions`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

#: silero threshold for onset detection.  Deliberately far below the outer
#: VAD's 0.5: this runs only on audio the model already produced text for,
#: so the question is not "is this speech" but "where in the window does
#: the sound start".
ONSET_VAD_THRESHOLD = 0.15
ONSET_MIN_SPEECH_MS = 80
ONSET_MIN_SILENCE_MS = 100
#: Lead the cue by this much so it is on screen as the first syllable lands.
ONSET_LEAD_S = 0.10


@dataclass
class OnsetStats:
    segments: int = 0
    snapped: int = 0
    shifts_s: list[float] = field(default_factory=list)

    def merge(self, other: "OnsetStats") -> None:
        self.segments += other.segments
        self.snapped += other.snapped
        self.shifts_s.extend(other.shifts_s)

    def summary(self) -> str:
        if not self.snapped:
            return f"Onset snap: 0 of {self.segments} cues moved"
        s = sorted(self.shifts_s)
        med = s[len(s) // 2]
        return (f"Onset snap: {self.snapped} of {self.segments} cues moved to the "
                f"audible onset (median +{med:.1f} s, max +{s[-1]:.1f} s)")


def speech_regions(audio, sr: int, *, threshold: float = ONSET_VAD_THRESHOLD) -> list[tuple[float, float]]:
    """Low-threshold silero regions on *audio*, in chunk-local seconds."""
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(
        threshold=threshold,
        min_speech_duration_ms=ONSET_MIN_SPEECH_MS,
        min_silence_duration_ms=ONSET_MIN_SILENCE_MS,
        speech_pad_ms=0,
    )
    return [(r["start"] / sr, r["end"] / sr) for r in get_speech_timestamps(audio, opts)]


def snap_to_onsets(
    spans: list[tuple[float, float]],
    regions: list[tuple[float, float]],
    chunk_len_s: float,
    *,
    lead_s: float = ONSET_LEAD_S,
) -> tuple[list[tuple[float, float]], OnsetStats]:
    """Move each span whose start is not inside speech to the next onset.

    Args:
        spans:       Chunk-local ``(start, end)`` pairs from the model, in
                     order.
        regions:     Chunk-local speech regions from :func:`speech_regions`.
        chunk_len_s: Chunk duration; ends are clamped to it.
        lead_s:      Subtracted from the onset so the cue leads the sound.

    Rules, precisely:
      * No regions → nothing moves (the VAD found no sound; leave the
        model's stamps rather than invent).
      * A span whose start lies within a region (or within ``lead_s`` before
        one) keeps its start.
      * Otherwise the first region start in ``[span.start, limit)`` is the
        new start, where ``limit`` is the next span's start (or the chunk
        end for the last span).  Duration is preserved; the end is clamped
        to the chunk.  No onset in that window → unchanged.
      * Starts never move earlier, and a span never crosses the next one.
    """
    stats = OnsetStats(segments=len(spans))
    if not spans or not regions:
        return list(spans), stats

    out: list[tuple[float, float]] = []
    for i, (start, end) in enumerate(spans):
        limit = spans[i + 1][0] if i + 1 < len(spans) else chunk_len_s
        inside = any(r0 - lead_s <= start <= r1 for r0, r1 in regions)
        if inside:
            out.append((start, end))
            continue
        onset = next((r0 for r0, _r1 in regions if start <= r0 < limit), None)
        if onset is None:
            out.append((start, end))
            continue
        new_start = max(start, onset - lead_s)
        shift = new_start - start
        if shift <= 0:
            out.append((start, end))
            continue
        new_end = min(end + shift, chunk_len_s)
        if i + 1 < len(spans):
            new_end = min(new_end, spans[i + 1][0])
        out.append((new_start, max(new_end, new_start + 0.2)))
        stats.snapped += 1
        stats.shifts_s.append(shift)
    return out, stats
