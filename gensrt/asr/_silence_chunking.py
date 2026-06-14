"""Silent-boundary chunking algorithm.

Implements the v1.2 rule: every chunk boundary lands in a silent slot,
no chunk exceeds ``max_chunk_s`` seconds, minimum chunk size is
``min_chunk_s`` seconds.  If no detectable silence exists within the
valid range, fall back to the audio energy minimum (lowest-RMS 20 ms
window).

This is the algorithmic core proven out in the
``chunked_vegam_silent_boundary.py`` test script during the I-2
investigation.  Only the algorithm proper lives here — orchestration
(model loading, per-chunk inference, SRT assembly) lives in
:mod:`gensrt.asr.monolingual_whisper`.

The progressive VAD sweep targets a known failure mode of fine-tuned
Whisper models (vegam in particular) where dense continuous speech
contains only brief inter-word silences that silero-VAD at default
threshold misses entirely.  Higher thresholds with tighter
``min_silence_duration_ms`` catch those gaps, at the cost of occasional
"soft consonant" cuts when no real pause exists.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


# Progressive VAD configs: (threshold, min_silence_ms, speech_pad_ms, label).
# Sorted by tier: lower-index = more reliable (less likely to land in
# soft-but-still-speech).  When multiple tiers find a candidate, we prefer
# the lowest-tier (most reliable) silence in the greedy-pack step.
PROGRESSIVE_VAD_CONFIGS: tuple[tuple[float, int, int, str], ...] = (
    # Tier 0 — reliable: standard threshold, longer min_silence
    (0.40, 500, 100, "loose"),
    (0.40, 200,  50, "default-200"),
    (0.40, 100,  50, "default-100"),
    (0.40,  50,  30, "default-50"),
    # Tier 1 — moderate: slightly stricter speech threshold
    (0.60, 200,  30, "strict-200"),
    (0.60, 100,  20, "strict-100"),
    (0.60,  50,  10, "strict-50"),
    # Tier 2 — aggressive: high threshold, tight pad.  Catches inter-word
    # gaps the looser passes miss, at risk of soft-speech cuts.
    (0.80, 100,  10, "very_strict-100"),
    (0.80,  50,  10, "very_strict-50"),
)

# Inner-VAD min_speech_duration_ms.  200ms matches the size sweep that
# successfully detected silences in Region 1 of MalayalamNews.mp4.
# Lower values paradoxically reduce detected silences because brief noise
# bursts get classified as speech and absorbed into adjacent regions.
INNER_VAD_MIN_SPEECH_MS: int = 200


# Defaults validated by the I-2 investigation on dense Malayalam news.
# Exposed as module constants so callers can override per-job.
DEFAULT_MAX_CHUNK_S: float = 6.0
DEFAULT_MIN_CHUNK_S: float = 2.0


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def plan_chunks(
    audio: "np.ndarray",
    sr: int,
    regions: list[tuple[float, float]],
    *,
    max_chunk_s: float = DEFAULT_MAX_CHUNK_S,
    min_chunk_s: float = DEFAULT_MIN_CHUNK_S,
) -> list[dict]:
    """Apply the silent-boundary rule across a list of speech regions.

    Args:
        audio:        Float32 audio samples in ``[-1, 1]`` (mono).
        sr:           Sample rate (16 000).
        regions:      Outer-VAD speech regions as ``(start_s, end_s)``
                      tuples.  Each region is processed independently —
                      no chunk ever spans across a region boundary.
        max_chunk_s:  No chunk may exceed this duration (default 6.0s).
        min_chunk_s:  Skip regions shorter than this (default 2.0s).

    Returns:
        List of chunk dicts, each with:
            * ``start_s`` (float)        — absolute start in audio
            * ``end_s`` (float)          — absolute end in audio
            * ``duration_s`` (float)
            * ``cut_method`` (str)       — "silence", "energy_min", or "region_end"
            * ``silence_tier`` (int|None) — index into PROGRESSIVE_VAD_CONFIGS,
                                            or None for non-silence cuts
            * ``silence_tier_label`` (str|None)
            * ``silence_dur_ms`` (int|None) — silence duration at the looser
                                              config that found this midpoint
            * ``rms_at_cut`` (float|None) — set for energy_min cuts only

        Sorted by ``start_s``.  Adjacent chunks abut (end of one ==
        start of next) when they share a silence boundary; for region
        starts/ends there are intentional gaps.
    """
    all_chunks: list[dict] = []
    for r_start, r_end in regions:
        all_chunks.extend(
            _chunk_region(audio, sr, r_start, r_end, max_chunk_s, min_chunk_s)
        )
    return all_chunks


# ──────────────────────────────────────────────────────────────────────────
# Inner helpers
# ──────────────────────────────────────────────────────────────────────────


def _chunk_region(
    audio: "np.ndarray",
    sr: int,
    region_start: float,
    region_end: float,
    max_chunk_s: float,
    min_chunk_s: float,
) -> list[dict]:
    """Apply the silent-boundary rule to a single speech region."""
    duration = region_end - region_start
    chunks: list[dict] = []

    if duration < min_chunk_s:
        # Region too short to emit as a chunk at all.
        return chunks

    if duration <= max_chunk_s:
        # Fits in a single chunk — no subdivision needed.
        chunks.append({
            "start_s":            round(region_start, 3),
            "end_s":              round(region_end, 3),
            "duration_s":         round(duration, 3),
            "cut_method":         "region_end",
            "silence_tier":       None,
            "silence_tier_label": None,
            "silence_dur_ms":     None,
            "rms_at_cut":         None,
        })
        return chunks

    # Subdivide: gather silence midpoints across all tiers, then greedy-pack.
    midpoints = _collect_silence_midpoints(audio, sr, region_start, region_end)

    cur_start = region_start
    while cur_start < region_end:
        remaining = region_end - cur_start
        if remaining <= max_chunk_s:
            # Final chunk fits — emit and done.
            chunks.append({
                "start_s":            round(cur_start, 3),
                "end_s":              round(region_end, 3),
                "duration_s":         round(remaining, 3),
                "cut_method":         "region_end",
                "silence_tier":       None,
                "silence_tier_label": None,
                "silence_dur_ms":     None,
                "rms_at_cut":         None,
            })
            break

        valid_lo = cur_start + min_chunk_s
        valid_hi = cur_start + max_chunk_s
        valid_cuts = [
            m for m in midpoints
            if valid_lo <= m["time"] <= valid_hi
        ]

        if valid_cuts:
            # Prefer most-reliable tier first.  Within that tier, pick the
            # LATEST midpoint (gives the biggest legal chunk).
            best_tier = min(m["loosest_tier"] for m in valid_cuts)
            tier_cuts = [m for m in valid_cuts if m["loosest_tier"] == best_tier]
            chosen = max(tier_cuts, key=lambda m: m["time"])
            chunks.append({
                "start_s":            round(cur_start, 3),
                "end_s":              round(chosen["time"], 3),
                "duration_s":         round(chosen["time"] - cur_start, 3),
                "cut_method":         "silence",
                "silence_tier":       chosen["loosest_tier"],
                "silence_tier_label": PROGRESSIVE_VAD_CONFIGS[chosen["loosest_tier"]][3],
                "silence_dur_ms":     chosen["silence_dur_ms"],
                "rms_at_cut":         None,
            })
            cur_start = chosen["time"]
        else:
            # No detected silence in range — fall back to energy minimum.
            cut_at, rms = _find_energy_min(audio, sr, valid_lo, valid_hi)
            chunks.append({
                "start_s":            round(cur_start, 3),
                "end_s":              round(cut_at, 3),
                "duration_s":         round(cut_at - cur_start, 3),
                "cut_method":         "energy_min",
                "silence_tier":       None,
                "silence_tier_label": None,
                "silence_dur_ms":     None,
                "rms_at_cut":         round(rms, 5),
            })
            cur_start = cut_at

    return chunks


def _collect_silence_midpoints(
    audio: "np.ndarray",
    sr: int,
    region_start: float,
    region_end: float,
) -> list[dict]:
    """Run VAD across all progressive configs.  Return sorted midpoints.

    Each returned dict has:
        * ``time`` (float)            — absolute time in audio
        * ``found_by`` (list[int])    — indices into PROGRESSIVE_VAD_CONFIGS
        * ``loosest_tier`` (int)      — lowest index that detected this midpoint
        * ``silence_dur_ms`` (int)    — silence duration at the loosest tier
    """
    from faster_whisper.vad import get_speech_timestamps, VadOptions

    s0 = int(region_start * sr)
    s1 = int(region_end * sr)
    sub_audio = audio[s0:s1]

    # Round midpoints to ms to deduplicate across tiers.
    midpoint_data: dict[float, dict] = {}
    for cfg_idx, (threshold, min_silence_ms, speech_pad_ms, _label) in enumerate(
            PROGRESSIVE_VAD_CONFIGS):
        opts = VadOptions(
            threshold=threshold,
            min_speech_duration_ms=INNER_VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        sub_regions = get_speech_timestamps(sub_audio, opts)
        for i in range(len(sub_regions) - 1):
            sil_start = sub_regions[i]["end"] / sr + region_start
            sil_end   = sub_regions[i + 1]["start"] / sr + region_start
            midpoint = round((sil_start + sil_end) / 2, 3)
            sil_dur_ms = int(round((sil_end - sil_start) * 1000))
            if midpoint not in midpoint_data:
                midpoint_data[midpoint] = {
                    "found_by": [],
                    "silence_dur_ms": sil_dur_ms,
                }
            midpoint_data[midpoint]["found_by"].append(cfg_idx)

    records = [
        {
            "time":           t,
            "found_by":       data["found_by"],
            "loosest_tier":   min(data["found_by"]),
            "silence_dur_ms": data["silence_dur_ms"],
        }
        for t, data in midpoint_data.items()
    ]
    records.sort(key=lambda r: r["time"])
    return records


def _find_energy_min(
    audio: "np.ndarray",
    sr: int,
    range_start_s: float,
    range_end_s: float,
    *,
    window_ms: float = 20.0,
) -> tuple[float, float]:
    """Find the time in ``[range_start_s, range_end_s]`` with minimum RMS.

    Used as the fallback when no detectable silence exists in the lookahead
    range during greedy packing.  Voiced phonemes have high RMS; unvoiced
    consonants and inter-syllable troughs are lower; the absolute minimum
    in the range is the least-bad cut point when no real pause exists.

    Args:
        audio:          Float32 audio.
        sr:             Sample rate.
        range_start_s:  Lower bound (inclusive) for the cut.
        range_end_s:    Upper bound (inclusive) for the cut.
        window_ms:      RMS analysis window size.  20ms is short enough to
                        find narrow troughs without being noisy.

    Returns:
        Tuple of (cut time in seconds, RMS at that point).  If the range
        is too short for meaningful analysis, returns the midpoint with
        RMS=0.0.
    """
    import numpy as np

    s0 = int(range_start_s * sr)
    s1 = int(range_end_s * sr)
    window_samples = max(1, int(window_ms * sr / 1000))
    if s1 - s0 < window_samples * 2:
        return (range_start_s + range_end_s) / 2, 0.0

    audio_slice = audio[s0:s1]
    n_windows = (s1 - s0) // window_samples
    if n_windows == 0:
        return (range_start_s + range_end_s) / 2, 0.0

    trimmed = audio_slice[: n_windows * window_samples].reshape(n_windows, window_samples)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1))
    min_idx = int(np.argmin(rms))
    cut_sample = s0 + min_idx * window_samples + window_samples // 2
    return cut_sample / sr, float(rms[min_idx])


# ──────────────────────────────────────────────────────────────────────────
# Diagnostic summary (used by monolingual engine for the one-line log)
# ──────────────────────────────────────────────────────────────────────────


def summarize_chunk_plan(chunks: list[dict]) -> dict:
    """Return aggregate stats for a chunk plan.

    Used for the one-line health log emitted by the monolingual engine.

    Returns:
        Dict with:
            * ``n_chunks``         — total chunks
            * ``n_silence``        — chunks cut at detected silence
            * ``n_energy_min``     — chunks cut at energy minimum (fallback)
            * ``n_region_end``     — chunks ending at region boundary
            * ``pct_silence``      — percentage of cuts at silence
            * ``pct_energy_min``   — percentage at fallback
    """
    n = len(chunks)
    n_silence  = sum(1 for c in chunks if c["cut_method"] == "silence")
    n_energy   = sum(1 for c in chunks if c["cut_method"] == "energy_min")
    n_regend   = sum(1 for c in chunks if c["cut_method"] == "region_end")
    return {
        "n_chunks":       n,
        "n_silence":      n_silence,
        "n_energy_min":   n_energy,
        "n_region_end":   n_regend,
        "pct_silence":    round(100.0 * n_silence / n, 1) if n else 0.0,
        "pct_energy_min": round(100.0 * n_energy  / n, 1) if n else 0.0,
    }
