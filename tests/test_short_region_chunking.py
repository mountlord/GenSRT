"""Short-region handling and chunk-size tuning (v1.2.7).

The bug being fixed: through v1.2.6, ``_chunk_region`` returned nothing
for any speech region shorter than ``min_chunk_s`` — the transcript
silently lost every utterance briefer than 2 seconds.  Measured cost: ~4
minutes of speech in a 10-minute sparse-dialogue excerpt, and worst on
material dominated by short exclamations, where a large share of
outer-VAD regions fall under the threshold.

The fix reframes ``min_chunk_s``: it is the minimum size of a chunk
produced by *cutting*, not a floor on which speech exists.  Short regions
are emitted whole (``cut_method="short_region"``); only sub-50ms VAD
artifacts are skipped.  Both knobs are now real configuration
(``max_chunk_s`` / ``min_chunk_s``), reachable from config, CLI, and the
GUI settings editor, and validated before any audio work runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from gensrt.asr._silence_chunking import (
    DEFAULT_MAX_CHUNK_S,
    DEFAULT_MIN_CHUNK_S,
    _MIN_EMITTABLE_REGION_S,
    plan_chunks,
    summarize_chunk_plan,
)
from gensrt.exceptions import ConfigError
from gensrt.models import TranscriptionConfig
from gensrt.pipeline import validate_chunking_config

SR = 16000


def _audio(seconds: float) -> np.ndarray:
    """Quiet noise floor — enough signal for RMS math, no real speech."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(seconds * SR)) * 0.001).astype(np.float32)


# ── The fix itself ────────────────────────────────────────────────────────

def test_short_region_is_emitted_whole_not_discarded():
    """A 1.5s utterance used to vanish; it must now become one chunk."""
    audio = _audio(10.0)
    chunks = plan_chunks(audio, SR, [(3.0, 4.5)])
    assert len(chunks) == 1
    c = chunks[0]
    assert c["cut_method"] == "short_region"
    assert c["start_s"] == 3.0 and c["end_s"] == 4.5
    assert c["duration_s"] == 1.5


def test_many_short_regions_all_survive():
    """The material that motivated this: lots of brief exclamations."""
    audio = _audio(60.0)
    regions = [(float(t), float(t) + 0.8) for t in range(0, 50, 5)]
    chunks = plan_chunks(audio, SR, regions)
    assert len(chunks) == len(regions)
    assert all(c["cut_method"] == "short_region" for c in chunks)


def test_degenerate_sliver_is_still_skipped():
    """Sub-50ms regions are VAD artifacts, not speech."""
    audio = _audio(5.0)
    chunks = plan_chunks(audio, SR, [(1.0, 1.0 + _MIN_EMITTABLE_REGION_S / 2)])
    assert chunks == []


def test_normal_region_behaviour_unchanged():
    """A region between min and max still becomes one region_end chunk."""
    audio = _audio(10.0)
    chunks = plan_chunks(audio, SR, [(2.0, 6.0)])
    assert len(chunks) == 1
    assert chunks[0]["cut_method"] == "region_end"


def test_mixed_regions_sorted_and_complete():
    audio = _audio(30.0)
    chunks = plan_chunks(audio, SR, [(1.0, 1.8), (5.0, 10.0), (20.0, 21.0)])
    methods = [c["cut_method"] for c in chunks]
    assert methods.count("short_region") == 2
    assert "region_end" in methods
    starts = [c["start_s"] for c in chunks]
    assert starts == sorted(starts)


# ── Tunable sizes ─────────────────────────────────────────────────────────

def test_min_chunk_s_moves_the_whole_vs_short_boundary():
    """With min_chunk_s=0.5 a 1.5s region is 'normal'; with 2.0 it's short."""
    audio = _audio(10.0)
    a = plan_chunks(audio, SR, [(3.0, 4.5)], min_chunk_s=0.5)
    b = plan_chunks(audio, SR, [(3.0, 4.5)], min_chunk_s=2.0)
    assert a[0]["cut_method"] == "region_end"
    assert b[0]["cut_method"] == "short_region"


def test_max_chunk_s_controls_subdivision():
    """A 10s region: max=12 leaves it whole; max=4 forces cutting."""
    audio = _audio(15.0)
    whole = plan_chunks(audio, SR, [(1.0, 11.0)], max_chunk_s=12.0)
    cut   = plan_chunks(audio, SR, [(1.0, 11.0)], max_chunk_s=4.0,
                        min_chunk_s=1.0)
    assert len(whole) == 1
    assert len(cut) >= 3
    assert all(c["duration_s"] <= 4.0 + 1e-6 for c in cut)


def test_summary_counts_short_regions():
    audio = _audio(20.0)
    chunks = plan_chunks(audio, SR, [(1.0, 1.5), (2.0, 2.5), (5.0, 9.0)])
    stats = summarize_chunk_plan(chunks)
    assert stats["n_short_region"] == 2
    assert stats["n_chunks"] == 3


# ── Config plumbing ───────────────────────────────────────────────────────

def test_defaults_flow_into_config():
    cfg = TranscriptionConfig()
    assert cfg.max_chunk_s == DEFAULT_MAX_CHUNK_S
    assert cfg.min_chunk_s == DEFAULT_MIN_CHUNK_S


def test_cli_exposes_chunk_sizes():
    from gensrt.cli import _build_parser

    args = _build_parser().parse_args(
        ["--input", "v.mkv", "--max-chunk-s", "8", "--min-chunk-s", "0.5"]
    )
    assert args.max_chunk_s == 8.0
    assert args.min_chunk_s == 0.5


def test_chunk_sizes_reach_the_built_config():
    from gensrt.config import merge_config
    from gensrt.operations import build_transcription_config

    merged = merge_config(
        {}, {"max_chunk_s": 8.0, "min_chunk_s": 0.5, "device": "cpu"}
    )
    built = build_transcription_config(merged)
    assert built.max_chunk_s == 8.0
    assert built.min_chunk_s == 0.5


# ── Validation ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bad", [
    {"min_chunk_s": 0.0},
    {"min_chunk_s": -1.0},
    {"min_chunk_s": 6.0, "max_chunk_s": 6.0},   # empty cut window
    {"min_chunk_s": 8.0, "max_chunk_s": 6.0},
    {"max_chunk_s": 31.0},                       # beyond Whisper's window
])
def test_invalid_chunk_sizes_rejected(bad):
    with pytest.raises(ConfigError):
        validate_chunking_config(TranscriptionConfig(**bad))


def test_valid_custom_sizes_accepted():
    validate_chunking_config(
        TranscriptionConfig(min_chunk_s=0.5, max_chunk_s=12.0)
    )
