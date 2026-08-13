"""Tests for gensrt.srt.builder — text integrity and cue timing.

These cover the two defects fixed in this drop:
  * _wrap_text silently discarding words past the line budget
  * build_srt producing cues that overlap their successor
"""

from __future__ import annotations

import pytest

from gensrt.models import SRTSegment
from gensrt.srt.builder import _wrap_text, build_srt, summarize_segment_durations


# ── Text integrity ────────────────────────────────────────────────────────

def test_wrap_never_drops_words():
    text = " ".join(f"word{i}" for i in range(60))
    wrapped = _wrap_text(text, max_line_chars=42, max_lines=2)
    assert wrapped.split() == text.split()


def test_wrap_respects_line_budget_where_possible():
    text = " ".join(f"word{i}" for i in range(60))
    for line in _wrap_text(text, max_line_chars=42, max_lines=2).split("\n"):
        assert len(line) <= 42


def test_wrap_exceeds_max_lines_rather_than_truncating():
    text = " ".join(f"word{i}" for i in range(60))
    wrapped = _wrap_text(text, max_line_chars=42, max_lines=2)
    assert len(wrapped.split("\n")) > 2
    assert "word59" in wrapped


def test_wrap_short_text_untouched():
    assert _wrap_text("short line", 42, 2) == "short line"


def test_wrap_never_splits_a_long_token():
    """A token longer than the budget overflows its line intact.

    Character-level splitting would break Indic grapheme clusters, so an
    over-long word is preferred to a corrupted one.
    """
    long_word = "a" * 80
    assert _wrap_text(f"{long_word} tail", 42, 2).split() == [long_word, "tail"]


def test_wrap_preserves_malayalam_text():
    text = "ആരോപിച്ചു അറിയിച്ചു ഉണ്ടാകും ആയിരുന്നു " * 4
    assert _wrap_text(text, 42, 2).split() == text.split()


def test_build_srt_preserves_all_text():
    long_text = " ".join(f"w{i}" for i in range(80))
    subs = build_srt([SRTSegment(1, 0.0, 5.0, long_text)])
    assert subs[0].content.split() == long_text.split()


# ── Cue timing ────────────────────────────────────────────────────────────

def test_no_overlapping_cues_after_floor():
    """The min-duration floor must not push a cue over its neighbour."""
    segs = [
        SRTSegment(1, 10.0, 10.42, "short"),
        SRTSegment(2, 10.5, 14.0, "next"),
    ]
    subs = build_srt(segs, max_duration_s=10, min_duration_s=1)
    assert subs[0].end <= subs[1].start


def test_floor_still_applies_when_there_is_room():
    segs = [
        SRTSegment(1, 10.0, 10.42, "short"),
        SRTSegment(2, 20.0, 24.0, "far away"),
    ]
    subs = build_srt(segs, min_duration_s=1)
    assert subs[0].end.total_seconds() == pytest.approx(11.0)


def test_no_overlaps_anywhere_in_a_dense_run():
    """Simulates chunked output: many short adjacent cues."""
    segs = [
        SRTSegment(i + 1, i * 0.6, i * 0.6 + 0.3, f"cue{i}")
        for i in range(50)
    ]
    subs = build_srt(segs, max_duration_s=10, min_duration_s=1)
    for a, b in zip(subs, subs[1:]):
        assert a.end <= b.start, f"cue overlap at {a.index}"


def test_cap_shortens_long_cues():
    subs = build_srt([SRTSegment(1, 0.0, 120.0, "hangs forever")], max_duration_s=3)
    assert subs[0].end.total_seconds() == pytest.approx(3.0)


def test_end_always_after_start():
    subs = build_srt([SRTSegment(1, 5.0, 5.0, "zero length")])
    assert subs[0].end > subs[0].start


def test_zero_floor_leaves_true_durations_intact():
    """The measurement path: floor off means the SRT keeps model timings."""
    segs = [SRTSegment(1, 0.0, 0.31, "a"), SRTSegment(2, 5.0, 5.77, "b")]
    subs = build_srt(segs, min_duration_s=0)
    assert (subs[0].end - subs[0].start).total_seconds() == pytest.approx(0.31)
    assert (subs[1].end - subs[1].start).total_seconds() == pytest.approx(0.77)


def test_floor_collapses_distinct_durations_to_one_value():
    """Documents WHY measurement must happen pre-build_srt.

    Two cues with visibly different model durations become indistinguishable
    once the floor has run.  This is the mechanism behind the 'exactly 1.00s'
    reading in INVESTIGATIONS.md I-7.
    """
    segs = [SRTSegment(1, 0.0, 0.31, "a"), SRTSegment(2, 5.0, 5.77, "b")]
    subs = build_srt(segs, min_duration_s=1.0)
    durations = {round((s.end - s.start).total_seconds(), 3) for s in subs}
    assert durations == {1.0}


# ── Measurement helper ────────────────────────────────────────────────────

def test_summarize_segment_durations_reports_true_values():
    segs = [SRTSegment(1, 0.0, 0.31, "a"), SRTSegment(2, 5.0, 5.77, "b")]
    summary = summarize_segment_durations(segs)
    assert summary["n"] == 2
    assert summary["min_s"] == pytest.approx(0.31)
    assert summary["max_s"] == pytest.approx(0.77)


def test_summarize_empty():
    assert summarize_segment_durations([]) == {"n": 0}
