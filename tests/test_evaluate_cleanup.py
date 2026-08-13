"""Tests for the cleanup-rule evaluation harness (Drop B)."""

from __future__ import annotations

import pytest

from gensrt.evaluate_cleanup import (
    Row,
    Score,
    load_labels,
    load_segments,
    main,
    score_rule,
)
from gensrt.models import SRTSegment
from gensrt.segment_dump import write_segment_dump


def _row(index, dur, lp=-0.3, tail=False, sole=False, cr=1.2, temp=0.0, start=0.0):
    return Row(index=index, start_s=start, duration_s=dur, avg_logprob=lp,
               compression_ratio=cr,
               no_speech_prob=0.01, temperature=temp, is_chunk_tail=tail,
               is_chunk_sole=sole, chars=10, text="x")


# ── Scoring arithmetic ────────────────────────────────────────────────────

def test_perfect_rule():
    rows = [_row(1, 0.1), _row(2, 3.0)]
    s = score_rule("d<1", lambda r: r.duration_s < 1.0, rows, {1})
    assert (s.tp, s.fp, s.fn, s.tn) == (1, 0, 0, 1)
    assert s.precision == 1.0 and s.recall == 1.0


def test_precision_and_recall_are_distinguished():
    # Flags everything: perfect recall, poor precision.
    rows = [_row(i, 0.1) for i in range(1, 5)]
    s = score_rule("all", lambda r: True, rows, {1})
    assert s.recall == 1.0
    assert s.precision == pytest.approx(0.25)


def test_fbeta_below_one_favours_precision():
    """The weighting that encodes 'deleting real speech is worse'."""
    precise = Score("p", tp=5, fp=0, fn=5, tn=90)    # prec 1.00, rec 0.50
    thorough = Score("r", tp=10, fp=10, fn=0, tn=80)  # prec 0.50, rec 1.00
    assert precise.fbeta(0.5) > thorough.fbeta(0.5)
    assert thorough.fbeta(2.0) > precise.fbeta(2.0)


def test_empty_rule_scores_zero():
    s = score_rule("none", lambda r: False, [_row(1, 0.1)], {1})
    assert s.precision == 0.0 and s.fbeta() == 0.0


# ── Label parsing ─────────────────────────────────────────────────────────

def test_labels_ignore_comments_and_blanks(tmp_path):
    p = tmp_path / "flagged.txt"
    p.write_text("# hallucinations\n10\n\n12  19\n48, 52\n56 # dup of 57\n")
    assert load_labels(p) == {10, 12, 19, 48, 52, 56}


def test_labels_handle_bom(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes("\ufeff7\n8\n".encode("utf-8"))
    assert load_labels(p) == {7, 8}


# ── Round trip through the dump ───────────────────────────────────────────

def test_dump_is_readable_by_the_evaluator(tmp_path):
    segs = [
        SRTSegment(1, 0.0, 3.0, "real speech", avg_logprob=-0.25,
                   chunk_index=1, chunk_position=1, chunk_n_segments=2),
        SRTSegment(2, 3.4, 3.56, "ആരോപിച്ചു", avg_logprob=-1.55,
                   compression_ratio=3.4, temperature=0.8,
                   chunk_index=1, chunk_position=2, chunk_n_segments=2),
    ]
    csv_path = tmp_path / "clip.segments.csv"
    write_segment_dump(segs, csv_path)

    rows = load_segments(csv_path)
    assert len(rows) == 2
    assert rows[1].avg_logprob == pytest.approx(-1.55)
    assert rows[1].is_chunk_tail and not rows[1].is_chunk_sole
    assert rows[1].duration_s == pytest.approx(0.16)


def test_missing_metrics_parse_as_none(tmp_path):
    csv_path = tmp_path / "d.csv"
    write_segment_dump([SRTSegment(1, 0, 1, "x")], csv_path)
    assert load_segments(csv_path)[0].avg_logprob is None


def test_rules_tolerate_absent_metrics():
    """A dump without decoder metrics must score, not crash."""
    rows = [_row(1, 0.1, lp=None), _row(2, 3.0, lp=None)]
    s = score_rule("lp<-1", lambda r: r.avg_logprob is not None
                   and r.avg_logprob < -1.0, rows, {1})
    assert s.tp == 0 and s.fp == 0


# ── End to end ────────────────────────────────────────────────────────────

def test_cli_runs_and_ranks(tmp_path, capsys):
    """Chunk-tail fragments are separable; the ranking should reflect it."""
    segs = []
    for i in range(1, 21):
        # Genuine speech.
        segs.append(SRTSegment(i, i * 5.0, i * 5.0 + 3.0, "real speech here",
                               avg_logprob=-0.3, chunk_index=i,
                               chunk_position=1, chunk_n_segments=2))
    spurious = []
    for j, i in enumerate(range(21, 27), start=1):
        segs.append(SRTSegment(i, i * 5.0, i * 5.0 + 0.16, "ആരോപിച്ചു",
                               avg_logprob=-1.6, compression_ratio=3.2,
                               chunk_index=j, chunk_position=2,
                               chunk_n_segments=2))
        spurious.append(i)

    csv_path = tmp_path / "c.segments.csv"
    write_segment_dump(segs, csv_path)
    labels = tmp_path / "c.flagged.txt"
    labels.write_text("\n".join(str(i) for i in spurious))

    assert main(["--segments", str(csv_path), "--labels", str(labels)]) == 0
    out = capsys.readouterr().out
    assert "Corpus" in out and "ranked by F0.5" in out
    assert "chunk-tail hypothesis: 6/6" in out


def test_cli_warns_about_labels_not_in_dump(tmp_path, capsys):
    csv_path = tmp_path / "c.csv"
    write_segment_dump([SRTSegment(1, 0, 1, "x")], csv_path)
    labels = tmp_path / "l.txt"
    labels.write_text("1\n999\n")
    main(["--segments", str(csv_path), "--labels", str(labels)])
    assert "not in the dump" in capsys.readouterr().out


# ── Label transfer across runs ────────────────────────────────────────────

def test_labels_transfer_by_text_and_time(tmp_path):
    """Cue numbers shift between runs; content does not.

    Simulates the real case: an extra cue appears early in a re-run, so every
    later index is off by one. Matching on text + timestamp must still find
    the right cues.
    """
    import srt as srt_lib
    from datetime import timedelta

    from gensrt.evaluate_cleanup import transfer_labels_from_srt

    old = [
        srt_lib.Subtitle(1, timedelta(seconds=10), timedelta(seconds=13), "speech one"),
        srt_lib.Subtitle(2, timedelta(seconds=14), timedelta(seconds=15), "ആരോപിച്ചു"),
        srt_lib.Subtitle(3, timedelta(seconds=20), timedelta(seconds=23), "speech two"),
    ]
    srt_path = tmp_path / "old.srt"
    srt_path.write_text(srt_lib.compose(old), encoding="utf-8")

    # Current run: an extra cue at the front shifts everything by one.
    rows = [
        _row(1, 2.0, start=5.0), _row(2, 3.0, start=10.0),
        _row(3, 0.16, start=14.02), _row(4, 3.0, start=20.0),
    ]
    rows[1].text, rows[2].text, rows[3].text = "speech one", "ആരോപിച്ചു", "speech two"

    matched, unmatched = transfer_labels_from_srt(srt_path, {2}, rows)
    assert matched == {3}          # old cue 2 is now dump index 3
    assert unmatched == []


def test_transfer_reports_unmatched(tmp_path):
    import srt as srt_lib
    from datetime import timedelta

    from gensrt.evaluate_cleanup import transfer_labels_from_srt

    srt_path = tmp_path / "old.srt"
    srt_path.write_text(srt_lib.compose([
        srt_lib.Subtitle(1, timedelta(seconds=10), timedelta(seconds=11), "vanished")
    ]), encoding="utf-8")

    matched, unmatched = transfer_labels_from_srt(srt_path, {1}, [_row(1, 3.0, start=10.0)])
    assert matched == set() and unmatched == [1]


def test_transfer_does_not_reuse_a_row_for_two_labels(tmp_path):
    """Repeated fragment text is common; each label must claim its own cue."""
    import srt as srt_lib
    from datetime import timedelta

    from gensrt.evaluate_cleanup import transfer_labels_from_srt

    srt_path = tmp_path / "old.srt"
    srt_path.write_text(srt_lib.compose([
        srt_lib.Subtitle(1, timedelta(seconds=10), timedelta(seconds=11), "ആരോപിച്ചു"),
        srt_lib.Subtitle(2, timedelta(seconds=10, milliseconds=300),
                         timedelta(seconds=11), "ആരോപിച്ചു"),
    ]), encoding="utf-8")

    rows = [_row(1, 0.2, start=10.0), _row(2, 0.2, start=10.3)]
    rows[0].text = rows[1].text = "ആരോപിച്ചു"

    matched, unmatched = transfer_labels_from_srt(srt_path, {1, 2}, rows)
    assert matched == {1, 2} and unmatched == []
