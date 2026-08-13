"""Tests for per-chunk decode diagnostics."""

from __future__ import annotations

import csv
import logging

import pytest

from gensrt.asr._chunk_debug import (
    ChunkRecord,
    ChunkRecorder,
    NullChunkRecorder,
    log_timing_summary,
)


class FakeSegment:
    def __init__(self, text, temperature=0.0, avg_logprob=-0.3,
                 compression_ratio=1.2, no_speech_prob=0.01):
        self.text = text
        self.temperature = temperature
        self.avg_logprob = avg_logprob
        self.compression_ratio = compression_ratio
        self.no_speech_prob = no_speech_prob


def _rec(index, start, end, wall, **kw):
    r = ChunkRecord(index, start, end)
    r.wall_s = wall
    r.observe_segments([FakeSegment("ആരോപിച്ചു", **kw)])
    return r


# ── Telemetry folding ─────────────────────────────────────────────────────

def test_worst_case_telemetry_across_segments():
    r = ChunkRecord(1, 0.0, 5.0)
    r.observe_segments([
        FakeSegment("a", temperature=0.0, avg_logprob=-0.2,
                    compression_ratio=1.1, no_speech_prob=0.01),
        FakeSegment("b", temperature=0.8, avg_logprob=-1.4,
                    compression_ratio=3.9, no_speech_prob=0.44),
    ])
    assert r.temperature == 0.8            # highest reached
    assert r.avg_logprob == -1.4           # least confident
    assert r.compression_ratio == 3.9      # most repetitive
    assert r.no_speech_prob == 0.44
    assert r.n_segments == 2
    assert r.text == "a b"


def test_missing_telemetry_fields_do_not_crash():
    """faster-whisper's Segment fields are not a stability guarantee."""
    class Bare:
        text = "hello"

    r = ChunkRecord(1, 0.0, 5.0)
    r.observe_segments([Bare()])
    assert r.temperature is None
    assert r.as_dict()["temperature"] is None


def test_realtime_factor():
    r = ChunkRecord(1, 10.0, 15.0)
    r.wall_s = 20.0
    assert r.realtime_factor == pytest.approx(4.0)


def test_zero_duration_does_not_divide_by_zero():
    r = ChunkRecord(1, 5.0, 5.0)
    r.wall_s = 1.0
    assert r.realtime_factor == 0.0


# ── Export ────────────────────────────────────────────────────────────────

def test_export_writes_audio_and_manifest(tmp_path):
    src = tmp_path / "chunk.wav"
    src.write_bytes(b"RIFFfake")

    rec_dir = tmp_path / "out"
    recorder = ChunkRecorder(rec_dir)
    r = _rec(7, 12.3, 17.6, 18.42, temperature=0.8)
    recorder.record(r, src)
    recorder.finalize("MalayalamNews.wav")

    wavs = list(rec_dir.glob("*.wav"))
    assert len(wavs) == 1
    name = wavs[0].name
    assert name.startswith("chunk_007_")
    assert "wall018.42s" in name
    assert "T0.8" in name
    assert wavs[0].read_bytes() == b"RIFFfake"

    assert (rec_dir / "chunks.json").exists()
    assert (rec_dir / "README.txt").exists()

    with (rec_dir / "chunks.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["index"] == "7"
    assert rows[0]["temperature"] == "0.8"
    assert float(rows[0]["realtime_factor"]) == pytest.approx(3.48, abs=0.01)


def test_null_recorder_writes_nothing(tmp_path):
    recorder = NullChunkRecorder()
    recorder.record(_rec(1, 0.0, 5.0, 1.0), tmp_path / "nope.wav")
    recorder.finalize("x")
    assert list(tmp_path.iterdir()) == []
    assert recorder.enabled is False


def test_export_survives_unreadable_source(tmp_path, caplog):
    recorder = ChunkRecorder(tmp_path / "out")
    recorder.record(_rec(1, 0.0, 5.0, 1.0), tmp_path / "does_not_exist.wav")
    recorder.finalize("x")
    # The manifest still lands even though the copy failed.
    assert (tmp_path / "out" / "chunks.csv").exists()


# ── Outlier summary ───────────────────────────────────────────────────────

def _fast_and_slow():
    fast = [_rec(i, i * 5.0, i * 5.0 + 4.0, 1.8) for i in range(1, 11)]
    slow = [_rec(i, i * 5.0, i * 5.0 + 4.0, 20.0, temperature=0.8,
                 compression_ratio=3.5) for i in range(11, 14)]
    return fast + slow


def test_summary_flags_outliers_and_names_temperature(caplog):
    with caplog.at_level(logging.INFO):
        log_timing_summary(_fast_and_slow(), "TestEngine")
    text = caplog.text
    assert "Decode time is concentrated" in text
    assert "temperature fallback" in text


def test_summary_silent_when_all_chunks_are_similar(caplog):
    recs = [_rec(i, i * 5.0, i * 5.0 + 4.0, 1.8) for i in range(1, 11)]
    with caplog.at_level(logging.INFO):
        log_timing_summary(recs, "TestEngine")
    assert caplog.text == ""


def test_summary_distinguishes_slow_without_fallback(caplog):
    """Zero temperature on the slow chunks rules the hypothesis OUT."""
    recs = [_rec(i, i * 5.0, i * 5.0 + 4.0, 1.8) for i in range(1, 11)]
    recs += [_rec(i, i * 5.0, i * 5.0 + 4.0, 20.0, temperature=0.0)
             for i in range(11, 14)]
    with caplog.at_level(logging.INFO):
        log_timing_summary(recs, "TestEngine")
    assert "is NOT temperature fallback" in caplog.text


def test_summary_ignores_failed_chunks(caplog):
    recs = [_rec(i, i * 5.0, i * 5.0 + 4.0, 1.8) for i in range(1, 11)]
    bad = ChunkRecord(11, 0.0, 4.0)
    bad.wall_s = 90.0
    bad.failed = True
    recs.append(bad)
    with caplog.at_level(logging.INFO):
        log_timing_summary(recs, "TestEngine")
    assert caplog.text == ""


def test_summary_needs_enough_samples(caplog):
    with caplog.at_level(logging.INFO):
        log_timing_summary([_rec(1, 0.0, 4.0, 1.0), _rec(2, 4.0, 8.0, 30.0)],
                           "TestEngine")
    assert caplog.text == ""


# ── Output directory naming ───────────────────────────────────────────────

def test_recorder_uses_the_directory_it_is_given(tmp_path):
    """Regression: the engine used to append the TEMP audio stem.

    That produced eval/gensrt_audio_73g7xhsu/ — a fresh randomly-named folder
    per run, with nothing tying it to the source video. The caller knows the
    source name; the engine does not, so the caller resolves the path and the
    recorder uses it verbatim.
    """
    out = tmp_path / "eval" / "MalayalamNews"
    recorder = ChunkRecorder(out)
    src = tmp_path / "chunk.wav"
    src.write_bytes(b"RIFF")
    recorder.record(_rec(1, 0.0, 4.0, 2.0), src)
    recorder.finalize("gensrt_audio_73g7xhsu.wav")

    assert (out / "chunks.csv").exists()
    assert not list(tmp_path.glob("eval/gensrt_audio_*"))
