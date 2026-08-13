"""Drop A: decoder telemetry and chunk provenance on SRTSegment.

None of this changes output. It exists so that cleanup rules can be evaluated
against evidence instead of chosen by feel — which needs the model's own
confidence numbers alongside each cue, and those are discarded at the
engine/SRT boundary in a plain faster-whisper pipeline.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from gensrt.asr.monolingual_whisper import MonolingualWhisperEngine
from gensrt.models import SRTSegment, TranscriptionConfig
from gensrt.segment_dump import COLUMNS, write_segment_dump
from gensrt.srt.builder import segments_from_whisper


# ── SRTSegment ────────────────────────────────────────────────────────────

def test_telemetry_defaults_to_none():
    seg = SRTSegment(1, 0.0, 1.0, "hi")
    assert seg.avg_logprob is None
    assert seg.chunk_index is None
    assert seg.is_chunk_tail is False


def test_duration_uses_model_timings():
    assert SRTSegment(1, 10.0, 10.16, "x").duration == pytest.approx(0.16)


def test_is_chunk_tail():
    tail = SRTSegment(1, 0, 1, "x", chunk_position=3, chunk_n_segments=3)
    mid = SRTSegment(2, 0, 1, "x", chunk_position=2, chunk_n_segments=3)
    assert tail.is_chunk_tail and not mid.is_chunk_tail


def test_sole_segment_is_distinguished_from_tail():
    """A one-segment chunk is both head and tail; conflating them would
    inflate any measurement of tail behaviour."""
    sole = SRTSegment(1, 0, 1, "x", chunk_position=1, chunk_n_segments=1)
    assert sole.is_chunk_tail and sole.is_chunk_sole


def test_from_dict_tolerates_unknown_and_missing_keys():
    seg = SRTSegment.from_dict(
        {"index": 1, "start": 0.0, "end": 1.0, "text": "x", "future_field": 9}
    )
    assert seg.text == "x"
    assert seg.avg_logprob is None


def test_roundtrip_preserves_telemetry():
    seg = SRTSegment(1, 0, 1, "x", avg_logprob=-1.4, temperature=0.8,
                     chunk_index=3, chunk_position=2, chunk_n_segments=2)
    assert SRTSegment.from_dict(seg.to_dict()) == seg


# ── Engine population ─────────────────────────────────────────────────────

class _Seg:
    def __init__(self, text, start, end, logprob=-0.3, cr=1.2, nsp=0.01, temp=0.0):
        self.text, self.start, self.end = text, start, end
        self.avg_logprob, self.compression_ratio = logprob, cr
        self.no_speech_prob, self.temperature = nsp, temp


class _Info:
    language = "ml"


def test_long_form_engine_carries_metrics_without_chunk_fields():
    segs = segments_from_whisper([_Seg("hello", 0.0, 2.0, logprob=-0.42)])
    assert segs[0].avg_logprob == pytest.approx(-0.42)
    # No chunk plan exists in long-form mode, so there is no such thing as a
    # chunk tail — the fields must stay unset rather than defaulting to 1.
    assert segs[0].chunk_index is None
    assert segs[0].is_chunk_tail is False


def test_missing_attributes_degrade_to_none():
    class Bare:
        text, start, end = "hi", 0.0, 1.0

    assert segments_from_whisper([Bare()])[0].avg_logprob is None


class _Model:
    """Chunk 1 emits two segments; chunk 2 emits one."""
    def __init__(self):
        self.calls = 0

    def transcribe(self, path, **kw):
        self.calls += 1
        if self.calls == 1:
            return iter([
                _Seg("real speech here", 0.2, 3.0, logprob=-0.25),
                _Seg("ആരോപിച്ചു", 3.4, 3.56, logprob=-1.55, cr=3.4, temp=0.8),
            ]), _Info()
        return iter([_Seg("more speech", 0.1, 3.2, logprob=-0.30)]), _Info()


def _run_two_chunks():
    audio = np.zeros(16000 * 12, dtype=np.float32)
    chunks = [{"start_s": 0.0, "end_s": 4.0}, {"start_s": 4.0, "end_s": 8.0}]
    return MonolingualWhisperEngine()._transcribe_chunks(
        _Model(), audio, 16000, chunks, "ml", Path("clip.wav"),
        config=TranscriptionConfig(device="cpu"),
    )[0]


def test_chunked_engine_records_provenance_and_metrics():
    segs = _run_two_chunks()
    assert len(segs) == 3

    assert [s.chunk_index for s in segs] == [1, 1, 2]
    assert [s.chunk_position for s in segs] == [1, 2, 1]
    assert [s.chunk_n_segments for s in segs] == [2, 2, 1]
    assert [s.is_chunk_tail for s in segs] == [False, True, True]
    assert segs[2].is_chunk_sole


def test_the_suspect_fragment_is_identifiable_without_duration():
    """The whole point of Drop A.

    The short fragment is flagged by confidence, repetition and chunk
    position — three signals that survive post-processing, unlike duration,
    which build_srt rewrites.
    """
    fragment = _run_two_chunks()[1]
    assert fragment.avg_logprob == pytest.approx(-1.55)
    assert fragment.compression_ratio == pytest.approx(3.4)
    assert fragment.temperature == pytest.approx(0.8)
    assert fragment.is_chunk_tail
    assert not fragment.is_chunk_sole


def test_timestamps_are_absolute_not_chunk_local():
    segs = _run_two_chunks()
    assert segs[2].start == pytest.approx(4.1)


def test_indices_are_monotonic_across_chunks():
    assert [s.index for s in _run_two_chunks()] == [1, 2, 3]


# ── Translation must not discard telemetry ────────────────────────────────

def test_translation_preserves_diagnostics():
    from dataclasses import replace

    seg = SRTSegment(1, 0, 1, "ആരോപിച്ചു", avg_logprob=-1.5, chunk_index=4,
                     chunk_position=2, chunk_n_segments=2)
    translated = replace(seg, text="alleged")
    assert translated.text == "alleged"
    assert translated.avg_logprob == pytest.approx(-1.5)
    assert translated.is_chunk_tail


# ── CSV dump ──────────────────────────────────────────────────────────────

def test_dump_writes_expected_columns(tmp_path):
    out = tmp_path / "clip.segments.csv"
    write_segment_dump(_run_two_chunks(), out)

    with out.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    assert list(rows[0].keys()) == COLUMNS
    assert len(rows) == 3
    assert rows[1]["avg_logprob"] == "-1.55"
    assert rows[1]["is_chunk_tail"] == "1"
    assert rows[2]["is_chunk_sole"] == "1"


def test_dump_records_true_durations(tmp_path):
    """Durations here must be the model's, not build_srt's floored values."""
    out = tmp_path / "d.csv"
    write_segment_dump([SRTSegment(1, 10.0, 10.16, "x")], out)
    with out.open(encoding="utf-8-sig") as fh:
        assert float(next(csv.DictReader(fh))["duration_s"]) == pytest.approx(0.16)


def test_dump_preserves_malayalam_text(tmp_path):
    out = tmp_path / "ml.csv"
    write_segment_dump([SRTSegment(1, 0, 1, "ആരോപിച്ചു")], out)
    with out.open(encoding="utf-8-sig") as fh:
        assert next(csv.DictReader(fh))["text"] == "ആരോപിച്ചു"


def test_dump_failure_does_not_raise(tmp_path):
    write_segment_dump([SRTSegment(1, 0, 1, "x")], tmp_path / "no" / "\0bad")


def test_dump_warns_when_no_metrics_present(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        write_segment_dump([SRTSegment(1, 0, 1, "x")], tmp_path / "x.csv")
    assert "decoder metrics" in caplog.text
