"""Fixed-window onset snapping (gensrt.asr._onset).

The core is a pure function over chunk-local spans and speech regions, so
these tests need no VAD model.  The measured pathology it targets: Whisper
stamping a lone short utterance 0.00→2.00 wherever it sits in a 5-8 s
window — 560 of 1,293 cues on one file.
"""

from __future__ import annotations

from gensrt.asr._onset import ONSET_LEAD_S, OnsetStats, snap_to_onsets


def test_lone_short_cue_at_window_start_moves_to_the_onset():
    spans, st = snap_to_onsets([(0.0, 2.0)], [(3.5, 4.6)], 6.0)
    (start, end), = spans
    assert abs(start - (3.5 - ONSET_LEAD_S)) < 1e-9
    assert abs(end - (2.0 + start)) < 1e-9          # duration preserved
    assert st.snapped == 1 and abs(st.shifts_s[0] - start) < 1e-9


def test_cue_already_inside_speech_is_left_alone():
    spans, st = snap_to_onsets([(1.2, 3.0)], [(1.0, 3.4)], 6.0)
    assert spans == [(1.2, 3.0)] and st.snapped == 0


def test_lead_window_counts_as_inside():
    """A start a hair before the region (within the lead) is the model
    being right; do not push it later."""
    spans, st = snap_to_onsets([(0.95, 2.0)], [(1.0, 2.5)], 6.0)
    assert spans == [(0.95, 2.0)] and st.snapped == 0


def test_no_regions_means_no_change():
    spans, st = snap_to_onsets([(0.0, 2.0)], [], 6.0)
    assert spans == [(0.0, 2.0)] and st.snapped == 0 and st.segments == 1


def test_onset_only_before_the_start_is_not_used():
    """Starts never move earlier."""
    spans, st = snap_to_onsets([(3.0, 5.0)], [(0.5, 1.0)], 6.0)
    assert spans == [(3.0, 5.0)] and st.snapped == 0


def test_end_is_clamped_to_the_chunk():
    spans, _ = snap_to_onsets([(0.0, 2.0)], [(5.5, 6.0)], 6.0)
    (start, end), = spans
    assert end == 6.0 and start < end


def test_multiple_segments_each_snap_within_their_own_window():
    """Segment 1 may only use onsets before segment 2 starts, and may not
    grow into it."""
    spans, st = snap_to_onsets([(0.0, 2.0), (4.0, 6.0)], [(1.5, 2.2), (4.0, 5.8)], 7.0)
    assert st.snapped == 1
    s1, s2 = spans
    assert abs(s1[0] - (1.5 - ONSET_LEAD_S)) < 1e-9 and s1[1] <= 4.0
    assert s2 == (4.0, 6.0)


def test_second_segment_uses_an_onset_in_its_own_window():
    spans, st = snap_to_onsets([(0.0, 1.0), (2.0, 3.0)], [(2.5, 3.0)], 6.0)
    assert spans[0] == (0.0, 1.0)            # nothing voiced in [0, 2) → unchanged
    assert st.snapped == 1 and abs(spans[1][0] - (2.5 - ONSET_LEAD_S)) < 1e-9


def test_stats_summary_reads_as_a_log_line():
    st = OnsetStats(segments=10, snapped=3, shifts_s=[1.0, 2.5, 3.0])
    assert st.summary() == ("Onset snap: 3 of 10 cues moved to the audible onset "
                            "(median +2.5 s, max +3.0 s)")
    assert OnsetStats(segments=4).summary() == "Onset snap: 0 of 4 cues moved"


# ── Engine wiring ─────────────────────────────────────────────────────────

class _Seg:
    def __init__(self, text, start, end):
        self.text, self.start, self.end = text, start, end
        self.avg_logprob = self.compression_ratio = self.no_speech_prob = self.temperature = None


class _Info:
    language = "ja"


class _Model:
    """Every chunk: one lone cue stamped 0.00→2.00, the measured pathology."""
    def transcribe(self, path, **kw):
        return iter([_Seg("ごめん", 0.0, 2.0)]), _Info()


def _run(monkeypatch, config, regions):
    import numpy as np
    from pathlib import Path
    import gensrt.asr.monolingual_whisper as mw

    calls = []
    monkeypatch.setattr(mw, "speech_regions",
                        lambda audio, sr: (calls.append(len(audio) / sr), regions)[1])
    audio = np.zeros(16000 * 12, dtype=np.float32)
    chunks = [{"start_s": 0.0, "end_s": 6.0}, {"start_s": 6.0, "end_s": 12.0}]
    segs = mw.MonolingualWhisperEngine()._transcribe_chunks(
        _Model(), audio, 16000, chunks, "ja", Path("clip.wav"), config=config,
    )[0]
    return segs, calls


def test_fixed_mode_snaps_each_chunk_and_offsets_by_chunk_start(monkeypatch, caplog):
    import logging
    from gensrt.models import TranscriptionConfig

    with caplog.at_level(logging.INFO, logger="gensrt.asr.monolingual_whisper"):
        segs, calls = _run(monkeypatch, TranscriptionConfig(device="cpu", chunk_mode="fixed"),
                           regions=[(3.5, 4.6)])
    assert calls == [6.0, 6.0]                       # chunk-local audio, both chunks
    assert [round(s.start, 2) for s in segs] == [3.4, 9.4]
    assert [round(s.end, 2) for s in segs] == [5.4, 11.4]
    assert any("Onset snap: 2 of 2 cues moved" in r.message for r in caplog.records)


def test_vad_mode_and_opt_out_leave_timestamps_alone(monkeypatch):
    from gensrt.models import TranscriptionConfig

    for cfg in (TranscriptionConfig(device="cpu", chunk_mode="vad"),
                TranscriptionConfig(device="cpu", chunk_mode="fixed", snap_onsets=False)):
        segs, calls = _run(monkeypatch, cfg, regions=[(3.5, 4.6)])
        assert calls == []
        assert [s.start for s in segs] == [0.0, 6.0]


def test_a_failing_vad_never_fails_the_chunk(monkeypatch):
    from gensrt.models import TranscriptionConfig
    import gensrt.asr.monolingual_whisper as mw

    def boom(audio, sr):
        raise RuntimeError("onnx exploded")
    monkeypatch.setattr(mw, "speech_regions", boom)
    import numpy as np
    from pathlib import Path
    segs = mw.MonolingualWhisperEngine()._transcribe_chunks(
        _Model(), np.zeros(16000 * 6, dtype=np.float32), 16000,
        [{"start_s": 0.0, "end_s": 6.0}], "ja", Path("clip.wav"),
        config=TranscriptionConfig(device="cpu", chunk_mode="fixed"),
    )[0]
    assert [s.start for s in segs] == [0.0]


def test_cli_and_gui_expose_the_switch():
    from gensrt.cli import _build_parser
    from gensrt.server import _validate_config_patch

    assert _build_parser().parse_args(["--input", "v.mkv"]).snap_onsets is None
    assert _build_parser().parse_args(["--input", "v.mkv", "--no-snap-onsets"]).snap_onsets is False
    assert _validate_config_patch({"snap_onsets": False}) == ({"snap_onsets": False}, {})
