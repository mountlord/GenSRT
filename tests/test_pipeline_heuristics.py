"""End-to-end wiring of the heuristics stages inside run_pipeline.

Audio extraction, ASR and the translation engine are stubbed; everything
between them — list drops, density, collapse, translation, subject
stripping, the report — runs for real.  This exists because the unit tests
prove each rule and this proves the pipeline actually calls them in the
right order with the right inputs.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gensrt.models import SRTSegment, TranscriptionConfig


def _seg(i, start, end, text):
    return SRTSegment(index=i, start=start, end=end, text=text)


_JA_TO_EN = {
    "ごめん": "I'm sorry.",
    "私は何も知りません": "I don't know anything.",
    "気持ちいい": "It feels good.",
    "早くしていきますよ": "I'm going to make it faster.",
    "何ですか?": "What is it?",
}


class _Engine:
    def translate_batch(self, texts, src, tgt):
        return [_JA_TO_EN.get(t, t) for t in texts]


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    import gensrt.pipeline as pl
    from gensrt.audio import extractor
    from gensrt.translation import factory

    wav = tmp_path / "x.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(extractor, "extract_audio", lambda p: wav)
    monkeypatch.setattr(pl, "ensure_translation_model", lambda c, status=None: c)
    monkeypatch.setattr(factory, "get_engine", lambda key, cfg: _Engine())

    # ASR output: 5 dense ごめん, an explicit-subject line, content, dialogue
    raw = [_seg(i, 2.0 * i, 2.0 * i + 1, "ごめん") for i in range(1, 6)]
    raw += [_seg(6, 20, 23, "私は何も知りません"), _seg(7, 25, 26, "気持ちいい"),
            _seg(8, 28, 30, "早くしていきますよ"), _seg(9, 32, 33, "ああ"),
            _seg(10, 40, 42, "何ですか?")]
    monkeypatch.setattr(pl, "_run_asr", lambda wav_path, config, status=None: (list(raw), "ja"))
    return pl, raw, tmp_path


def _config(**kw):
    base = dict(
        model="large-v3", device="cpu", compute_type="int8",
        translate=True, translation_engine="nllb", target_language="en",
        source_language="ja",
    )
    return TranscriptionConfig(**{**base, **kw})


def test_pipeline_runs_all_three_rules_and_reports(stubbed):
    pl, raw, tmp_path = stubbed
    out = tmp_path / "out.srt"
    result = pl.run_pipeline(Path("movie.mp4"), out, _config())

    texts = [s.text for s in result.segments]
    # ごめん: hallucination list; ああ: drop list — gone before translation.
    assert "I'm sorry." not in texts and "Sorry." not in texts
    # Subject stripped only where the source names none.
    assert "Feels good." in texts and "Going to make it faster." in texts
    assert "I don't know anything." in texts            # 私は … keeps its subject
    assert "What is it?" in texts

    rep = result.heuristics_report
    assert rep is not None
    by = {r.source: r for r in rep.rows}
    assert by["ごめん"].reason == "hallucination" and by["ごめん"].dropped == 5
    assert by["ああ"].reason == "drop"
    assert rep.subject_stripped == 2
    assert ("It feels good.", "Feels good.") in rep.subject_examples
    assert out.exists()


def test_density_catches_what_the_lists_do_not(stubbed, tmp_path, monkeypatch):
    """Empty the ja hallucination list: the five ごめん in ten seconds still go,
    this time by the density rule, and the report says so."""
    import json

    pl, raw, _ = stubbed
    rules = tmp_path / "gensrt-heuristics.json"
    rules.write_text(json.dumps({"interjections": {"languages": {"ja": {
        "hallucination": [], "drop": ["ああ"]}}}}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = pl.run_pipeline(Path("movie.mp4"), tmp_path / "o.srt", _config())
    row = {r.source: r for r in result.heuristics_report.rows}["ごめん"]
    assert row.reason == "density" and row.dropped == 5


def test_subject_rule_does_not_run_without_translation(stubbed, tmp_path):
    pl, raw, _ = stubbed
    result = pl.run_pipeline(Path("movie.mp4"), tmp_path / "o.srt",
                             _config(translate=False))
    assert "気持ちいい" in [s.text for s in result.segments]
    assert result.heuristics_report.subject_stripped == 0


def test_pipeline_writes_the_report_when_a_dir_is_configured(stubbed, tmp_path):
    pl, raw, _ = stubbed
    out_dir = tmp_path / "reports"
    pl.run_pipeline(Path("movie.mp4"), tmp_path / "o.srt",
                    _config(heuristics_report_dir=str(out_dir)))
    assert (out_dir / "movie.heuristics.txt").exists()
    assert (out_dir / "movie.heuristics.json").exists()
