"""Output files: which formats are written, and the source-language
subtitles written beside the translation from the same cues."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gensrt.models import SRTSegment, TranscriptionConfig
from gensrt.srt.builder import build_srt, write_outputs
from gensrt.utils.media_files import source_output_path


@pytest.mark.parametrize("out, src, want", [
    ("movie.srt", "ja", "movie.ja.srt"),           # English target: unsuffixed main file
    ("movie.ko.srt", "ja", "movie.ja.srt"),        # non-English target: tag replaced
    ("run7.srt", "ja", "run7.ja.srt"),             # --output-filename
    ("movie.nllb.srt", "ja", "movie.nllb.ja.srt"), # 4-letter tail is not a language tag
    ("a.b.en.srt", "ml", "a.b.ml.srt"),
    ("movie.srt", "JA", "movie.ja.srt"),
])
def test_source_output_path(out, src, want):
    assert source_output_path(Path("/x") / out, src) == Path("/x") / want


def _subs():
    return build_srt([SRTSegment(1, 0.0, 1.0, "a"), SRTSegment(2, 2.0, 3.5, "b")],
                     max_duration_s=4, min_duration_s=0, max_line_chars=42, max_lines=2)


@pytest.mark.parametrize("fmt, names", [
    ("srt", ["o.srt"]), ("vtt", ["o.vtt"]), ("both", ["o.srt", "o.vtt"]),
    ("", ["o.srt"]), ("bogus", ["o.srt"]),
])
def test_write_outputs_formats(tmp_path, fmt, names):
    written = write_outputs(_subs(), tmp_path / "o.srt", fmt)
    assert [p.name for p in written] == names
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(names)
    if "o.vtt" in names:
        assert (tmp_path / "o.vtt").read_text(encoding="utf-8").startswith("WEBVTT")


# ── through the pipeline ───────────────────────────────────────────────────

_JA = {"大丈夫じゃないか": "Are you okay?", "行きます": "I'm going."}


class _Engine:
    def translate_batch(self, texts, src, tgt):
        return [_JA.get(t, t) for t in texts]


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
    # get_shared_engine caches per process; a stub left behind here would
    # translate for the next test file.
    factory.clear_shared_engines()
    cues = [SRTSegment(1, 1.0, 3.0, "大丈夫じゃないか"), SRTSegment(2, 10.0, 12.0, "行きます")]
    monkeypatch.setattr(pl, "_run_asr", lambda wav_path, config, status=None: (list(cues), "ja"))
    yield pl, tmp_path
    factory.clear_shared_engines()


def _cfg(**kw):
    base = dict(model="large-v3", device="cpu", compute_type="int8",
                translation_engine="nllb", target_language="en",
                source_language="ja", min_subtitle_duration_s=0)
    return TranscriptionConfig(**{**base, **kw})


def _texts(p: Path) -> list[str]:
    import srt
    return [s.content for s in srt.parse(p.read_text(encoding="utf-8"))]


def test_source_srt_is_written_from_the_same_cues(stubbed):
    pl, tmp = stubbed
    out = tmp / "movie.srt"
    pl.run_pipeline(Path("movie.mp4"), out, _cfg(source_srt=True))
    assert _texts(out) == ["Are you okay?", "Going."]          # translated (+ subject rule)
    src = tmp / "movie.ja.srt"
    assert src.exists()
    assert _texts(src) == ["大丈夫じゃないか", "行きます"]
    # identical timings, cue for cue
    import srt
    a = list(srt.parse(out.read_text(encoding="utf-8")))
    b = list(srt.parse(src.read_text(encoding="utf-8")))
    assert [(s.start, s.end) for s in a] == [(s.start, s.end) for s in b]
    assert not (tmp / "movie.vtt").exists()                     # default formats: srt only


def test_source_srt_writes_nothing_extra_without_a_translation(stubbed, caplog):
    import logging

    pl, tmp = stubbed
    out = tmp / "movie.srt"
    caplog.set_level(logging.INFO, logger="gensrt.pipeline")
    pl.run_pipeline(Path("movie.mp4"), out, _cfg(source_srt=True, translate=False))
    assert _texts(out) == ["大丈夫じゃないか", "行きます"]
    assert not (tmp / "movie.ja.srt").exists()
    assert "already holds the source language" in caplog.text


def test_source_srt_off_by_default_and_formats_both(stubbed):
    pl, tmp = stubbed
    out = tmp / "movie.srt"
    pl.run_pipeline(Path("movie.mp4"), out, _cfg(subtitle_formats="both"))
    assert out.exists() and (tmp / "movie.vtt").exists()
    assert not (tmp / "movie.ja.srt").exists()
    pl.run_pipeline(Path("movie.mp4"), tmp / "v.srt", _cfg(subtitle_formats="vtt", source_srt=True))
    assert not (tmp / "v.srt").exists() and (tmp / "v.vtt").exists() and (tmp / "v.ja.vtt").exists()


def test_non_english_target_names_both_files_by_language(stubbed):
    pl, tmp = stubbed
    out = tmp / "movie.ko.srt"
    pl.run_pipeline(Path("movie.mp4"), out, _cfg(source_srt=True, target_language="ko"))
    assert out.exists() and (tmp / "movie.ja.srt").exists()


# ── CLI and config plumbing ────────────────────────────────────────────────

def test_cli_flags_leave_the_config_file_in_charge_when_absent(tmp_path):
    from gensrt.cli import _build_parser, _resolve_settings

    cfg = tmp_path / "gensrt-config.json"
    cfg.write_text(json.dumps({"source_srt": True, "subtitle_formats": "both"}))
    m = _resolve_settings(_build_parser().parse_args(["--config", str(cfg)]))
    assert m["source_srt"] is True and m["subtitle_formats"] == "both"
    m = _resolve_settings(_build_parser().parse_args(["--config", str(cfg), "--formats", "srt"]))
    assert m["subtitle_formats"] == "srt"
    cfg.write_text("{}")
    m = _resolve_settings(_build_parser().parse_args(["--config", str(cfg), "--source-srt"]))
    assert m["source_srt"] is True and m["subtitle_formats"] == "srt"
    m = _resolve_settings(_build_parser().parse_args(["--config", str(cfg)]))
    assert m["source_srt"] is False


def test_editor_save_writes_vtt_only_when_formats_say_so(tmp_path, monkeypatch):
    from gensrt.api import media
    from gensrt.server import app

    app.config["TESTING"] = True
    c = app.test_client()
    body = {"path": str(tmp_path / "e.srt"), "segments": [{"start_time": 0, "end_time": 1, "text": "x"}]}
    monkeypatch.setattr(media, "read_config_file", lambda **kw: {"subtitle_formats": "srt"})
    r = c.post("/api/srt", json=body).get_json()
    assert r["vtt_path"] is None and not (tmp_path / "e.vtt").exists()
    monkeypatch.setattr(media, "read_config_file", lambda **kw: {"subtitle_formats": "both"})
    r = c.post("/api/srt", json=body).get_json()
    assert r["vtt_path"] and (tmp_path / "e.vtt").exists()


# ── --translate-srt ────────────────────────────────────────────────────────

def _write_ja_srt(path: Path):
    path.write_text("1\n00:00:01,000 --> 00:00:03,000\n大丈夫じゃないか\n\n"
                    "2\n00:00:10,000 --> 00:00:12,000\n行きます\n\n", encoding="utf-8")


def test_translate_srt_keeps_timings_and_names_the_output(stubbed):
    pl, tmp = stubbed
    src = tmp / "movie.ja.srt"
    _write_ja_srt(src)
    res = pl.translate_srt(src, _cfg())
    assert res.output_path == tmp / "movie.srt" and res.detected_language == "ja"
    assert _texts(res.output_path) == ["Are you okay?", "Going."]
    import srt
    a = list(srt.parse(src.read_text(encoding="utf-8")))
    b = list(srt.parse(res.output_path.read_text(encoding="utf-8")))
    assert [(s.start, s.end) for s in a] == [(s.start, s.end) for s in b]
    # non-English target is tagged; explicit output honoured
    res = pl.translate_srt(src, _cfg(target_language="ko"))
    assert res.output_path == tmp / "movie.ko.srt"
    res = pl.translate_srt(src, _cfg(), output_path=tmp / "movie.nllb.srt")
    assert (tmp / "movie.nllb.srt").exists()


def test_translate_srt_language_comes_from_the_tag_or_the_flag(stubbed):
    from gensrt.exceptions import ConfigError

    pl, tmp = stubbed
    untagged = tmp / "movie.srt"
    _write_ja_srt(untagged)
    with pytest.raises(ConfigError, match="language"):
        pl.translate_srt(untagged, _cfg(source_language="auto"))
    pl.translate_srt(untagged, _cfg(source_language="ja"), output_path=tmp / "out.srt")
    assert _texts(tmp / "out.srt") == ["Are you okay?", "Going."]
    with pytest.raises(ConfigError, match="already in the target language"):
        pl.translate_srt(untagged, _cfg(source_language="en"))
    with pytest.raises(ConfigError, match="engine"):
        pl.translate_srt(untagged, _cfg(source_language="ja", translation_engine="none"))


def test_translate_srt_cli(stubbed, tmp_path, monkeypatch, capsys):
    from gensrt import cli

    pl, tmp = stubbed
    src = tmp / "movie.ja.srt"
    _write_ja_srt(src)
    cfg = tmp / "gensrt-config.json"
    cfg.write_text(json.dumps({"translation_engine": "nllb", "device": "cpu", "compute_type": "int8"}))
    with pytest.raises(SystemExit) as ei:
        cli.main(["--config", str(cfg), "--translate-srt", str(src), "--output-filename", "movie.nllb.srt"])
    assert ei.value.code == 0, capsys.readouterr().err
    assert _texts(tmp / "movie.nllb.srt") == ["Are you okay?", "Going."]
    with pytest.raises(SystemExit) as ei:
        cli.main(["--config", str(cfg), "--translate-srt", str(tmp / "missing.ja.srt")])
    assert ei.value.code == 1
