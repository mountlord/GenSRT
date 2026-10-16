"""Profiles: named default sets between the config file and explicit flags."""

from __future__ import annotations

import json

import pytest

from gensrt import profiles as P
from gensrt.config import BUILTIN_DEFAULTS
from gensrt.exceptions import ConfigError


def test_builtin_profiles_only_set_real_config_keys_and_no_paths():
    for name, d in P.BUILTIN_PROFILES.items():
        assert d["description"]
        for k in d["settings"]:
            assert k in BUILTIN_DEFAULTS, (name, k)
            assert k not in P._NOT_IN_PROFILE, (name, k)


def test_the_two_shipped_profiles_match_the_two_kinds_of_material():
    b = P.BUILTIN_PROFILES["broadcast"]["settings"]
    s = P.BUILTIN_PROFILES["soft-speech"]["settings"]
    assert b["model"] == "large-v3-turbo" and b["chunk_mode"] == "vad"
    assert s["model"] == "large-v3-turbo" and s["chunk_mode"] == "fixed" and s["snap_onsets"] is True
    # Neither profile fixes the language: soft speech is a property of the
    # material, and the language picks the model.
    assert "source_language" not in b and "source_language" not in s
    assert P.BUILTIN_PROFILES["soft-speech"]["model_by_language"]["ja"].startswith("kotoba-tech/")


def test_unknown_profile_names_the_available_ones():
    with pytest.raises(ConfigError) as ei:
        P.load_profiles(path=None).get("brodcast")
    assert "brodcast" in str(ei.value) and "broadcast" in str(ei.value) and "soft-speech" in str(ei.value)


def test_user_file_adds_and_overrides_by_name(tmp_path):
    f = tmp_path / "gensrt-profiles.json"
    f.write_text(json.dumps({"profiles": {
        "Korean-Drama": {"description": "my set", "settings": {"model": "large-v3", "source_language": "ko"}},
        "broadcast": {"settings": {"chunk_mode": "fixed"}},
    }}), encoding="utf-8")
    ps = P.load_profiles(f)
    assert ps.names() == ["broadcast", "korean-drama", "soft-speech"]
    assert ps.get("korean-drama").settings == {"model": "large-v3", "source_language": "ko"}
    assert ps.get("KOREAN-DRAMA").source == str(f)
    assert ps.get("broadcast").settings == {"chunk_mode": "fixed"}   # replaced, not merged


def test_no_shipped_profile_names_the_translation_engine():
    """The engine is decided by VRAM and licence — per machine, in the
    config file.  A profile that set it silently undid a config edit
    (2026-10-07), so neither shipped profile may name it, and user
    profiles are refused if they try."""
    for name, d in P.BUILTIN_PROFILES.items():
        assert "translation_engine" not in d["settings"], name


@pytest.mark.parametrize("settings, needle", [
    ({"modle": "x"}, "modle"),                       # typo
    ({"output": "D:/out"}, "per machine"),           # a path
    ({"device": "cuda"}, "per machine"),             # a machine setting
    ({"translation_engine": "nllb"}, "translation engine"),
    ("large-v3", "must be an object"),
])
def test_bad_profile_settings_are_errors_not_silence(tmp_path, settings, needle):
    f = tmp_path / "gensrt-profiles.json"
    f.write_text(json.dumps({"profiles": {"p": {"settings": settings}}}), encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        P.load_profiles(f)
    assert needle in str(ei.value)


def test_malformed_profiles_file_is_an_error(tmp_path):
    f = tmp_path / "gensrt-profiles.json"
    f.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError):
        P.load_profiles(f)
    f.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(ConfigError):
        P.load_profiles(f)


def test_apply_profile_fills_defaults_and_keeps_what_the_user_set():
    """A profile fills in; it never overrules a value set in the file."""
    file_cfg = {"model": "medium", "chunk_mode": "vad", "vad_threshold": 0.4, "translation_engine": "nllb"}
    out, rec = P.apply_profile(file_cfg, "soft-speech")
    assert rec.profile.name == "soft-speech"
    assert out["model"] == "medium"                      # user-set: kept
    assert out["chunk_mode"] == "fixed"                  # file had the default: filled
    assert out["snap_onsets"] is True                    # absent from file: filled
    assert out["translation_engine"] == "nllb"           # profile does not name it
    assert out["vad_threshold"] == 0.4                   # untouched
    assert rec.kept == {"model": "medium"}
    assert rec.applied == {"asr_engine": "chunked", "chunk_mode": "fixed", "snap_onsets": True,
                           "heuristics_mode": "soft-speech"}
    assert "kept from gensrt-config.json: model=medium" in rec.summary()
    assert file_cfg["model"] == "medium"                 # input not mutated
    out, rec = P.apply_profile(file_cfg, "")
    assert out == file_cfg and rec.profile is None and rec.summary() == ""
    assert P.apply_profile(file_cfg, None)[1].profile is None


def test_precedence_defaults_profile_file_flags(tmp_path, monkeypatch):
    """built-in < profile < values set in gensrt-config.json < explicit flag."""
    from gensrt.cli import _build_parser, _resolve_settings

    cfg = tmp_path / "gensrt-config.json"
    cfg.write_text(json.dumps({"model": "medium", "chunk_mode": "vad", "max_chunk_s": 8.0}))
    monkeypatch.chdir(tmp_path)

    args = _build_parser().parse_args(["--config", str(cfg), "--profile", "soft-speech", "--model", "large-v3"])
    m = _resolve_settings(args)
    assert m["model"] == "large-v3"             # flag beats everything
    assert m["chunk_mode"] == "fixed"           # file had the default → profile fills it
    assert m["max_chunk_s"] == 8.0              # file value, profile does not name it
    assert m["profile"] == "soft-speech"

    # A model the user set in the file beats the profile and its
    # per-language map; the note says so.
    m = _resolve_settings(_build_parser().parse_args(["--config", str(cfg), "--profile", "soft-speech",
                                                      "--source-language", "ja"]))
    assert m["model"] == "medium" and "kept from gensrt-config.json: model=medium" in m["_profile_note"]

    # The config file can name a default profile; --profile overrides it.
    cfg.write_text(json.dumps({"profile": "soft-speech"}))
    m = _resolve_settings(_build_parser().parse_args(["--config", str(cfg)]))
    assert m["chunk_mode"] == "fixed" and m["profile"] == "soft-speech"
    m = _resolve_settings(_build_parser().parse_args(["--config", str(cfg), "--profile", "broadcast"]))
    assert m["chunk_mode"] == "vad" and m["model"] == "large-v3-turbo"


def test_the_engine_in_the_config_file_is_what_runs(tmp_path):
    """The 2026-10-07 case: the file says nllb, the file names the
    soft-speech profile, the user expects nllb — and gets it."""
    from gensrt.cli import _build_parser, _resolve_settings

    cfg = tmp_path / "gensrt-config.json"
    cfg.write_text(json.dumps({"profile": "soft-speech", "translation_engine": "nllb"}))
    m = _resolve_settings(_build_parser().parse_args(["--config", str(cfg), "--source-language", "ja"]))
    assert m["translation_engine"] == "nllb"
    assert m["model"].startswith("kotoba-tech/") and m["chunk_mode"] == "fixed"
    # and the explicit flag still wins over the file
    m = _resolve_settings(_build_parser().parse_args(["--config", str(cfg), "--source-language", "ja",
                                                      "--translation-engine", "madlad"]))
    assert m["translation_engine"] == "madlad"


def test_soft_speech_picks_the_model_from_the_language(tmp_path):
    """The misuse this replaces: kotoba fed Korean.  Now the language picks
    the model — kotoba for ja, large-v3-turbo for anything else — and the
    guard never has to fire."""
    from gensrt.cli import _build_parser, _resolve_settings
    from gensrt.operations import build_transcription_config
    from gensrt.pipeline import validate_model_language

    cfg = tmp_path / "gensrt-config.json"
    cfg.write_text("{}")

    def resolve(*flags):
        return _resolve_settings(_build_parser().parse_args(["--config", str(cfg), "--profile", "soft-speech", *flags]))

    ja = resolve("--source-language", "ja")
    assert ja["model"] == "kotoba-tech/kotoba-whisper-v2.0-faster" and ja["chunk_mode"] == "fixed"
    ko = resolve("--source-language", "ko")
    assert ko["model"] == "large-v3-turbo" and ko["chunk_mode"] == "fixed"
    validate_model_language(build_transcription_config(ko, auto_detect_backend=False))   # no error
    assert resolve()["model"] == "large-v3-turbo"                                        # auto-detect: safe model
    assert resolve("--source-language", "ja", "--model", "large-v3")["model"] == "large-v3"   # explicit wins
    # The file's language counts too.
    cfg.write_text(json.dumps({"source_language": "ja"}))
    assert resolve()["model"].startswith("kotoba-tech/")


def test_user_profile_model_by_language_is_validated(tmp_path):
    f = tmp_path / "gensrt-profiles.json"
    f.write_text(json.dumps({"profiles": {"p": {"settings": {}, "model_by_language": {"ml": 3}}}}), encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        P.load_profiles(f)
    assert "model_by_language" in str(ei.value)
    f.write_text(json.dumps({"profiles": {"p": {"settings": {}, "model_by_language": {"ML": "adalat-ai/ct2-whisper-medium-ml-rmft"}}}}), encoding="utf-8")
    prof = P.load_profiles(f).get("p")
    assert prof.model_for("ml-IN") == "adalat-ai/ct2-whisper-medium-ml-rmft"
    assert prof.model_for("ko") is None


def test_list_profiles_and_init_profiles(tmp_path, monkeypatch, capsys):
    from gensrt import cli

    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as ei:
        cli.main(["--list-profiles"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    assert "broadcast" in out and "soft-speech" in out and "model = large-v3-turbo" in out

    monkeypatch.setattr("gensrt.model_paths.sidecar_dir", lambda: tmp_path)
    with pytest.raises(SystemExit):
        cli.main(["--init-profiles"])
    written = json.loads((tmp_path / "gensrt-profiles.json").read_text(encoding="utf-8"))
    assert set(written["profiles"]) == {"broadcast", "soft-speech"}
    assert P.load_profiles(tmp_path / "gensrt-profiles.json").names() == ["broadcast", "soft-speech"]


def test_dump_config_shows_the_profile(tmp_path, monkeypatch, capsys):
    from gensrt import cli

    cfg = tmp_path / "gensrt-config.json"
    cfg.write_text("{}")
    with pytest.raises(SystemExit) as ei:
        cli.main(["--config", str(cfg), "--profile", "broadcast", "--dump-config"])
    assert ei.value.code == 0
    out = capsys.readouterr().out
    dumped = json.loads(out[out.index("{"):])      # log lines may precede the JSON
    assert dumped["profile"] == "broadcast" and dumped["model"] == "large-v3-turbo"


def test_api_transcribe_applies_a_profile(tmp_path, monkeypatch):
    from gensrt.api import jobs
    from gensrt.server import app

    video = tmp_path / "v.mp4"
    video.write_bytes(b"\x00")
    seen = {}

    class R:
        translation_error = None

    def fake_run(**kw):
        seen["config"] = kw["config"]
        return R()
    monkeypatch.setattr(jobs, "run_transcription", fake_run)
    monkeypatch.setattr(jobs, "read_config_file", lambda **kw: {"vad_threshold": 0.4})
    app.config["TESTING"] = True
    c = app.test_client()
    r = c.post("/api/transcribe", json={"input_path": str(video), "profile": "broadcast", "no_translate": True})
    assert r.status_code == 200, r.get_json()
    assert seen["config"].model == "large-v3-turbo" and seen["config"].chunk_mode == "vad"
    r = c.post("/api/transcribe", json={"input_path": str(video), "profile": "soft-speech",
                                        "source_language": "ja", "no_translate": True})
    assert r.status_code == 200 and seen["config"].model.startswith("kotoba-tech/")
    r = c.post("/api/transcribe", json={"input_path": str(video), "profile": "soft-speech",
                                        "source_language": "ja", "model": "large-v3", "no_translate": True})
    assert r.status_code == 200 and seen["config"].model == "large-v3"
    r = c.post("/api/transcribe", json={"input_path": str(video), "profile": "nope"})
    assert r.status_code == 400 and "nope" in r.get_json()["message"]


def test_profile_is_a_saveable_config_field():
    from gensrt.api.config import _CONFIG_VALIDATORS
    assert _CONFIG_VALIDATORS["profile"]("broadcast")[0] and _CONFIG_VALIDATORS["profile"](None)[0]


def test_api_profiles_lists_everything_and_survives_a_broken_user_file(tmp_path, monkeypatch):
    from gensrt.server import app
    app.config["TESTING"] = True
    c = app.test_client()
    body = c.get("/api/profiles").get_json()
    names = [p["name"] for p in body["profiles"]]
    assert names == ["broadcast", "soft-speech"] and body["error"] is None
    soft = body["profiles"][1]
    assert soft["settings"]["chunk_mode"] == "fixed" and soft["model_by_language"]["ja"].startswith("kotoba-tech/")
    assert soft["description"] and soft["source"] == "built-in"

    bad = tmp_path / "gensrt-profiles.json"
    bad.write_text("{oops", encoding="utf-8")
    monkeypatch.setattr(P, "_find_profiles_file", lambda: bad)
    body = c.get("/api/profiles").get_json()
    assert [p["name"] for p in body["profiles"]] == ["broadcast", "soft-speech"]
    assert body["error"] and "gensrt-profiles.json" in body["error"]


def test_soft_speech_profile_turns_the_heuristics_mode_on_all_the_way_to_the_pipeline(tmp_path, monkeypatch):
    """The mode was implemented and tested in isolation while the profile
    never set it — the CLI produced heuristics_mode="" and nothing changed
    on the user's file (2026-09-30).  This follows the setting from the
    profile through _resolve_settings, build_transcription_config and
    run_pipeline to the cue list."""
    from pathlib import Path

    import gensrt.pipeline as pl
    from gensrt.audio import extractor
    from gensrt.cli import _build_parser, _resolve_settings
    from gensrt.models import SRTSegment
    from gensrt.operations import build_transcription_config

    cfg_file = tmp_path / "gensrt-config.json"
    cfg_file.write_text("{}")
    merged = _resolve_settings(_build_parser().parse_args(
        ["--config", str(cfg_file), "--profile", "soft-speech", "--source-language", "ja"]))
    assert merged["heuristics_mode"] == "soft-speech"
    merged["translate"] = False
    config = build_transcription_config(merged, auto_detect_backend=False)
    assert config.heuristics_mode == "soft-speech"

    wav = tmp_path / "x.wav"
    wav.write_bytes(b"")
    monkeypatch.setattr(extractor, "extract_audio", lambda p: wav)
    cues = [SRTSegment(1, 0.0, 1.0, "うん"), SRTSegment(2, 20.0, 23.0, "大丈夫じゃないか"),
            SRTSegment(3, 40.0, 42.0, "アハハハ"), SRTSegment(4, 60.0, 61.0, "うまい")]
    monkeypatch.setattr(pl, "_run_asr", lambda wav_path, config, status=None: (list(cues), "ja"))
    result = pl.run_pipeline(Path("movie.mp4"), tmp_path / "o.srt", config)
    assert [s.text for s in result.segments] == ["大丈夫じゃないか"]

    # And without the profile the same cues keep their うん: mode is off.
    merged = _resolve_settings(_build_parser().parse_args(["--config", str(cfg_file), "--source-language", "ja"]))
    merged["translate"] = False
    config = build_transcription_config(merged, auto_detect_backend=False)
    assert config.heuristics_mode == ""
    result = pl.run_pipeline(Path("movie.mp4"), tmp_path / "o2.srt", config)
    assert "うん" in [s.text for s in result.segments]
