"""The translation engine roster, as of v1.3.0.

History matters here.  v1.2.5 removed the torch-based offline engines
(NLLB-on-transformers and MarianMT): each dragged in ~2.5 GB of PyTorch,
could only produce English, and neither was ever confirmed working.  v1.2.7
brought NLLB *back* on CTranslate2 and added MADLAD-400 beside it.  v1.3.0
removed Google GTX: the endpoint blocks IPs that translate at subtitle
volumes and the block outlives an IP change, so the engine, its MyMemory
per-cue fallback and ``translation_fallback`` all went.

So the roster is: nllb, madlad, none — all offline.  A leftover ``"google"``
or ``"marian"`` in an old config gets an explanation rather than a bare
"unknown engine".  The old module paths must stay gone.
"""

from __future__ import annotations

import pytest

from gensrt.exceptions import ConfigError
from gensrt.models import TranscriptionConfig, TranslationEngineKey
from gensrt.pipeline import validate_translation_config
from gensrt.translation.factory import available_engines, get_engine


# ── The roster ────────────────────────────────────────────────────────────

def test_roster_matches_the_factory():
    """Written as a fixed set before MADLAD existed; it must follow the enum."""
    expected = {"nllb", "madlad", "none"}
    assert {e.value for e in TranslationEngineKey} == expected
    assert set(available_engines()) == expected


def test_nllb_resolves_without_loading_the_model():
    """Construction must be cheap: no download, no model load, no network.

    The factory builds the engine at validation time, before any audio
    work, so an expensive constructor would tax every run.
    """
    engine = get_engine("nllb")
    assert engine.name == "nllb"
    assert engine._translator is None
    assert engine._tokenizer is None


def test_none_is_passthrough():
    engine = get_engine("none")
    assert engine.translate_batch(["ആരോപിച്ചു"], "ml", "en") == ["ആരോപിച്ചു"]


# ── Config plumbing ───────────────────────────────────────────────────────

def test_default_engine_is_nllb():
    assert TranscriptionConfig().translation_engine == "nllb"


def test_config_no_longer_has_a_fallback_field():
    assert not hasattr(TranscriptionConfig(), "translation_fallback")
    with pytest.raises(TypeError):
        TranscriptionConfig(translation_fallback="none")


# ── What stays removed ────────────────────────────────────────────────────

@pytest.mark.parametrize("removed,version", [
    ("marian", "v1.2.5"), ("Marian", "v1.2.5"),
    ("google", "v1.3.0"), ("Google", "v1.3.0"),
])
def test_removed_engines_explain_themselves(removed, version):
    """A leftover config must get an explanation, not 'unknown engine'."""
    with pytest.raises(ConfigError) as exc:
        get_engine(removed)
    msg = str(exc.value)
    assert f"removed in {version}" in msg
    assert "madlad" in msg and "none" in msg and "nllb" in msg


def test_google_module_stays_gone():
    with pytest.raises(ImportError):
        __import__("gensrt.translation.google_gtx")


def test_genuinely_unknown_engine_still_errors():
    with pytest.raises(ConfigError) as exc:
        get_engine("deepl")
    assert "removed" not in str(exc.value)


def test_old_torch_engine_modules_stay_gone():
    for mod in ("gensrt.translation.nllb", "gensrt.translation.marian"):
        with pytest.raises(ImportError):
            __import__(mod)


# ── CLI and config plumbing (carried forward from the removal-era file) ───

def test_validation_noop_when_not_translating():
    validate_translation_config(
        TranscriptionConfig(translate=False, translation_engine="deepl")
    )


def test_cli_exposes_target_language():
    from gensrt.cli import _build_parser

    args = _build_parser().parse_args(
        ["--input", "v.mkv", "--source-language", "ml", "--target-language", "ko"]
    )
    assert args.source_language == "ml"
    assert args.target_language == "ko"


def test_cli_accepts_nllb_engine():
    """v1.2.5 rejected --translation-engine nllb; v1.2.7 accepts it."""
    from gensrt.cli import _build_parser

    args = _build_parser().parse_args(
        ["--input", "v.mkv", "--translation-engine", "nllb"]
    )
    assert args.translation_engine == "nllb"


def test_cli_no_longer_accepts_the_fallback_flag_or_google():
    from gensrt.cli import _build_parser

    for argv in (["--input", "v.mkv", "--translation-fallback", "none"],
                 ["--input", "v.mkv", "--translation-engine", "google"]):
        with pytest.raises(SystemExit):
            _build_parser().parse_args(argv)


def test_target_language_reaches_the_config():
    from gensrt.config import merge_config
    from gensrt.operations import build_transcription_config

    merged = merge_config({}, {"target_language": "ko", "device": "cpu"})
    assert build_transcription_config(merged).target_language == "ko"


def test_stale_fallback_key_in_an_old_config_file_is_ignored():
    """merge_config only carries keys the dataclass knows, so an upgraded
    install with translation_fallback still in gensrt-config.json runs."""
    from gensrt.config import merge_config
    from gensrt.operations import build_transcription_config

    merged = merge_config({"translation_fallback": "nllb"}, {"device": "cpu"})
    assert "translation_fallback" not in merged
    built = build_transcription_config(merged)
    assert built.translation_model    # default flows through non-empty


# ── Shared engine (v1.3.0) ────────────────────────────────────────────────

def test_shared_engine_is_one_instance_per_model_and_device():
    from gensrt.translation.factory import clear_shared_engines, get_shared_engine

    clear_shared_engines()
    a = get_shared_engine("nllb", TranscriptionConfig(device="cpu"))
    b = get_shared_engine("nllb", TranscriptionConfig(device="cpu"))
    c = get_shared_engine("nllb", TranscriptionConfig(device="cpu", translation_model="other/model"))
    d = get_shared_engine("madlad", TranscriptionConfig(device="cpu"))
    assert a is b and a is not c and a is not d
    clear_shared_engines()
    assert get_shared_engine("nllb", TranscriptionConfig(device="cpu")) is not a


def test_pipeline_translation_and_report_share_one_engine(monkeypatch, tmp_path):
    """MADLAD was loaded twice per file — once to translate, once for the
    report — and never freed between files."""
    import gensrt.pipeline as pl
    from gensrt.translation import factory

    built = []

    class _E:
        def translate_batch(self, texts, s, t):
            return [f"en:{x}" for x in texts]

    monkeypatch.setattr(factory, "get_engine", lambda key, cfg: (built.append(key), _E())[1])
    factory.clear_shared_engines()
    cfg = TranscriptionConfig(device="cpu", translation_engine="madlad", target_language="en")
    from gensrt.models import SRTSegment
    segs = [SRTSegment(index=1, start=0, end=1, text="ごめん")]
    pl._maybe_translate(segs, "ja", cfg, True)
    pl._maybe_translate(segs, "ja", cfg, True)
    assert built == ["madlad"]
    factory.clear_shared_engines()
