"""The translation engine roster, as of v1.2.7.

History matters here.  v1.2.5 removed the torch-based offline engines
(NLLB-on-transformers and MarianMT): each dragged in ~2.5 GB of PyTorch,
could only produce English, and neither was ever confirmed working.  v1.2.7
brings NLLB *back* — on CTranslate2 this time, sharing the runtime Whisper
already uses, with any mapped target language and zero new dependencies.

So the roster is: google, nllb, none.  Marian stays removed, and a leftover
``"marian"`` in an old config still gets an explanation rather than a bare
"unknown engine".  The old torch-based module paths must stay gone — the
new engine lives at ``gensrt.translation.nllb_ct2``, deliberately a
different name so nothing can half-import the old world.
"""

from __future__ import annotations

import pytest

from gensrt.exceptions import ConfigError
from gensrt.models import TranscriptionConfig, TranslationEngineKey
from gensrt.pipeline import validate_translation_config
from gensrt.translation.factory import available_engines, get_engine


# ── The roster ────────────────────────────────────────────────────────────

def test_roster_is_google_nllb_none():
    assert {e.value for e in TranslationEngineKey} == {"google", "nllb", "none"}
    assert set(available_engines()) == {"google", "nllb", "none"}


def test_google_resolves():
    assert get_engine("google").name == "google-gtx"


def test_nllb_resolves_without_loading_the_model():
    """Construction must be cheap: no download, no model load, no network.

    The factory builds an NLLB instance whenever the fallback is 'nllb' —
    which is the default — so an expensive constructor would tax every
    translating run whether or not the fallback ever fires.
    """
    engine = get_engine("nllb")
    assert engine.name == "nllb"
    assert engine._translator is None
    assert engine._tokenizer is None


def test_none_is_passthrough():
    engine = get_engine("none")
    assert engine.translate_batch(["ആരോപിച്ചു"], "ml", "en") == ["ആരോപിച്ചു"]


# ── Config plumbing ───────────────────────────────────────────────────────

def test_google_engine_receives_configured_fallback():
    config = TranscriptionConfig(translation_fallback="none")
    engine = get_engine("google", config)
    assert engine._fallback == "none"


def test_google_default_fallback_without_config_is_mymemory():
    """Direct construction (tests, scripts) keeps pre-v1.2.7 behaviour."""
    assert get_engine("google")._fallback == "mymemory"


def test_config_default_fallback_is_nllb_with_factory_injected():
    config = TranscriptionConfig()
    assert config.translation_fallback == "nllb"
    engine = get_engine("google", config)
    assert engine._fallback == "nllb"
    assert engine._fallback_engine_factory is not None
    # And the injection is lazy: nothing constructed yet.
    assert engine._fallback_engine is None


def test_unknown_fallback_is_a_config_error():
    config = TranscriptionConfig(translation_fallback="deepl")
    with pytest.raises(ConfigError) as exc:
        get_engine("google", config)
    assert "translation_fallback" in str(exc.value)


def test_validate_translation_config_checks_fallback_too():
    config = TranscriptionConfig(translation_fallback="bogus")
    with pytest.raises(ConfigError):
        validate_translation_config(config)


# ── What stays removed ────────────────────────────────────────────────────

@pytest.mark.parametrize("removed", ["marian", "Marian"])
def test_marian_explains_itself(removed):
    """A leftover config must get an explanation, not 'unknown engine'."""
    with pytest.raises(ConfigError) as exc:
        get_engine(removed)
    msg = str(exc.value)
    assert "removed in v1.2.5" in msg
    assert "google" in msg and "none" in msg and "nllb" in msg


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
        TranscriptionConfig(
            translate=False, translation_engine="deepl",
            translation_fallback="bogus",
        )
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


def test_cli_exposes_translation_fallback():
    from gensrt.cli import _build_parser

    args = _build_parser().parse_args(
        ["--input", "v.mkv", "--translation-fallback", "none"]
    )
    assert args.translation_fallback == "none"


def test_cli_rejects_unknown_fallback():
    from gensrt.cli import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["--input", "v.mkv", "--translation-fallback", "deepl"]
        )


def test_target_language_reaches_the_config():
    from gensrt.config import merge_config
    from gensrt.operations import build_transcription_config

    merged = merge_config({}, {"target_language": "ko", "device": "cpu"})
    assert build_transcription_config(merged).target_language == "ko"


def test_fallback_and_model_reach_the_config():
    from gensrt.config import merge_config
    from gensrt.operations import build_transcription_config

    merged = merge_config({}, {"translation_fallback": "none", "device": "cpu"})
    built = build_transcription_config(merged)
    assert built.translation_fallback == "none"
    assert built.translation_model    # default flows through non-empty
