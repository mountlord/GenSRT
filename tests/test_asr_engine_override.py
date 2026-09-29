"""Forcing the ASR engine regardless of model.

Routing is a heuristic about how a model was trained, and a checkpoint cannot
be introspected to find out. So the heuristic can be wrong, and a user who
knows their material should be able to say so — for instance to test whether
chunked inference improves results on a multilingual model, which the routing
would never choose on its own.
"""

from __future__ import annotations

import logging

import pytest

from gensrt.asr.factory import get_engine_for_model
from gensrt.models import TranscriptionConfig

CHUNKED = "MonolingualWhisperEngine"
LONGFORM = "MultilingualWhisperEngine"


# ── The override ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("model", [
    "large-v3-turbo", "medium", "small", "tiny",
])
def test_builtin_models_can_be_forced_to_chunk(model):
    """The case that motivated this: chunking a general multilingual model."""
    assert get_engine_for_model(model, "chunked").name == CHUNKED


def test_finetunes_can_be_forced_to_longform():
    assert get_engine_for_model(
        "smcproject/vegam-whisper-medium-ml-int8_float16", "longform"
    ).name == LONGFORM


def test_local_path_model_can_be_forced_to_longform(tmp_path):
    assert get_engine_for_model(str(tmp_path), "longform").name == LONGFORM


# ── Automatic routing must not regress ────────────────────────────────────

@pytest.mark.parametrize("override", [None, "", "auto", "AUTO", "  auto  "])
def test_auto_preserves_existing_routing(override):
    assert get_engine_for_model("large-v3-turbo", override).name == LONGFORM
    assert get_engine_for_model("some/custom-model", override).name == CHUNKED
    assert get_engine_for_model(
        "adalat-ai/ct2-whisper-medium-ml-rmft", override
    ).name == CHUNKED


def test_unrecognised_value_falls_back_to_auto(caplog):
    """A typo must not silently change how audio is processed."""
    with caplog.at_level(logging.WARNING):
        assert get_engine_for_model("large-v3-turbo", "chunk").name == LONGFORM
    assert "asr_engine" in caplog.text


def test_default_is_auto():
    assert TranscriptionConfig().asr_engine == "auto"


def test_config_key_reaches_the_pipeline():
    from gensrt.config import merge_config

    merged = merge_config({}, {"asr_engine": "chunked"})
    assert TranscriptionConfig.from_dict(merged).asr_engine == "chunked"


def test_cli_flag_parses():
    from gensrt.cli import _build_parser

    args = _build_parser().parse_args(["--input", "v.mkv", "--asr-engine", "chunked"])
    assert args.asr_engine == "chunked"


def test_cli_rejects_bad_value():
    from gensrt.cli import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--input", "v.mkv", "--asr-engine", "nonsense"])


def test_server_validator_accepts_the_choices():
    from gensrt.api.config import _CONFIG_VALIDATORS

    v = _CONFIG_VALIDATORS["asr_engine"]
    for good in ("auto", "chunked", "longform"):
        ok, _ = v(good)
        assert ok, good
    ok, msg = v("nonsense")
    assert not ok and "chunked" in msg


# ── Model/language mismatch guard (v1.3.0) ────────────────────────────────

def test_kotoba_is_registered_as_japanese_only():
    from gensrt.asr.factory import get_known_language_for_model

    assert get_known_language_for_model("kotoba-tech/kotoba-whisper-v2.0-faster") == "ja"
    assert get_known_language_for_model("kotoba-tech/kotoba-whisper-v1.0") == "ja"
    assert get_known_language_for_model("kotoba-tech/kotoba-whisper-bilingual-v1.0") is None


@pytest.mark.parametrize("model,lang,ok", [
    ("kotoba-tech/kotoba-whisper-v2.0-faster", "ko", False),   # the real case
    ("kotoba-tech/kotoba-whisper-v2.0-faster", "ja", True),
    ("kotoba-tech/kotoba-whisper-v2.0-faster", "ja-JP", True),
    ("kotoba-tech/kotoba-whisper-v2.0-faster", "auto", True),  # engine substitutes ja
    ("smcproject/vegam-whisper-medium-ml-int8_float16", "ko", False),
    ("large-v3-turbo", "ko", True),                             # multilingual: anything
    ("some/unregistered-model", "ko", True),                    # unknown: not our call
])
def test_registered_model_refuses_a_language_it_cannot_produce(model, lang, ok):
    from gensrt.exceptions import ConfigError
    from gensrt.models import TranscriptionConfig
    from gensrt.pipeline import validate_model_language

    cfg = TranscriptionConfig(model=model, source_language=lang)
    if ok:
        validate_model_language(cfg)
    else:
        with pytest.raises(ConfigError) as exc:
            validate_model_language(cfg)
        assert "large-v3-turbo" in str(exc.value) and lang in str(exc.value)
