"""Removal of the offline translation engines (v1.2.5).

NLLB-200 and MarianMT were dropped. Both could only translate *to English*,
each required a ~2.5 GB PyTorch dependency for that one direction, and neither
was ever confirmed working. Their removal takes torch out of the project
entirely.

These tests pin the removal itself and the CLI flag that was missing.
"""

from __future__ import annotations

import pytest

from gensrt.exceptions import ConfigError
from gensrt.models import TranscriptionConfig, TranslationEngineKey
from gensrt.pipeline import validate_translation_config
from gensrt.translation.factory import get_engine


# ── What remains ──────────────────────────────────────────────────────────

def test_only_google_and_none_remain():
    assert {e.value for e in TranslationEngineKey} == {"google", "none"}


def test_google_resolves():
    assert get_engine("google").name


def test_none_is_passthrough():
    engine = get_engine("none")
    assert engine.translate_batch(["ആരോപിച്ചു"], "ml", "en") == ["ആരോപിച്ചു"]


# ── What was removed ──────────────────────────────────────────────────────

@pytest.mark.parametrize("removed", ["nllb", "marian", "NLLB", "Marian"])
def test_removed_engines_explain_themselves(removed):
    """A leftover config must get an explanation, not 'unknown engine'."""
    with pytest.raises(ConfigError) as exc:
        get_engine(removed)
    msg = str(exc.value)
    assert "removed in v1.2.5" in msg
    assert "google" in msg and "none" in msg


def test_genuinely_unknown_engine_still_errors():
    with pytest.raises(ConfigError) as exc:
        get_engine("deepl")
    assert "removed" not in str(exc.value)


def test_offline_engine_modules_are_gone():
    for mod in ("gensrt.translation.nllb", "gensrt.translation.marian"):
        with pytest.raises(ImportError):
            __import__(mod)


# ── Target language ───────────────────────────────────────────────────────

def test_any_target_language_is_accepted():
    """The whole point of the change: ml -> ko must not be rejected."""
    validate_translation_config(
        TranscriptionConfig(translation_engine="google",
                            source_language="ml", target_language="ko")
    )


def test_removed_engine_rejected_before_expensive_work():
    """Validation runs before audio extract and model load."""
    with pytest.raises(ConfigError):
        validate_translation_config(
            TranscriptionConfig(translation_engine="nllb", target_language="en")
        )


def test_validation_noop_when_not_translating():
    validate_translation_config(
        TranscriptionConfig(translate=False, translation_engine="nllb")
    )


def test_cli_exposes_target_language():
    from gensrt.cli import _build_parser

    args = _build_parser().parse_args(
        ["--input", "v.mkv", "--source-language", "ml", "--target-language", "ko"]
    )
    assert args.source_language == "ml"
    assert args.target_language == "ko"


def test_cli_rejects_removed_engines():
    from gensrt.cli import _build_parser

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--input", "v.mkv", "--translation-engine", "nllb"])


def test_target_language_reaches_the_config():
    from gensrt.config import merge_config
    from gensrt.operations import build_transcription_config

    merged = merge_config({}, {"target_language": "ko", "device": "cpu"})
    assert build_transcription_config(merged).target_language == "ko"
