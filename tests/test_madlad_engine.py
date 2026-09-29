"""MADLAD-400 translation engine.

The two MADLAD-specific behaviours pinned here both fail silently rather
than loudly, which is why they are tested at all:

* the ``<2xx>`` target token must go INSIDE the source string before
  tokenisation — prepending it to the token list loses the word-boundary
  piece and mangles the output, raising nothing;
* the SOURCE language is ignored on purpose (MADLAD infers it), so a wrong
  or absent source language must not change the request.
"""

from __future__ import annotations

import pytest

from gensrt.exceptions import TranslationError
from gensrt.models import TranscriptionConfig, TranslationEngineKey
from gensrt.translation.factory import ENGINE_KEYS, available_engines, get_engine
from gensrt.translation.madlad_ct2 import (
    DEFAULT_MODEL,
    MADLADCT2Engine,
    is_model_present,
    model_dir_for,
    target_token,
)


# ── Registration ──────────────────────────────────────────────────────────

def test_madlad_is_in_the_roster():
    assert "madlad" in ENGINE_KEYS
    assert "madlad" in available_engines()
    assert TranslationEngineKey.MADLAD.value == "madlad"


def test_madlad_resolves_without_loading_the_model():
    """Construction must be cheap: no download, no model load, no network.

    The factory builds this whenever the fallback is 'madlad', including on
    runs where Google never fails.
    """
    engine = get_engine("madlad", TranscriptionConfig())
    assert engine.name == "madlad"
    assert engine._translator is None
    assert engine._tokenizer is None


def test_madlad_uses_its_own_model_setting():
    """madlad_model, NOT translation_model — the latter names an NLLB repo."""
    cfg = TranscriptionConfig(
        translation_model="some/nllb-repo",
        madlad_model="some/madlad-repo",
    )
    assert get_engine("madlad", cfg)._model_ref == "some/madlad-repo"
    assert get_engine("nllb", cfg)._model_ref == "some/nllb-repo"


def test_madlad_falls_back_to_its_default_model():
    assert MADLADCT2Engine(None)._model_ref == DEFAULT_MODEL


# ── Target token ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("iso,expected", [
    ("en", "<2en>"), ("ja", "<2ja>"), ("ml", "<2ml>"),
    ("EN", "<2en>"), ("  ko  ", "<2ko>"),
])
def test_target_token(iso, expected):
    assert target_token(iso) == expected


@pytest.mark.parametrize("bad", ["", "   ", "auto", None])
def test_target_must_be_concrete(bad):
    """MADLAD infers the SOURCE but cannot infer the target."""
    with pytest.raises(TranslationError):
        target_token(bad)


# ── Encoding: the silent-failure case ─────────────────────────────────────

class _FakeTokenizer:
    """Records what it was asked to encode."""

    def __init__(self):
        self.seen = []

    def encode(self, text, add_special_tokens=True):
        self.seen.append(text)
        return type("E", (), {"tokens": text.split()})()


def test_target_token_is_inside_the_string_before_tokenising():
    """The whole point. Prepending it to the token list instead produces
    mangled translations and raises nothing."""
    engine = MADLADCT2Engine(None)
    engine._tokenizer = _FakeTokenizer()

    tokens = engine._encode("こんにちは", "<2en>")

    assert engine._tokenizer.seen == ["<2en> こんにちは"], \
        "the target token must be tokenised as part of the source string"
    assert tokens[0] == "<2en>"


def test_source_language_is_ignored(monkeypatch):
    """MADLAD infers the source, so the argument must not reach the model."""
    engine = MADLADCT2Engine(None)
    engine._tokenizer = _FakeTokenizer()
    sent = {}

    class _T:
        def translate_batch(self, sources, **kw):
            sent["sources"] = sources
            return [type("R", (), {"hypotheses": [["ok"]]})() for _ in sources]

    engine._translator = _T()
    monkeypatch.setattr(engine, "_load", lambda: None)
    monkeypatch.setattr(engine, "_decode", lambda toks: " ".join(toks))

    engine.translate_batch(["text"], "ja", "en")
    first = list(sent["sources"])
    engine.translate_batch(["text"], "this-is-not-a-language", "en")
    assert list(sent["sources"]) == first


# ── Batch behaviour, matching NLLB's ──────────────────────────────────────

def test_empty_batch_returns_empty():
    assert MADLADCT2Engine(None).translate_batch([], "ja", "en") == []


def test_blank_cues_pass_through_without_reaching_the_model(monkeypatch):
    """An empty source sequence invites the decoder to invent something."""
    engine = MADLADCT2Engine(None)
    engine._tokenizer = _FakeTokenizer()
    called = {"n": 0}

    class _T:
        def translate_batch(self, sources, **kw):
            called["n"] += 1
            return [type("R", (), {"hypotheses": [["translated"]]})() for _ in sources]

    engine._translator = _T()
    monkeypatch.setattr(engine, "_load", lambda: None)
    monkeypatch.setattr(engine, "_decode", lambda toks: " ".join(toks))

    out = engine.translate_batch(["", "   ", "real"], "ja", "en")
    assert out[0] == "" and out[1] == "   "
    assert out[2] == "translated"
    assert called["n"] == 1

    called["n"] = 0
    assert engine.translate_batch(["", "  "], "ja", "en") == ["", "  "]
    assert called["n"] == 0, "an all-blank batch must not reach the model"


# ── Model paths ───────────────────────────────────────────────────────────

def test_model_dir_is_under_models(tmp_path, monkeypatch):
    monkeypatch.setattr("gensrt.model_paths.models_dir", lambda: tmp_path)
    monkeypatch.setattr("gensrt.model_paths.model_search_dirs", lambda: [tmp_path])
    d = model_dir_for("olob0/madlad400-3b-mt-ct2-int8_float16")
    assert d.name == "madlad400-3b-mt-ct2-int8_float16"


def test_model_present_needs_weights_and_a_tokenizer(tmp_path, monkeypatch):
    monkeypatch.setattr("gensrt.model_paths.models_dir", lambda: tmp_path)
    monkeypatch.setattr("gensrt.model_paths.model_search_dirs", lambda: [tmp_path])
    d = tmp_path / "madlad400-3b-mt-ct2-int8_float16"
    d.mkdir()
    ref = "olob0/madlad400-3b-mt-ct2-int8_float16"

    assert not is_model_present(ref)
    (d / "model.bin").write_bytes(b"x")
    assert not is_model_present(ref), "weights alone are not enough"
    (d / "tokenizer.json").write_text("{}")
    assert is_model_present(ref)


# ── Pipeline pre-download ─────────────────────────────────────────────────

@pytest.mark.parametrize("engine,expected", [
    ("madlad", "madlad"),
    ("nllb",   "nllb"),
    ("none",   None),
])
def test_offline_engine_needed(engine, expected):
    from gensrt.pipeline import _offline_engine_needed

    cfg = TranscriptionConfig(translate=True, translation_engine=engine)
    assert _offline_engine_needed(cfg) == expected


def test_no_offline_engine_when_not_translating():
    from gensrt.pipeline import _offline_engine_needed

    cfg = TranscriptionConfig(translate=False, translation_engine="madlad")
    assert _offline_engine_needed(cfg) is None
