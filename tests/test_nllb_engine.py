"""NLLB-CT2 engine: everything testable without the 650 MB model.

The model itself is exercised by hand on real content (that judgement needs
a native reader, not an assertion).  What CAN be pinned mechanically is the
plumbing around it — the parts that fail silently or confusingly if they
regress:

* the ISO→FLORES language mapping, which covers every language GenSRT's
  own UI offers (an unmapped UI language would fail at translation time,
  after transcription already spent its minutes);
* the token sequence sent to the model — ``[src_lang] pieces </s>`` with a
  ``[tgt_lang]`` prefix — which produces *wrong translations, not errors*
  if it drifts;
* model directory resolution (the ``models/`` convention shared with
  Whisper models);
* the download guard: present → no network call; bare name absent → a
  clear error, never a download attempt.
"""

from __future__ import annotations

import logging

import pytest

from gensrt.exceptions import TranslationError
from gensrt.translation import nllb_ct2
from gensrt.translation.nllb_ct2 import (
    DEFAULT_MODEL,
    ISO_TO_FLORES,
    LICENSE_NOTICE,
    NLLBCT2Engine,
    ensure_model,
    flores_code,
    is_model_present,
    model_dir_for,
)


# ── Language mapping ──────────────────────────────────────────────────────

def test_primary_languages_map_correctly():
    """The three languages GenSRT was built around, plus English."""
    assert flores_code("ml") == "mal_Mlym"
    assert flores_code("ko") == "kor_Hang"
    assert flores_code("ja") == "jpn_Jpan"
    assert flores_code("en") == "eng_Latn"


def test_every_gui_language_is_mapped():
    """Every code offered in the GUI source/target dropdowns must map.

    An unmapped dropdown entry is a translation-time failure on a run that
    already paid for transcription.  This list mirrors review.html and
    config.js — extend BOTH when adding a language there.
    """
    gui_codes = {
        "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko",
        "hi", "ml", "ta", "te", "bn", "ar", "tr", "vi", "th", "id",
        "ms", "pl", "nl", "sv", "da", "no", "fi", "cs", "he", "ur", "fa",
    }
    unmapped = {c for c in gui_codes if c not in ISO_TO_FLORES}
    assert not unmapped, f"GUI languages missing a FLORES mapping: {unmapped}"


def test_flores_code_passes_flores_through():
    assert flores_code("mal_Mlym") == "mal_Mlym"


def test_auto_is_rejected_with_guidance():
    with pytest.raises(TranslationError) as exc:
        flores_code("auto")
    assert "concrete source language" in str(exc.value)


def test_unmapped_language_error_names_the_file():
    with pytest.raises(TranslationError) as exc:
        flores_code("xx")
    assert "nllb_ct2.py" in str(exc.value)


# ── Token plumbing (fake tokenizer — no model needed) ─────────────────────

class _FakeTokenizer:
    """Stands in for tokenizers.Tokenizer with a tiny fixed vocabulary."""

    _VOCAB = {"▁hello": 1, "▁world": 2, "▁안녕": 3}

    class _Enc:
        def __init__(self, tokens):
            self.tokens = tokens

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return self._Enc([f"▁{w}" for w in text.split()])

    def token_to_id(self, tok):
        return self._VOCAB.get(tok)

    def decode(self, ids, skip_special_tokens=True):
        rev = {v: k for k, v in self._VOCAB.items()}
        return "".join(rev[i] for i in ids).replace("▁", " ").strip()


def _engine_with_fake_tokenizer() -> NLLBCT2Engine:
    engine = NLLBCT2Engine()
    engine._tokenizer = _FakeTokenizer()
    return engine


def test_encode_produces_lang_pieces_eos():
    engine = _engine_with_fake_tokenizer()
    assert engine._encode("hello world", "kor_Hang") == \
        ["kor_Hang", "▁hello", "▁world", "</s>"]


def test_decode_strips_language_and_special_tokens():
    engine = _engine_with_fake_tokenizer()
    # CTranslate2 echoes the target prefix back as the first hypothesis
    # token, and </s> may trail.
    assert engine._decode(["eng_Latn", "▁hello", "▁world", "</s>"]) == \
        "hello world"


def test_decode_drops_tokens_outside_the_vocab():
    engine = _engine_with_fake_tokenizer()
    assert engine._decode(["eng_Latn", "▁hello", "<unk>"]) == "hello"


def test_translate_batch_preserves_empty_cues(monkeypatch):
    """Empty/whitespace cues pass through and never reach the model —
    an empty source invites the decoder to invent text."""
    engine = _engine_with_fake_tokenizer()

    class _Hyp:
        def __init__(self, tokens):
            self.hypotheses = [tokens]

    class _FakeTranslator:
        def __init__(self):
            self.calls = []

        def translate_batch(self, sources, target_prefix, **kw):
            self.calls.append(sources)
            return [_Hyp(["eng_Latn", "▁hello", "</s>"]) for _ in sources]

    engine._translator = _FakeTranslator()
    monkeypatch.setattr(engine, "_load", lambda: None)

    out = engine.translate_batch(["hello", "", "   ", "world"], "ko", "en")
    assert out[1] == "" and out[2] == "   "
    assert out[0] == "hello" and out[3] == "hello"
    # Only the two non-empty cues were sent.
    assert len(engine._translator.calls[0]) == 2


# ── Model location & download guard ───────────────────────────────────────

def test_repo_id_resolves_to_models_dir_leaf(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gensrt.model_paths.model_search_dirs", lambda: [tmp_path / "models"]
    )
    monkeypatch.setattr(
        "gensrt.model_paths.models_dir", lambda: tmp_path / "models"
    )
    d = model_dir_for("someone/nllb-200-distilled-600M-ct2-int8")
    assert d == tmp_path / "models" / "nllb-200-distilled-600M-ct2-int8"


def test_bare_name_found_under_models_dir_wins(tmp_path, monkeypatch):
    local = tmp_path / "models" / "my-nllb"
    local.mkdir(parents=True)
    monkeypatch.setattr(
        "gensrt.model_paths.model_search_dirs", lambda: [tmp_path / "models"]
    )
    assert model_dir_for("my-nllb") == local


def test_is_model_present_requires_model_and_tokenizer(tmp_path, monkeypatch):
    local = tmp_path / "models" / "m"
    local.mkdir(parents=True)
    monkeypatch.setattr(
        "gensrt.model_paths.model_search_dirs", lambda: [tmp_path / "models"]
    )
    assert not is_model_present("m")
    (local / "model.bin").write_bytes(b"x")
    assert not is_model_present("m")          # tokenizer still missing
    (local / "tokenizer.json").write_text("{}")
    assert is_model_present("m")


def test_ensure_model_skips_download_when_present(tmp_path, monkeypatch):
    local = tmp_path / "models" / "m"
    local.mkdir(parents=True)
    (local / "model.bin").write_bytes(b"x")
    (local / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(
        "gensrt.model_paths.model_search_dirs", lambda: [tmp_path / "models"]
    )

    def _boom(**kw):   # any network attempt is a test failure
        raise AssertionError("snapshot_download must not be called")

    monkeypatch.setattr("huggingface_hub.snapshot_download", _boom,
                        raising=False)
    assert ensure_model("m") == local


def test_ensure_model_rejects_absent_bare_name(tmp_path, monkeypatch):
    """A bare name is the local-directory convention, not a repo ID — it
    must never trigger a download attempt."""
    monkeypatch.setattr(
        "gensrt.model_paths.model_search_dirs", lambda: [tmp_path / "models"]
    )
    monkeypatch.setattr(
        "gensrt.model_paths.models_dir", lambda: tmp_path / "models"
    )
    with pytest.raises(TranslationError) as exc:
        ensure_model("not-on-disk")
    assert "cannot be downloaded" in str(exc.value)


# ── License notice ────────────────────────────────────────────────────────

def test_license_notice_names_the_terms():
    assert "CC-BY-NC-4.0" in LICENSE_NOTICE
    assert "non-commercial" in LICENSE_NOTICE.lower()


def test_load_logs_the_license_notice(tmp_path, monkeypatch, caplog):
    """The notice must exist at the point of use, not only in the README."""
    local = tmp_path / "models" / "m"
    local.mkdir(parents=True)
    (local / "model.bin").write_bytes(b"x")
    (local / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(
        "gensrt.model_paths.model_search_dirs", lambda: [tmp_path / "models"]
    )

    engine = NLLBCT2Engine()
    engine._model_ref = "m"
    monkeypatch.setattr(engine, "_load_tokenizer", lambda d: None)

    class _FakeCT2:
        @staticmethod
        def get_cuda_device_count():
            return 0

        class Translator:
            def __init__(self, path, device, compute_type):
                assert device == "cpu" and compute_type == "int8"

    import sys
    monkeypatch.setitem(sys.modules, "ctranslate2", _FakeCT2)

    with caplog.at_level(logging.INFO, logger="gensrt.translation.nllb_ct2"):
        engine._load()
    assert any(LICENSE_NOTICE in r.message for r in caplog.records)


def test_default_model_is_the_documented_repo():
    """README, config.js hint and INSTRUCTIONS all name this repo — if the
    default moves, they all need the same edit."""
    assert DEFAULT_MODEL == "mijuanlo/nllb-200-distilled-600M-ct2-int8"
