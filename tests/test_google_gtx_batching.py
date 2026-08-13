"""Batch-delimiter handling in the Google GTX engine.

Translating Malayalam to Korean turned the old "{{SPLIT}}" delimiter into
"{{분할}}" — SPLIT is an English word, so Google translated it. The split found
one part instead of eighteen, the mismatch path padded the tail with the
ORIGINAL text, and 93% of a 102-cue file came out untranslated behind a single
WARNING no GUI user ever sees.

Two fixes, both pinned here: a delimiter with no word in it, and a mismatch
path that translates per cue instead of padding.
"""

from __future__ import annotations

import pytest

from gensrt.translation.google_gtx import _GLUE, _GLUE_RE, GoogleGTXEngine


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("gensrt.translation.google_gtx.time.sleep", lambda _: None)


# ── The delimiter ─────────────────────────────────────────────────────────

def test_delimiter_contains_no_translatable_word():
    assert not any(c.isalpha() for c in _GLUE)


def test_delimiter_regex_tolerates_reflowed_whitespace():
    assert _GLUE_RE.split("one @@@ two\n@@@\nthree@@@four") == \
        ["one", "two", "three", "four"]


# ── The mismatch path ─────────────────────────────────────────────────────

class _Fake:
    """Stands in for _fetch_gtx. Batch requests come back as one blob."""

    def __init__(self):
        self.batch_calls = 0
        self.single_calls = 0

    def __call__(self, params):
        q = params["q"]
        if _GLUE.strip() in q:                     # a batched request
            self.batch_calls += 1
            return [[["<merged blob with no delimiter>", q, None, None]]]
        self.single_calls += 1
        return [[[f"KO:{q}", q, None, None]]]


def _engine(monkeypatch):
    eng = GoogleGTXEngine()
    fake = _Fake()
    monkeypatch.setattr(eng, "_fetch_gtx", fake)
    return eng, fake


def test_lost_delimiter_falls_back_to_per_cue(monkeypatch):
    eng, fake = _engine(monkeypatch)
    texts = [f"cue {i}" for i in range(5)]

    out = eng.translate_batch(texts, "ml", "ko")

    assert out == [f"KO:{t}" for t in texts]      # every cue translated
    assert fake.single_calls == 5


def test_originals_are_never_silently_kept(monkeypatch):
    """The actual v1.2.4 failure: untranslated text passed off as output."""
    eng, fake = _engine(monkeypatch)
    texts = ["ആരോപിച്ചു", "അറിയിച്ചു", "ഉണ്ടാകും"]

    out = eng.translate_batch(texts, "ml", "ko")

    assert not set(out) & set(texts)


def test_glue_is_abandoned_after_the_first_failure(monkeypatch):
    """Do not burn a doomed batch request per chunk."""
    eng, fake = _engine(monkeypatch)
    eng.translate_batch([f"cue {i}" for i in range(60)], "ml", "ko")
    assert fake.batch_calls == 1


def test_flag_resets_between_jobs(monkeypatch):
    """Whether glue survives depends on the target language."""
    eng, _ = _engine(monkeypatch)
    eng.translate_batch(["a", "b"], "ml", "ko")
    assert eng._glue_disabled
    eng.translate_batch(["a", "b"], "ml", "en")
    # reset at entry; this run's own outcome sets it again
    assert eng._glue_disabled is True


def test_a_surviving_delimiter_still_batches(monkeypatch):
    eng = GoogleGTXEngine()
    calls = {"n": 0}

    def fetch(params):
        calls["n"] += 1
        parts = params["q"].split(_GLUE)
        return [[[_GLUE.join(f"KO:{p}" for p in parts), "", None, None]]]

    monkeypatch.setattr(eng, "_fetch_gtx", fetch)
    texts = [f"cue {i}" for i in range(6)]
    assert eng.translate_batch(texts, "ml", "ko") == [f"KO:{t}" for t in texts]
    assert calls["n"] == 1


def test_one_failing_cue_does_not_lose_the_rest(monkeypatch):
    eng = GoogleGTXEngine()

    def fetch(params):
        if _GLUE.strip() in params["q"]:
            return [[["blob", "", None, None]]]
        if "bad" in params["q"]:
            raise RuntimeError("HTTP 500")
        return [[[f"KO:{params['q']}", "", None, None]]]

    monkeypatch.setattr(eng, "_fetch_gtx", fetch)
    out = eng.translate_batch(["good1", "bad", "good2"], "ml", "ko")
    assert out == ["KO:good1", "bad", "KO:good2"]
