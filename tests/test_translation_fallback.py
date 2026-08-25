"""Google GTX batch-failure handling and rate-limit behaviour (v1.2.7).

The scenario that motivated all of this: a nightly batch of multi-hour
recordings got the IP throttled by the GTX endpoint, after which EVERY
batch failed with HTTP 429 — ~80 warnings per file, ~50 seconds of
MyMemory round-trips per batch, and output quality the user had already
judged unusable.  Three behaviours are pinned here:

* the fallback is configurable and dispatches correctly
  (``nllb`` / ``mymemory`` / ``none``), degrades instead of raising, and
  constructs the injected NLLB engine lazily — never on the happy path;
* the log does not flood: first failure at WARNING, the rest at DEBUG,
  one summary WARNING with totals at the end;
* HTTP 429/503 back off on the long ladder (honouring Retry-After), while
  ordinary transient errors keep the fast ladder — retrying a rate
  limiter at 0.25s only deepens the hole the IP is in.
"""

from __future__ import annotations

import logging

import pytest

from gensrt.exceptions import TranslationError
from gensrt.translation.base import TranslationEngine
from gensrt.translation.google_gtx import (
    _RATE_LIMIT_DELAYS_S,
    _RETRY_BASE_S,
    GoogleGTXEngine,
    _parse_retry_after,
)


@pytest.fixture()
def sleeps(monkeypatch):
    """Capture time.sleep calls instead of sleeping."""
    calls: list[float] = []
    monkeypatch.setattr(
        "gensrt.translation.google_gtx.time.sleep", calls.append
    )
    return calls


def _always_failing(engine, monkeypatch):
    """Make every GTX request fail like a throttled IP does."""
    def _boom(params):
        raise TranslationError("google", "GTX failed after 3 attempts: HTTP 429")
    monkeypatch.setattr(engine, "_fetch_gtx", _boom)


class _RecordingEngine(TranslationEngine):
    """Fake NLLB fallback that records what it was asked to translate."""

    def __init__(self):
        self.batches: list[list[str]] = []

    def translate(self, text, source_language, target_language="en"):
        return self.translate_batch([text], source_language, target_language)[0]

    def translate_batch(self, texts, source_language, target_language="en"):
        self.batches.append(list(texts))
        return [f"<{t}>" for t in texts]

    def is_available(self):
        return True


# ── Fallback dispatch ─────────────────────────────────────────────────────

def test_fallback_none_keeps_originals(sleeps, monkeypatch):
    engine = GoogleGTXEngine(fallback="none")
    _always_failing(engine, monkeypatch)
    out = engine.translate_batch(["하나", "둘"], "ko", "en")
    assert out == ["하나", "둘"]


def test_fallback_nllb_translates_failed_batches(sleeps, monkeypatch):
    fake = _RecordingEngine()
    engine = GoogleGTXEngine(fallback="nllb", fallback_engine_factory=lambda: fake)
    _always_failing(engine, monkeypatch)
    out = engine.translate_batch(["하나", "둘"], "ko", "en")
    assert out == ["<하나>", "<둘>"]
    assert fake.batches == [["하나", "둘"]]


def test_nllb_factory_called_once_and_only_on_failure(sleeps, monkeypatch):
    constructed = []

    def factory():
        constructed.append(1)
        return _RecordingEngine()

    engine = GoogleGTXEngine(fallback="nllb", fallback_engine_factory=factory)

    # Happy path first: GTX succeeds, factory must not fire.
    monkeypatch.setattr(
        engine, "_gtx_glue_batch", lambda texts, src, tgt: [t.upper() for t in texts]
    )
    assert engine.translate_batch(["ab"], "ko", "en") == ["AB"]
    assert constructed == []

    # Now fail everything: one construction, however many batches fail.
    # (Patch the same seam the happy path patched — _always_failing patches
    # one level lower and would be masked by the patch above.)
    def _boom(texts, src, tgt):
        raise TranslationError("google", "GTX failed after 3 attempts: HTTP 429")

    monkeypatch.setattr(engine, "_gtx_glue_batch", _boom)
    engine.translate_batch(["one"], "ko", "en")
    engine.translate_batch(["two"], "ko", "en")
    assert constructed == [1]


def test_nllb_fallback_failure_degrades_to_originals(sleeps, monkeypatch):
    class _Broken(TranslationEngine):
        def translate(self, *a, **k):
            raise RuntimeError("no model")

        def translate_batch(self, *a, **k):
            raise RuntimeError("no model")

        def is_available(self):
            return False

    engine = GoogleGTXEngine(
        fallback="nllb", fallback_engine_factory=lambda: _Broken()
    )
    _always_failing(engine, monkeypatch)
    assert engine.translate_batch(["하나"], "ko", "en") == ["하나"]


def test_nllb_fallback_without_factory_degrades_to_originals(sleeps, monkeypatch):
    """Reachable only via direct construction — must not crash."""
    engine = GoogleGTXEngine(fallback="nllb")
    _always_failing(engine, monkeypatch)
    assert engine.translate_batch(["하나"], "ko", "en") == ["하나"]


def test_fallback_mymemory_uses_mymemory(sleeps, monkeypatch):
    engine = GoogleGTXEngine(fallback="mymemory")
    _always_failing(engine, monkeypatch)
    monkeypatch.setattr(
        engine, "_mymemory_single", lambda t, s, tgt: f"mm:{t}"
    )
    assert engine.translate_batch(["하나"], "ko", "en") == ["mm:하나"]


def test_default_construction_is_mymemory():
    """Pre-v1.2.7 behaviour for anyone constructing the engine directly."""
    assert GoogleGTXEngine()._fallback == "mymemory"


# ── Warning-flood throttle ────────────────────────────────────────────────

def test_one_warning_per_run_plus_summary(sleeps, monkeypatch, caplog):
    """80 failed batches used to mean 80 WARNINGs. Now: first failure at
    WARNING, the rest at DEBUG, and one summary WARNING with the totals."""
    engine = GoogleGTXEngine(fallback="none")
    _always_failing(engine, monkeypatch)

    # Force many single-item batches so several batches fail.
    monkeypatch.setattr(
        "gensrt.translation.google_gtx._make_chunks",
        lambda texts: [([i], [t]) for i, t in enumerate(texts)],
    )

    with caplog.at_level(logging.DEBUG, logger="gensrt.translation.google_gtx"):
        engine.translate_batch([f"cue{i}" for i in range(10)], "ko", "en")

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2                       # first failure + summary
    summary = warnings[-1].message
    assert "10 of 10 batches failed" in summary
    assert "'none'" in summary


def test_summary_absent_when_nothing_failed(monkeypatch, caplog):
    engine = GoogleGTXEngine(fallback="none")
    monkeypatch.setattr(
        "gensrt.translation.google_gtx.time.sleep", lambda _: None
    )
    monkeypatch.setattr(
        engine, "_gtx_glue_batch", lambda texts, src, tgt: list(texts)
    )
    with caplog.at_level(logging.DEBUG, logger="gensrt.translation.google_gtx"):
        engine.translate_batch(["a", "b"], "ko", "en")
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


# ── Batch pacing ──────────────────────────────────────────────────────────

def test_batches_are_paced(sleeps, monkeypatch):
    engine = GoogleGTXEngine(fallback="none")
    monkeypatch.setattr(
        "gensrt.translation.google_gtx._make_chunks",
        lambda texts: [([i], [t]) for i, t in enumerate(texts)],
    )
    monkeypatch.setattr(
        engine, "_gtx_glue_batch", lambda texts, src, tgt: list(texts)
    )
    engine.translate_batch(["a", "b", "c"], "ko", "en")
    # Two pauses for three batches: none before the first.
    from gensrt.translation.google_gtx import _BATCH_DELAY_S
    assert sleeps == [_BATCH_DELAY_S, _BATCH_DELAY_S]


# ── Rate-limit backoff in _fetch_gtx ──────────────────────────────────────

class _Resp:
    def __init__(self, status_code, headers=None, payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def _run_fetch(monkeypatch, responses, sleeps):
    """Drive _fetch_gtx through a scripted sequence of responses."""
    engine = GoogleGTXEngine()
    it = iter(responses)
    monkeypatch.setattr(
        "gensrt.translation.google_gtx.requests.get",
        lambda *a, **k: next(it),
    )
    monkeypatch.setattr(
        "gensrt.translation.google_gtx.time.sleep", sleeps.append
    )
    return engine._fetch_gtx({"q": "x"})


def test_429_uses_long_ladder(monkeypatch):
    sleeps: list[float] = []
    ok = _Resp(200, payload=[[["hola", "hello", None]]])
    result = _run_fetch(monkeypatch, [_Resp(429), _Resp(429), ok], sleeps)
    assert result[0][0][0] == "hola"
    assert sleeps == [_RATE_LIMIT_DELAYS_S[0], _RATE_LIMIT_DELAYS_S[1]]


def test_429_honours_retry_after(monkeypatch):
    sleeps: list[float] = []
    ok = _Resp(200, payload=[[["hola", "hello", None]]])
    _run_fetch(
        monkeypatch, [_Resp(429, headers={"Retry-After": "7"}), ok], sleeps
    )
    assert sleeps == [7.0]


def test_retry_after_is_capped(monkeypatch):
    from gensrt.translation.google_gtx import _RATE_LIMIT_MAX_WAIT_S

    sleeps: list[float] = []
    ok = _Resp(200, payload=[[["hola", "hello", None]]])
    _run_fetch(
        monkeypatch, [_Resp(429, headers={"Retry-After": "3600"}), ok], sleeps
    )
    assert sleeps == [_RATE_LIMIT_MAX_WAIT_S]


def test_transient_errors_keep_fast_ladder(monkeypatch):
    sleeps: list[float] = []
    ok = _Resp(200, payload=[[["hola", "hello", None]]])
    _run_fetch(monkeypatch, [_Resp(500), ok], sleeps)
    assert sleeps == [_RETRY_BASE_S]


def test_exhausted_retries_raise_translation_error(monkeypatch):
    sleeps: list[float] = []
    with pytest.raises(TranslationError):
        _run_fetch(monkeypatch, [_Resp(429)] * 3, sleeps)


def test_parse_retry_after():
    assert _parse_retry_after("5") == 5.0
    assert _parse_retry_after(" 2.5 ") == 2.5
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("Wed, 21 Oct 2026 07:28:00 GMT") is None
    assert _parse_retry_after("-1") is None
