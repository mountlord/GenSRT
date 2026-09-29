"""What an 8 GB GPU taught us (Tesla P4, MADLAD-400 3B + large-v3-turbo).

Observed in one run:
  file 1: Whisper on GPU fine → MADLAD loaded on GPU → translate_batch
          "CUDA failed with error out of memory" → file written UNTRANSLATED
          behind a one-line warning.
  file 2: MADLAD still resident → Whisper load OOM → CPU for 80 minutes.

Three ladders fix three defects: shrink the translation batch then move the
translator to CPU; free the resident translation engine before Whisper
leaves the GPU; never let an untranslated file pass quietly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gensrt.models import SRTSegment, TranscriptionConfig


def _oom():
    return RuntimeError("CUDA failed with error out of memory")


# ── 1. Translation OOM ladder ─────────────────────────────────────────────

def test_ladder_halves_the_token_batch_until_it_fits(caplog):
    import logging
    from gensrt.translation._oom import translate_with_oom_ladder

    seen = []

    def run(batch):
        seen.append(batch)
        if batch > 256:
            raise _oom()
        return ["ok"]

    moved = []
    with caplog.at_level(logging.WARNING, logger="gensrt.translation._oom"):
        out, held = translate_with_oom_ladder(run, max_batch_tokens=1024,
                                              move_to_cpu=lambda: moved.append(1), engine_name="X")
    assert out == ["ok"] and held == 256 and seen == [1024, 512, 256] and not moved
    assert any("max_batch_size=512" in r.message for r in caplog.records)


def test_ladder_moves_to_cpu_when_even_the_floor_does_not_fit():
    from gensrt.translation._oom import translate_with_oom_ladder

    on_cpu = {"v": False}

    def run(batch):
        if not on_cpu["v"]:
            raise _oom()
        return ["cpu-ok"]

    def move():
        on_cpu["v"] = True

    assert translate_with_oom_ladder(run, max_batch_tokens=1024, move_to_cpu=move,
                                     engine_name="X") == (["cpu-ok"], 1024)


def test_ladder_does_not_swallow_other_errors():
    from gensrt.translation._oom import translate_with_oom_ladder

    def run(batch):
        raise ValueError("bad token")

    with pytest.raises(ValueError):
        translate_with_oom_ladder(run, max_batch_tokens=1024, move_to_cpu=lambda: None,
                                  engine_name="X")


def test_madlad_translate_batch_uses_the_ladder(monkeypatch):
    from gensrt.translation.madlad_ct2 import MADLADCT2Engine

    eng = MADLADCT2Engine(None)
    calls = []

    class _Hyp:
        def __init__(self, toks): self.hypotheses = [toks]

    class _Tr:
        def translate_batch(self, sources, **kw):
            calls.append(kw["max_batch_size"])
            if kw["max_batch_size"] > 512:
                raise _oom()
            return [_Hyp(["▁hi"]) for _ in sources]

    monkeypatch.setattr(eng, "_load", lambda: None)
    monkeypatch.setattr(eng, "_encode", lambda text, tgt: ["x"])
    monkeypatch.setattr(eng, "_decode", lambda toks: "hi")
    eng._translator = _Tr()
    assert eng.translate_batch(["안녕"], "ko", "en") == ["hi"]
    assert calls == [1024, 512]


# ── 2. Whisper load frees resident engines before leaving the GPU ─────────

def test_whisper_load_oom_frees_shared_engines_and_retries_cuda(monkeypatch, caplog):
    import logging
    from gensrt.asr._model_loader import load_whisper_model
    from gensrt.translation import factory

    factory._shared[("madlad", "", "", "cuda")] = object()      # a resident engine
    attempts = []

    def fake_model(ref, device, compute_type):
        attempts.append((device, compute_type, bool(factory._shared)))
        if device == "cuda" and factory._shared:
            raise _oom()
        return object()

    cfg = TranscriptionConfig(model="large-v3-turbo", device="cuda", compute_type="int8")
    with caplog.at_level(logging.INFO, logger="gensrt.asr._model_loader"):
        load_whisper_model(Path("x.wav"), cfg, fake_model)
    assert attempts == [("cuda", "int8", True), ("cuda", "int8", False)]
    assert not factory._shared
    assert not any("fell back to CPU" in r.message for r in caplog.records)
    assert any("released the resident translation model" in r.message for r in caplog.records)


def test_whisper_load_oom_with_nothing_to_free_still_walks_the_ladder():
    from gensrt.asr._model_loader import load_whisper_model
    from gensrt.translation import factory

    factory.clear_shared_engines()
    attempts = []

    def fake_model(ref, device, compute_type):
        attempts.append((device, compute_type))
        if device == "cuda":
            raise _oom()
        return object()

    cfg = TranscriptionConfig(model="large-v3-turbo", device="cuda", compute_type="int8")
    load_whisper_model(Path("x.wav"), cfg, fake_model)
    assert attempts == [("cuda", "int8"), ("cuda", "int8_float16"), ("cpu", "int8")]


# ── 3. An untranslated file is loud ───────────────────────────────────────

def test_untranslated_file_is_recorded_on_the_result(monkeypatch, tmp_path, caplog):
    import logging
    import gensrt.pipeline as pl
    from gensrt.audio import extractor
    from gensrt.translation import factory

    class _Broken:
        def translate_batch(self, *a, **k):
            raise _oom()

    wav = tmp_path / "x.wav"; wav.write_bytes(b"")
    monkeypatch.setattr(extractor, "extract_audio", lambda p: wav)
    monkeypatch.setattr(pl, "ensure_translation_model", lambda c, status=None: c)
    monkeypatch.setattr(factory, "get_engine", lambda key, cfg: _Broken())
    factory.clear_shared_engines()
    monkeypatch.setattr(pl, "_run_asr", lambda wav_path, config, status=None:
                        ([SRTSegment(index=1, start=0, end=2, text="안녕하세요")], "ko"))
    statuses = []
    cfg = TranscriptionConfig(model="large-v3-turbo", device="cpu", translate=True,
                              translation_engine="madlad", target_language="en",
                              source_language="ko")
    with caplog.at_level(logging.ERROR, logger="gensrt.pipeline"):
        result = pl.run_pipeline(Path("m.mp4"), tmp_path / "o.srt", cfg, status=statuses.append)
    assert result.translation_error and "out of memory" in result.translation_error
    assert result.segments[0].text == "안녕하세요"                      # source kept
    assert any("TRANSLATION FAILED" in s for s in statuses)
    assert any("TRANSLATION FAILED" in r.message for r in caplog.records)
    factory.clear_shared_engines()


def test_successful_translation_leaves_no_error(monkeypatch, tmp_path):
    import gensrt.pipeline as pl
    from gensrt.audio import extractor
    from gensrt.translation import factory

    class _Ok:
        def translate_batch(self, texts, s, t):
            return ["Hello."] * len(texts)

    wav = tmp_path / "x.wav"; wav.write_bytes(b"")
    monkeypatch.setattr(extractor, "extract_audio", lambda p: wav)
    monkeypatch.setattr(pl, "ensure_translation_model", lambda c, status=None: c)
    monkeypatch.setattr(factory, "get_engine", lambda key, cfg: _Ok())
    factory.clear_shared_engines()
    monkeypatch.setattr(pl, "_run_asr", lambda wav_path, config, status=None:
                        ([SRTSegment(index=1, start=0, end=2, text="안녕하세요")], "ko"))
    cfg = TranscriptionConfig(model="large-v3-turbo", device="cpu", translate=True,
                              translation_engine="madlad", target_language="en",
                              source_language="ko")
    result = pl.run_pipeline(Path("m.mp4"), tmp_path / "o.srt", cfg)
    assert result.translation_error is None and result.segments[0].text == "Hello."
    factory.clear_shared_engines()


# ── Ladder memory and slicing (second P4 run) ─────────────────────────────

def test_engine_starts_the_next_call_at_the_batch_that_held(monkeypatch):
    """Files 2 and 3 re-walked 1024 → 512 → 256 → 128 from the top each
    time.  The engine now remembers."""
    from gensrt.translation.madlad_ct2 import MADLADCT2Engine

    eng = MADLADCT2Engine(None)
    calls = []

    class _Hyp:
        def __init__(self, toks): self.hypotheses = [toks]

    class _Tr:
        def translate_batch(self, sources, **kw):
            calls.append(kw["max_batch_size"])
            if kw["max_batch_size"] > 256:
                raise _oom()
            return [_Hyp(["▁hi"]) for _ in sources]

    monkeypatch.setattr(eng, "_load", lambda: None)
    monkeypatch.setattr(eng, "_encode", lambda text, tgt: ["x"])
    monkeypatch.setattr(eng, "_decode", lambda toks: "hi")
    eng._translator = _Tr()
    eng.translate_batch(["안녕"], "ko", "en")
    assert calls == [1024, 512, 256]
    calls.clear()
    eng.translate_batch(["안녕"], "ko", "en")
    assert calls == [256]                       # no re-walk


def test_translation_runs_in_slices_so_a_retry_redoes_only_the_slice(monkeypatch):
    from gensrt.translation._oom import SLICE_CUES
    from gensrt.translation.nllb_ct2 import NLLBCT2Engine

    eng = NLLBCT2Engine(None)
    sizes = []

    class _Hyp:
        def __init__(self, toks): self.hypotheses = [toks]

    class _Tr:
        def translate_batch(self, sources, **kw):
            sizes.append(len(sources))
            return [_Hyp(["▁hi"]) for _ in sources]

    monkeypatch.setattr(eng, "_load", lambda: None)
    monkeypatch.setattr(eng, "_encode", lambda text, src: ["x"])
    monkeypatch.setattr(eng, "_decode", lambda toks: "hi")
    eng._translator = _Tr()
    out = eng.translate_batch(["a"] * (SLICE_CUES * 2 + 5), "ko", "en")
    assert sizes == [SLICE_CUES, SLICE_CUES, 5] and out == ["hi"] * (SLICE_CUES * 2 + 5)


def test_whisper_model_is_released_before_the_engine_returns(monkeypatch, caplog):
    """The floor after Whisper climbed 603 → 2,247 MiB over five files: the
    model was dropped but not collected before the next load.  Release is
    now explicit, and the run logs what the card still holds."""
    import logging
    import numpy as np
    import gensrt.asr.monolingual_whisper as mw
    import gensrt.gpu_mem as gm

    released = []
    monkeypatch.setattr(gm, "used_mib", lambda: 603)
    monkeypatch.setattr(mw, "_whisper_model_class", lambda wav: object)
    monkeypatch.setattr(mw, "load_whisper_model", lambda *a, **k: object())
    monkeypatch.setattr(mw.MonolingualWhisperEngine, "_transcribe_chunks",
                        lambda self, model, *a, **k: ([], "ko"))
    orig = gm.release
    monkeypatch.setattr(gm, "release", lambda what, *objs: (released.append(what), orig(what, *objs)))
    eng = mw.MonolingualWhisperEngine()
    with caplog.at_level(logging.INFO, logger="gensrt.gpu_mem"):
        eng._transcribe_with_cpu_retry(np.zeros(16000, dtype=np.float32), 16000, [], "ko",
                                       Path("x.wav"), TranscriptionConfig(device="cuda"))
    assert released == ["Whisper model"]
    assert any("after releasing Whisper model: 603 MiB" in r.message for r in caplog.records)


def test_gpu_memory_reading_never_raises(monkeypatch):
    import subprocess
    from gensrt.gpu_mem import used_mib

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")
    monkeypatch.setattr(subprocess, "run", boom)
    assert used_mib() is None
