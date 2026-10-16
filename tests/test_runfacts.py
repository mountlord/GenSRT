"""The one-line run summary, the extraction timing and the translator preflight."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from gensrt import runfacts
from gensrt.models import SRTSegment, TranscriptionConfig


# ── runfacts ────────────────────────────────────────────────────────────────

def test_summary_names_only_what_was_recorded():
    runfacts.start()
    assert runfacts.summary_line() == ""
    runfacts.timing("asr", 384.2)
    runfacts.note(asr_device="cuda", asr_compute="float16")
    assert runfacts.summary_line() == "ASR 6.4 min on cuda/float16"
    runfacts.timing("extract", 31.2)
    runfacts.note(extract_mb_s=314.0)
    runfacts.timing("translate", 10.04)
    runfacts.note(translate_engine="madlad", translate_device="cuda")
    runfacts.note(vram_after_asr_mib=667)
    runfacts.timing("total", 534)
    assert runfacts.summary_line() == (
        "extract 31.2s at 314 MB/s | ASR 6.4 min on cuda/float16 | "
        "translate 10.0s via madlad on cuda | VRAM after ASR release 667 MiB | total 8.9 min"
    )


def test_start_resets_and_snapshot_is_a_copy():
    runfacts.start()
    runfacts.note(asr_device="cpu")
    snap = runfacts.snapshot()
    runfacts.start()
    assert runfacts.snapshot() == {}
    assert snap == {"asr_device": "cpu"}          # the earlier copy survives
    # a snapshot formats the same as the live notebook
    assert runfacts.summary_line({"t_total": 5}) == "total 5.0s"


# ── extraction timing ───────────────────────────────────────────────────────

def test_extraction_line_warns_only_for_a_slow_long_read(tmp_path, caplog):
    import gensrt.pipeline as pl

    f = tmp_path / "movie.ts"
    with f.open("wb") as fh:                             # sparse 50 MB: size without disk
        fh.truncate(50 * 1024 * 1024)
    runfacts.start()
    with caplog.at_level(logging.INFO, logger="gensrt.pipeline"):
        pl._log_extraction(f, 2.0)                      # 25 MB/s, fast
    assert "Audio extracted in 2.0s (50 MB read at 25 MB/s)" in caplog.text
    assert "network share" not in caplog.text
    assert runfacts.snapshot() == {"t_extract": 2.0, "extract_mb_s": 25.0}

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gensrt.pipeline"):
        pl._log_extraction(f, 100.0)                    # 0.5 MB/s for 100 s
    assert "network share or a slow disk" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gensrt.pipeline"):
        pl._log_extraction(f, 30.0)                     # slow but short: no warning
    assert "network share" not in caplog.text


# ── translator preflight ────────────────────────────────────────────────────

def _model_dir(tmp_path: Path, mib: int) -> Path:
    d = tmp_path / "model"
    d.mkdir()
    with (d / "model.bin").open("wb") as fh:            # sparse: st_size without the bytes
        fh.truncate(mib * 1024 * 1024)
    return d


@pytest.mark.parametrize("free, want", [(None, True), (6000, True), (3400, True), (3399, False), (500, False)])
def test_gpu_can_hold_refuses_only_when_the_weights_exceed_free_vram(tmp_path, monkeypatch, caplog, free, want):
    from gensrt import gpu_mem
    from gensrt.translation import _preflight

    monkeypatch.setattr(gpu_mem, "free_mib", lambda: free)
    d = _model_dir(tmp_path, 3400)
    with caplog.at_level(logging.WARNING):
        assert _preflight.gpu_can_hold(d, "MADLAD") is want
    if want:
        assert "cannot fit on the GPU" not in caplog.text
    else:
        assert "MADLAD: the model's weights are 3400 MiB on disk and only" in caplog.text
        assert "translation_engine=nllb" in caplog.text


def test_model_size_counts_everything_when_there_is_no_model_bin(tmp_path):
    from gensrt.translation._preflight import model_size_mib

    d = tmp_path / "m"
    d.mkdir()
    (d / "a.weights").write_bytes(b"\0" * (2 * 1024 * 1024))
    (d / "sub").mkdir()
    (d / "sub" / "b.weights").write_bytes(b"\0" * (3 * 1024 * 1024))
    assert model_size_mib(d) == 5


def test_free_mib_is_none_without_nvidia_smi(monkeypatch):
    import subprocess

    from gensrt import gpu_mem

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(subprocess, "run", boom)
    assert gpu_mem.free_mib() is None


@pytest.mark.parametrize("fits, want_devices", [(False, ["cpu"]), (True, ["cuda"])])
def test_madlad_asks_the_preflight_before_its_cuda_attempts(tmp_path, monkeypatch, fits, want_devices):
    """Refused → straight to cpu/int8, no CUDA attempt; allowed → the ladder as before.
    Either way the device it ended on is in the run facts."""
    import sys
    import types

    from gensrt.translation import _preflight, madlad_ct2

    monkeypatch.setattr(_preflight, "gpu_can_hold", lambda d, n: fits)
    seen = []

    class _Translator:
        def __init__(self, path, device, compute_type, **kw):
            seen.append(device)
            self.device = device

    fake = types.ModuleType("ctranslate2")
    fake.Translator = _Translator
    fake.get_cuda_device_count = lambda: 1
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)

    d = _model_dir(tmp_path, 1)
    monkeypatch.setattr(madlad_ct2, "ensure_model", lambda ref: d)
    monkeypatch.setattr(madlad_ct2.MADLADCT2Engine, "_load_tokenizer", lambda self, md: None)
    cfg = TranscriptionConfig(model="large-v3", device="cuda", compute_type="float16",
                              translation_engine="madlad", madlad_model=str(d),
                              target_language="en", source_language="ja")
    eng = madlad_ct2.MADLADCT2Engine(cfg)
    runfacts.start()
    eng._load()
    assert seen == want_devices
    facts = runfacts.snapshot()
    assert facts["translate_device"] == want_devices[0]
    assert facts["translate_engine"] == "madlad"


# ── the CLI footer ─────────────────────────────────────────────────────────

def test_cli_prints_the_summary_from_the_result(capsys):
    from gensrt.cli import _facts_summary
    from gensrt.models import TranscriptionResult

    r = TranscriptionResult(
        input_path=Path("a.mp4"), output_path=Path("a.srt"), detected_language="ja",
        segments=[SRTSegment(1, 0, 1, "x")], config=TranscriptionConfig(model="large-v3"),
        elapsed_s=1.0, facts={"t_asr": 3.0, "asr_device": "cpu", "asr_compute": "int8"},
    )
    assert _facts_summary(r) == "ASR 3.0s on cpu/int8"
    import dataclasses
    assert _facts_summary(dataclasses.replace(r, facts={})) == ""
