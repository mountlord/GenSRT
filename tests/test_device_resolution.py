"""Tests for device/compute-type resolution in operations.build_transcription_config.

Covers the fix for the review's §3.5: an explicitly requested device used to be
overwritten unconditionally by the GPU probe, so `--device cpu` did nothing and
users with a broken CUDA install had no escape hatch.
"""

from __future__ import annotations

import pytest

from gensrt import gpu_probe
from gensrt.operations import build_transcription_config


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(gpu_probe, "_cuda_device_count", lambda: 0)
    return monkeypatch


@pytest.fixture
def with_cuda(monkeypatch):
    monkeypatch.setattr(gpu_probe, "_cuda_device_count", lambda: 1)
    monkeypatch.setattr(gpu_probe, "cuda_device_name", lambda gpu_id=0: "Fake RTX")
    return monkeypatch


def test_explicit_cpu_is_honoured_even_when_cuda_exists(with_cuda):
    cfg = build_transcription_config({"device": "cpu"})
    assert cfg.device == "cpu"
    assert cfg.backend == "cpu"


def test_explicit_cpu_does_not_probe(monkeypatch):
    """Forcing CPU must not touch the GPU probe at all."""
    called = []
    monkeypatch.setattr(
        gpu_probe, "_cuda_device_count", lambda: called.append(1) or 1
    )
    cfg = build_transcription_config({"device": "cpu"})
    assert cfg.device == "cpu"
    assert called == []


def test_auto_picks_cuda_when_available(with_cuda):
    cfg = build_transcription_config({"device": "auto"})
    assert cfg.device == "cuda"
    assert cfg.backend == "cuda"


def test_auto_falls_back_to_cpu_without_cuda(no_cuda):
    cfg = build_transcription_config({"device": "auto"})
    assert cfg.device == "cpu"
    assert cfg.backend == "cpu"


def test_explicit_cuda_is_honoured_when_available(with_cuda):
    cfg = build_transcription_config({"device": "cuda"})
    assert cfg.device == "cuda"


def test_explicit_cuda_degrades_to_cpu_when_unavailable(no_cuda, caplog):
    """Honouring an impossible request helps nobody — warn and use CPU."""
    cfg = build_transcription_config({"device": "cuda"})
    assert cfg.device == "cpu"
    assert any("cuda" in r.message.lower() for r in caplog.records)


def test_explicit_compute_type_is_preserved(with_cuda):
    cfg = build_transcription_config({"device": "cuda", "compute_type": "int8"})
    assert cfg.compute_type == "int8"


def test_auto_compute_type_is_derived_for_cpu(no_cuda):
    cfg = build_transcription_config({"device": "auto", "compute_type": "auto"})
    assert cfg.device == "cpu"
    # CTranslate2 does not support float16 on CPU; the derived default must
    # be something the device can actually run.
    assert cfg.compute_type in ("int8", "float32")


def test_unrecognised_device_treated_as_auto(no_cuda, caplog):
    cfg = build_transcription_config({"device": "banana"})
    assert cfg.device == "cpu"
    assert any("banana" in r.message.lower() for r in caplog.records)


def test_no_probe_when_auto_detect_disabled():
    cfg = build_transcription_config(
        {"device": "cuda", "compute_type": "float16", "backend": "cuda"},
        auto_detect_backend=False,
    )
    assert cfg.device == "cuda"
    assert cfg.compute_type == "float16"


def test_default_config_device_is_auto():
    """The shipped default must probe, not assume."""
    from gensrt.models import TranscriptionConfig

    assert TranscriptionConfig().device == "auto"


# ── Run banner ────────────────────────────────────────────────────────────

def test_banner_reports_resolved_values_not_the_request(no_cuda, capsys):
    """The banner must show what will run, not what was asked for.

    Regression guard: the banner was printed from the pre-resolution settings
    dict, so with device/compute_type defaulting to "auto" it reported "auto"
    — and reported `backend` straight from gensrt-config.json, which on a
    CPU-only machine could claim "cuda".
    """
    from gensrt.cli import _print_banner

    cfg = build_transcription_config(
        {"device": "auto", "compute_type": "auto", "backend": "cuda"}
    )
    _print_banner(cfg)
    banner = capsys.readouterr().err

    assert "auto" not in banner
    assert "Device : cpu (cpu)" in banner
