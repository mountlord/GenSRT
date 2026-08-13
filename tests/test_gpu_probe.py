"""Tests for gensrt.gpu_probe — the torch-free CUDA probe."""

from __future__ import annotations

import sys

from gensrt import gpu_probe
from gensrt.models import GPUBackend


def test_probe_uses_ctranslate2_not_torch(monkeypatch):
    """The whole point of the change: no torch import on the probe path."""
    import ctranslate2

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 2)
    monkeypatch.setattr(gpu_probe, "cuda_device_name", lambda gpu_id=0: "Fake")

    # Poison torch so importing it would fail loudly.
    monkeypatch.setitem(sys.modules, "torch", None)

    assert gpu_probe._probe_cuda(0) is True
    assert gpu_probe.detect_backend(0) is GPUBackend.CUDA


def test_no_devices_means_cpu(monkeypatch):
    import ctranslate2

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 0)
    assert gpu_probe._probe_cuda(0) is False
    assert gpu_probe.detect_backend(0) is GPUBackend.CPU


def test_gpu_id_beyond_device_count_is_rejected(monkeypatch):
    import ctranslate2

    monkeypatch.setattr(ctranslate2, "get_cuda_device_count", lambda: 1)
    assert gpu_probe._probe_cuda(3) is False


def test_default_compute_type_cpu_is_supported():
    """CTranslate2 has no float16 on CPU — the default must be runnable."""
    assert gpu_probe.default_compute_type_for("cpu") in ("int8", "float32")


def test_backend_to_ct2_device_mapping():
    assert gpu_probe.backend_to_ct2_device(GPUBackend.CUDA) == ("cuda", "float16")
    assert gpu_probe.backend_to_ct2_device(GPUBackend.CPU) == ("cpu", "int8")


def test_cuda_device_name_degrades_to_none(monkeypatch):
    """Cosmetic only — must never raise when torch and nvidia-smi are absent."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(
        gpu_probe.subprocess, "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()),
    )
    assert gpu_probe.cuda_device_name(0) is None
