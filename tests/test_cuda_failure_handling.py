"""Regression tests for the clean-machine CUDA failure.

A packaged build on a machine with only the NVIDIA display driver failed every
chunk with "Library cublas64_12.dll is not found or cannot be loaded".  The
model *constructed* fine — CTranslate2 resolves cuBLAS lazily at first
inference — so the load-time device fallback never fired, and the per-chunk
"skip bad chunks" guard swallowed all 43 failures.  The run finished with no
error, no fallback, and no subtitles.

Telemetry from that run: GPU load never above 14%, GPU memory delta 0.00 GB,
CPU load max 26.6%.  It did no work of any kind.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from gensrt.asr._model_loader import is_environment_error
from gensrt.asr.monolingual_whisper import MonolingualWhisperEngine
from gensrt.exceptions import TranscriptionError
from gensrt.models import TranscriptionConfig

CUBLAS_ERROR = "Library cublas64_12.dll is not found or cannot be loaded"


# ── Error classification ──────────────────────────────────────────────────

@pytest.mark.parametrize("msg", [
    CUBLAS_ERROR,
    "Library cudnn_ops64_9.dll is not found or cannot be loaded",
    "CUDA failed with error out of memory",
    "cuda runtime error (35)",
    "no kernel image is available for execution on the device",
])
def test_environment_errors_are_recognised(msg):
    assert is_environment_error(RuntimeError(msg))


@pytest.mark.parametrize("msg", [
    "could not decode audio frame",
    "invalid sample rate",
    "unexpected token in output",
])
def test_data_errors_are_not_environment_errors(msg):
    assert not is_environment_error(ValueError(msg))


# ── Fakes ─────────────────────────────────────────────────────────────────

class _Seg:
    def __init__(self, text):
        self.text = text
        self.start, self.end = 0.1, 1.2
        self.temperature, self.avg_logprob = 0.0, -0.3
        self.compression_ratio, self.no_speech_prob = 1.1, 0.01


class _Info:
    language = "ml"


class AlwaysCublasFails:
    """Every transcribe() raises the observed cuBLAS error."""
    def __init__(self):
        self.calls = 0

    def transcribe(self, path, **kw):
        self.calls += 1
        raise RuntimeError(CUBLAS_ERROR)


class OneBadChunk:
    """A single data-level failure — the case skipping was designed for."""
    def __init__(self):
        self.calls = 0

    def transcribe(self, path, **kw):
        self.calls += 1
        if self.calls == 2:
            raise ValueError("could not decode audio frame")
        return iter([_Seg(f"cue {self.calls}")]), _Info()


class Working:
    def __init__(self):
        self.calls = 0

    def transcribe(self, path, **kw):
        self.calls += 1
        return iter([_Seg(f"cue {self.calls}")]), _Info()


SR = 16000


def _audio_and_chunks(n=8):
    audio = np.zeros(SR * (n * 4 + 4), dtype=np.float32)
    chunks = [{"start_s": i * 4.0, "end_s": i * 4.0 + 4.0} for i in range(n)]
    return audio, chunks


def _run(model, config=None):
    audio, chunks = _audio_and_chunks()
    cfg = config or TranscriptionConfig(device="cuda", compute_type="float16")
    return MonolingualWhisperEngine()._transcribe_chunks(
        model, audio, SR, chunks, "ml", Path("clip.wav"), config=cfg,
    )


# ── The actual regression ─────────────────────────────────────────────────

def test_systemic_failure_raises_instead_of_returning_nothing():
    """The whole point: an empty result must not be reported as success."""
    with pytest.raises(TranscriptionError):
        _run(AlwaysCublasFails())


def test_systemic_failure_aborts_on_the_first_chunk():
    """Do not grind through 43 identical failures before giving up."""
    model = AlwaysCublasFails()
    with pytest.raises(TranscriptionError):
        _run(model)
    assert model.calls == 1


def test_error_message_is_actionable():
    with pytest.raises(TranscriptionError) as exc:
        _run(AlwaysCublasFails())
    text = str(exc.value)
    assert "cublas64_12.dll" in text
    assert "device" in text and "cpu" in text     # names the workaround
    assert "requirements-cuda.txt" in text        # names the real fix


def test_cpu_device_does_not_claim_cuda_libraries_are_missing():
    cfg = TranscriptionConfig(device="cpu", compute_type="int8")
    with pytest.raises(TranscriptionError) as exc:
        _run(AlwaysCublasFails(), cfg)
    assert "requirements-cuda.txt" not in str(exc.value)


# ── Resilience that must NOT regress ──────────────────────────────────────

def test_single_bad_chunk_is_still_skipped():
    """One unreadable chunk must not fail the file — the original intent."""
    segments, lang = _run(OneBadChunk())
    assert lang == "ml"
    assert len(segments) == 7        # 8 chunks, 1 skipped


def test_healthy_run_is_unaffected():
    segments, lang = _run(Working())
    assert len(segments) == 8


def test_majority_failure_refuses_to_write_a_partial_file(caplog):
    """A file covering a third of the audio looks finished. It isn't."""
    class MostlyBroken:
        def __init__(self):
            self.calls = 0

        def transcribe(self, path, **kw):
            self.calls += 1
            if self.calls % 4 != 0:
                raise ValueError("could not decode audio frame")
            return iter([_Seg("ok")]), _Info()

    with pytest.raises(TranscriptionError) as exc:
        _run(MostlyBroken())
    assert "mostly" in str(exc.value).lower()


# ── Inference-time CPU fallback ───────────────────────────────────────────

def test_engine_retries_on_cpu_after_gpu_inference_failure(monkeypatch, caplog):
    """GPU loads, GPU cannot run, CPU finishes the job."""
    from gensrt.asr import monolingual_whisper as mw

    loaded: list[str] = []

    def fake_loader(wav_path, config, WhisperModel, status=None):
        loaded.append(config.device)
        return Working() if config.device == "cpu" else AlwaysCublasFails()

    monkeypatch.setattr(mw, "load_whisper_model", fake_loader)
    monkeypatch.setattr(mw, "_whisper_model_class", lambda wav_path: object)

    audio, chunks = _audio_and_chunks()
    cfg = TranscriptionConfig(device="cuda", compute_type="float16")
    engine = MonolingualWhisperEngine()

    messages: list[str] = []
    with caplog.at_level(logging.WARNING):
        segments, lang = engine._transcribe_with_cpu_retry(
            audio, SR, chunks, "ml", Path("clip.wav"), cfg, messages.append
        )

    assert loaded == ["cuda", "cpu"]
    assert len(segments) == 8
    assert any("CPU" in m for m in messages)
    assert "retrying the whole file on CPU" in caplog.text


def test_no_retry_when_already_on_cpu(monkeypatch):
    from gensrt.asr import monolingual_whisper as mw

    loaded: list[str] = []

    def fake_loader(wav_path, config, WhisperModel, status=None):
        loaded.append(config.device)
        return AlwaysCublasFails()

    monkeypatch.setattr(mw, "load_whisper_model", fake_loader)
    monkeypatch.setattr(mw, "_whisper_model_class", lambda wav_path: object)

    audio, chunks = _audio_and_chunks()
    cfg = TranscriptionConfig(device="cpu", compute_type="int8")
    with pytest.raises(TranscriptionError):
        MonolingualWhisperEngine()._transcribe_with_cpu_retry(
            audio, SR, chunks, "ml", Path("clip.wav"), cfg, None
        )
    assert loaded == ["cpu"]


# ── DLL directory registration ────────────────────────────────────────────

def test_registration_is_a_noop_off_windows():
    """Must not raise or misbehave on the platform the tests run on."""
    from gensrt import _cuda_dlls

    _cuda_dlls._done = False
    _cuda_dlls._registered = []
    assert _cuda_dlls.register_cuda_dll_directories() == []


def test_registration_finds_nvidia_bin_dirs(monkeypatch, tmp_path):
    """The layout the nvidia-*-cu12 wheels actually install."""
    from gensrt import _cuda_dlls

    site = tmp_path / "site-packages"
    for comp in ("cublas", "cudnn", "cuda_runtime"):
        (site / "nvidia" / comp / "bin").mkdir(parents=True)
    (site / "nvidia" / "cublas" / "include").mkdir()   # must be ignored

    added: list[str] = []
    monkeypatch.setattr(_cuda_dlls.sys, "platform", "win32")
    monkeypatch.setattr(_cuda_dlls.os, "add_dll_directory",
                        lambda d: added.append(d), raising=False)
    monkeypatch.setattr(_cuda_dlls.sys, "path", [str(site)])
    monkeypatch.setattr(_cuda_dlls.sys, "executable", str(tmp_path / "python.exe"))
    _cuda_dlls._done = False
    _cuda_dlls._registered = []

    result = _cuda_dlls.register_cuda_dll_directories()

    assert len(result) == 3
    assert all(d.endswith("bin") for d in result)
    assert not any("include" in d for d in result)


def test_registration_never_raises_on_a_broken_path(monkeypatch, tmp_path):
    from gensrt import _cuda_dlls

    monkeypatch.setattr(_cuda_dlls.sys, "platform", "win32")
    monkeypatch.setattr(
        _cuda_dlls.os, "add_dll_directory",
        lambda d: (_ for _ in ()).throw(OSError("access denied")), raising=False,
    )
    site = tmp_path / "site-packages"
    (site / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    monkeypatch.setattr(_cuda_dlls.sys, "path", [str(site)])
    _cuda_dlls._done = False
    _cuda_dlls._registered = []

    assert _cuda_dlls.register_cuda_dll_directories() == []
