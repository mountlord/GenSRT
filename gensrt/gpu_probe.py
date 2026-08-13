"""GPU backend detection and selection.

Probes the system for available hardware acceleration in priority order:
  1. CUDA  (NVIDIA)           — Phase I  ✅
  2. ROCm  (AMD)              — Phase II 🔲 stub
  3. XPU   (Intel ARC)        — Phase III 🔲 stub
  4. CPU   (int8 quantization) — always available

Usage::

    from gensrt.gpu_probe import detect_backend
    backend = detect_backend(gpu_id=0)

Why CTranslate2 and not PyTorch
-------------------------------
GenSRT used to answer "is there a GPU?" with ``torch.cuda.is_available()``.
That worked, but it made PyTorch — a ~2.5 GB dependency — mandatory for every
user, including CPU-only users who could never benefit from it, purely to
evaluate one boolean.  For a tool distributed to places where bandwidth is
metered, that is a real cost paid by the people least able to absorb it.

``ctranslate2.get_cuda_device_count()`` answers the same question and is a
*better* answer, because CTranslate2 is the library that actually runs
inference.  Asking PyTorch whether CUDA works tells you whether *PyTorch's*
CUDA build is healthy, which is only correlated with what GenSRT needs.
CTranslate2 is already a hard dependency via faster-whisper, so this probe
costs nothing.

Since v1.2.5 nothing in GenSRT uses PyTorch at all — the offline translation
engines that needed it were removed.  The torch branches below are retained
only as an opportunistic fallback: if torch happens to be present in the
environment for unrelated reasons, its CUDA probe and device-name lookup are
used.  Nothing requires it, and a torch-free install is the normal case.

Caveat this probe cannot cover
------------------------------
A positive result means the NVIDIA *driver* is present and exposes at least
one device.  It does not prove that cuBLAS and cuDNN 9 are installed, which
CTranslate2 also needs for GPU inference.  That gap is covered at model-load
time by :func:`gensrt.asr._model_loader.load_whisper_model`, which falls back
to CPU with a loud warning rather than failing the run.
"""

from __future__ import annotations

import logging
import subprocess

from gensrt.models import GPUBackend

logger = logging.getLogger(__name__)


def _cuda_device_count() -> int | None:
    """Return the number of visible CUDA devices, or ``None`` if unknowable.

    Tries CTranslate2 first (the library that actually runs inference), then
    PyTorch if it happens to be installed.  ``None`` means neither probe could
    run — distinct from ``0``, which means "probed successfully, no devices".
    """
    try:
        import ctranslate2

        count = int(ctranslate2.get_cuda_device_count())
        logger.debug("CTranslate2 reports %d CUDA device(s).", count)
        return count
    except ImportError:
        logger.debug("ctranslate2 not importable; trying torch.")
    except Exception as exc:
        logger.debug("CTranslate2 CUDA probe failed: %s", exc)

    try:
        import torch

        if not torch.cuda.is_available():
            logger.debug("torch.cuda.is_available() returned False.")
            return 0
        count = int(torch.cuda.device_count())
        logger.debug("PyTorch reports %d CUDA device(s).", count)
        return count
    except ImportError:
        logger.debug("PyTorch is not installed either; CUDA count unknown.")
        return None
    except Exception as exc:
        logger.debug("PyTorch CUDA probe failed: %s", exc)
        return None


def cuda_device_name(gpu_id: int = 0) -> str | None:
    """Best-effort human-readable name for CUDA device *gpu_id*.

    Purely cosmetic — used to make the startup log useful.  Tries PyTorch if
    it is installed, then ``nvidia-smi``.  Returns ``None`` when neither is
    available; callers should degrade to something like ``"CUDA device 0"``.
    """
    try:
        import torch

        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            return str(torch.cuda.get_device_name(gpu_id))
    except Exception:
        pass

    try:
        from gensrt.ffmpeg_util import get_subprocess_creationflags

        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
            encoding="utf-8", errors="replace",
            creationflags=get_subprocess_creationflags(),
        )
        if proc.returncode == 0:
            names = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
            if gpu_id < len(names):
                return names[gpu_id]
    except Exception as exc:
        logger.debug("nvidia-smi name lookup unavailable: %s", exc)

    return None


def _probe_cuda(gpu_id: int = 0) -> bool:
    """Check whether a usable CUDA device *gpu_id* is present."""
    count = _cuda_device_count()

    if count is None:
        logger.debug("No CUDA probe available; assuming no CUDA.")
        return False
    if count == 0:
        logger.debug("No CUDA devices visible.")
        return False
    if gpu_id >= count:
        logger.warning(
            "Requested gpu_id=%d but only %d CUDA device(s) visible.", gpu_id, count
        )
        return False

    name = cuda_device_name(gpu_id) or f"CUDA device {gpu_id}"
    logger.info("CUDA device %d: %s  (total devices: %d)", gpu_id, name, count)
    return True


def _probe_rocm() -> bool:  # pragma: no cover
    """Check whether ROCm is available.  Phase II stub — always returns False."""
    logger.debug("ROCm probe: stubbed (Phase II).")
    return False


def _probe_xpu() -> bool:  # pragma: no cover
    """Check whether Intel XPU is available.  Phase III stub — always returns False."""
    logger.debug("XPU probe: stubbed (Phase III).")
    return False


def detect_backend(gpu_id: int = 0) -> GPUBackend:
    """Detect the best available compute backend.

    Probes in priority order and returns the first working backend.
    Falls back to CPU if no GPU backend is found.

    Args:
        gpu_id: CUDA device ordinal for the NVIDIA probe.

    Returns:
        The :class:`GPUBackend` enum member for the best available backend.
    """
    if _probe_cuda(gpu_id):
        logger.info("Selected backend: CUDA (gpu_id=%d)", gpu_id)
        return GPUBackend.CUDA

    if _probe_rocm():
        logger.info("Selected backend: ROCm")
        return GPUBackend.ROCM

    if _probe_xpu():
        logger.info("Selected backend: XPU")
        return GPUBackend.XPU

    logger.warning(
        "No GPU backend detected — falling back to CPU (int8 quantization). "
        "Transcription will be significantly slower."
    )
    return GPUBackend.CPU


def cuda_is_available(gpu_id: int = 0) -> bool:
    """Public wrapper around the CUDA probe, for callers that want a bool."""
    return _probe_cuda(gpu_id)


def backend_to_ct2_device(backend: GPUBackend) -> tuple[str, str]:
    """Map a :class:`GPUBackend` to a (device, compute_type) pair for CTranslate2.

    Args:
        backend: Detected backend.

    Returns:
        ``(device_str, compute_type_str)`` suitable for
        ``faster_whisper.WhisperModel``.
    """
    match backend:
        case GPUBackend.CUDA:
            return "cuda", "float16"
        case GPUBackend.ROCM:
            return "cuda", "float16"   # ROCm exposes a CUDA-compat layer
        case GPUBackend.XPU:
            return "cpu", "int8"       # Placeholder until OpenVINO path is wired
        case GPUBackend.CPU:
            return "cpu", "int8"
        case _:
            return "cpu", "int8"


def default_compute_type_for(device: str) -> str:
    """Return a sensible default compute type for *device*.

    Asks CTranslate2 which compute types the device actually supports rather
    than assuming, so an older GPU without efficient float16 support gets a
    working default instead of relying on the load-time fallback chain.
    """
    preferred = ["float16", "int8_float16", "int8"] if device == "cuda" else ["int8", "float32"]
    try:
        import ctranslate2

        supported = set(ctranslate2.get_supported_compute_types(device))
        for ct in preferred:
            if ct in supported:
                return ct
        logger.debug(
            "No preferred compute type supported on %s (have: %s); using int8.",
            device, sorted(supported),
        )
    except Exception as exc:
        logger.debug("Could not query supported compute types for %s: %s", device, exc)

    return preferred[0]
