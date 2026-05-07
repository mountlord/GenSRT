"""GPU backend detection and selection.

Probes the system for available hardware acceleration in priority order:
  1. CUDA  (NVIDIA)           — Phase I  ✅
  2. ROCm  (AMD)              — Phase II 🔲 stub
  3. XPU   (Intel ARC)        — Phase III 🔲 stub
  4. CPU   (int8 quantization) — always available

Usage::

    from gensrt.gpu_probe import detect_backend
    backend = detect_backend(gpu_id=0)
"""

from __future__ import annotations

import logging

from gensrt.models import GPUBackend

logger = logging.getLogger(__name__)


def _probe_cuda(gpu_id: int = 0) -> bool:
    """Check whether CUDA is available via PyTorch."""
    try:
        import torch

        if not torch.cuda.is_available():
            logger.debug("torch.cuda.is_available() returned False.")
            return False

        count = torch.cuda.device_count()
        if count == 0:
            logger.debug("CUDA available but no devices found.")
            return False

        if gpu_id >= count:
            logger.warning(
                "Requested gpu_id=%d but only %d CUDA device(s) visible.", gpu_id, count
            )
            return False

        name = torch.cuda.get_device_name(gpu_id)
        logger.info("CUDA device %d: %s  (total devices: %d)", gpu_id, name, count)
        return True

    except ImportError:
        logger.debug("PyTorch is not installed; CUDA unavailable.")
        return False
    except Exception as exc:
        logger.debug("CUDA probe failed: %s", exc)
        return False


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
