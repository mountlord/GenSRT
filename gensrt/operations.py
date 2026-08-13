"""Shared business logic for GenSRT.

This module is the single source of truth for all non-trivial operations.
Both :mod:`gensrt.cli` and :mod:`gensrt.server` call these functions —
neither duplicates config loading, GPU detection, or pipeline execution.

Architecture quote:
    "The server is a thin HTTP adapter over operations.py, not a second pipeline."
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from gensrt.models import TranscriptionConfig, TranscriptionResult

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
StatusCallback = Callable[[str], None]


# ── Config file I/O ────────────────────────────────────────────────────────

def read_config_file(
    path: Path | None = None,
    *,
    default_if_missing: bool = True,
) -> dict[str, Any]:
    """Load ``gensrt-config.json`` from *path* (or auto-discovered location).

    Args:
        path:               Explicit path override.  Auto-discovers if ``None``.
        default_if_missing: If ``True``, return ``{}`` when the file is absent.
                            If ``False``, raise :class:`~gensrt.exceptions.ConfigError`.

    Returns:
        Parsed config dict.

    Raises:
        ConfigError:      If file is missing and *default_if_missing* is ``False``.
        ConfigParseError: If the file contains invalid JSON.
    """
    from gensrt.config import load_config
    return load_config(path, strict=not default_if_missing)


def write_config_file(path: Path, data: dict[str, Any]) -> None:
    """Serialise *data* to a JSON config file at *path*.

    Args:
        path: Destination path.
        data: Dict to serialise.

    Raises:
        ConfigError: If the file cannot be written.
    """
    from gensrt.exceptions import ConfigError

    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        logger.info("Config written: %s", path)
    except OSError as exc:
        raise ConfigError(f"Cannot write config file {path}: {exc}") from exc


# ── GPU / backend ──────────────────────────────────────────────────────────

def detect_gpu_backend(gpu_id: int = 0) -> str:
    """Probe available compute backends and return the best one as a string.

    Returns:
        One of ``"cuda"``, ``"rocm"``, ``"xpu"``, ``"cpu"``.
    """
    from gensrt.gpu_probe import detect_backend
    backend = detect_backend(gpu_id)
    return backend.name.lower()


# ── Config resolution ─────────────────────────────────────────────────────

def build_transcription_config(
    merged: dict[str, Any],
    *,
    auto_detect_backend: bool = True,
) -> TranscriptionConfig:
    """Build a :class:`TranscriptionConfig` from a merged settings dict.

    Device resolution honours what the user actually asked for:

    ``device: "auto"`` (the default)
        Probe the hardware and use whatever is best.

    ``device: "cpu"``
        Use the CPU.  No probe runs.  This is the escape hatch for a machine
        with a working NVIDIA driver but a broken CUDA/cuDNN install, and for
        anyone who simply wants the deterministic-but-slow path.  Through
        v1.2.1 this setting was silently discarded, which left those users
        with no way out.

    ``device: "cuda"``
        Use CUDA — but verify first.  If no CUDA device is visible we warn
        and use the CPU rather than proceeding into a guaranteed failure.
        Honouring an impossible request helps nobody.

    ``compute_type`` is only defaulted when unset, and the default is derived
    from what CTranslate2 reports the chosen device actually supports.

    Args:
        merged:              Dict from :func:`~gensrt.config.merge_config`.
        auto_detect_backend: Whether hardware probing is permitted at all.
                             ``False`` takes the dict at face value — used by
                             callers that have already resolved the device.

    Returns:
        Fully resolved :class:`TranscriptionConfig`.
    """
    if not auto_detect_backend:
        return TranscriptionConfig.from_dict(merged)

    from gensrt.gpu_probe import (
        GPUBackend,
        backend_to_ct2_device,
        cuda_is_available,
        default_compute_type_for,
    )

    gpu_id = int(merged.get("gpu_id", 0) or 0)
    requested = str(merged.get("device") or "auto").strip().lower()

    if requested == "cpu":
        logger.info("Device explicitly set to CPU — skipping GPU probe.")
        device, backend_str = "cpu", "cpu"

    elif requested == "cuda":
        if cuda_is_available(gpu_id):
            device, backend_str = "cuda", "cuda"
        else:
            logger.warning(
                "Device explicitly set to 'cuda' but no usable CUDA device was "
                "found (gpu_id=%d) — using CPU instead. Transcription will be "
                "significantly slower. Set \"device\": \"auto\" in "
                "gensrt-config.json to silence this.",
                gpu_id,
            )
            device, backend_str = "cpu", "cpu"

    else:
        # "auto" or anything unrecognised — probe and take the best available.
        if requested not in ("auto", ""):
            logger.warning(
                "Unrecognised device %r — treating as 'auto'.", requested
            )
        backend_str = detect_gpu_backend(gpu_id)
        try:
            backend = GPUBackend[backend_str.upper()]
        except KeyError:
            backend = GPUBackend.CPU
        device, _ = backend_to_ct2_device(backend)

    requested_ct = str(merged.get("compute_type") or "auto").strip().lower()
    if requested_ct in ("auto", ""):
        compute_type = default_compute_type_for(device)
    else:
        compute_type = requested_ct

    merged = {
        **merged,
        "backend": backend_str,
        "device": device,
        "compute_type": compute_type,
    }

    return TranscriptionConfig.from_dict(merged)


# ── Core transcription ────────────────────────────────────────────────────

def run_transcription(
    input_path: Path,
    output_path: Path,
    config: TranscriptionConfig,
    *,
    progress: ProgressCallback | None = None,
    status: StatusCallback | None = None,
) -> TranscriptionResult:
    """Run the full pipeline on a single media file.

    This is the single function that both CLI and server call.  It is a
    thin error-normalising wrapper around :func:`gensrt.pipeline.run_pipeline`.

    Args:
        input_path:  Path to the input media file.
        output_path: Path for the output ``.srt`` file.
        config:      Fully resolved transcription config.
        progress:    Optional ``(current, total)`` callback.
        status:      Optional human-readable status callback.

    Returns:
        :class:`TranscriptionResult` for this file.

    Raises:
        GenSRTError: Any pipeline error, re-raised with a consistent type.
    """
    from gensrt.exceptions import GenSRTError, TranscriptionError
    from gensrt.pipeline import run_pipeline

    try:
        return run_pipeline(
            input_path=input_path,
            output_path=output_path,
            config=config,
            progress=progress,
            status=status,
        )
    except GenSRTError:
        raise
    except Exception as exc:
        raise TranscriptionError(str(input_path), str(exc)) from exc


# ── Output path resolution ────────────────────────────────────────────────

def resolve_output_path(
    input_path: Path,
    output_dir: Path | None,
    output_filename: str | None,
    target_language: str = "en",
) -> Path:
    """Thin wrapper around :func:`~gensrt.utils.media_files.resolve_output_path`."""
    from gensrt.utils.media_files import resolve_output_path as _resolve
    return _resolve(input_path, output_dir, output_filename, target_language)
