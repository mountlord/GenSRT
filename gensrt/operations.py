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
    from gensrt.gpu_probe import GPUBackend, detect_backend
    backend = detect_backend(gpu_id)
    return backend.name.lower()


# ── Config resolution ─────────────────────────────────────────────────────

def build_transcription_config(
    merged: dict[str, Any],
    *,
    auto_detect_backend: bool = True,
) -> TranscriptionConfig:
    """Build a :class:`TranscriptionConfig` from a merged settings dict.

    If *auto_detect_backend* is ``True``, probe the GPU and override the
    ``backend`` / ``device`` / ``compute_type`` fields accordingly.

    Args:
        merged:              Dict from :func:`~gensrt.config.merge_config`.
        auto_detect_backend: Whether to probe and override GPU settings.

    Returns:
        Fully resolved :class:`TranscriptionConfig`.
    """
    if auto_detect_backend:
        gpu_id = int(merged.get("gpu_id", 0))
        backend_str = detect_gpu_backend(gpu_id)

        from gensrt.gpu_probe import GPUBackend, backend_to_ct2_device
        try:
            backend = GPUBackend[backend_str.upper()]
        except KeyError:
            backend = GPUBackend.CPU

        device, compute_type = backend_to_ct2_device(backend)
        merged = {
            **merged,
            "backend": backend_str,
            "device": device,
            "compute_type": merged.get("compute_type") or compute_type,
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
) -> Path:
    """Thin wrapper around :func:`~gensrt.utils.media_files.resolve_output_path`."""
    from gensrt.utils.media_files import resolve_output_path as _resolve
    return _resolve(input_path, output_dir, output_filename)
