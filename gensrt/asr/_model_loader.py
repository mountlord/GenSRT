"""Shared Whisper model loading with compute-type and device fallback.

Both ASR engines need identical loading behaviour, and both need the *same*
fallback ladder, so the logic lives here rather than being duplicated.  (It
was duplicated verbatim in both engines through v1.2.1, with a comment noting
the duplication was deliberate in case the engines diverged.  They did not
diverge, and the CPU-fallback work below needed to land in both places, so it
is now shared.  If an engine ever genuinely needs different loading, it can
stop calling this.)

Two fallback ladders run, outer to inner:

    device:        requested → cpu
    compute_type:  requested → int8_float16 → int8   (on CUDA)
                   requested → int8                  (on CPU)

The device fallback matters more than it used to.  GenSRT no longer ships
PyTorch (see :mod:`gensrt.gpu_probe`), and PyTorch was previously the thing
that dragged the CUDA runtime DLLs onto the machine as a side effect.  A user
whose driver is fine but whose cuBLAS/cuDNN install is missing or mismatched
now gets a slow-but-working CPU run and a loud warning, instead of a failed
job and a stack trace.  Slower is recoverable; failed is not.
"""

from __future__ import annotations

import logging
from pathlib import Path

from gensrt.exceptions import TranscriptionError
from gensrt.models import TranscriptionConfig

logger = logging.getLogger(__name__)


# Substrings that identify an environment-level failure — a missing library, a
# broken CUDA install, an exhausted GPU — as opposed to a problem with the
# audio being decoded.  The distinction matters because the two need opposite
# responses: a bad chunk should be skipped so the rest of the file still
# works, while a broken environment will fail identically on every chunk and
# must trigger a fallback or a hard error instead of 43 skipped chunks and an
# empty subtitle file.
_ENVIRONMENT_ERROR_MARKERS = (
    "is not found or cannot be loaded",
    "cublas",
    "cublaslt",
    "cudnn",
    "cudart",
    "nvrtc",
    "cuda driver",
    "cuda runtime",
    "cuda error",
    "cuda failure",
    "no kernel image",
    "out of memory",
    "invalid device",
    "device ordinal",
    "unknown compute type",
)


def is_environment_error(exc: BaseException) -> bool:
    """True when *exc* looks like a broken environment rather than bad input.

    Deliberately matches on message text.  CTranslate2 surfaces these as plain
    ``RuntimeError``, so there is no exception type to key on, and being
    slightly over-inclusive is the safe direction: treating a data problem as
    an environment problem costs one unnecessary CPU fallback, while treating
    an environment problem as a data problem produces an empty output file
    and no clear reason why.
    """
    text = str(exc).lower()
    return any(marker in text for marker in _ENVIRONMENT_ERROR_MARKERS)


def _compute_type_ladder(device: str, requested: str) -> list[str]:
    """Return the compute types to try, in order, for *device*."""
    ladder = [requested]
    if device == "cuda":
        for ct in ("int8_float16", "int8"):
            if ct not in ladder:
                ladder.append(ct)
    else:
        # float16 is not a CPU compute type in CTranslate2; int8 always is.
        if requested in ("float16", "int8_float16"):
            ladder = ["int8"]
        elif "int8" not in ladder:
            ladder.append("int8")
    return ladder


def load_whisper_model(
    wav_path: Path,
    config: TranscriptionConfig,
    WhisperModel,  # noqa: N803 — capitalised to mirror the upstream class
    *,
    status=None,
):
    """Load a faster-whisper model, degrading compute type then device.

    Args:
        wav_path:     Only used to build a useful error message.
        config:       Supplies ``model``, ``device``, ``compute_type``.
        WhisperModel: The ``faster_whisper.WhisperModel`` class, passed in so
                      this module never imports faster-whisper at module load.
        status:       Optional ``(str) -> None`` callback used to surface a
                      GPU→CPU fallback in the GUI, where nobody reads the log.

    Returns:
        A loaded ``WhisperModel``.

    Raises:
        TranscriptionError: If every device/compute-type combination failed.
    """
    from gensrt.model_paths import resolve_model

    # A bare name may refer to a directory under <app dir>/models; anything
    # else passes through as a HuggingFace repo ID.
    model_ref = resolve_model(config.model)

    requested_device = (config.device or "cpu").strip().lower()
    requested_ct = config.compute_type

    device_ladder = [requested_device]
    if requested_device != "cpu":
        device_ladder.append("cpu")

    logger.info(
        "Loading Whisper model: %s  (device=%s, compute=%s)",
        model_ref, requested_device, requested_ct,
    )

    last_exc: Exception | None = None

    for device in device_ladder:
        for ct in _compute_type_ladder(device, requested_ct):
            try:
                model = WhisperModel(model_ref, device=device, compute_type=ct)
            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "WhisperModel load failed (device=%r, compute_type=%r): %s",
                    device, ct, exc,
                )
                continue

            if device != requested_device:
                msg = (
                    f"GPU unavailable for inference — fell back to {device.upper()}. "
                    f"Transcription will be substantially slower. "
                    f"(Cause: {last_exc})"
                )
                logger.warning("%s", msg)
                if callable(status):
                    status(
                        f"GPU unavailable — running on {device.upper()} "
                        f"(much slower)."
                    )
            elif ct != requested_ct:
                logger.warning(
                    "compute_type=%r unsupported on %s — fell back to %r.",
                    requested_ct, device, ct,
                )

            return model

    detail = (
        f"Failed to load Whisper model {config.model!r} on any of "
        f"{device_ladder}: {last_exc}"
    )

    # "Unable to open file 'model.bin'" is CTranslate2 saying the directory
    # is not one of its models. By far the most common cause is pointing
    # GenSRT at an unconverted PyTorch repo, and the raw message gives a user
    # nothing to act on.
    if "model.bin" in str(last_exc):
        name = config.model
        suggestion = ""
        if "/" in name and not Path(name).exists():
            org, repo = name.split("/", 1)
            suggestion = f" Try '{org}/ct2-{repo}' if it exists."
        detail += (
            "\n\nThis usually means the model is a PyTorch/transformers "
            "repository rather than a CTranslate2 conversion. GenSRT runs on "
            f"CTranslate2 and needs the converted form.{suggestion} "
            "To convert it yourself:\n"
            f"  ct2-transformers-converter --model {name} "
            "--output_dir <dir> --quantization float16"
        )

    raise TranscriptionError(str(wav_path), detail)
