"""Best-effort GPU memory readings for the log.

Measured on an 8 GB card over five files: the memory still held after the
Whisper model was released climbed 603 → 1,017 → 1,445 → 1,865 → 2,247 MiB,
until a 15-cue translation no longer fit.  Inferring that from a separate
monitor is slow; the run should say it.  ``nvidia-smi`` ships with the
driver on every machine that has a CUDA GPU, so ask it — and never let the
asking fail a run.
"""

from __future__ import annotations

import gc
import logging
import subprocess

logger = logging.getLogger(__name__)


def used_mib() -> int | None:
    """Used MiB on GPU 0, or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return int(out.stdout.strip().splitlines()[0])
    except Exception:
        pass
    return None


def release(what: str, *objs) -> None:
    """Drop references, collect, and log what the card holds afterwards.

    faster-whisper's model sits in Python reference cycles; without an
    explicit collection it is freed at the next GC pass — which may be after
    the next model has already loaded.  Collect now, so the release is
    deterministic, then say what is left.
    """
    del objs
    gc.collect()
    used = used_mib()
    if used is not None:
        logger.info("GPU memory after releasing %s: %d MiB in use", what, used)
