"""Decide, before loading a translator, whether it can go on the GPU at all.

The OOM ladder (``_oom.py``) handles a model that fits but whose batches
do not.  A model that does not fit *as a whole* used to walk the same
ladder — four CUDA out-of-memory errors, then a reload on the CPU — and
the user saw a two-hour "hang" instead of one sentence.  So: compare the
model's size on disk with the free VRAM first.

The test is deliberately a floor, not an estimate.  CTranslate2 holds at
least the weights' on-disk size in device memory (the int8_float16
MADLAD conversion is ~3.3 GB on disk and ~5.1 GB resident on a card that
has to keep its float32 tensors), so ``free < on_disk`` is a certain no;
anything above that is handed to the ladder, which has the real answer.
Measured: MADLAD-400 3B loads and translates on a 6 GB RTX 3060 Laptop
with ~5.4 GB free (2026-10-07), so this never refuses that card.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_WEIGHT_FILES = ("model.bin",)


def model_size_mib(model_dir: Path) -> int:
    """Size on disk of the weight files in *model_dir*, in MiB."""
    total = 0
    for name in _WEIGHT_FILES:
        p = Path(model_dir) / name
        if p.is_file():
            total += p.stat().st_size
    if total == 0:   # unfamiliar layout: count everything
        total = sum(p.stat().st_size for p in Path(model_dir).rglob("*") if p.is_file())
    return total // (1024 * 1024)


def gpu_can_hold(model_dir: Path, engine_name: str) -> bool:
    """False only when the weights alone exceed the free VRAM.

    Logs one WARNING with the numbers and the way out when it refuses;
    says nothing when it cannot tell (no nvidia-smi) and lets the ladder
    decide.
    """
    from gensrt.gpu_mem import free_mib

    free = free_mib()
    if free is None:
        return True
    need = model_size_mib(model_dir)
    if need <= 0 or free >= need:
        return True
    logger.warning(
        "%s: the model's weights are %d MiB on disk and only %d MiB of VRAM is "
        "free — it cannot fit on the GPU, so translation will run on the CPU "
        "(hours for a feature film with a 3B model).  A smaller engine "
        "(translation_engine=nllb, ~650 MB) stays on the GPU.",
        engine_name, need, free,
    )
    return False
