"""Out-of-memory ladder for the CTranslate2 translation engines.

Measured on a Tesla P4 (8 GB): MADLAD-400 3B loaded on CUDA in int8, then
the first ``translate_batch`` of 38 subtitle cues failed with ``CUDA failed
with error out of memory`` — and the file was written UNTRANSLATED behind a
one-line warning.  On a pre-Volta card the activations run in float32 (no
FP16 path), so a 1024-token batch at beam 4 is what did not fit; the weights
did.

Rather than give up, walk a ladder:

1. Retry with the token batch halved, down to 128 — activations shrink,
   weights stay put, the GPU is still used.
2. Reload the translator on CPU and translate there.  Slow, but the output
   is right, and correctness beats speed for a subtitle file.
3. Only then let the caller keep the source text — and the caller must say
   so loudly, not in a debug line.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

_OOM_MARKERS = ("out of memory", "cuda_error_out_of_memory", "cudaErrorMemoryAllocation".lower())


def is_oom(exc: BaseException) -> bool:
    t = str(exc).lower()
    return any(m in t for m in _OOM_MARKERS)


#: Cues per engine call.  A whole 550-cue file used to go through one call,
#: so an OOM part-way redid everything from the start at the next rung —
#: files 2 and 3 of the measured run spent ~60 s each re-translating.  A
#: slice bounds what a retry throws away.
SLICE_CUES = 64


def translate_with_oom_ladder(
    run: Callable[[int], list],
    *,
    max_batch_tokens: int,
    move_to_cpu: Callable[[], None],
    engine_name: str,
    floor: int = 128,
) -> tuple[list, int]:
    """Call ``run(max_batch_tokens)``; on CUDA OOM shrink the batch, then move to CPU.

    Returns ``(result, batch_that_held)`` so the caller can START there next
    time instead of re-walking the ladder from the top on every call.

    Args:
        run:              Performs the translation with the given token batch cap.
        max_batch_tokens: Starting cap.
        move_to_cpu:      Reloads the engine's translator on CPU.
        engine_name:      For log lines.
        floor:            Smallest batch to try before leaving the GPU.
    """
    batch = max_batch_tokens
    while True:
        try:
            return run(batch), batch
        except Exception as exc:
            if not is_oom(exc):
                raise
            if batch > floor:
                batch = max(floor, batch // 2)
                logger.warning(
                    "%s: CUDA out of memory — retrying with max_batch_size=%d tokens.",
                    engine_name, batch,
                )
                continue
            logger.warning(
                "%s: CUDA out of memory even at %d-token batches — reloading on "
                "CPU for this run. Translation will be much slower but complete. "
                "(A 3B model plus Whisper does not fit this GPU; consider the "
                "nllb engine, ~650 MB.)",
                engine_name, batch,
            )
            move_to_cpu()
            return run(max_batch_tokens), max_batch_tokens
