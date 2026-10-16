"""Facts about the current run that the stages know and the summary needs.

The pipeline is a sequence of long functions and engines that each learn
something the user wants in one line at the end — which device Whisper
ended up on after its ladder, which device the translator ended up on,
what the card held after the model was released, how long the audio read
took.  Threading those back through return values would touch every
signature; this is a per-run notebook the stages write into and the
pipeline reads at the end.

One run at a time per process (the CLI batch loop and the server's
single-operation gate both guarantee that), so a module-level dict reset
by :func:`start` is enough.  Writes never fail the run.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_facts: dict[str, Any] = {}
_t0: float = 0.0


def start() -> None:
    """Reset for a new run."""
    global _t0
    _facts.clear()
    _t0 = time.perf_counter()


def note(**kw: Any) -> None:
    """Record facts: ``note(asr_device="cuda", asr_compute="float16")``."""
    _facts.update(kw)


def timing(stage: str, seconds: float) -> None:
    _facts[f"t_{stage}"] = round(float(seconds), 1)


def snapshot() -> dict[str, Any]:
    return dict(_facts)


def _fmt_s(v: Any) -> str:
    v = float(v)
    return f"{v / 60:.1f} min" if v >= 120 else f"{v:.1f}s"


def summary_line(facts: dict[str, Any] | None = None) -> str:
    """One line for the log and the batch footer; only the facts present.

    Formats *facts* when given (a ``TranscriptionResult.facts`` snapshot),
    else the current run's notebook."""
    f = _facts if facts is None else facts
    parts: list[str] = []
    if "t_extract" in f:
        rate = f" at {f['extract_mb_s']:.0f} MB/s" if f.get("extract_mb_s") else ""
        parts.append(f"extract {_fmt_s(f['t_extract'])}{rate}")
    if "t_asr" in f:
        where = f.get("asr_device", "")
        if where and f.get("asr_compute"):
            where += f"/{f['asr_compute']}"
        parts.append(f"ASR {_fmt_s(f['t_asr'])}" + (f" on {where}" if where else ""))
    if "t_translate" in f:
        where = f.get("translate_device", "")
        eng = f.get("translate_engine", "")
        parts.append(f"translate {_fmt_s(f['t_translate'])}"
                     + (f" via {eng}" if eng else "") + (f" on {where}" if where else ""))
    if "vram_after_asr_mib" in f:
        parts.append(f"VRAM after ASR release {f['vram_after_asr_mib']} MiB")
    if "t_total" in f:
        parts.append(f"total {_fmt_s(f['t_total'])}")
    return " | ".join(parts)
