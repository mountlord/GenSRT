"""Per-chunk decode diagnostics for the chunked ASR path.

Motivation
----------
On a 197s Malayalam news clip split into 43 chunks, ten of those chunks
consumed 78% of total decode time — 15-32 seconds each, against a median of
under two seconds.  The slow set was *identical* across repeated runs, so the
cause is a property of the audio content, not scheduling noise.

Two questions follow, and this module exists to answer both:

**What is the decoder doing?**  faster-whisper falls back through a
temperature ladder (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) whenever a decode fails its
``compression_ratio_threshold`` or ``log_prob_threshold`` check, re-decoding
from scratch at each step.  Six passes at ``beam_size=5`` would account for
roughly the observed penalty.  Segment objects carry the temperature that
finally succeeded, along with the quality metrics that drove the fallback, so
recording them settles the question from data rather than inference.

**Why is that audio hard?**  No amount of telemetry answers this.  It needs a
person who speaks the language to listen to the chunk.  So the recorder can
also export each chunk's audio, named with its timing and decode telemetry, so
the expensive ones are obvious in a file listing.

Cost when disabled
------------------
:class:`NullChunkRecorder` is used unless a debug directory is configured.  It
has the same interface and does nothing.  The one thing that runs regardless
is the wall-clock timing of each decode — a ``perf_counter`` call per chunk —
because the outlier summary it enables is worth having in every run log.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

# A chunk is called out in the summary when it takes this many times the
# median chunk's wall time.  3x is well clear of ordinary variation (the fast
# chunks in the observed run clustered within about 2x of each other) while
# still catching every member of the slow set.
OUTLIER_FACTOR = 3.0
# Cap on how many outliers get listed, so a pathological file cannot flood
# the log with a hundred lines.
MAX_OUTLIERS_LOGGED = 12


class ChunkRecord:
    """Telemetry for a single decoded chunk."""

    __slots__ = (
        "index", "start_s", "end_s", "wall_s", "n_segments", "text",
        "temperature", "avg_logprob", "compression_ratio", "no_speech_prob",
        "failed",
    )

    def __init__(self, index: int, start_s: float, end_s: float):
        self.index = index
        self.start_s = start_s
        self.end_s = end_s
        self.wall_s = 0.0
        self.n_segments = 0
        self.text = ""
        # Worst-case values across the chunk's segments.  "Worst" is the
        # direction that indicates difficulty: highest temperature reached,
        # lowest average log-probability, highest compression ratio (the
        # repetition signal), highest no-speech probability.
        self.temperature: float | None = None
        self.avg_logprob: float | None = None
        self.compression_ratio: float | None = None
        self.no_speech_prob: float | None = None
        self.failed = False

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def realtime_factor(self) -> float:
        """Wall seconds per second of audio.  Below 1.0 is faster than real time."""
        return self.wall_s / self.duration_s if self.duration_s > 0 else 0.0

    def observe_segments(self, segments) -> None:
        """Fold a chunk's segment list into the worst-case telemetry."""
        self.n_segments = len(segments)
        self.text = " ".join(s.text.strip() for s in segments if s.text.strip())

        for seg in segments:
            # getattr throughout: these fields are present in faster-whisper
            # 1.x but are not part of any stability guarantee, and
            # `temperature` is Optional even when present.
            t = getattr(seg, "temperature", None)
            if t is not None:
                self.temperature = t if self.temperature is None else max(self.temperature, t)

            lp = getattr(seg, "avg_logprob", None)
            if lp is not None:
                self.avg_logprob = lp if self.avg_logprob is None else min(self.avg_logprob, lp)

            cr = getattr(seg, "compression_ratio", None)
            if cr is not None:
                self.compression_ratio = (
                    cr if self.compression_ratio is None else max(self.compression_ratio, cr)
                )

            ns = getattr(seg, "no_speech_prob", None)
            if ns is not None:
                self.no_speech_prob = (
                    ns if self.no_speech_prob is None else max(self.no_speech_prob, ns)
                )

    def as_dict(self) -> dict:
        def r(v, n=3):
            return None if v is None else round(v, n)

        return {
            "index": self.index,
            "start_s": r(self.start_s),
            "end_s": r(self.end_s),
            "duration_s": r(self.duration_s),
            "wall_s": r(self.wall_s),
            "realtime_factor": r(self.realtime_factor, 2),
            "temperature": r(self.temperature, 2),
            "avg_logprob": r(self.avg_logprob),
            "compression_ratio": r(self.compression_ratio),
            "no_speech_prob": r(self.no_speech_prob),
            "n_segments": self.n_segments,
            "chars": len(self.text),
            "failed": self.failed,
            "text": self.text,
        }


class NullChunkRecorder:
    """No-op recorder used when chunk export is disabled."""

    enabled = False

    def record(self, rec: ChunkRecord, chunk_wav: Path) -> None:
        pass

    def finalize(self, source_name: str) -> None:
        pass


class ChunkRecorder:
    """Exports chunk audio plus a telemetry manifest to a directory."""

    enabled = True

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[ChunkRecord] = []

    def record(self, rec: ChunkRecord, chunk_wav: Path) -> None:
        """Store *rec* and copy its audio out before the temp dir is reaped.

        The filename carries index, wall time, temperature, and source
        timestamps, so a plain directory listing sorts chronologically while
        making the expensive chunks visually obvious — no need to consult the
        CSV to decide what to listen to first.
        """
        self.records.append(rec)

        temp_tag = "Tna" if rec.temperature is None else f"T{rec.temperature:.1f}"
        name = (
            f"chunk_{rec.index:03d}"
            f"_wall{rec.wall_s:06.2f}s"
            f"_{temp_tag}"
            f"_{rec.start_s:07.2f}-{rec.end_s:07.2f}.wav"
        )
        try:
            shutil.copyfile(chunk_wav, self.out_dir / name)
        except OSError as exc:
            logger.warning("Could not export chunk %d audio: %s", rec.index, exc)

    def finalize(self, source_name: str) -> None:
        """Write ``chunks.csv``, ``chunks.json``, and a README."""
        if not self.records:
            return

        rows = [r.as_dict() for r in self.records]

        csv_path = self.out_dir / "chunks.csv"
        try:
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            logger.warning("Could not write %s: %s", csv_path, exc)

        json_path = self.out_dir / "chunks.json"
        try:
            json_path.write_text(
                json.dumps({"source": source_name, "chunks": rows}, indent=2,
                           ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not write %s: %s", json_path, exc)

        try:
            (self.out_dir / "README.txt").write_text(_README, encoding="utf-8")
        except OSError:
            pass

        logger.info(
            "Chunk diagnostics written: %d chunk(s) + chunks.csv → %s",
            len(self.records), self.out_dir,
        )


def log_timing_summary(records: list[ChunkRecord], engine_name: str) -> None:
    """Log decode-time outliers, and what the decoder was doing on them.

    Runs on every chunked transcription, export enabled or not, because the
    single most useful thing to know about a slow run is which chunks were
    slow and what temperature they finished at.  Emits nothing when no chunk
    is an outlier, so a healthy run stays quiet.
    """
    timed = [r for r in records if r.wall_s > 0 and not r.failed]
    if len(timed) < 4:
        return

    walls = sorted(r.wall_s for r in timed)
    median = walls[len(walls) // 2]
    total = sum(walls)
    if median <= 0:
        return

    outliers = sorted(
        (r for r in timed if r.wall_s >= median * OUTLIER_FACTOR),
        key=lambda r: r.wall_s, reverse=True,
    )
    if not outliers:
        return

    shown = outliers[:MAX_OUTLIERS_LOGGED]
    outlier_total = sum(r.wall_s for r in outliers)

    logger.info(
        "[%s] Decode time is concentrated: %d/%d chunks (%.0f%%) took >=%.1fx the "
        "median (%.2fs) and account for %.0f%% of decode time.",
        engine_name, len(outliers), len(timed),
        100.0 * len(outliers) / len(timed), OUTLIER_FACTOR, median,
        100.0 * outlier_total / total if total else 0.0,
    )
    logger.info(
        "[%s]   %-5s %-9s %-8s %-7s %-6s %-7s %s",
        engine_name, "chunk", "wall", "audio", "x-rt", "temp", "cmprsn", "start",
    )
    for r in shown:
        logger.info(
            "[%s]   %-5d %-9s %-8s %-7s %-6s %-7s %.1fs",
            engine_name, r.index,
            f"{r.wall_s:.2f}s", f"{r.duration_s:.2f}s",
            f"{r.realtime_factor:.1f}x",
            "n/a" if r.temperature is None else f"{r.temperature:.1f}",
            "n/a" if r.compression_ratio is None else f"{r.compression_ratio:.2f}",
            r.start_s,
        )

    # The interpretive line.  faster-whisper only raises temperature when a
    # decode fails its quality gates, so a non-zero temperature on the slow
    # chunks is direct evidence of fallback rather than an inference from
    # timing alone.
    temps = [r.temperature for r in outliers if r.temperature is not None]
    if temps and max(temps) > 0.0:
        logger.info(
            "[%s]   Non-zero temperature on %d/%d slow chunk(s) (max %.1f) — these "
            "hit faster-whisper's temperature fallback, which re-decodes from "
            "scratch at each step and explains the cost.",
            engine_name, sum(1 for t in temps if t > 0.0), len(outliers), max(temps),
        )
    elif temps:
        logger.info(
            "[%s]   All slow chunks finished at temperature 0.0 — the cost is NOT "
            "temperature fallback; look at chunk length or token count instead.",
            engine_name,
        )


_README = """\
GenSRT — chunk diagnostics
==========================

One WAV per chunk fed to the ASR model, plus chunks.csv / chunks.json.

Filename format
---------------
    chunk_007_wall018.42s_T0.8_0012.30-0017.60.wav
          |         |        |        |
          |         |        |        +-- start-end in the source audio (s)
          |         |        +----------- temperature the decode finished at
          |         +-------------------- wall-clock decode time
          +------------------------------ chunk index (1-based, chronological)

Sorting by name gives chronological order; the wall time in the name makes
expensive chunks obvious at a glance.

Columns
-------
index               1-based, chronological
start_s / end_s     position in the source audio
duration_s          chunk length
wall_s              decode wall-clock time
realtime_factor     wall_s / duration_s.  Below 1.0 = faster than real time.
temperature         temperature the successful decode ran at.  0.0 means the
                    first attempt passed.  Anything above 0.0 means
                    faster-whisper's quality gates rejected earlier attempts
                    and re-decoded — the usual reason a chunk is slow.
avg_logprob         lowest across the chunk's segments.  More negative = the
                    model was less confident.
compression_ratio   highest across segments.  High values indicate repetition;
                    above 2.4 triggers the fallback by default.
no_speech_prob      highest across segments.  High = the model suspects there
                    is no speech here.
n_segments          cues emitted from this chunk
chars               characters of text emitted
failed              True if the decode raised and the chunk was skipped
text                what the model produced

Where to start
--------------
Sort chunks.csv by wall_s descending and listen to the top few. Compare what
you hear against the `text` column — whether the model produced repetition,
whether the audio is genuinely hard (overlapping speakers, music under
speech, phone-line audio, a name or English loanword), or whether the chunk
boundary cut a word in half.

If temperature is above 0.0 on the slow chunks, the decoder was retrying. The
question then becomes *what about that audio* made the first attempt fail.
That part needs ears, not telemetry.
"""
