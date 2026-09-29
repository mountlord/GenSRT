"""Remux MPEG-TS recordings to MP4 so the review window can play them.

Why this exists
---------------
The review window is a Chromium ``<video>`` element inside WebView2, and
its demuxer is Edge's trimmed build of FFmpeg's libavformat.  Desktop
Chromium has never compiled the MPEG-TS demuxer in; Edge shipped it for a
while, which is how ``.ts`` recordings played at all, and an Edge update
took it away.  Symptom, 2026-09-29, on a clean H.264/AAC transport stream
(ffprobe probe_score 100)::

    MEDIA_ERR_SRC_NOT_SUPPORTED — PipelineStatus::DEMUXER_ERROR_COULD_NOT_OPEN:
    FFmpegDemuxer: open context failed

Edge itself now *downloads* a dropped ``.ts`` instead of playing it.  The
container support lived in the runtime, and the runtime moved.

What this does
--------------
Rewrap the stream once, with our own ffmpeg, into MP4 — ``-c copy``, no
re-encode, so the cost is one read and one write of the file (disk-bound,
roughly a minute or two for a 2-hour 1440p recording) — into a cache under
the system temp folder.  ``/api/media`` then serves the MP4 with the same
HTTP Range support the player already uses, so seeking from a cue click
works exactly as it does for a native MP4.  Timestamps survive a stream
copy, so the subtitles stay aligned with the picture.

The cache is keyed by path, size and mtime: an unchanged recording is
instant the second time, a re-download is redone.  A size cap with
oldest-first eviction keeps it from eating the drive.

``build_command``, ``parse_progress`` and ``cache_key`` are pure and
tested without ffmpeg; :class:`RemuxManager` owns the threads and the
subprocesses.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Containers the embedded browser cannot open and we rewrap.
REMUX_EXTS: frozenset[str] = frozenset({".ts", ".m2ts", ".mts"})
CACHE_DIR_NAME = "gensrt_remux"
#: Total cache size before oldest outputs are evicted.
CACHE_MAX_BYTES = 20 * 1024 ** 3
#: An in-progress output older than this with no live job is a leftover
#: from a crash and is swept.
PART_MAX_AGE_S = 24 * 60 * 60
_PART_SUFFIX = ".part"


def needs_remux(path: Path) -> bool:
    return path.suffix.lower() in REMUX_EXTS


def cache_dir() -> Path:
    return Path(tempfile.gettempdir()) / CACHE_DIR_NAME


def cache_key(src: Path, size: int, mtime_ns: int) -> str:
    """Stable id for one version of one file.

    Path, size and mtime together: the same recording downloaded again has
    a new mtime and usually a new size, and gets a fresh remux; an
    unchanged file hits the cache however many times the window opens it.
    """
    h = hashlib.sha1()
    h.update(str(src).encode("utf-8", "surrogateescape"))
    h.update(f"|{size}|{mtime_ns}".encode())
    return h.hexdigest()[:20]


def cached_output(src: Path, directory: Optional[Path] = None) -> Path:
    st = src.stat()
    return (directory or cache_dir()) / f"{cache_key(src, st.st_size, st.st_mtime_ns)}.mp4"


def build_command(ffmpeg: str, src: Path, dst: Path) -> list[str]:
    """ffmpeg arguments for a stream-copy rewrap into a seekable MP4.

    - first video and (if present) first audio stream only; data and
      subtitle streams in a TS are not playable in the element anyway
    - AAC in a transport stream is ADTS-framed and MP4 wants raw AAC; the
      mp4 muxer inserts ``aac_adtstoasc`` itself, and naming it explicitly
      would break a stream whose audio is not AAC.
    - ``+faststart``: move the index to the front after writing, so the
      element can start before it has read the whole file.  Costs a second
      pass over the output at the end; progress sits at 100 % meanwhile.
    - ``-progress pipe:1``: machine-readable ``out_time_us=`` lines on
      stdout, which :func:`parse_progress` reads.
    """
    return [
        ffmpeg, "-hide_banner", "-nostdin", "-y",
        "-loglevel", "error",
        "-fflags", "+genpts+discardcorrupt",
        "-i", str(src),
        "-map", "0:v:0", "-map", "0:a:0?",
        "-dn", "-sn",
        "-c", "copy",
        "-movflags", "+faststart",
        "-progress", "pipe:1", "-nostats",
        "-f", "mp4",
        str(dst),
    ]


def parse_progress(line: str, duration_s: Optional[float]) -> Optional[float]:
    """Fraction complete from one ``-progress`` line, or ``None``.

    ffmpeg ≥ 5 writes ``out_time_us``; older builds write ``out_time_ms``,
    which — despite the name — is also microseconds.  Both are accepted.
    Without a known duration there is nothing to divide by.
    """
    if not duration_s or duration_s <= 0:
        return None
    key, sep, value = line.strip().partition("=")
    if not sep or key not in ("out_time_us", "out_time_ms"):
        return None
    try:
        us = int(value)
    except ValueError:
        return None
    if us < 0:
        return None
    return max(0.0, min(1.0, (us / 1_000_000) / duration_s))


def probe_duration(ffprobe: str, src: Path, *, creationflags: int = 0) -> Optional[float]:
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
            capture_output=True, text=True, timeout=30, creationflags=creationflags,
        ).stdout.strip()
        return float(out) if out else None
    except Exception:
        return None


@dataclass
class RemuxStatus:
    state: str                      # "ready" | "preparing" | "error" | "unprepared"
    progress: Optional[float] = None
    error: Optional[str] = None
    output: Optional[Path] = None

    def as_dict(self) -> dict:
        return {
            "status": self.state,
            "progress": self.progress,
            "error": self.error,
        }


@dataclass
class _Job:
    src: Path
    dst: Path
    status: RemuxStatus
    cancel: threading.Event = field(default_factory=threading.Event)
    proc: Optional[subprocess.Popen] = None
    thread: Optional[threading.Thread] = None


class RemuxManager:
    """One remux per source file, in a background thread, results cached."""

    def __init__(self, *, directory: Optional[Path] = None,
                 ffmpeg: Optional[str] = None, ffprobe: Optional[str] = None,
                 max_bytes: int = CACHE_MAX_BYTES):
        self._dir = directory
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe
        self._max_bytes = max_bytes
        self._jobs: dict[Path, _Job] = {}
        self._lock = threading.Lock()

    # ── binaries, resolved late so tests can inject fakes ──────────────
    def _exes(self) -> tuple[str, str, int]:
        from gensrt.ffmpeg_util import (get_ffmpeg_exe, get_ffprobe_exe,
                                        get_subprocess_creationflags)
        return (self._ffmpeg or get_ffmpeg_exe(),
                self._ffprobe or get_ffprobe_exe(),
                get_subprocess_creationflags())

    @property
    def directory(self) -> Path:
        return self._dir or cache_dir()

    # ── public ─────────────────────────────────────────────────────────
    def status(self, src: Path) -> RemuxStatus:
        """Current state for *src* without starting anything."""
        dst = cached_output(src, self.directory)
        with self._lock:
            job = self._jobs.get(dst)
            if job is not None:
                return job.status
        if dst.is_file():
            try:
                os.utime(dst)          # LRU by last use, for the sweep
            except OSError:
                pass
            return RemuxStatus("ready", 1.0, output=dst)
        return RemuxStatus("unprepared", None)

    def prepare(self, src: Path) -> RemuxStatus:
        """Start a remux for *src* unless one is running or cached."""
        dst = cached_output(src, self.directory)
        with self._lock:
            job = self._jobs.get(dst)
            if job is not None and job.status.state != "error":
                return job.status
            # A failed job is retried by the next prepare(); polling goes
            # through status(), which reports the error as it stands.
            if dst.is_file():
                self._jobs.pop(dst, None)
                return RemuxStatus("ready", 1.0, output=dst)
            job = _Job(src, dst, RemuxStatus("preparing", 0.0))
            self._jobs[dst] = job
            job.thread = threading.Thread(target=self._run, args=(job,),
                                          name="gensrt-remux", daemon=True)
            job.thread.start()
            return job.status

    def cancel(self, src: Path) -> None:
        dst = cached_output(src, self.directory)
        with self._lock:
            job = self._jobs.get(dst)
        if job is not None:
            job.cancel.set()
            if job.proc is not None and job.proc.poll() is None:
                job.proc.terminate()

    def cancel_all(self) -> None:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.cancel.set()
            if job.proc is not None and job.proc.poll() is None:
                job.proc.terminate()

    def sweep(self) -> int:
        """Delete stale partials and, above the size cap, the least
        recently used outputs.  Returns bytes freed."""
        d = self.directory
        if not d.is_dir():
            return 0
        with self._lock:
            active = {j.dst for j in self._jobs.values()}
        freed = 0
        now = time.time()
        entries = []
        for p in d.iterdir():
            if not p.is_file():
                continue
            try:
                st = p.stat()
            except OSError:
                continue
            if p.name.endswith(_PART_SUFFIX):
                final = p.with_name(p.name[:-len(_PART_SUFFIX)])
                if final not in active and now - st.st_mtime > PART_MAX_AGE_S:
                    freed += self._unlink(p, st.st_size)
                continue
            if p.suffix == ".mp4" and p not in active:
                entries.append((st.st_mtime, st.st_size, p))
        total = sum(size for _, size, _ in entries)
        for mtime, size, p in sorted(entries):
            if total <= self._max_bytes:
                break
            freed += self._unlink(p, size)
            total -= size
        return freed

    # ── worker ─────────────────────────────────────────────────────────
    def _run(self, job: _Job) -> None:
        part = job.dst.with_name(job.dst.name + _PART_SUFFIX)
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.sweep()
            ffmpeg, ffprobe, flags = self._exes()
            duration = probe_duration(ffprobe, job.src, creationflags=flags)
            cmd = build_command(ffmpeg, job.src, part)
            logger.info("Remuxing %s → %s (%s)", job.src.name, job.dst.name,
                        f"{duration:.0f} s" if duration else "duration unknown")
            t0 = time.monotonic()
            job.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", errors="replace", creationflags=flags,
            )
            assert job.proc.stdout is not None
            for line in job.proc.stdout:
                if job.cancel.is_set():
                    break
                frac = parse_progress(line, duration)
                if frac is not None:
                    job.status.progress = frac
            _out, err = job.proc.communicate()
            if job.cancel.is_set():
                job.status = RemuxStatus("error", None, "cancelled")
                return
            if job.proc.returncode != 0:
                tail = (err or "").strip().splitlines()[-3:]
                msg = " / ".join(tail) or f"ffmpeg exit {job.proc.returncode}"
                logger.error("Remux of %s failed: %s", job.src.name, msg)
                job.status = RemuxStatus("error", None, msg)
                return
            os.replace(part, job.dst)
            job.status = RemuxStatus("ready", 1.0, output=job.dst)
            logger.info("Remux of %s done in %.0f s (%.0f MiB)", job.src.name,
                        time.monotonic() - t0, job.dst.stat().st_size / 2**20)
        except FileNotFoundError as exc:
            job.status = RemuxStatus("error", None, f"ffmpeg not found: {exc}")
            logger.error("Remux of %s: %s", job.src.name, job.status.error)
        except Exception as exc:
            job.status = RemuxStatus("error", None, str(exc))
            logger.exception("Remux of %s failed", job.src.name)
        finally:
            if job.status.state != "ready":
                try:
                    part.unlink(missing_ok=True)
                except OSError:
                    pass
            with self._lock:
                # A finished job is dropped so the cache file is the truth;
                # a failed one stays so the poller sees the error once,
                # and prepare() replaces it on the next request.
                if job.status.state == "ready":
                    self._jobs.pop(job.dst, None)

    @staticmethod
    def _unlink(p: Path, size: int) -> int:
        try:
            p.unlink()
            logger.debug("Remux cache: removed %s", p.name)
            return size
        except OSError:
            return 0


#: Process-wide manager used by the media routes.
manager = RemuxManager()
