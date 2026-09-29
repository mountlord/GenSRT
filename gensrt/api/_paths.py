"""Path validation and media/SRT lookup helpers for the API routes.

Every caller-supplied path goes through :func:`validate_readable_path` or
:func:`validate_srt_save_path` before any disk access; nothing in the routes
touches the filesystem with a raw request string.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ── Media path validation & metadata helpers ──────────────────────────────
#
# Used by /api/media (serves video bytes with HTTP Range support) and
# /api/video_info (returns fps/duration via ffprobe).  All caller-supplied
# paths go through _validate_readable_path before any disk access.

_VIDEO_EXTS: frozenset[str] = frozenset(
    {".mp4", ".mkv", ".webm", ".avi", ".mov", ".ts", ".m2ts", ".mts", ".m4v"}
)
_READABLE_EXTS: frozenset[str] = _VIDEO_EXTS | frozenset({".srt"})


def _find_sibling_video(srt_path: Path) -> Path | None:
    """Return the first sibling video file next to *srt_path*, or None.

    Handles both naming conventions:
      * ``movie.srt``      → looks for ``movie.{mp4,mkv,...}``
      * ``movie.ml.srt``   → tries ``movie.ml.{...}`` first (unlikely
                             but possible if user really has that file),
                             then strips the ``.ml`` and tries
                             ``movie.{mp4,mkv,...}`` — the common case.

    The language-code detection is heuristic: the second-to-last suffix
    must be 2-3 lowercase ASCII letters.  ``movie.korean-cut.srt`` is
    left alone (``korean-cut`` is longer than 3 chars and contains a
    hyphen) — we only look for ``movie.korean-cut.{ext}`` and don't
    strip anything.

    Implementation note: we use string concatenation rather than
    :meth:`Path.with_suffix` because Python treats ``.korean-cut`` as a
    valid suffix and would happily strip it — that's not what we want
    when probing the literal stem.
    """
    # Strip exactly the trailing '.srt' from the filename — no clever
    # suffix handling.
    full_name = srt_path.name
    literal_stem = full_name[: -len(srt_path.suffix)] if srt_path.suffix else full_name
    base_stems: list[str] = [literal_stem]

    # If srt_path looks like 'foo.ml.srt', also consider 'foo' as a base.
    suffixes = srt_path.suffixes  # e.g. ['.ml', '.srt']
    if len(suffixes) >= 2:
        possible_lang = suffixes[-2].lstrip(".")
        if (2 <= len(possible_lang) <= 3
                and possible_lang.isalpha()
                and possible_lang.islower()):
            stem_no_lang = literal_stem[: -len(suffixes[-2])]
            base_stems.append(stem_no_lang)

    for base in base_stems:
        for ext in _VIDEO_EXTS:
            candidate = srt_path.parent / (base + ext)
            if candidate.exists() and candidate.is_file():
                return candidate
    return None


def _validate_readable_path(path_str: str) -> tuple[Path, str | None]:
    """Resolve and validate a caller-supplied absolute path for read-only endpoints.

    Returns ``(resolved_path, error_msg_or_None)``.  Only files with allowed
    extensions that exist on disk are accepted.
    """
    if not path_str or not isinstance(path_str, str):
        return Path(), "path must be a non-empty string"
    try:
        p = Path(path_str).expanduser().resolve()
    except Exception as exc:
        return Path(path_str), f"Invalid path: {exc}"
    if p.suffix.lower() not in _READABLE_EXTS:
        return p, (
            f"Unsupported file type '{p.suffix}'. "
            f"Accepted: {', '.join(sorted(_READABLE_EXTS))}"
        )
    if not p.exists() or not p.is_file():
        return p, f"File not found: {p}"
    return p, None


def _guess_mime(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".mp4", ".m4v"}:
        return "video/mp4"
    if ext == ".webm":
        return "video/webm"
    if ext == ".mkv":
        return "video/x-matroska"
    if ext == ".ts":
        return "video/mp2t"
    if ext == ".mov":
        return "video/quicktime"
    if ext == ".srt":
        return "application/x-subrip"
    return "application/octet-stream"


def _parse_rate_to_float(rate: str) -> Optional[float]:
    """Parse ffprobe frame-rate fields like ``'60000/1001'`` into a float."""
    if not rate:
        return None
    s = str(rate).strip()
    if not s:
        return None
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            num = float(a.strip())
            den = float(b.strip())
            return num / den if den else None
        return float(s)
    except Exception:
        return None


# ── SRT file helpers ──────────────────────────────────────────────────────


def _find_sidecar_srt(video_path: Path) -> Optional[Path]:
    """Look for an SRT next to a video file.

    Discovery order (matches Plex/Jellyfin sidecar conventions):
      1. ``<basename>.srt``           — preferred (treated as the default
                                        / English track)
      2. ``<basename>.<any>.srt``     — first language-suffixed SRT we find
                                        (e.g. ``movie.ml.srt``, ``movie.ko.srt``)

    The simple `<basename>.srt` always wins when present, regardless of
    what target language the user currently has selected.  Use the file
    picker or drag the specific .lang.srt to load a different track.

    Returns the SRT path if one is found, else ``None``.
    """
    # Preferred: unsuffixed sidecar.
    canonical = video_path.with_suffix(".srt")
    if canonical.exists() and canonical.is_file():
        return canonical

    # Fallback: any <basename>.*.srt in the same directory.  We don't try
    # to guess "best language" — first match wins.  Glob is bounded by
    # basename so this is fast even on large folders.
    base_stem = video_path.stem
    parent = video_path.parent
    try:
        for candidate in parent.glob(f"{base_stem}.*.srt"):
            if candidate.is_file():
                return candidate
    except OSError:
        # Some filesystems object to glob on certain characters; be quiet.
        pass
    return None


def _validate_save_path(path_str: str, suffix: str) -> tuple[Path, str | None]:
    """Resolve a caller-supplied destination path for writes.

    The file may not exist yet (that's the common Save case), but the parent
    directory must, and the suffix must be *suffix*.
    """
    if not path_str or not isinstance(path_str, str):
        return Path(), "path must be a non-empty string"
    try:
        p = Path(path_str).expanduser().resolve()
    except Exception as exc:
        return Path(path_str), f"Invalid path: {exc}"
    if p.suffix.lower() != suffix:
        return p, f"Destination path must end with {suffix}"
    if not p.parent.exists() or not p.parent.is_dir():
        return p, f"Parent directory does not exist: {p.parent}"
    return p, None


def _validate_srt_save_path(path_str: str) -> tuple[Path, str | None]:
    return _validate_save_path(path_str, ".srt")

