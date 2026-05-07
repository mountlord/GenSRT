"""Media file discovery and output path resolution utilities."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from gensrt.constants import MEDIA_EXTENSIONS

logger = logging.getLogger(__name__)


def is_media_file(path: Path) -> bool:
    """Return ``True`` if *path* has a recognised media extension."""
    return path.suffix.lower() in MEDIA_EXTENSIONS


def collect_media_files(
    path: Path,
    *,
    recurse: bool = False,
) -> list[Path]:
    """Return an ordered list of media files rooted at *path*.

    Args:
        path:    A file or directory.
        recurse: If ``True`` and *path* is a directory, walk subdirectories.

    Returns:
        Sorted list of :class:`Path` objects.  If *path* is a single file and
        it passes the extension check, returns ``[path]``.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    path = Path(path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    if path.is_file():
        if not is_media_file(path):
            logger.warning(
                "%s has an unrecognised extension — attempting to process anyway "
                "(FFmpeg will be the final arbiter).",
                path.name,
            )
        return [path]

    # Directory
    pattern = "**/*" if recurse else "*"
    found = sorted(p for p in path.glob(pattern) if p.is_file() and is_media_file(p))

    if not found:
        logger.warning("No media files found in: %s", path)

    return found


def resolve_output_path(
    input_path: Path,
    output_dir: Path | None,
    output_filename: str | None,
) -> Path:
    """Determine the output .srt path for a given input file.

    Resolution rules (from REQUIREMENTS §2.6):
      1. ``output_filename`` set, ``output_dir`` also set
         → warn that ``--output`` is ignored; use ``input_path.parent / output_filename``
      2. ``output_filename`` set, ``output_dir`` not set
         → ``input_path.parent / output_filename``
      3. ``output_dir`` set, no ``output_filename``
         → ``output_dir / <stem>.srt``
      4. Neither set
         → ``input_path.parent / <stem>.srt``

    Args:
        input_path:       Resolved path to the input media file.
        output_dir:       Optional directory override from ``--output``.
        output_filename:  Optional filename override from ``--output-filename``.

    Returns:
        Resolved :class:`Path` for the .srt output file.
    """
    if output_filename:
        if output_dir:
            warnings.warn(
                "--output is ignored when --output-filename is set. "
                f"Writing to: {input_path.parent / output_filename}",
                stacklevel=2,
            )
        return input_path.parent / output_filename

    stem = input_path.stem
    target_dir = output_dir or input_path.parent
    return Path(target_dir) / f"{stem}.srt"
