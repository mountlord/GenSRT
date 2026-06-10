"""Locate the ffmpeg / ffprobe binaries GenSRT uses.

GenSRT bundles ffmpeg and ffprobe (gyan.dev essentials build) so end users
don't need to install ffmpeg separately or have it on ``PATH``.  The
binaries live next to the Python package at ``gensrt/bin/`` and are
copied into the installer by ``Pack-gensrt.ps1``.

This module is the single source of truth for the binary paths — every
place GenSRT shells out to ffmpeg or ffprobe should resolve through
:func:`get_ffmpeg_exe` and :func:`get_ffprobe_exe`.

Resolution order:
  1. Bundled at ``gensrt/bin/<name>.exe`` (relative to this module).
     This is the canonical path for installed users and for devs who
     have placed the binaries there.
  2. ``PATH`` fallback.  Lets devs with their own ffmpeg run GenSRT
     before dropping binaries into ``gensrt/bin/``, and provides a last-
     resort path if the bundled binary is somehow missing post-install.

If neither resolves, the bare command name is returned and the
subsequent subprocess call will raise ``FileNotFoundError`` — the
caller's existing error handling kicks in (e.g. ``/api/burn`` returns a
clear "ffmpeg not found" message).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Directory where bundled binaries live.  Anchored to the package, NOT
# to the current working directory — so this works whether GenSRT is run
# from the dev tree, from a `pip install -e .` editable install, or from
# the PyInstaller --onedir layout (where the package ends up under
# `_internal/gensrt/` and __file__ resolves correctly inside it).
_BIN_DIR = Path(__file__).parent / "bin"


def _resolve(exe_name: str) -> str:
    """Return the path to *exe_name* (e.g. "ffmpeg.exe"), bundled or PATH.

    Bundled wins.  Logs which resolution path was taken at DEBUG so we
    have a record in the server log without spamming INFO.
    """
    bundled = _BIN_DIR / exe_name
    if bundled.is_file():
        logger.debug("%s: using bundled binary at %s", exe_name, bundled)
        return str(bundled)

    on_path = shutil.which(exe_name)
    if on_path:
        logger.debug("%s: bundled missing, falling back to PATH (%s)", exe_name, on_path)
        return on_path

    logger.warning(
        "%s: not found bundled (looked in %s) and not on PATH. "
        "Subsequent calls will fail with FileNotFoundError.",
        exe_name, _BIN_DIR,
    )
    # Return the bare name; let subprocess raise FileNotFoundError so the
    # caller's existing error path handles it.
    return exe_name


@lru_cache(maxsize=1)
def get_ffmpeg_exe() -> str:
    """Absolute path (or bare name) of the ffmpeg binary to invoke."""
    return _resolve("ffmpeg.exe" if sys.platform.startswith("win") else "ffmpeg")


@lru_cache(maxsize=1)
def get_ffprobe_exe() -> str:
    """Absolute path (or bare name) of the ffprobe binary to invoke."""
    return _resolve("ffprobe.exe" if sys.platform.startswith("win") else "ffprobe")


def get_subprocess_creationflags() -> int:
    """Windows-only flag to suppress the console-window pop-up.

    Used when invoking ffmpeg/ffprobe from the pywebview GUI process,
    which has no console of its own — without this, Windows opens a
    fleeting black box on each ffmpeg/ffprobe call.  No effect on POSIX.
    """
    if sys.platform.startswith("win"):
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0
