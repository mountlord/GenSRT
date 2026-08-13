"""Register NVIDIA CUDA DLL directories on Windows.

The problem
-----------
CTranslate2 needs ``cublas64_12.dll``, ``cudnn*.dll``, ``cudart64_12.dll`` and
``nvrtc64_120_0.dll`` for GPU inference.  When those come from the
``nvidia-*-cu12`` pip wheels, they are installed to::

    <site-packages>/nvidia/<component>/bin/*.dll

and **nothing puts those directories on the Windows DLL search path.**  The
wheels ship no ``__init__.py`` and no ``.pth`` hook — verified by inspecting
the wheel contents.  ``ctranslate2/__init__.py`` calls
``os.add_dll_directory`` only for its *own* package directory.

So a machine that works does so for an incidental reason: a CUDA Toolkit
install, a leftover ``PATH`` entry, or another application (PyTorch being the
usual one) having put the libraries somewhere Windows already searches.  A
clean machine with only the NVIDIA display driver has none of that, and every
inference call fails with::

    Library cublas64_12.dll is not found or cannot be loaded

This is exactly the failure mode that hides during development.  Removing
PyTorch from GenSRT made it reachable, because PyTorch was one of the things
that used to make the libraries findable as a side effect.

The fix
-------
Find every ``nvidia/*/bin`` directory that ships with the application or the
environment, and register it explicitly with :func:`os.add_dll_directory`.
This must happen *before* ``ctranslate2`` is imported, which is why it is
invoked from :mod:`gensrt.__init__`.

Search locations, in order:

1. ``sys._MEIPASS`` — the PyInstaller bundle root.  Covers frozen builds.
2. The directory containing the executable.  Covers DLLs placed next to
   ``gensrt.exe``.
3. Every ``site-packages`` on ``sys.path``.  Covers ordinary pip installs and
   development venvs.

No-ops on non-Windows platforms, where shared-library resolution uses
``RPATH``/``LD_LIBRARY_PATH`` and this problem does not arise.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Populated by register_cuda_dll_directories() so diagnostics can report what
# was actually registered without re-running the search.
_registered: list[str] = []
_done = False


def _candidate_roots() -> list[Path]:
    """Directories that might contain an ``nvidia/`` DLL tree."""
    roots: list[Path] = []

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        roots.append(Path(meipass))

    try:
        roots.append(Path(sys.executable).parent)
    except (TypeError, ValueError):
        pass

    for entry in sys.path:
        if not entry:
            continue
        p = Path(entry)
        if p.name in ("site-packages", "dist-packages"):
            roots.append(p)

    # Deduplicate, preserving order.
    seen: set[str] = set()
    unique: list[Path] = []
    for r in roots:
        key = str(r).lower()
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def register_cuda_dll_directories() -> list[str]:
    """Add any bundled NVIDIA library directories to the DLL search path.

    Idempotent — repeated calls return the cached result.  Never raises: a
    failure here should degrade to "GPU not available", which the rest of the
    stack already handles, not crash the application.

    Returns:
        The directories that were successfully registered.
    """
    global _done

    if _done:
        return _registered
    _done = True

    if sys.platform != "win32":
        return _registered

    add_dll_directory = getattr(os, "add_dll_directory", None)
    if add_dll_directory is None:  # pragma: no cover — Python < 3.8
        return _registered

    for root in _candidate_roots():
        nvidia_dir = root / "nvidia"
        try:
            if not nvidia_dir.is_dir():
                continue
            components = sorted(nvidia_dir.iterdir())
        except OSError:
            continue

        for component in components:
            bin_dir = component / "bin"
            try:
                if not bin_dir.is_dir():
                    continue
                add_dll_directory(str(bin_dir))
            except OSError as exc:
                logger.debug("Could not register %s: %s", bin_dir, exc)
                continue
            _registered.append(str(bin_dir))
            logger.debug("Registered CUDA DLL directory: %s", bin_dir)

    # The bundle root itself, for builds that place the DLLs flat rather than
    # under an nvidia/ tree.  PyInstaller normally does this already; doing it
    # again is harmless and covers layouts that do not.
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        try:
            add_dll_directory(str(meipass))
            _registered.append(str(meipass))
        except OSError:
            pass

    if _registered:
        logger.debug(
            "Registered %d CUDA DLL director(ies).", len(_registered)
        )
    return _registered


def registered_directories() -> list[str]:
    """Directories registered by :func:`register_cuda_dll_directories`."""
    return list(_registered)


def find_cuda_libraries() -> dict[str, str | None]:
    """Locate the CUDA libraries CTranslate2 needs, for diagnostics.

    Returns a mapping of library name to the path it was found at, or ``None``
    when it could not be found in any registered directory.  Used by
    ``--check-cuda`` to tell a user precisely which library is missing rather
    than leaving them with a bare loader error.
    """
    wanted = ["cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"]
    # cuDNN and NVRTC carry version numbers that move between releases, so
    # they are matched by prefix rather than exact name.
    prefixes = ["cudnn64_", "cudnn_", "nvrtc64_"]

    found: dict[str, str | None] = {name: None for name in wanted}
    for prefix in prefixes:
        found[prefix + "*.dll"] = None

    search_dirs = [Path(d) for d in _registered]
    for raw in os.environ.get("PATH", "").split(os.pathsep):
        if raw:
            search_dirs.append(Path(raw))

    for d in search_dirs:
        try:
            if not d.is_dir():
                continue
            entries = list(d.iterdir())
        except OSError:
            continue

        for entry in entries:
            name = entry.name
            if name in found and found[name] is None:
                found[name] = str(entry)
            for prefix in prefixes:
                key = prefix + "*.dll"
                if found[key] is None and name.lower().startswith(prefix) \
                        and name.lower().endswith(".dll"):
                    found[key] = str(entry)

    return found
