"""Verify that an installation — especially a packaged build — is complete.

Why this exists
---------------
Two release-blocking bugs shipped in a row, and both were invisible until
transcription had already started:

* ``cublas64_12.dll`` was bundled in a directory Windows does not search, so
  the app started, detected the GPU, loaded the model, and then failed on
  every chunk.
* ``gensrt.asr._chunk_debug`` was missing from the bundle entirely, so the run
  died on the first import inside the ASR engine.

Both share a shape: **GenSRT defers almost everything.**  faster-whisper,
CTranslate2, the translation engines and the ASR engines are all imported
inside functions, deliberately, to keep CLI startup fast.  That makes
PyInstaller's static analysis fragile, and it means ``gensrt.exe --version``
succeeding tells you almost nothing about whether the build works.

This module does eagerly, in a couple of seconds, what a real run does lazily
over several minutes: import every module, resolve every binary, and load the
CUDA libraries.  Run it as a build step and these failures surface on the
build machine instead of a user's.

Usage::

    gensrt --self-check                 # imports, ffmpeg, CUDA reported
    gensrt --self-check --require-cuda  # CUDA failures are fatal too
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

# Third-party modules that a working install must be able to import.  Listed
# explicitly because every one of them is imported lazily somewhere in the
# codebase, which is exactly why PyInstaller can miss them.
_REQUIRED_THIRD_PARTY = [
    "ctranslate2",
    "faster_whisper",
    "srt",
    "flask",
    "requests",
    "tqdm",
    "rich",
    "numpy",
]

# Imported by the GUI only; a headless-capable build without it is degraded
# but not broken, so its absence is a warning rather than an error.
_OPTIONAL_THIRD_PARTY = ["webview", "onnxruntime", "hf_xet"]

# The CUDA libraries CTranslate2 loads at first inference.  Checked by
# actually calling into the OS loader, because "the file exists" and "the
# loader can load it" are different questions — a DLL present at a path
# Windows does not search fails exactly like a missing one.
_CUDA_LIBRARIES = ["cublas64_12.dll", "cudart64_12.dll"]
_CUDA_LIBRARY_PREFIXES = ["cudnn64_", "nvrtc64_"]


def _version_of(name: str) -> str:
    """Best-effort version string for a package.

    Uses importlib.metadata rather than the module's ``__version__``, which
    several packages (Flask among them) now emit a DeprecationWarning for.
    """
    try:
        from importlib.metadata import version

        return version(name.replace("_", "-"))
    except Exception:
        try:
            return str(getattr(importlib.import_module(name), "__version__", ""))
        except Exception:
            return ""


class _Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.lines: list[str] = []

    def ok(self, msg: str) -> None:
        self.lines.append(f"  [ ok ] {msg}")

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.lines.append(f"  [warn] {msg}")

    def fail(self, msg: str) -> None:
        self.errors.append(msg)
        self.lines.append(f"  [FAIL] {msg}")

    def section(self, title: str) -> None:
        self.lines.append("")
        self.lines.append(title)


def _check_gensrt_modules(report: _Report) -> None:
    """Import every ``gensrt.*`` submodule.

    Walks the package rather than checking a hand-maintained list, so a module
    added later is covered automatically — the hand-maintained list is what
    failed last time.
    """
    report.section("GenSRT modules")

    import gensrt

    failed = 0
    checked = 0
    for mod in pkgutil.walk_packages(gensrt.__path__, prefix="gensrt."):
        name = mod.name
        # gensrt.__main__ is an entry point, not a library module.  Importing
        # it is never useful and, before it grew an `if __name__` guard, it
        # re-ran the CLI from inside this very check.
        if name == "gensrt.__main__":
            continue
        checked += 1
        try:
            importlib.import_module(name)
        except Exception as exc:
            failed += 1
            report.fail(f"{name}: {type(exc).__name__}: {exc}")

    if failed == 0:
        report.ok(f"all {checked} gensrt modules import cleanly")


def _check_third_party(report: _Report) -> None:
    report.section("Dependencies")

    for name in _REQUIRED_THIRD_PARTY:
        try:
            importlib.import_module(name)
            report.ok(f"{name} {_version_of(name)}".rstrip())
        except Exception as exc:
            report.fail(f"{name}: {type(exc).__name__}: {exc}")

    for name in _OPTIONAL_THIRD_PARTY:
        try:
            importlib.import_module(name)
            report.ok(f"{name} (optional)")
        except Exception:
            report.warn(f"{name} not available (optional)")


def _check_ffmpeg(report: _Report) -> None:
    """Resolve and actually execute the bundled binaries."""
    report.section("FFmpeg")

    import subprocess

    from gensrt.ffmpeg_util import (
        get_ffmpeg_exe,
        get_ffprobe_exe,
        get_subprocess_creationflags,
    )

    for label, getter in (("ffmpeg", get_ffmpeg_exe), ("ffprobe", get_ffprobe_exe)):
        try:
            exe = getter()
        except Exception as exc:
            report.fail(f"{label}: could not resolve ({exc})")
            continue

        try:
            proc = subprocess.run(
                [exe, "-version"], capture_output=True, text=True, timeout=15,
                encoding="utf-8", errors="replace",
                creationflags=get_subprocess_creationflags(),
            )
        except Exception as exc:
            report.fail(f"{label}: cannot execute {exe} ({exc})")
            continue

        if proc.returncode != 0:
            report.fail(f"{label}: {exe} exited {proc.returncode}")
        else:
            first = (proc.stdout or "").splitlines()[:1]
            report.ok(f"{label}: {first[0] if first else exe}")


def _check_cuda(report: _Report, *, required: bool) -> None:
    """Register the CUDA directories and try to actually load the libraries.

    ``ctypes.CDLL`` is the point of this check.  It asks the operating system
    the same question CTranslate2 will ask at first inference, and gets the
    same answer — including the case where the file is present in the bundle
    but sitting somewhere the loader does not look.
    """
    report.section("CUDA")

    note = report.fail if required else report.warn

    from gensrt._cuda_dlls import (
        find_cuda_libraries,
        register_cuda_dll_directories,
    )

    dirs = register_cuda_dll_directories()
    if sys.platform == "win32":
        if dirs:
            report.ok(f"{len(dirs)} CUDA DLL director(ies) registered")
        else:
            note("no CUDA DLL directories found")

    try:
        import ctranslate2

        count = ctranslate2.get_cuda_device_count()
    except Exception as exc:
        report.fail(f"ctranslate2 CUDA probe failed: {exc}")
        return

    if count == 0:
        note("no CUDA devices visible (driver absent, or no NVIDIA GPU)")
    else:
        report.ok(f"{count} CUDA device(s) visible")

    if sys.platform != "win32":
        return

    import ctypes

    found = find_cuda_libraries()
    wanted = list(_CUDA_LIBRARIES) + [p + "*.dll" for p in _CUDA_LIBRARY_PREFIXES]

    for key in wanted:
        path = found.get(key)
        if path is None:
            note(f"{key}: not found in any searched directory")
            continue
        # Loading by bare name exercises the real search path, which is what
        # CTranslate2 does.  Loading succeeded from a full path but failing by
        # name is precisely the bug that shipped.
        name = Path(path).name
        try:
            ctypes.CDLL(name)
            report.ok(f"{name}: loadable")
        except OSError as exc:
            note(f"{name}: found at {path} but NOT loadable by name ({exc})")


def _check_network(report: _Report) -> None:
    """Verify HTTPS to HuggingFace actually works, certificates included.

    Every model GenSRT does not already have cached comes over TLS, and a
    certificate failure is invisible until the moment a user tries to add a
    model.  Reported as a warning rather than an error: an offline machine
    with a pre-populated model cache is a perfectly valid installation, and
    failing the build over no network would be wrong.
    """
    report.section("Network")

    try:
        import certifi

        report.ok(f"certifi CA bundle: {certifi.where()}")
    except Exception as exc:
        report.warn(f"certifi not available ({exc}) — HTTPS may fail")

    try:
        import requests

        resp = requests.get(
            "https://huggingface.co/api/models/openai/whisper-tiny",
            headers={"User-Agent": "GenSRT/selfcheck"}, timeout=15,
        )
        if resp.status_code == 200:
            report.ok("HuggingFace reachable over HTTPS")
        else:
            report.warn(f"HuggingFace returned HTTP {resp.status_code}")
    except Exception as exc:
        name = type(exc).__name__
        if "SSL" in name or "Certificate" in str(exc):
            report.warn(
                f"TLS certificate verification FAILED ({exc}). Model downloads "
                f"will not work on this machine until this is resolved — open "
                f"https://huggingface.co in a browser once, or check the "
                f"system clock."
            )
        else:
            report.warn(f"HuggingFace unreachable ({name}: {exc}) — offline?")


def run_self_check(*, require_cuda: bool = False) -> int:
    """Run every check and print a report.

    Returns:
        ``0`` when everything required passed, ``1`` otherwise — so a build
        script can simply test the exit code.
    """
    from gensrt import __version__

    report = _Report()

    print("")
    print("=" * 60)
    print(f"  GenSRT self-check — version {__version__}")
    frozen = getattr(sys, "frozen", False)
    print(f"  {'packaged build' if frozen else 'source install'}")
    print("=" * 60)

    _check_gensrt_modules(report)
    _check_third_party(report)
    _check_ffmpeg(report)
    _check_cuda(report, required=require_cuda)
    _check_network(report)

    for line in report.lines:
        print(line)

    print("")
    print("=" * 60)
    if report.errors:
        print(f"  FAILED — {len(report.errors)} error(s), "
              f"{len(report.warnings)} warning(s)")
        print("=" * 60)
        print("")
        for e in report.errors:
            print(f"  ERROR: {e}")
        print("")
        return 1

    print(f"  PASSED — {len(report.warnings)} warning(s)")
    print("=" * 60)
    print("")
    return 0
