"""GenSRT — GPU-accelerated subtitle generation using Whisper."""

from __future__ import annotations

# Single source of truth for the version.  pyproject.toml reads this via
# `dynamic = ["version"]` + `[tool.setuptools.dynamic]`, so the package
# metadata, `gensrt --version`, and GET /api/status cannot drift apart.
#
# Bump here — and only here — as part of the release ritual.
__version__ = "1.2.7"

# Register the bundled NVIDIA DLL directories on Windows.  This MUST run
# before anything imports ctranslate2, because CTranslate2 resolves cuBLAS and
# cuDNN through the Windows DLL search path, and the nvidia-*-cu12 wheels do
# not put themselves on it.  See gensrt/_cuda_dlls.py for the full story.
# No-op on non-Windows.
from gensrt._cuda_dlls import register_cuda_dll_directories as _register_cuda

_register_cuda()
del _register_cuda

__all__ = ["__version__"]
