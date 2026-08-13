"""Tests for the build-time self-check.

The self-check exists because two bundles shipped that started fine and failed
at first use.  These tests pin the properties that make it worth running.
"""

from __future__ import annotations


import pytest

from gensrt.selfcheck import _Report, run_self_check


def test_passes_on_a_working_install(capsys):
    assert run_self_check() == 0
    assert "PASSED" in capsys.readouterr().out


def test_detects_a_missing_gensrt_submodule(monkeypatch, capsys):
    """The exact v1.2.3 failure: _chunk_debug absent from the bundle."""
    import importlib

    real = importlib.import_module

    def broken(name, *a, **kw):
        if name == "gensrt.asr._chunk_debug":
            raise ModuleNotFoundError("No module named 'gensrt.asr._chunk_debug'")
        return real(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", broken)
    assert run_self_check() == 1
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "_chunk_debug" in out


def test_detects_a_missing_dependency(monkeypatch, capsys):
    import importlib

    real = importlib.import_module

    def broken(name, *a, **kw):
        if name == "faster_whisper":
            raise ModuleNotFoundError("No module named 'faster_whisper'")
        return real(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", broken)
    assert run_self_check() == 1
    assert "faster_whisper" in capsys.readouterr().out


def test_detects_unusable_ffmpeg(monkeypatch, capsys):
    from gensrt import ffmpeg_util

    monkeypatch.setattr(ffmpeg_util, "get_ffmpeg_exe",
                        lambda: "/nonexistent/ffmpeg")
    assert run_self_check() == 1
    assert "ffmpeg" in capsys.readouterr().out.lower()


def test_cuda_absence_is_a_warning_by_default(capsys):
    """A CPU build must pass on a machine with no GPU."""
    assert run_self_check(require_cuda=False) == 0


def test_require_cuda_fails_without_a_device(capsys):
    """The CUDA build must not ship if CUDA is not actually usable."""
    import ctranslate2

    if ctranslate2.get_cuda_device_count() > 0:
        pytest.skip("this machine has CUDA")
    assert run_self_check(require_cuda=True) == 1


def test_optional_dependency_absence_does_not_fail(monkeypatch, capsys):
    import importlib

    real = importlib.import_module

    def broken(name, *a, **kw):
        if name == "webview":
            raise ModuleNotFoundError("no webview")
        return real(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", broken)
    assert run_self_check() == 0


def test_report_accumulates_correctly():
    r = _Report()
    r.ok("fine")
    r.warn("hmm")
    r.fail("bad")
    assert r.errors == ["bad"]
    assert r.warnings == ["hmm"]


def test_main_module_has_an_entry_guard():
    """Without it, importing gensrt.__main__ re-runs the CLI.

    That is not hypothetical: it made --self-check execute itself twice,
    because the check walks and imports every module in the package.
    """
    from pathlib import Path

    import gensrt

    source = (Path(gensrt.__file__).parent / "__main__.py").read_text()
    assert '__name__ == "__main__"' in source
    assert "sys.exit(main())" in source


def test_walk_skips_main_module(capsys):
    """A second guard on the same problem, from the other side."""
    run_self_check()
    out = capsys.readouterr().out
    assert out.count("GenSRT self-check") == 1
