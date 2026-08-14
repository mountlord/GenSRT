"""Where per-installation files get created.

gensrt-config.json and gensrt-known-models.json searched both the application
directory and the working directory, but CREATED in the working directory. A
user launching GenSRT from an unexpected folder therefore got their settings
written there, and found them apparently lost next time.
"""

from __future__ import annotations


import pytest

from gensrt import known_models, model_paths


@pytest.fixture
def app(tmp_path, monkeypatch):
    appd = tmp_path / "install"
    appd.mkdir()
    work = tmp_path / "elsewhere"
    work.mkdir()
    monkeypatch.setattr(model_paths, "app_dir", lambda: appd)
    monkeypatch.chdir(work)
    return appd, work


# ── Creation location ─────────────────────────────────────────────────────

def test_new_files_go_beside_the_executable(app):
    appd, work = app
    assert model_paths.sidecar_dir() == appd


def test_known_models_created_in_the_app_dir(app):
    appd, work = app
    p = known_models._resolve_known_models_path()
    assert p.parent == appd


def test_falls_back_to_cwd_when_install_dir_is_read_only(app, monkeypatch):
    """Program Files without elevation, or a read-only share."""
    appd, work = app
    monkeypatch.setattr(model_paths, "_is_writable", lambda d: d != appd)
    assert model_paths.sidecar_dir() == work


# ── Existing files are never relocated ────────────────────────────────────

def test_existing_file_in_cwd_still_wins(app):
    """Moving a user's remembered models out from under them would be worse
    than an unusual location."""
    appd, work = app
    existing = work / known_models.KNOWN_MODELS_FILENAME
    existing.write_text('{"models": ["org/mine"]}')
    assert known_models._resolve_known_models_path() == existing


def test_app_dir_file_preferred_over_cwd(app):
    appd, work = app
    (appd / known_models.KNOWN_MODELS_FILENAME).write_text('{"models": []}')
    (work / known_models.KNOWN_MODELS_FILENAME).write_text('{"models": []}')
    assert known_models._resolve_known_models_path().parent == appd


# ── Round trip ────────────────────────────────────────────────────────────

def test_saved_models_are_read_back(app):
    appd, _ = app
    known_models.save_known_models(["org/a", "org/b"])
    assert (appd / known_models.KNOWN_MODELS_FILENAME).is_file()
    assert known_models.load_known_models() == ["org/a", "org/b"]


def test_missing_file_degrades_to_empty(app):
    assert known_models.load_known_models() == []


def test_builtins_available_without_any_file(app):
    """A fresh install has no side file and must still offer models."""
    assert "large-v3-turbo" in known_models.BUILTIN_RECOMMENDED
    assert known_models.load_known_models() == []


def test_default_config_written_beside_the_executable(app):
    from gensrt.config import DEFAULT_CONFIG_NAME, generate_default_config

    appd, work = app
    p = generate_default_config()
    assert p.parent == appd
    assert p.name == DEFAULT_CONFIG_NAME


def test_writable_probe_leaves_nothing_behind(tmp_path):
    assert model_paths._is_writable(tmp_path)
    assert list(tmp_path.iterdir()) == []
