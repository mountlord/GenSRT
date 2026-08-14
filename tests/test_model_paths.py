"""Local model directory convention.

"I converted a model — where do I put it?" had no answer: WhisperModel takes
any path, so GenSRT accepted anything and suggested nothing. A user who has to
invent a convention invents a different one each time.
"""

from __future__ import annotations

import pytest

from gensrt import model_paths


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(model_paths, "app_dir", lambda: tmp_path)
    return tmp_path


def _make_model(root, name, with_bin=True):
    d = root / "models" / name
    d.mkdir(parents=True)
    if with_bin:
        (d / "model.bin").write_bytes(b"x")
        (d / "config.json").write_text("{}")
    return d


# ── Resolution ────────────────────────────────────────────────────────────

def test_bare_name_resolves_under_models_dir(app):
    d = _make_model(app, "ct2-whisper-small-ml-punct")
    assert model_paths.resolve_model("ct2-whisper-small-ml-punct") == str(d)


def test_hf_repo_id_passes_through_untouched(app):
    """The critical non-regression: repo IDs must not be mangled."""
    for repo in ("smcproject/vegam-whisper-medium-ml-int8_float16",
                 "adalat-ai/ct2-whisper-medium-ml-rmft",
                 "large-v3-turbo"):
        assert model_paths.resolve_model(repo) == repo


def test_absolute_path_is_used_as_given(app, tmp_path):
    elsewhere = tmp_path / "somewhere" / "mymodel"
    elsewhere.mkdir(parents=True)
    assert model_paths.resolve_model(str(elsewhere)) == str(elsewhere)


def test_local_directory_wins_over_a_repo_id_of_the_same_name(app):
    """Someone who deliberately placed a directory there meant it."""
    d = _make_model(app, "large-v3-turbo")
    assert model_paths.resolve_model("large-v3-turbo") == str(d)


def test_unknown_bare_name_passes_through(app):
    assert model_paths.resolve_model("not-here") == "not-here"


def test_empty_input(app):
    assert model_paths.resolve_model("") == ""
    assert model_paths.resolve_model("  ") == ""


# ── Listing ───────────────────────────────────────────────────────────────

def test_listing_requires_model_bin(app):
    _make_model(app, "good")
    _make_model(app, "half-converted", with_bin=False)
    assert model_paths.list_local_models() == ["good"]


def test_listing_when_directory_absent(app):
    assert model_paths.list_local_models() == []


# ── Guidance text ─────────────────────────────────────────────────────────

def test_guidance_names_the_directory(app):
    text = model_paths.describe_model_locations()
    assert str(app / "models") in text
    assert "does not exist" in text


def test_guidance_lists_what_is_there(app):
    _make_model(app, "ct2-ml-punct")
    text = model_paths.describe_model_locations()
    assert "ct2-ml-punct" in text


def test_guidance_flags_an_empty_models_dir(app):
    (app / "models").mkdir()
    assert "no CTranslate2 models" in model_paths.describe_model_locations()


def test_models_dir_is_not_under_internal(app):
    """PyInstaller owns _internal and wipes it on reinstall."""
    assert "_internal" not in str(model_paths.models_dir())


# ── Directory creation ────────────────────────────────────────────────────

def test_ensure_creates_the_directory(app):
    assert not (app / "models").exists()
    assert model_paths.ensure_models_dir() == app / "models"
    assert (app / "models").is_dir()


def test_ensure_is_idempotent(app):
    model_paths.ensure_models_dir()
    assert model_paths.ensure_models_dir() == app / "models"


def test_ensure_survives_an_unwritable_location(app, monkeypatch):
    """Program Files without elevation, or a read-only share."""
    def boom(*a, **kw):
        raise PermissionError("access denied")

    monkeypatch.setattr(model_paths.Path, "mkdir", boom)
    assert model_paths.ensure_models_dir() is None


# ── Conversion command ────────────────────────────────────────────────────

def test_command_names_a_real_path_not_a_placeholder(app):
    cmd = model_paths.conversion_command("adalat-ai/whisper-small-ml-x")
    assert "<dir>" not in cmd
    assert str(app / "models") in cmd


def test_command_output_folder_gets_a_ct2_prefix(app):
    d = model_paths.suggested_output_dir("adalat-ai/whisper-small-ml-x")
    assert d.name == "ct2-whisper-small-ml-x"


def test_command_does_not_double_prefix(app):
    d = model_paths.suggested_output_dir("adalat-ai/ct2-whisper-small-ml-x")
    assert d.name == "ct2-whisper-small-ml-x"


def test_command_quotes_the_path(app):
    """Install paths contain spaces — 'Program Files', 'My Programs'."""
    assert '"' in model_paths.conversion_command("org/repo")


def test_suggested_dir_is_where_resolve_will_look(app):
    """The command's output must be findable by the name it tells you to use."""
    repo = "adalat-ai/whisper-small-ml-x"
    d = model_paths.suggested_output_dir(repo)
    d.mkdir(parents=True)
    (d / "model.bin").write_bytes(b"x")
    assert model_paths.resolve_model(d.name) == str(d)



# ── Search path (the venv gotcha) ─────────────────────────────────────────

def test_working_directory_is_searched_too(tmp_path, monkeypatch):
    """Running from source, sys.argv[0] is <venv>/Scripts/gensrt.exe.

    Without a cwd fallback, a model converted into <project>/models/ is
    invisible because GenSRT looks in <project>/venv/Scripts/models/ — and
    reports the name as an unknown HuggingFace repo, which sends the user
    looking in entirely the wrong direction.
    """
    monkeypatch.setattr(
        model_paths, "app_dir", lambda: tmp_path / "venv" / "Scripts"
    )
    monkeypatch.chdir(tmp_path)

    d = tmp_path / "models" / "ct2-test"
    d.mkdir(parents=True)
    (d / "model.bin").write_bytes(b"x")

    assert model_paths.resolve_model("ct2-test") == str(d)
    assert model_paths.list_local_models() == ["ct2-test"]


def test_guidance_lists_every_searched_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_paths, "app_dir", lambda: tmp_path / "venv" / "Scripts"
    )
    monkeypatch.chdir(tmp_path)
    text = model_paths.describe_model_locations()
    assert str(tmp_path / "venv" / "Scripts" / "models") in text
    assert str(tmp_path / "models") in text


def test_app_dir_wins_when_both_hold_the_same_name(tmp_path, monkeypatch):
    """Most-specific first: the install folder beats the working directory."""
    appd = tmp_path / "app"
    (appd / "models" / "dup").mkdir(parents=True)
    (appd / "models" / "dup" / "model.bin").write_bytes(b"x")
    (tmp_path / "models" / "dup").mkdir(parents=True)
    (tmp_path / "models" / "dup" / "model.bin").write_bytes(b"x")

    monkeypatch.setattr(model_paths, "app_dir", lambda: appd)
    monkeypatch.chdir(tmp_path)
    assert model_paths.resolve_model("dup") == str(appd / "models" / "dup")
