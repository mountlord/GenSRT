"""Where GenSRT looks for locally-converted models.

The problem this solves
-----------------------
``faster_whisper.WhisperModel`` accepts either a HuggingFace repo ID or a
filesystem path, and GenSRT passed whatever the user typed straight through.
That is maximally flexible and gives the user no answer to a reasonable
question: *I converted a model — where do I put it?*  "Anywhere, use an
absolute path" is a non-answer, and a user who has to invent a convention will
invent a different one each time.

So there is now a convention: a ``models`` directory beside the executable,
the same place ``gensrt-config.json`` is discovered.  A bare name is resolved
against it.

    C:\\MyPrograms\\gensrt\\
        gensrt.exe
        gensrt-config.json
        models\\
            ct2-whisper-small-ml-punct\\      <- "ct2-whisper-small-ml-punct"

Deliberately NOT ``_internal\\``.  That directory belongs to PyInstaller, is
wiped on reinstall, and mixing user data into it loses models on upgrade.

Resolution order for a model string:

1. An absolute path, or one containing a separator, is used as given.  Nothing
   clever — if the user typed a path they meant that path.
2. A bare name matching a directory under ``models`` resolves to it.
3. Anything else is passed through untouched, and is treated as a HuggingFace
   repo ID.

Rule 3 is what keeps ``smcproject/vegam-whisper-medium-ml-int8_float16``
working: it contains a separator, but the local check in rule 1 fails, so it
falls through to HuggingFace.  A local directory always wins over a repo ID of
the same name, which is the behaviour someone who deliberately placed a
directory there would expect.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR_NAME = "models"


def app_dir() -> Path:
    """The directory holding the GenSRT executable (or entry script)."""
    try:
        return Path(sys.argv[0]).resolve().parent
    except (OSError, ValueError):  # pragma: no cover — defensive
        return Path.cwd()


def _is_writable(d: Path) -> bool:
    """Whether a file can actually be created in *d*.

    Tested by writing, not by inspecting permissions: on Windows the useful
    answer depends on UAC virtualisation, the directory's ACL and whether the
    process is elevated, and none of those are reliably readable in advance.
    """
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe = d / ".gensrt-write-test"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def sidecar_dir() -> Path:
    """Where GenSRT should CREATE its per-installation files.

    Used for ``gensrt-config.json`` and ``gensrt-known-models.json``.  Reading
    them searches both the application directory and the working directory;
    this decides where a *new* one goes.

    The application directory is preferred, because that is where a user
    looks and where the file stays put across sessions.  Falling back to the
    working directory — as these files did until now — means a user who
    launches GenSRT from somewhere unexpected gets their settings written to
    whatever folder they happened to be in, and finds them apparently lost
    next time.

    The working directory remains the fallback for a genuinely unwritable
    install location, which is the case that presumably motivated it
    originally: an installation under Program Files, or on a read-only share,
    cannot take a settings file at all.
    """
    app = app_dir()
    if _is_writable(app):
        return app
    logger.debug("Application directory %s is not writable; using cwd.", app)
    return Path.cwd()


def model_search_dirs() -> list[Path]:
    """Every directory a local model might live in, most-specific first.

    Mirrors ``config._find_config_file``, and for the same reason.  In a
    packaged build ``sys.argv[0]`` is the executable, so ``app_dir()`` is the
    install folder and is correct.  Running from a source checkout it is
    ``<venv>/Scripts/gensrt.exe``, so ``app_dir()`` points inside the virtual
    environment — nowhere a person would think to put a model.

    Without the working-directory fallback, a developer who converts a model
    into ``<project>/models/`` finds GenSRT looking in
    ``<project>/venv/Scripts/models/`` and reporting the name as an unknown
    HuggingFace repo.  Config discovery has had this fallback all along; this
    did not, and the failure was silent in exactly the confusing direction.
    """
    seen: set[str] = set()
    out: list[Path] = []
    for base in (app_dir(), Path.cwd()):
        d = base / MODELS_DIR_NAME
        key = str(d).lower()
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def models_dir() -> Path:
    """The directory to CREATE and to name in guidance.

    Where more than one candidate exists, prefer one that is already there —
    telling a user to use a folder they have not got, while ignoring the one
    they have, is the failure this function exists to avoid.  Otherwise the
    first candidate wins.
    """
    candidates = model_search_dirs()
    for d in candidates:
        if d.is_dir():
            return d
    return candidates[0]


def ensure_models_dir() -> Path | None:
    """Create the models directory if it does not exist.

    Called at startup so the folder is visible in the install directory before
    anyone needs it — a convention nobody can see is not much of a convention,
    and "where do I put this?" is easier to answer when the answer is already
    sitting there.

    Returns the path, or ``None`` if it could not be created.  Never raises:
    an install directory that is read-only (Program Files without elevation,
    a network share) is unusual but survivable — models can still be loaded
    from an explicit path elsewhere.
    """
    root = models_dir()
    try:
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
            logger.info("Created models directory: %s", root)
        return root
    except OSError as exc:
        logger.debug("Could not create models directory %s: %s", root, exc)
        return None


def suggested_output_dir(repo_id: str) -> Path:
    """Where a converted copy of *repo_id* should be written.

    Used to build a conversion command the user can paste without editing.  A
    command containing ``<dir>`` requires the reader to already know the
    answer to the question they are asking.
    """
    leaf = repo_id.rsplit("/", 1)[-1].strip() or "converted-model"
    if not leaf.startswith("ct2-"):
        leaf = f"ct2-{leaf}"
    return models_dir() / leaf


def conversion_command(repo_id: str) -> str:
    """A ready-to-run ct2-transformers-converter invocation for *repo_id*.

    Single source, so the same text appears wherever GenSRT suggests
    converting a model.
    """
    return (
        f"ct2-transformers-converter --model {repo_id} "
        f'--output_dir "{suggested_output_dir(repo_id)}" '
        f"--quantization float16"
    )


def list_local_models() -> list[str]:
    """Names of directories under :func:`models_dir`, sorted.

    Only directories containing a CTranslate2 ``model.bin`` are returned, so a
    half-finished conversion or a stray folder is not offered as a model.
    """
    found: dict[str, None] = {}
    for root in model_search_dirs():
        try:
            if not root.is_dir():
                continue
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "model.bin").is_file():
                    found.setdefault(d.name, None)
        except OSError:
            continue
    return list(found)


def resolve_model(name: str) -> str:
    """Resolve *name* to a path or leave it as a HuggingFace repo ID.

    See the module docstring for the ordering.  Returns a string because that
    is what ``WhisperModel`` takes, and because leaving repo IDs untouched
    matters more than type tidiness here.
    """
    name = (name or "").strip()
    if not name:
        return name

    candidate = Path(name)

    # 1. Explicit path — absolute, or relative with a separator in it.
    if candidate.is_absolute() or any(sep in name for sep in ("\\", "/")):
        if candidate.is_dir():
            logger.debug("Model resolved as an explicit path: %s", candidate)
            return str(candidate)
        # Falls through: "org/repo" looks path-like on POSIX but is a repo ID.

    # 2. Bare name under any of the conventional models directories.
    if "/" not in name and "\\" not in name:
        for root in model_search_dirs():
            local = root / name
            if local.is_dir():
                logger.info("Model resolved to local directory: %s", local)
                return str(local)

    # 3. HuggingFace repo ID.
    return name


def describe_model_locations() -> str:
    """Human-readable text for error messages and the GUI.

    Kept here so every surface says the same thing; a convention explained
    three different ways is not a convention.
    """
    roots = model_search_dirs()
    found = list_local_models()

    lines = ["GenSRT looks for locally-converted models in:"]
    lines += [f"    {r}" + ("" if r.is_dir() else "   (does not exist)")
              for r in roots]
    root = models_dir()
    lines += [
        "",
        "Put the converted model in its own subdirectory there, then enter "
        "just the folder name — for example a folder named "
        "'ct2-whisper-small-ml-punct' is entered as "
        "'ct2-whisper-small-ml-punct'.",
    ]
    if found:
        lines += ["", "Models currently found:"]
        lines += [f"    {n}" for n in found]
    elif any(r.is_dir() for r in roots):
        lines += ["", "Those directories contain no CTranslate2 models "
                      "(a model directory must contain 'model.bin')."]
    else:
        lines += ["", f"Create {root} and put the model there."]

    lines += ["", "A full path to a model directory anywhere on disk also "
                  "works, if you prefer to keep models elsewhere."]
    return "\n".join(lines)
