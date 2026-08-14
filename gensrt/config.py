"""Configuration file support for GenSRT.

Loads settings from ``gensrt-config.json`` and merges them with CLI
arguments.  Precedence (highest wins):

    built-in defaults  →  config file  →  CLI arguments

Usage::

    gensrt --init-config           # create default config file
    gensrt --config my.json ...    # use custom config path
    gensrt --dump-config           # show resolved config and exit
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_NAME = "gensrt-config.json"


def _derive_transcription_defaults() -> dict[str, Any]:
    """Return defaults auto-derived from :class:`~gensrt.models.TranscriptionConfig` fields.

    This is the *single source of truth* for default values.  Any change to a
    ``TranscriptionConfig`` field default is automatically reflected here — no
    manual sync required.
    """
    import dataclasses
    from gensrt.models import TranscriptionConfig

    out: dict[str, Any] = {}
    for f in dataclasses.fields(TranscriptionConfig):
        if f.default is not dataclasses.MISSING:
            val = f.default
        elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
            val = f.default_factory()  # type: ignore[misc]
        else:
            continue  # required field with no default — skip

        if isinstance(val, tuple):
            val = list(val)

        out[f.name] = val

    return out


# ── Non-model defaults (ops, logging) ─────────────────────────────────────
_STATIC_DEFAULTS: dict[str, Any] = {
    "output": None,
    "output_filename": None,
    "recurse": False,
    "log_level": "INFO",
}


# ── Built-in defaults — auto-derived + static ─────────────────────────────
BUILTIN_DEFAULTS: dict[str, Any] = {
    **_STATIC_DEFAULTS,
    **_derive_transcription_defaults(),
}


def _find_config_file() -> Path | None:
    """Auto-discover ``gensrt-config.json``.

    Search order:
      1. Next to the installed ``gensrt`` executable (sys.argv[0] directory).
      2. Current working directory.
    """
    import sys

    candidates = [
        Path(sys.argv[0]).resolve().parent / DEFAULT_CONFIG_NAME,
        Path.cwd() / DEFAULT_CONFIG_NAME,
    ]  # both searched; creation is handled by model_paths.sidecar_dir()
    for p in candidates:
        if p.is_file():
            logger.debug("Auto-discovered config: %s", p)
            return p
    return None


def load_config(path: Path | None = None, *, strict: bool = False) -> dict[str, Any]:
    """Load ``gensrt-config.json`` as a flat dict.

    Args:
        path:   Explicit path override.  If ``None``, auto-discover.
        strict: If ``True``, raise on parse/read errors.  If ``False``
                (default), log a warning and return ``{}``.

    Returns:
        Parsed config dict, or ``{}`` if file is absent (non-strict).
    """
    from gensrt.exceptions import ConfigError, ConfigParseError

    resolved = path or _find_config_file()
    if resolved is None:
        logger.debug("No config file found; using built-in defaults.")
        return {}

    resolved = Path(resolved)
    if not resolved.exists():
        if strict:
            raise ConfigError(f"Config file not found: {resolved}")
        logger.debug("Config file not found: %s", resolved)
        return {}

    try:
        text = resolved.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Cannot read config file {resolved}: {exc}"
        if strict:
            raise ConfigError(msg) from exc
        logger.warning("%s", msg)
        return {}

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigParseError(
            str(resolved), exc.msg, exc.lineno, exc.colno
        ) from exc

    if not isinstance(data, dict):
        msg = f"Config file {resolved} must contain a JSON object, not {type(data).__name__}."
        if strict:
            raise ConfigError(msg)
        logger.warning("%s", msg)
        return {}

    logger.info("Loaded config: %s", resolved)
    return data


def generate_default_config(path: Path | None = None) -> Path:
    """Write a fully-populated default config file.

    Args:
        path: Destination path.  Defaults to ``./gensrt-config.json``.

    Returns:
        The path that was written.
    """
    if path is None:
        # Next to the executable rather than the working directory: a config
        # written to wherever the user happened to run from is a config they
        # will not find again. See model_paths.sidecar_dir().
        from gensrt.model_paths import sidecar_dir

        path = sidecar_dir() / DEFAULT_CONFIG_NAME

    path = Path(path)
    path.write_text(json.dumps(BUILTIN_DEFAULTS, indent=2) + "\n", encoding="utf-8")
    logger.info("Default config written: %s", path)
    return path


def merge_config(
    config: dict[str, Any],
    cli_args: dict[str, Any],
) -> dict[str, Any]:
    """Merge config file values with CLI arguments.

    Precedence: built-in defaults < config file < CLI args.

    CLI args that are ``None`` (not explicitly provided) fall through to the
    config file value, which in turn falls through to the built-in default.

    Args:
        config:   Dict from :func:`load_config`.
        cli_args: Dict from ``vars(argparse.Namespace)``.

    Returns:
        Merged dict with all values resolved.
    """
    merged: dict[str, Any] = {}

    for key, builtin_default in BUILTIN_DEFAULTS.items():
        cli_val = cli_args.get(key)
        cfg_val = config.get(key)

        if cli_val is not None:
            merged[key] = cli_val
        elif cfg_val is not None:
            merged[key] = cfg_val
        else:
            merged[key] = builtin_default

    # Pass through CLI-only keys not in BUILTIN_DEFAULTS
    # (e.g. "inputs", "verbose", "quiet", "config", "init_config")
    for key, val in cli_args.items():
        if key not in merged:
            merged[key] = val

    return merged
