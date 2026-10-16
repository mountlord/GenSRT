"""Named default sets — one word instead of a command line to remember.

Why
---
Two kinds of material want different defaults, and until now the user
carried the difference in their head and in a shell script::

    gensrt --input F:\\Live --source-language ko --model large-v3-turbo
           --translation-engine madlad ...
    gensrt --input F:\\Films --source-language ja
           --model kotoba-tech/kotoba-whisper-v2.0-faster --chunk-mode fixed ...

The second line, mis-typed with ``--source-language ko``, fed Korean to a
Japanese-only model and produced fluent Japanese for a Korean broadcast
(2026-09-28).  A profile bundles the model with the language it is for,
so that mistake is caught by the model/language guard before any work.

What a profile is
-----------------
A name, a one-line description, and a dict of config keys — the same keys
as ``gensrt-config.json``.  A profile FILLS IN, it does not overrule: a
key it names is applied only when the config file still has the built-in
default for it.  A value the user set in ``gensrt-config.json`` is kept,
and the log says so (``kept from gensrt-config.json: …``).  Explicit
flags beat both.  So:

    built-in defaults  <  profile  <  values set in gensrt-config.json  <  flags

(2026-10-07: a profile that set ``translation_engine`` silently undid the
user's edit of that key in the config file, and an engine A/B never
happened.  The engine is a property of the machine — VRAM, licence — not
of the material, so no shipped profile names it any more either.)

A profile never sets paths (output, report and dump directories are per
run) and never sets the device.

Two ship (``BUILTIN_PROFILES``); ``gensrt-profiles.json`` next to the
config adds or overrides by name (``--init-profiles`` writes the built-ins
out for editing).  A profile naming a key that is not a config key is a
``ConfigError`` — silently ignoring a typo would be worse than stopping.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_NAME = "gensrt-profiles.json"

BUILTIN_PROFILES: dict[str, dict[str, Any]] = {
    "broadcast": {
        "description": (
            "Studio and broadcast speech — news, live streams, interviews. "
            "Whisper large-v3-turbo with VAD chunking (fast; silence is not "
            "decoded). Any source language."
        ),
        "settings": {
            "model": "large-v3-turbo",
            "asr_engine": "chunked",
            "chunk_mode": "vad",
        },
    },
    "soft-speech": {
        "description": (
            "Soft speech and speech under other vocal sounds, any language. "
            "Fixed 5-8 s windows with no voice detector in front (recovers what "
            "VAD drops categorically; several times the decode time), cues "
            "snapped to the audible onset, and the soft-speech heuristics (lone "
            "cues, laughter, repetition loops removed) take out what the recall "
            "brings. Japanese runs on "
            "kotoba-whisper v2.0, everything else on large-v3-turbo."
        ),
        "settings": {
            "model": "large-v3-turbo",
            "asr_engine": "chunked",
            "chunk_mode": "fixed",
            "snap_onsets": True,
            "heuristics_mode": "soft-speech",
        },
        # Chosen AFTER the source language is known (flag > profile > file),
        # and only when no explicit --model was given: the material decides
        # the mode, the language decides the model.
        "model_by_language": {
            "ja": "kotoba-tech/kotoba-whisper-v2.0-faster",
        },
    },
}

#: Config keys a profile may not set: they are per run or per machine.
_NOT_IN_PROFILE = frozenset({
    "output", "output_filename", "recurse", "log_level", "profile",
    "device", "backend", "gpu_id", "compute_type",
    # Which translator fits is decided by VRAM and licence, per machine.
    "translation_engine", "translation_model", "madlad_model",
    "debug_chunk_dir", "dump_segments_dir", "heuristics_report_dir",
    "remux_cache_keep",
})


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    settings: dict[str, Any]
    source: str = "built-in"          # "built-in" or the file it came from
    #: language code → model, applied by :func:`choose_model` once the
    #: effective source language is known.  Lets one profile describe the
    #: material while the language picks the model.
    model_by_language: dict[str, str] = field(default_factory=dict)

    def model_for(self, language: str | None) -> str | None:
        code = (language or "").strip().lower()
        for key in (code, code.split("-")[0]):
            if key and key in self.model_by_language:
                return self.model_by_language[key]
        return None


@dataclass
class Profiles:
    by_name: dict[str, Profile] = field(default_factory=dict)
    source_path: Path | None = None

    def get(self, name: str) -> Profile:
        """The profile called *name*, or a ConfigError naming the ones that exist."""
        from gensrt.exceptions import ConfigError

        key = (name or "").strip().lower()
        if key in self.by_name:
            return self.by_name[key]
        known = ", ".join(sorted(self.by_name)) or "(none)"
        raise ConfigError(
            f"Unknown profile {name!r}. Available: {known}. "
            f"Run 'gensrt --list-profiles' to see what each one sets."
        )

    def names(self) -> list[str]:
        return sorted(self.by_name)


def _find_profiles_file() -> Path | None:
    """Same search order as the config file: next to the exe, then cwd."""
    import sys

    for p in (Path(sys.argv[0]).resolve().parent / DEFAULT_PROFILES_NAME,
              Path.cwd() / DEFAULT_PROFILES_NAME):
        if p.is_file():
            return p
    return None


def _validate_settings(name: str, settings: Any, source: str) -> dict[str, Any]:
    from gensrt.config import BUILTIN_DEFAULTS
    from gensrt.exceptions import ConfigError

    if not isinstance(settings, dict):
        raise ConfigError(f"Profile {name!r} ({source}): 'settings' must be an object.")
    unknown = sorted(k for k in settings if k not in BUILTIN_DEFAULTS)
    if unknown:
        raise ConfigError(
            f"Profile {name!r} ({source}) sets keys that are not config keys: "
            f"{', '.join(unknown)}."
        )
    barred = sorted(k for k in settings if k in _NOT_IN_PROFILE)
    if barred:
        raise ConfigError(
            f"Profile {name!r} ({source}) may not set {', '.join(barred)}: "
            f"paths, the device and the translation engine are per machine or "
            f"per run, not per profile — set them in gensrt-config.json."
        )
    return dict(settings)


def _from_dict(name: str, d: Any, source: str) -> Profile:
    from gensrt.exceptions import ConfigError

    if not isinstance(d, dict):
        raise ConfigError(f"Profile {name!r} ({source}) must be an object with 'settings'.")
    mbl = d.get("model_by_language") or {}
    if not isinstance(mbl, dict) or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in mbl.items()
    ):
        raise ConfigError(
            f"Profile {name!r} ({source}): 'model_by_language' must map language "
            f"codes to model names."
        )
    return Profile(
        name=name.strip().lower(),
        description=str(d.get("description") or ""),
        settings=_validate_settings(name, d.get("settings", {}), source),
        source=source,
        model_by_language={k.strip().lower(): v for k, v in mbl.items()},
    )


def load_profiles(path: Path | None = None) -> Profiles:
    """Built-in profiles plus those in *path* (or the discovered file).

    Unlike the heuristics file, a malformed profiles file is an error, not
    a warning: the user asked for a profile by name and must get exactly
    what it says or nothing.
    """
    from gensrt.exceptions import ConfigError

    profiles = Profiles(by_name={n: _from_dict(n, d, "built-in")
                                 for n, d in BUILTIN_PROFILES.items()})
    path = path or _find_profiles_file()
    if path is None:
        return profiles
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ConfigError(f"Could not read {path}: {exc}") from exc
    entries = loaded.get("profiles") if isinstance(loaded, dict) else None
    if not isinstance(entries, dict):
        raise ConfigError(f"{path}: expected {{\"profiles\": {{name: {{...}}}}}}.")
    for name, d in entries.items():
        profiles.by_name[name.strip().lower()] = _from_dict(name, d, str(path))
    profiles.source_path = Path(path)
    return profiles


@dataclass
class Applied:
    """What a profile did to the config-file dict, for the log and banner."""
    profile: Profile | None = None
    applied: dict[str, Any] = field(default_factory=dict)   # key → profile value used
    kept: dict[str, Any] = field(default_factory=dict)      # key → file value kept

    def summary(self) -> str:
        if self.profile is None:
            return ""
        parts = [", ".join(f"{k}={v}" for k, v in self.applied.items()) or "(nothing to fill in)"]
        if self.kept:
            parts.append("kept from gensrt-config.json: " + ", ".join(
                f"{k}={v} (profile has {self.profile.settings[k]})" for k, v in self.kept.items()))
        return "; ".join(parts)


def _is_user_set(file_config: dict[str, Any], key: str) -> bool:
    """True when the config file holds a value for *key* that differs from
    the built-in default — i.e. the user chose it."""
    from gensrt.config import BUILTIN_DEFAULTS

    if key not in file_config or file_config[key] is None:
        return False
    return file_config[key] != BUILTIN_DEFAULTS.get(key)


def apply_profile(file_config: dict[str, Any], name: str | None,
                  profiles: Profiles | None = None) -> tuple[dict[str, Any], Applied]:
    """Fill the named profile's settings into the config-file dict.

    A key the user set in the file (value differs from the built-in
    default) is kept and reported; the rest take the profile's value.
    Returns the new dict and an :class:`Applied` record.  Explicit CLI
    flags are merged on top by the caller.
    """
    if not name or not str(name).strip():
        return dict(file_config), Applied()
    prof = (profiles or load_profiles()).get(str(name))
    merged = dict(file_config)
    rec = Applied(profile=prof)
    for key, value in prof.settings.items():
        if _is_user_set(file_config, key):
            rec.kept[key] = file_config[key]
        else:
            merged[key] = value
            rec.applied[key] = value
    logger.info("Profile %r: %s", prof.name, rec.summary())
    return merged, rec


def choose_model(merged: dict[str, Any], prof: Profile | None, *,
                 explicit_model: bool, file_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Apply the profile's per-language model to the fully merged settings.

    Runs after flags are merged, so the language it reads is the effective
    one.  An explicit ``--model`` (or request ``model``), or a model the
    user set in the config file, always wins.
    """
    if prof is None or explicit_model:
        return merged
    if file_config is not None and _is_user_set(file_config, "model"):
        logger.info("Profile %r: kept model=%s from gensrt-config.json", prof.name, merged.get("model"))
        return merged
    chosen = prof.model_for(merged.get("source_language"))
    if chosen and chosen != merged.get("model"):
        logger.info("Profile %r: model %s for source language %s",
                    prof.name, chosen, merged.get("source_language"))
        return {**merged, "model": chosen}
    return merged


def describe(profiles: Profiles) -> str:
    """Text for ``--list-profiles``."""
    lines = []
    for name in profiles.names():
        p = profiles.by_name[name]
        lines.append(f"{name}  [{p.source}]")
        if p.description:
            lines.append(f"    {p.description}")
        for k, v in p.settings.items():
            lines.append(f"    {k} = {v}")
        for lang, model in p.model_by_language.items():
            lines.append(f"    model when source_language={lang}: {model}")
        lines.append("")
    if profiles.source_path:
        lines.append(f"User profiles: {profiles.source_path}")
    else:
        lines.append(f"No {DEFAULT_PROFILES_NAME} found; --init-profiles writes the built-ins for editing.")
    return "\n".join(lines)


def write_default_profiles(path: Path | None = None) -> Path:
    """Write the built-in profiles out for editing (``--init-profiles``)."""
    from gensrt.model_paths import sidecar_dir

    path = path or (sidecar_dir() / DEFAULT_PROFILES_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"profiles": BUILTIN_PROFILES}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
