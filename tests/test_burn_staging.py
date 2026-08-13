"""Tests for the burn-in SRT staging helper.

Covers the review's §3.9: the SRT basename was interpolated raw into an
ffmpeg filtergraph, where [ ] , ; ' : are metacharacters.  Rather than
escaping across three parsing layers, the file is staged under a
guaranteed-safe ASCII name.
"""

from __future__ import annotations

import re

from gensrt.server import _BURN_STAGE_PREFIX, _stage_srt_for_burn

SAFE_NAME = re.compile(r"^[A-Za-z0-9_]+\.srt$")

HOSTILE_NAMES = [
    "Movie [1080p].srt",              # bracket — filtergraph link label
    "Show, Season 1.srt",             # comma — filter separator
    "Title; Subtitle.srt",            # semicolon — chain separator
    "It's a Movie.srt",               # apostrophe — quoting
    "Name=Value.srt",                 # equals — option separator
    "ചലച്ചിത്രം.srt",                     # non-ASCII Malayalam
    "Movie [2020], Part 1; it's.srt",  # all at once
]


def _write(tmp_path, name, body="1\n00:00:01,000 --> 00:00:02,000\nhello\n"):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return p


def test_staged_name_is_filtergraph_safe(tmp_path):
    for name in HOSTILE_NAMES:
        staged = _stage_srt_for_burn(_write(tmp_path, name))
        assert SAFE_NAME.match(staged.name), f"{name} → unsafe {staged.name}"
        assert staged.name.startswith(_BURN_STAGE_PREFIX)


def test_staging_preserves_bytes_exactly(tmp_path):
    body = "1\n00:00:01,000 --> 00:00:02,000\nആരോപിച്ചു\n\n"
    src = _write(tmp_path, "Movie [1080p].srt", body)
    staged = _stage_srt_for_burn(src)
    assert staged.read_bytes() == src.read_bytes()
    assert staged.read_text(encoding="utf-8") == body


def test_staged_names_are_unique(tmp_path):
    src = _write(tmp_path, "same.srt")
    names = {_stage_srt_for_burn(src).name for _ in range(10)}
    assert len(names) == 10


def test_sweep_removes_only_stale_staged_files(tmp_path, monkeypatch):
    import os
    import tempfile
    import time

    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    stale = tmp_path / f"{_BURN_STAGE_PREFIX}old12345.srt"
    stale.write_text("stale")
    old = time.time() - (48 * 3600)
    os.utime(stale, (old, old))

    fresh = tmp_path / f"{_BURN_STAGE_PREFIX}new12345.srt"
    fresh.write_text("fresh")

    unrelated = tmp_path / "someones_actual_subtitles.srt"
    unrelated.write_text("do not touch")

    _stage_srt_for_burn(_write(tmp_path, "input.srt"))

    assert not stale.exists()
    assert fresh.exists()
    assert unrelated.exists()
