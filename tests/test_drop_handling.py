"""Drag-and-drop routing in the pywebview handler.

Dropping MalayalamNews-2.ml.srt loaded the video and then displayed
MalayalamNews-2.srt instead. The handler discarded the dropped path on the
assumption that sidecar discovery would "re-discover and load this same SRT"
after the video loaded — true only when the dropped file happens to be
<basename>.srt, because discovery prefers the canonical name.
"""

from __future__ import annotations

import pytest

from gensrt.server import resolve_drop_targets


@pytest.fixture
def clip(tmp_path):
    for name in ("clip.mp4", "clip.srt", "clip.ml.srt", "clip.ko.srt"):
        (tmp_path / name).write_text("x")
    return tmp_path


# ── The regression ────────────────────────────────────────────────────────

def test_dropped_language_variant_is_kept(clip):
    """The actual bug: .ml.srt must not be replaced by .srt."""
    video, srt, explicit = resolve_drop_targets([str(clip / "clip.ml.srt")])
    assert srt == str(clip / "clip.ml.srt")
    assert explicit is True


def test_sibling_video_still_loads_alongside_it(clip):
    video, srt, _ = resolve_drop_targets([str(clip / "clip.ml.srt")])
    assert video == str(clip / "clip.mp4")


def test_korean_variant_too(clip):
    _, srt, explicit = resolve_drop_targets([str(clip / "clip.ko.srt")])
    assert srt.endswith("clip.ko.srt") and explicit


# ── Ordinary paths must not regress ───────────────────────────────────────

def test_video_alone_leaves_discovery_to_run(clip):
    """No SRT given, so sidecar discovery is exactly what we want."""
    video, srt, explicit = resolve_drop_targets([str(clip / "clip.mp4")])
    assert video == str(clip / "clip.mp4")
    assert srt is None
    assert explicit is False


def test_canonical_srt_alone(clip):
    video, srt, explicit = resolve_drop_targets([str(clip / "clip.srt")])
    assert srt == str(clip / "clip.srt")
    assert video == str(clip / "clip.mp4")
    assert explicit is True


def test_both_dropped_together(clip):
    video, srt, explicit = resolve_drop_targets(
        [str(clip / "clip.mp4"), str(clip / "clip.ko.srt")]
    )
    assert video == str(clip / "clip.mp4")
    assert srt == str(clip / "clip.ko.srt")
    assert explicit is True


def test_srt_with_no_sibling_video(tmp_path):
    orphan = tmp_path / "orphan.ml.srt"
    orphan.write_text("x")
    video, srt, explicit = resolve_drop_targets([str(orphan)])
    assert video is None
    assert srt == str(orphan)
    assert explicit is True


def test_unrelated_files_are_ignored(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    assert resolve_drop_targets([str(tmp_path / "notes.txt")]) == (None, None, False)


def test_empty_drop():
    assert resolve_drop_targets([]) == (None, None, False)


def test_extension_matching_is_case_insensitive(tmp_path):
    (tmp_path / "Clip.MP4").write_text("x")
    (tmp_path / "Clip.SRT").write_text("x")
    video, srt, explicit = resolve_drop_targets(
        [str(tmp_path / "Clip.MP4"), str(tmp_path / "Clip.SRT")]
    )
    assert video and srt and explicit
