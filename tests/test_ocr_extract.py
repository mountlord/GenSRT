"""Subtitle extraction: segmentation, voting, and the sampling command.

The engine is driven end to end here with a fake detector, a fake recogniser
and a fake ffmpeg, because the interesting behaviour is the STATE MACHINE —
when a cue opens, when it closes, and what text it settles on — and none of
that needs a real model or a real video.

What these tests pin, in rough order of how badly each hurts when it breaks:

* Cue boundaries. An off-by-one-sample error at the closing edge shifts
  every subtitle in the file.
* Jitter tolerance. OCR does not read identical pixels identically. Without
  the similarity comparison, one caption becomes three cues — the failure
  mode that would dominate a first real run.
* Voting. Several readings of one subtitle beat any single reading, and
  majority must outrank confidence: confidence tracks how HARD a crop was
  as much as whether the reading was right (observed on real frames: a
  correct reading at 0.39 beside an incorrect one at 0.65).
* Multi-line grouping. Two boxes on one frame are one two-line cue, not two
  overlapping cues.
"""

from __future__ import annotations

import pytest

from gensrt.exceptions import ConfigError
from gensrt.ocr.extract import (
    ExtractSettings,
    _OpenCue,
    comparison_form,
    extract_subtitles,
    same_subtitle,
)

np = pytest.importorskip("numpy")


# ── Comparison form ───────────────────────────────────────────────────────

def test_comparison_form_folds_fullwidth_and_drops_punctuation():
    """Both are things OCR disagrees with itself about on identical pixels."""
    assert comparison_form("０１２、こんにちは。") == "012こんにちは"


def test_comparison_form_ignores_whitespace():
    assert comparison_form("hello world") == comparison_form("helloworld")


@pytest.mark.parametrize("a,b,expected", [
    ("今日で8本目です", "今日で8本目です", True),      # identical
    ("こんにちは、元気ですか", "こんにちは元気ですか", True),  # punctuation only
    ("今日で8本目です", "今日で8本ヨです", True),        # one character misread
    ("今日で8本目です", "まったく違う文章です", False),   # a different caption
    ("", "", True),
    ("something", "", False),
])
def test_same_subtitle(a, b, expected):
    assert same_subtitle(a, b, 0.85) is expected


def test_threshold_is_honoured():
    a, b = "abcdefghij", "abcdefghXY"
    assert same_subtitle(a, b, 0.5) is True
    assert same_subtitle(a, b, 0.99) is False


# ── Voting ────────────────────────────────────────────────────────────────

def test_majority_beats_a_confident_outlier():
    cue = _OpenCue(0.0, 0.0, [("AAA", 0.40), ("AAA", 0.50), ("AAA", 0.45),
                              ("BBB", 0.99)])
    text, _ = cue.vote()
    assert text == "AAA"


def test_confidence_breaks_a_tie():
    cue = _OpenCue(0.0, 0.0, [("AAA", 0.40), ("BBB", 0.90)])
    assert cue.vote()[0] == "BBB"


def test_variants_of_the_same_text_vote_together():
    """Punctuation-only differences are one group, so three near-identical
    readings outvote one genuinely different one."""
    cue = _OpenCue(0.0, 0.0, [("hello, world", 0.5), ("hello world", 0.6),
                              ("hello world.", 0.55), ("goodbye", 0.95)])
    assert cue.vote()[0].startswith("hello")


def test_empty_cue_votes_to_nothing():
    assert _OpenCue(0.0, 0.0, []).vote() == ("", None)


# ── Settings validation ───────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs", [
    {"sample_fps": 0},
    {"sample_fps": -1},
    {"sample_fps": 60},                      # would cost hours for no gain
    {"similarity": 0},
    {"similarity": 1.5},
    {"min_duration_s": -1},
    {"start_time": -5},
    {"start_time": 10, "end_time": 5},
    {"region": (0, 0, 2, 2)},                # too small to hold text
    {"region": (-5, 0, 100, 100)},
])
def test_invalid_settings_are_rejected(kwargs):
    with pytest.raises(ConfigError):
        ExtractSettings(**kwargs).validate()


def test_sensible_settings_pass():
    ExtractSettings(region=(40, 820, 1840, 220), sample_fps=2.0,
                    start_time=60, end_time=180).validate()


# ── ffmpeg command construction ───────────────────────────────────────────

def test_sample_command_crops_before_resampling():
    """Cropping first means the pixels outside the region are never decoded
    into this process, and detection runs on a strip instead of a frame."""
    from gensrt.ocr.extract import _ffmpeg_sample_command
    from pathlib import Path

    cmd = _ffmpeg_sample_command(
        Path("v.mp4"),
        ExtractSettings(region=(10, 20, 300, 80), sample_fps=2.5),
        300, 80,
    )
    joined = " ".join(cmd)
    assert "crop=300:80:10:20,fps=2.5" in joined
    assert "rawvideo" in joined and "bgr24" in joined


def test_sample_command_seeks_before_input():
    """-ss before -i is the fast seek; after -i it decodes everything up to
    the start point."""
    from gensrt.ocr.extract import _ffmpeg_sample_command
    from pathlib import Path

    cmd = _ffmpeg_sample_command(
        Path("v.mp4"), ExtractSettings(start_time=90, end_time=150), 100, 50)
    assert cmd.index("-ss") < cmd.index("-i")
    assert "-t" in cmd and cmd[cmd.index("-t") + 1].startswith("60")


# ── End-to-end state machine, with everything faked ───────────────────────

class _FakeRegion:
    def __init__(self, index):
        self.index = index
        self.quad = [(0, 0), (10, 0), (10, 5), (0, 5)]


class _ScriptedOCR:
    """Returns a scripted reading per frame.

    Each script entry is the list of text lines detected on that frame, so
    [] is a blank frame and ["a", "b"] is a two-line subtitle.
    """

    def __init__(self, script):
        self.script = list(script)
        self.frame = -1

    def detect(self, _image):
        self.frame += 1
        lines = self.script[self.frame] if self.frame < len(self.script) else []
        return [_FakeRegion(i) for i in range(1, len(lines) + 1)]

    def recognize(self, crop):
        lines = self.script[self.frame]
        idx = getattr(crop, "_line_index", 0)
        return (lines[idx], 0.9) if idx < len(lines) else ("", None)


def _run(monkeypatch, script, **settings_kwargs):
    """Drive extract_subtitles over a scripted sequence of frames."""
    ocr = _ScriptedOCR(script)
    line_counter = {"i": 0}

    def _crop(_frame, _quad):
        arr = np.zeros((5, 10, 3), dtype=np.uint8).view(_TaggedArray)
        arr._line_index = line_counter["i"]
        line_counter["i"] += 1
        return arr

    def _detect(image):
        line_counter["i"] = 0            # reset per frame
        return ocr.detect(image)

    class _Det:
        name = "fake"
        detect = staticmethod(_detect)

    monkeypatch.setattr("gensrt.ocr.factory.get_detector", lambda: _Det())
    monkeypatch.setattr("gensrt.ocr.factory.get_recognizer",
                        lambda code, status=None: ocr)
    monkeypatch.setattr("gensrt.ocr.ppocr_onnx.crop_region", _crop)
    # probe_video returns (width, height, duration); the region below is the
    # whole 10x5 "frame", and a 600s duration keeps range validation happy.
    monkeypatch.setattr("gensrt.ocr.extract.probe_video",
                        lambda video: (10, 5, 600.0))

    frame_bytes = 10 * 5 * 3
    payload = b"\x00" * frame_bytes * len(script)
    monkeypatch.setattr("gensrt.ocr.extract.subprocess.Popen",
                        lambda *a, **k: _FakeProc(payload))
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)

    settings_kwargs.setdefault("min_duration_s", 0.0)
    settings = ExtractSettings(sample_fps=2.0, **settings_kwargs)
    return extract_subtitles("fake.mp4", settings)


class _TaggedArray(np.ndarray):
    """ndarray that can carry which detected line it came from."""
    _line_index = 0


class _FakeProc:
    def __init__(self, payload):
        import io

        self.stdout = io.BytesIO(payload)
        self.stderr = io.BytesIO(b"")
        self._done = False

    def wait(self):
        self._done = True
        return 0

    def poll(self):
        return 0 if self._done else None

    def kill(self):
        self._done = True


def test_one_caption_across_several_frames_is_one_cue(monkeypatch):
    cues = _run(monkeypatch, [["hello"], ["hello"], ["hello"], []])
    assert len(cues) == 1
    assert cues[0].text == "hello"
    assert cues[0].start == 0.0
    assert cues[0].end == pytest.approx(1.5)      # last seen 1.0s + one step


def test_ocr_jitter_does_not_split_a_caption(monkeypatch):
    """The failure mode that would dominate a first real run."""
    cues = _run(monkeypatch, [
        ["今日で8本目です"], ["今日で8本ヨです"], ["今日で8本目です"], [],
    ])
    assert len(cues) == 1
    assert cues[0].text == "今日で8本目です"        # the majority reading


def test_different_captions_become_separate_cues(monkeypatch):
    cues = _run(monkeypatch, [["first"], ["first"], ["second"], ["second"], []])
    assert [c.text for c in cues] == ["first", "second"]
    assert cues[0].end == pytest.approx(1.0)      # closes where the next opens
    assert cues[1].start == pytest.approx(1.0)


def test_blank_frames_close_a_cue_and_leave_a_gap(monkeypatch):
    cues = _run(monkeypatch, [["a"], [], [], ["b"], []])
    assert [c.text for c in cues] == ["a", "b"]
    assert cues[0].end == pytest.approx(0.5)
    assert cues[1].start == pytest.approx(1.5)


def test_two_boxes_on_one_frame_make_one_two_line_cue(monkeypatch):
    """Not two overlapping cues — that is what the overlap-clamp would then
    have to mangle."""
    cues = _run(monkeypatch, [["line one", "line two"], []])
    assert len(cues) == 1
    assert cues[0].text == "line one\nline two"


def test_short_cues_are_dropped(monkeypatch):
    """Fade-in/out produces partial reads at the boundaries."""
    cues = _run(monkeypatch,
                [["blip"], ["real"], ["real"], ["real"], []],
                min_duration_s=0.9)
    assert [c.text for c in cues] == ["real"]


def test_a_cue_still_open_at_the_end_is_closed(monkeypatch):
    cues = _run(monkeypatch, [["last"], ["last"]])
    assert len(cues) == 1
    assert cues[0].end == pytest.approx(1.0)


def test_no_text_anywhere_yields_no_cues(monkeypatch):
    assert _run(monkeypatch, [[], [], []]) == []


def test_cue_indices_are_contiguous(monkeypatch):
    cues = _run(monkeypatch, [["a"], [], ["b"], [], ["c"], []])
    assert [c.index for c in cues] == [1, 2, 3]


def test_start_offset_shifts_all_timings(monkeypatch):
    cues = _run(monkeypatch, [["x"], ["x"], []], start_time=60.0)
    assert cues[0].start == pytest.approx(60.0)
    assert cues[0].end == pytest.approx(61.0)


# ── CLI wiring ────────────────────────────────────────────────────────────
#
# The unit tests above all call extract_subtitles() directly, so none of them
# touched the CLI runner — and the first real invocation failed instantly on
# `args.input` (the argparse dest is `inputs`, because --input is repeatable).
# These drive the runner the way the terminal does.

def _extract_args(argv):
    from gensrt.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(argv)
    # main() does this merge before dispatching; mirror it here.
    if not args.inputs and getattr(args, "inputs_pos", None):
        args.inputs = list(args.inputs_pos)
    return args


def test_cli_runner_reaches_extraction_with_the_parsed_settings(monkeypatch, tmp_path):
    from gensrt import cli

    video = tmp_path / "movie.mkv"
    video.write_bytes(b"x")
    seen = {}

    def _fake_extract(path, settings, progress=None):
        seen["path"] = path
        seen["settings"] = settings
        return [__import__("gensrt.models", fromlist=["SRTSegment"]).SRTSegment(
            index=1, start=0.0, end=1.0, text="ok")]

    monkeypatch.setattr("gensrt.ocr.extract.extract_subtitles", _fake_extract)

    args = _extract_args([
        "--input", str(video), "--extract-subtitles",
        "--region", "40,820,1840,220", "--ocr-language", "zh",
        "--extract-from", "600", "--extract-to", "720",
        "--output-filename", "out.srt", "--output", str(tmp_path),
    ])
    assert cli._run_extract(args) == 0

    settings = seen["settings"]
    assert seen["path"] == video
    assert settings.region == (40, 820, 1840, 220)
    assert settings.language == "zh"
    assert settings.start_time == 600.0
    assert settings.end_time == 720.0
    assert (tmp_path / "out.srt").is_file()


def test_cli_runner_rejects_zero_or_many_inputs(monkeypatch, tmp_path):
    from gensrt import cli

    assert cli._run_extract(_extract_args(["--extract-subtitles"])) == 2

    a = tmp_path / "a.mkv"; a.write_bytes(b"x")
    b = tmp_path / "b.mkv"; b.write_bytes(b"x")
    args = _extract_args(["--input", str(a), "--input", str(b),
                          "--extract-subtitles"])
    assert cli._run_extract(args) == 2


def test_cli_runner_reports_when_nothing_was_found(monkeypatch, tmp_path):
    from gensrt import cli

    video = tmp_path / "movie.mkv"
    video.write_bytes(b"x")
    monkeypatch.setattr("gensrt.ocr.extract.extract_subtitles",
                        lambda *a, **k: [])
    args = _extract_args(["--input", str(video), "--extract-subtitles",
                          "--output", str(tmp_path)])
    assert cli._run_extract(args) == 1


@pytest.mark.parametrize("bad", ["1,2,3", "a,b,c,d", "", "1,2,3,4,5"])
def test_bad_region_strings_are_rejected(bad):
    from gensrt.cli import _parse_region

    if bad == "":
        assert _parse_region(bad) is None
        return
    with pytest.raises(SystemExit):
        _parse_region(bad)


# ── Range and region validation against the real file ─────────────────────
#
# The bug these exist for: --extract-from 600 on a video shorter than ten
# minutes. ffmpeg seeks past the end, exits CLEANLY, writes nothing, and the
# empty pipe is indistinguishable from "the region contained no text". The
# first real run reported a region problem that did not exist.

def _patch_probe(monkeypatch, width=1920, height=1080, duration=300.0,
                 allow_models=False):
    """Patch the probe, and by default make loading a model an error.

    Validation has to happen BEFORE any model is fetched — being told the
    time range is wrong after an 11 MB download is a bad trade, and the
    first version got this backwards. Tests that exercise validation leave
    allow_models=False so the ordering cannot regress silently; tests that
    need to reach ffmpeg pass True.
    """
    monkeypatch.setattr("gensrt.ocr.extract.probe_video",
                        lambda video: (width, height, duration))
    monkeypatch.setattr("pathlib.Path.is_file", lambda self: True)

    if allow_models:
        monkeypatch.setattr("gensrt.ocr.factory.get_recognizer",
                            lambda code, status=None: object())
        monkeypatch.setattr("gensrt.ocr.factory.get_detector", lambda: object())
    else:
        def _must_not_load(*_a, **_k):
            raise AssertionError(
                "a model was loaded before the request was validated")

        monkeypatch.setattr("gensrt.ocr.factory.get_recognizer", _must_not_load)
        monkeypatch.setattr("gensrt.ocr.factory.get_detector", _must_not_load)


def test_start_past_end_of_video_is_a_clear_error(monkeypatch):
    from gensrt.ocr.extract import OCRError

    _patch_probe(monkeypatch, duration=300.0)
    with pytest.raises(OCRError) as exc:
        extract_subtitles("v.mp4", ExtractSettings(start_time=600.0, end_time=720.0))
    message = str(exc.value)
    assert "past the end" in message
    assert "300" in message          # tells the user the actual length


def test_region_larger_than_the_frame_is_rejected(monkeypatch):
    from gensrt.ocr.extract import OCRError

    _patch_probe(monkeypatch, width=1920, height=1080)
    with pytest.raises(OCRError) as exc:
        extract_subtitles("v.mp4", ExtractSettings(region=(40, 820, 1840, 400)))
    message = str(exc.value)
    assert "1920" in message and "1080" in message


def test_region_that_fits_exactly_is_accepted(monkeypatch):
    from gensrt.ocr.extract import _validate_against_video

    _validate_against_video(
        ExtractSettings(region=(0, 881, 1920, 178)), 1920, 1059, 300.0)


def test_end_beyond_the_video_warns_but_proceeds(monkeypatch, caplog):
    import logging

    from gensrt.ocr.extract import _validate_against_video

    with caplog.at_level(logging.WARNING, logger="gensrt.ocr.extract"):
        _validate_against_video(
            ExtractSettings(start_time=0, end_time=9999), 1920, 1080, 300.0)
    assert any("beyond the video length" in r.message for r in caplog.records)


def test_zero_sampled_frames_does_not_blame_the_region(monkeypatch):
    """ffmpeg yielding nothing must not be reported as an OCR miss — no
    pixel was ever examined."""
    from gensrt.ocr.extract import OCRError

    _patch_probe(monkeypatch, duration=300.0, allow_models=True)
    monkeypatch.setattr("gensrt.ocr.extract.subprocess.Popen",
                        lambda *a, **k: _FakeProc(b""))

    with pytest.raises(OCRError) as exc:
        extract_subtitles("v.mp4", ExtractSettings(start_time=0, end_time=60))
    message = str(exc.value)
    assert "sampled no frames" in message
    assert "region" not in message.lower()


# ── Translation of extracted cues ─────────────────────────────────────────
#
# SRTSegment is a FROZEN dataclass. The first version assigned cue.text
# directly, which raises; the assignment sat inside the same try/except that
# guards engine failures, so a real extraction produced a full file of
# untranslated source text and one warning line that scrolled past.

def _cue(text, index=1):
    from gensrt.models import SRTSegment

    return SRTSegment(index=index, start=0.0, end=1.0, text=text)


def test_translation_replaces_frozen_cues(monkeypatch):
    from gensrt.ocr.extract import _translate_cues

    class _Engine:
        def translate_batch(self, texts, src, tgt):
            assert (src, tgt) == ("zh", "en")
            return [f"EN:{t}" for t in texts]

    monkeypatch.setattr("gensrt.translation.factory.get_engine",
                        lambda key, config=None: _Engine())
    cues = [_cue("你現在要回家", 1), _cue("是", 2)]
    _translate_cues(cues, ExtractSettings(language="zh", target_language="en"))
    assert [c.text for c in cues] == ["EN:你現在要回家", "EN:是"]


def test_translation_preserves_timings_and_indices(monkeypatch):
    from gensrt.models import SRTSegment
    from gensrt.ocr.extract import _translate_cues

    monkeypatch.setattr(
        "gensrt.translation.factory.get_engine",
        lambda key, config=None: type("E", (), {
            "translate_batch": lambda self, t, s, g: ["translated"] * len(t)
        })())
    cues = [SRTSegment(index=7, start=12.5, end=15.25, text="原文")]
    _translate_cues(cues, ExtractSettings(language="zh"))
    assert (cues[0].index, cues[0].start, cues[0].end) == (7, 12.5, 15.25)
    assert cues[0].text == "translated"


def test_engine_failure_keeps_the_extraction(monkeypatch, caplog):
    """Twenty minutes of extraction must not be lost to a translation fault —
    source-language cues are still a usable result."""
    import logging

    from gensrt.ocr.extract import _translate_cues

    def _boom(*_a, **_k):
        raise RuntimeError("no model")

    monkeypatch.setattr("gensrt.translation.factory.get_engine", _boom)
    cues = [_cue("原文")]
    with caplog.at_level(logging.WARNING, logger="gensrt.ocr.extract"):
        _translate_cues(cues, ExtractSettings(language="zh"))
    assert cues[0].text == "原文"
    assert any("Translation failed" in r.message for r in caplog.records)


def test_partial_translation_warns(monkeypatch, caplog):
    """Blank results are not a crash, but they are not success either."""
    import logging

    from gensrt.ocr.extract import _translate_cues

    monkeypatch.setattr(
        "gensrt.translation.factory.get_engine",
        lambda key, config=None: type("E", (), {
            "translate_batch": lambda self, t, s, g: ["ok", "", "  "]
        })())
    cues = [_cue("a", 1), _cue("b", 2), _cue("c", 3)]
    with caplog.at_level(logging.WARNING, logger="gensrt.ocr.extract"):
        _translate_cues(cues, ExtractSettings(language="zh"))
    assert [c.text for c in cues] == ["ok", "b", "c"]
    assert any("Translated 1 of 3" in r.message for r in caplog.records)
