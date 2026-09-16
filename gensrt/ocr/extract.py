"""Extract burned-in subtitles from a video into cues.

The problem, and why it is not an OCR problem
----------------------------------------------
Recognising the text on one frame is already solved (:mod:`gensrt.ocr`).
The unsolved part is TIME: knowing that a caption appeared at 14:02.5 and
vanished at 14:06.0, when all you can do is look at individual frames.

The obvious approach — diff the pixels in the subtitle band and re-read
when they change — fails on the common case.  Most burned-in subtitles are
white text with an outline sitting directly over moving video, so the band
changes on *every* frame and a pixel diff fires constantly.  Making it work
means binarising, thresholding, and tuning against compression noise.

Comparing the recognised TEXT instead is invariant to whatever the picture
is doing behind the caption, and it buys something a pixel diff cannot:
several independent readings of the same subtitle.  At 2 fps a three-second
caption is read six times, and those six readings can be voted on.  That
matters because OCR is not deterministic on hard material — the Japanese
model has been observed reading the same on-screen logo three different
ways across three frames.  Voting turns that liability into an advantage.

The pipeline
------------
1. ffmpeg samples the video at ``sample_fps``, cropped to the subtitle
   region, as raw BGR frames on a pipe.  Cropping in ffmpeg rather than in
   Python means the pixels outside the region are never decoded into our
   process at all, and detection runs on a strip instead of a full frame.
2. Each frame goes through the existing detector + recogniser.
3. A one-cue state machine (see :class:`_OpenCue`) accumulates readings
   while the text stays the same and closes the cue when it changes.
4. Each closed cue votes across its readings for a final text.
5. Cues shorter than ``min_duration_s`` are dropped as fade artifacts.

Nothing here is Whisper-adjacent: the output is a list of
:class:`~gensrt.models.SRTSegment`, which the normal SRT writer consumes.
"""

from __future__ import annotations

import logging
import re
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from gensrt.exceptions import ConfigError
from gensrt.models import SRTSegment
from gensrt.ocr.ppocr_onnx import OCRError

logger = logging.getLogger(__name__)

#: Characters ignored when deciding whether two readings are "the same
#: subtitle".  OCR routinely disagrees with itself about punctuation and
#: spacing on identical pixels, and those disagreements should not split one
#: caption into three cues.
_IGNORE_FOR_COMPARISON = re.compile(r"[\s.,!?;:''\"「」『』、。・…\-—~〜]+")


@dataclass
class ExtractSettings:
    """Everything the extraction run needs beyond the video itself.

    Args:
        region:         ``(x, y, w, h)`` in VIDEO pixels — the subtitle area.
                        ``None`` reads the whole frame, which works but is
                        slower and sweeps in logos and watermarks.
        language:       ISO 639-1 code for the OCR recognition model.
        sample_fps:     Frames inspected per second.  Sets both the boundary
                        precision (±1/(2·fps)) and how many readings each cue
                        gets to vote with.  2.0 is a reasonable default: ±250 ms
                        boundaries, and 6 readings for a three-second caption.
        start_time:     Where to begin, in seconds.
        end_time:       Where to stop; ``None`` means end of video.
        min_duration_s: Cues shorter than this are discarded.  Subtitles
                        fading in and out produce partial reads at the
                        boundaries, which surface as sub-second junk cues.
        similarity:     0-1.  Two readings whose comparison forms are at least
                        this similar are the same subtitle.  Below ~0.7 distinct
                        captions start merging; above ~0.95 OCR jitter splits
                        single captions apart.
        translate:      Translate each finished cue.
        target_language: Translation target.
    """

    region: tuple[int, int, int, int] | None = None
    language: str = "ja"
    sample_fps: float = 2.0
    start_time: float = 0.0
    end_time: float | None = None
    min_duration_s: float = 0.5
    similarity: float = 0.85
    translate: bool = False
    target_language: str = "en"

    def validate(self) -> None:
        if self.sample_fps <= 0:
            raise ConfigError(f"sample_fps must be > 0 (got {self.sample_fps}).")
        if self.sample_fps > 30:
            raise ConfigError(
                f"sample_fps of {self.sample_fps} is higher than any subtitle "
                f"needs and would cost hours; 1-4 is the useful range."
            )
        if not (0.0 < self.similarity <= 1.0):
            raise ConfigError(
                f"similarity must be in (0, 1] (got {self.similarity})."
            )
        if self.min_duration_s < 0:
            raise ConfigError("min_duration_s cannot be negative.")
        if self.start_time < 0:
            raise ConfigError("start_time cannot be negative.")
        if self.end_time is not None and self.end_time <= self.start_time:
            raise ConfigError(
                f"end_time ({self.end_time}) must be after start_time "
                f"({self.start_time})."
            )
        if self.region is not None:
            x, y, w, h = self.region
            if w < 8 or h < 8:
                raise ConfigError(
                    f"region is {w}×{h} px — too small to contain text."
                )
            if x < 0 or y < 0:
                raise ConfigError("region origin cannot be negative.")


def comparison_form(text: str) -> str:
    """Normalise text for the "is this the same subtitle?" test.

    NFKC folds the full-width forms PP-OCR emits; punctuation and whitespace
    are dropped because they are exactly what OCR disagrees with itself
    about on identical pixels.
    """
    return _IGNORE_FOR_COMPARISON.sub("", unicodedata.normalize("NFKC", text))


def same_subtitle(a: str, b: str, threshold: float) -> bool:
    """Whether two readings are the same on-screen subtitle."""
    ca, cb = comparison_form(a), comparison_form(b)
    if not ca or not cb:
        return ca == cb
    if ca == cb:
        return True
    return SequenceMatcher(None, ca, cb).ratio() >= threshold


@dataclass
class _OpenCue:
    """The single subtitle currently on screen, accumulating readings.

    This is the whole state machine.  It is one variable rather than a stack
    because only one subtitle is ever open at a time: a new reading either
    continues this cue or ends it.
    """

    start: float
    last_seen: float
    readings: list[tuple[str, float | None]] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The most recent reading — what new frames are compared against."""
        return self.readings[-1][0] if self.readings else ""

    def vote(self) -> tuple[str, float | None]:
        """Pick the best text from every reading of this subtitle.

        Majority first: if four of six readings agree, that is the answer,
        and it beats any single confidence score.  Confidence only breaks
        ties, which is the right ordering — confidence tracks how *hard* a
        crop was to read at least as much as whether the reading is right.
        (Observed: a correct reading at 0.39 next to an incorrect one at
        0.65 on the same frame.)
        """
        if not self.readings:
            return "", None

        groups: dict[str, list[tuple[str, float | None]]] = {}
        for text, confidence in self.readings:
            groups.setdefault(comparison_form(text), []).append((text, confidence))

        def group_rank(item):
            _key, entries = item
            scored = [c for _t, c in entries if c is not None]
            return (len(entries), sum(scored) / len(scored) if scored else 0.0)

        _key, best_group = max(groups.items(), key=group_rank)

        # Within the winning group, prefer the highest-confidence rendering —
        # the variants differ only in punctuation and spacing by construction.
        def entry_rank(entry):
            _text, confidence = entry
            return confidence if confidence is not None else -1.0

        text, confidence = max(best_group, key=entry_rank)
        scored = [c for _t, c in best_group if c is not None]
        return text, (sum(scored) / len(scored) if scored else None)


def _ffmpeg_sample_command(
    video: Path, settings: ExtractSettings, width: int, height: int
) -> list[str]:
    """Build the ffmpeg call that streams cropped frames at the sample rate.

    ``-ss`` before ``-i`` seeks fast (keyframe-accurate, which is plenty:
    the sample grid is reconstructed from the frame index, not from ffmpeg's
    idea of presentation time).  Output is rawvideo rather than encoded
    images so frames arrive as fixed-size byte blocks — no container parsing,
    no framing ambiguity.
    """
    from gensrt.ffmpeg_util import get_ffmpeg_exe

    filters = []
    if settings.region:
        x, y, w, h = settings.region
        filters.append(f"crop={w}:{h}:{x}:{y}")
    filters.append(f"fps={settings.sample_fps}")

    cmd = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin"]
    if settings.start_time > 0:
        cmd += ["-ss", f"{settings.start_time:.3f}"]
    cmd += ["-i", str(video)]
    if settings.end_time is not None:
        cmd += ["-t", f"{settings.end_time - settings.start_time:.3f}"]
    cmd += [
        "-vf", ",".join(filters),
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    return cmd


def probe_video(video: Path) -> tuple[int, int, float]:
    """Return ``(width, height, duration_seconds)`` for *video*.

    Duration matters as much as the dimensions: seeking past the end of a
    file makes ffmpeg exit CLEANLY having written nothing, which is
    indistinguishable at the pipe from "the region contained no text".
    Knowing the duration up front turns that into a precise error.
    """
    from gensrt.ffmpeg_util import get_ffprobe_exe, get_subprocess_creationflags

    try:
        out = subprocess.run(
            [get_ffprobe_exe(), "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height:format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1",
             str(video)],
            capture_output=True, text=True, check=True,
            creationflags=get_subprocess_creationflags(),
        ).stdout.split()
    except Exception as exc:
        raise OCRError(f"ffprobe could not read {video.name}: {exc}") from exc

    try:
        width, height = int(out[0]), int(out[1])
        duration = float(out[2]) if len(out) > 2 else 0.0
    except (IndexError, ValueError) as exc:
        raise OCRError(
            f"Could not parse video properties from ffprobe: {out!r}"
        ) from exc
    return width, height, duration


def _validate_against_video(
    settings: ExtractSettings, width: int, height: int, duration: float
) -> None:
    """Check the request against the actual file, before spawning ffmpeg."""
    if duration > 0 and settings.start_time >= duration:
        raise OCRError(
            f"--extract-from {settings.start_time:.0f}s is past the end of "
            f"this video, which is {duration:.0f}s "
            f"({duration / 60:.1f} min) long. Nothing would be sampled."
        )
    if settings.end_time is not None and duration > 0 and settings.end_time > duration + 1:
        logger.warning(
            "Requested end %.0fs is beyond the video length %.0fs — "
            "extraction will stop at the end of the file.",
            settings.end_time, duration,
        )
    if settings.region:
        x, y, w, h = settings.region
        if x + w > width or y + h > height:
            raise OCRError(
                f"Region {w}×{h} at ({x}, {y}) does not fit inside this "
                f"video's {width}×{height} frame — it would need "
                f"{x + w}×{y + h}. Re-select the region, or check that the "
                f"numbers came from this video."
            )


def extract_subtitles(
    video_path: str | Path,
    settings: ExtractSettings,
    *,
    progress=None,
) -> list[SRTSegment]:
    """Read burned-in subtitles out of *video_path*.

    Args:
        video_path: The video to read.
        settings:   See :class:`ExtractSettings`.
        progress:   Optional ``(frames_done, frames_total, seconds_position,
                    cues_so_far) -> None`` callback.  ``frames_total`` is an
                    estimate when the video duration is unknown.

    Returns:
        Subtitle segments in timeline order, ready for ``build_srt``.

    Raises:
        ConfigError: Bad settings.
        OCRError:    ffmpeg failed, or the OCR stack is unavailable.
    """
    import numpy as np

    from gensrt.ffmpeg_util import get_subprocess_creationflags
    from gensrt.ocr.factory import get_detector, get_recognizer
    from gensrt.ocr.ppocr_onnx import crop_region

    settings.validate()
    video = Path(video_path)
    if not video.is_file():
        raise OCRError(f"Video not found: {video}")

    # Cheap local checks BEFORE anything expensive. ffprobe costs
    # milliseconds; get_recognizer may download a model. Being told the time
    # range is invalid after an 11 MB download is a bad trade, and this is
    # the order the first real failure argued for.
    video_w, video_h, duration = probe_video(video)
    _validate_against_video(settings, video_w, video_h, duration)

    # Then the model — still before ffmpeg spawns, so a missing model fails
    # fast rather than after the first frames arrive.
    recognizer = get_recognizer(settings.language)
    detector = get_detector()

    if settings.region:
        width, height = settings.region[2], settings.region[3]
    else:
        width, height = video_w, video_h
    frame_bytes = width * height * 3
    step = 1.0 / settings.sample_fps

    total_frames = 0
    if settings.end_time is not None:
        total_frames = int((settings.end_time - settings.start_time) * settings.sample_fps)

    logger.info(
        "Extracting subtitles: %s, region %s, %.1f fps, %s",
        video.name,
        f"{width}x{height}" + (f" at ({settings.region[0]}, {settings.region[1]})"
                               if settings.region else " (full frame)"),
        settings.sample_fps,
        f"{settings.start_time:.1f}s-{settings.end_time:.1f}s"
        if settings.end_time is not None else f"from {settings.start_time:.1f}s",
    )

    cmd = _ffmpeg_sample_command(video, settings, width, height)
    logger.debug("ffmpeg: %s", " ".join(cmd))

    cues: list[SRTSegment] = []
    open_cue: _OpenCue | None = None
    frame_index = 0

    def close(cue: _OpenCue, end_time: float) -> None:
        text, confidence = cue.vote()
        if not text:
            return
        duration = end_time - cue.start
        if duration < settings.min_duration_s:
            logger.debug("dropping %.2fs cue at %.2fs: %r",
                         duration, cue.start, text[:40])
            return
        cues.append(SRTSegment(
            index=len(cues) + 1,
            start=round(cue.start, 3),
            end=round(end_time, 3),
            text=text,
            avg_logprob=confidence,     # reused as the OCR confidence slot
        ))

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=get_subprocess_creationflags(),
    )
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break

            timestamp = settings.start_time + frame_index * step
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)

            # Detection returns one box per text line; a two-line subtitle is
            # two boxes on the same frame and must become ONE cue, so the
            # readings are joined in reading order.
            lines: list[str] = []
            scores: list[float] = []
            for region in detector.detect(frame):
                text, confidence = recognizer.recognize(
                    crop_region(frame, region.quad)
                )
                text = unicodedata.normalize("NFKC", text).strip()
                if text:
                    lines.append(text)
                    if confidence is not None:
                        scores.append(confidence)

            reading = "\n".join(lines)
            mean_score = sum(scores) / len(scores) if scores else None

            if not reading:
                # Blank frame ends whatever was on screen. The cue ends at the
                # last frame that still showed it, not at this empty one.
                if open_cue is not None:
                    close(open_cue, open_cue.last_seen + step)
                    open_cue = None
            elif open_cue is None:
                open_cue = _OpenCue(start=timestamp, last_seen=timestamp,
                                    readings=[(reading, mean_score)])
            elif same_subtitle(reading, open_cue.text, settings.similarity):
                open_cue.readings.append((reading, mean_score))
                open_cue.last_seen = timestamp
            else:
                # Different text: the previous subtitle ended somewhere in the
                # interval between the two samples. Splitting the difference is
                # the best available estimate and bounds the error at one
                # sample step.
                close(open_cue, timestamp)
                open_cue = _OpenCue(start=timestamp, last_seen=timestamp,
                                    readings=[(reading, mean_score)])

            frame_index += 1
            if progress and frame_index % 10 == 0:
                progress(frame_index, total_frames, timestamp, len(cues))

        if open_cue is not None:
            close(open_cue, open_cue.last_seen + step)

        stderr = proc.stderr.read().decode("utf-8", "replace").strip()
        exit_code = proc.wait()
        if exit_code != 0 and not cues:
            raise OCRError(f"ffmpeg failed while sampling frames: {stderr[:400]}")
        if stderr:
            logger.debug("ffmpeg stderr: %s", stderr[:400])

        if frame_index == 0:
            # ffmpeg produced nothing at all. This is NOT "the region had no
            # text" — no pixel was ever examined — and conflating the two
            # sends the user hunting for a region problem that is not there.
            raise OCRError(
                f"ffmpeg sampled no frames from {video.name} "
                f"({duration:.0f}s long) over "
                f"{settings.start_time:.0f}s-"
                f"{settings.end_time if settings.end_time is not None else duration:.0f}s. "
                f"The requested range produced no output; check that it falls "
                f"inside the video."
            )
    finally:
        if proc.poll() is None:
            proc.kill()
        for stream in (proc.stdout, proc.stderr):
            try:
                stream.close()
            except Exception:
                pass

    logger.info("Extracted %d cue(s) from %d sampled frame(s)",
                len(cues), frame_index)

    if settings.translate and cues:
        _translate_cues(cues, settings)

    return cues


def _translate_cues(cues: list[SRTSegment], settings: ExtractSettings) -> None:
    """Translate cue text in place, keeping the source on failure.

    A translation failure must not destroy an extraction that may have taken
    twenty minutes — the source-language cues are still a usable result.
    """
    from gensrt.models import TranscriptionConfig
    from gensrt.translation.factory import get_engine

    from dataclasses import replace

    try:
        engine = get_engine("nllb", TranscriptionConfig())
        texts = [c.text for c in cues]
        translated = engine.translate_batch(
            texts, settings.language, settings.target_language
        )
    except Exception as exc:
        logger.warning(
            "Translation failed (%s) — keeping %s text. The extraction itself "
            "is unaffected.", exc, settings.language,
        )
        return

    # SRTSegment is frozen, so each cue is REPLACED rather than mutated.
    # Assigning to cue.text raises, and because the whole block used to sit
    # inside one try/except that failure surfaced as a log line and a file
    # full of untranslated source text.
    done = 0
    for i, (cue, text) in enumerate(zip(cues, translated)):
        if text and text.strip():
            cues[i] = replace(cue, text=text)
            done += 1

    if done == len(cues):
        logger.info("Translated %d cue(s) to %s", done, settings.target_language)
    else:
        # A partial result is not a crash, but it is not a success either.
        logger.warning(
            "Translated %d of %d cue(s) to %s — the rest kept their %s text.",
            done, len(cues), settings.target_language, settings.language,
        )
