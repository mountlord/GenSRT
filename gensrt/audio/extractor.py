"""Audio extraction using ffmpeg-python.

Converts any FFmpeg-supported input to a 16 kHz mono WAV temporary file
suitable for faster-whisper / Silero VAD.

Usage::

    from gensrt.audio.extractor import extract_audio
    wav_path = extract_audio(Path("video.mp4"))
    # ... process wav_path ...
    wav_path.unlink()   # caller is responsible for cleanup
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from gensrt.constants import AUDIO_CHANNELS, AUDIO_FORMAT, AUDIO_SAMPLE_RATE
from gensrt.exceptions import AudioExtractionError

logger = logging.getLogger(__name__)


def extract_audio(
    input_path: Path,
    *,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = AUDIO_CHANNELS,
) -> Path:
    """Extract audio from *input_path* to a temporary WAV file.

    The caller is responsible for deleting the returned file when done.

    Args:
        input_path:   Path to the media file (any FFmpeg-supported format).
        sample_rate:  Target sample rate in Hz (default: 16 000).
        channels:     Number of audio channels (default: 1 = mono).

    Returns:
        :class:`Path` to the temporary WAV file.

    Raises:
        AudioExtractionError: If FFmpeg fails or the file has no audio stream.
    """
    try:
        import ffmpeg
    except ImportError as exc:
        raise AudioExtractionError(
            str(input_path),
            "ffmpeg-python is not installed. Run: pip install ffmpeg-python",
        ) from exc

    input_path = Path(input_path).resolve()

    # Create a named temp file — keep it around until the caller deletes it.
    tmp = tempfile.NamedTemporaryFile(
        suffix=f".{AUDIO_FORMAT}",
        delete=False,
        prefix="gensrt_audio_",
    )
    tmp.close()
    wav_path = Path(tmp.name)

    logger.debug(
        "Extracting audio: %s  →  %s  (%d Hz, %d ch)",
        input_path.name,
        wav_path.name,
        sample_rate,
        channels,
    )

    try:
        (
            ffmpeg
            .input(str(input_path))
            .audio
            .output(
                str(wav_path),
                ar=sample_rate,
                ac=channels,
                format=AUDIO_FORMAT,
                acodec="pcm_s16le",
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as exc:
        # Clean up temp file on failure
        wav_path.unlink(missing_ok=True)

        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else str(exc)
        logger.debug("FFmpeg stderr: %s", stderr)
        raise AudioExtractionError(
            str(input_path),
            f"FFmpeg exited with error: {stderr[:500]}",
        ) from exc
    except Exception as exc:
        wav_path.unlink(missing_ok=True)
        raise AudioExtractionError(str(input_path), str(exc)) from exc

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        wav_path.unlink(missing_ok=True)
        raise AudioExtractionError(
            str(input_path),
            "FFmpeg produced an empty or missing output file. "
            "The input may have no audio stream.",
        )

    logger.info(
        "Audio extracted: %s  (%.1f MB)",
        wav_path.name,
        wav_path.stat().st_size / 1_048_576,
    )
    return wav_path
