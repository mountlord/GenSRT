"""Audio loader — read a 16 kHz mono PCM WAV into a numpy array.

The chunking pipeline needs sample-level access to the audio waveform
(for VAD slicing and energy-min detection).  faster-whisper handles its
own loading internally, but the silent-boundary chunker we run in
:mod:`gensrt.asr.monolingual_whisper` needs to slice the audio before
each per-chunk inference call.

This loader assumes input was produced by
:func:`gensrt.audio.extractor.extract_audio` — 16 kHz, mono, 16-bit
PCM.  It validates these properties and fails loudly if the input
doesn't match (rather than silently mis-decoding).
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path
from typing import TYPE_CHECKING

from gensrt.constants import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE
from gensrt.exceptions import AudioExtractionError

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def load_wav_float32(wav_path: Path) -> tuple["np.ndarray", int]:
    """Load a 16 kHz mono 16-bit PCM WAV into a float32 numpy array.

    Args:
        wav_path: Path to the WAV file (typically the temp file produced
                  by :func:`gensrt.audio.extractor.extract_audio`).

    Returns:
        Tuple of:
            * 1-D float32 ndarray with samples in range ``[-1.0, 1.0]``.
            * Sample rate (always 16 000 — included for downstream
              convenience).

    Raises:
        AudioExtractionError: If the file is missing, has unexpected
                              sample width or rate, or cannot be opened.
                              We raise an extraction error rather than a
                              generic one because the WAV is part of the
                              audio-extraction stage's output contract.
    """
    import numpy as np

    wav_path = Path(wav_path)
    if not wav_path.is_file():
        raise AudioExtractionError(
            str(wav_path),
            "WAV file not found.  Audio extraction may have silently failed.",
        )

    try:
        with wave.open(str(wav_path), "rb") as wf:
            sr = wf.getframerate()
            sample_width = wf.getsampwidth()
            n_channels = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
    except wave.Error as exc:
        raise AudioExtractionError(str(wav_path), f"wave.open failed: {exc}") from exc

    if sample_width != 2:
        raise AudioExtractionError(
            str(wav_path),
            f"Expected 16-bit PCM, got {sample_width * 8}-bit.  "
            f"Audio extractor should always produce 16-bit.",
        )
    if sr != AUDIO_SAMPLE_RATE:
        raise AudioExtractionError(
            str(wav_path),
            f"Expected {AUDIO_SAMPLE_RATE} Hz, got {sr} Hz.  "
            f"Audio extractor should always produce {AUDIO_SAMPLE_RATE} Hz.",
        )

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > AUDIO_CHANNELS:
        # Down-mix to mono.  Extractor should already do this; defensive.
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    logger.debug(
        "Loaded WAV: %s  (%.1fs, %d samples)",
        wav_path.name,
        len(audio) / sr,
        len(audio),
    )
    return audio, sr
