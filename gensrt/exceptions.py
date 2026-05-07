"""Custom exception hierarchy for GenSRT.

All domain-specific exceptions inherit from :class:`GenSRTError` so callers
can catch a single base class when they need broad error handling.
"""

from __future__ import annotations


class GenSRTError(Exception):
    """Base exception for all GenSRT errors."""


# ── Audio ─────────────────────────────────────────────────────────────────

class AudioExtractionError(GenSRTError):
    """FFmpeg failed to extract or convert audio."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Audio extraction failed for {path!r}: {reason}")


# ── Transcription ─────────────────────────────────────────────────────────

class TranscriptionError(GenSRTError):
    """faster-whisper failed to transcribe audio."""

    def __init__(self, path: str, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"Transcription failed for {path!r}: {reason}")


# ── Translation ───────────────────────────────────────────────────────────

class TranslationError(GenSRTError):
    """Translation engine failed."""

    def __init__(self, engine: str, reason: str) -> None:
        self.engine = engine
        self.reason = reason
        super().__init__(f"Translation engine {engine!r} failed: {reason}")


class TranslationEngineUnavailableError(TranslationError):
    """Translation engine is not installed or not reachable."""


# ── GPU / Backend ─────────────────────────────────────────────────────────

class NoGPUBackendError(GenSRTError):
    """No usable compute backend could be initialised."""


# ── Input / Output ────────────────────────────────────────────────────────

class InputError(GenSRTError):
    """Invalid or missing input file/directory."""


class OutputError(GenSRTError):
    """Cannot write output .srt file."""


# ── Config ────────────────────────────────────────────────────────────────

class ConfigError(GenSRTError):
    """An error occurred while reading or writing GenSRT configuration."""


class ConfigParseError(ConfigError):
    """A config file contained invalid JSON."""

    def __init__(
        self,
        path: str,
        message: str,
        line: int | None = None,
        column: int | None = None,
    ) -> None:
        self.path = path
        self.message = message
        self.line = line
        self.column = column

        loc = ""
        if line is not None and column is not None:
            loc = f" (line {line}, column {column})"

        super().__init__(f"Invalid JSON in config file: {path}{loc}: {message}")
