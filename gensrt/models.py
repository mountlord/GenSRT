"""Core data models for GenSRT.

All models are immutable (frozen) dataclasses with JSON round-trip support.
``TranscriptionConfig`` is the single source of truth for configuration
defaults — ``config.py`` derives ``BUILTIN_DEFAULTS`` from its field values
automatically so the two never drift apart.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any


# ── Enumerations ──────────────────────────────────────────────────────────

class GPUBackend(Enum):
    """Detected compute backend."""

    CUDA = auto()
    """NVIDIA CUDA via CTranslate2."""

    ROCM = auto()
    """AMD ROCm via CTranslate2 (Phase II — stub)."""

    XPU = auto()
    """Intel XPU / ARC via OpenVINO or IPEX (Phase III — stub)."""

    CPU = auto()
    """CPU with int8 quantization via CTranslate2."""


class TranslationEngineKey(str, Enum):
    """Translation engine selector."""

    GOOGLE = "google"
    NLLB = "nllb"
    MARIAN = "marian"
    NONE = "none"


class ComputeType(str, Enum):
    """CTranslate2 compute type."""

    FLOAT16 = "float16"
    INT8_FLOAT16 = "int8_float16"
    INT8 = "int8"


# ── SRT segment ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SRTSegment:
    """A single subtitle entry before SRT formatting.

    Attributes:
        index:      1-based subtitle index.
        start:      Start time in seconds.
        end:        End time in seconds.
        text:       Subtitle text (translated or original).
    """

    index: int
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SRTSegment:
        return cls(**data)


# ── Transcription configuration ───────────────────────────────────────────

@dataclass(frozen=True)
class TranscriptionConfig:
    """All knobs that control a transcription run.

    This dataclass is the *single source of truth* for built-in defaults.
    ``config.py`` derives ``BUILTIN_DEFAULTS`` from these field values
    automatically — no manual sync required.
    """

    # Model
    model: str = "large-v3-turbo"
    device: str = "cuda"
    compute_type: str = "float16"
    gpu_id: int = 0

    # Language
    source_language: str = "auto"

    # VAD
    vad_enabled: bool = True
    vad_threshold: float = 0.5          # speech probability threshold (0–1)
    vad_min_speech_ms: int = 250        # minimum speech segment duration
    vad_min_silence_ms: int = 2000      # minimum silence gap that splits segments

    # SRT output
    max_subtitle_duration_s: float = 10.0   # cap subtitle display time; 0 = no cap
    min_subtitle_duration_s: float = 1.0    # floor subtitle display time; 0 = no floor

    # Translation
    translation_engine: str = "google"
    translate: bool = True

    # Backend (set by gpu_probe, not directly by user)
    backend: str = "cuda"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptionConfig:
        import dataclasses
        valid = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ── Transcription result ──────────────────────────────────────────────────

@dataclass(frozen=True)
class TranscriptionResult:
    """Complete output for a single transcription job.

    Attributes:
        input_path:         Absolute path to the input media file.
        output_path:        Absolute path to the written .srt file.
        detected_language:  ISO 639-1 language code detected by Whisper.
        segments:           Ordered list of subtitle segments.
        config:             Configuration used for this run.
        elapsed_s:          Wall-clock seconds for the full pipeline.
    """

    input_path: Path
    output_path: Path
    detected_language: str
    segments: list[SRTSegment]
    config: TranscriptionConfig
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": str(self.input_path),
            "output_path": str(self.output_path),
            "detected_language": self.detected_language,
            "segments": [s.to_dict() for s in self.segments],
            "config": self.config.to_dict(),
            "elapsed_s": round(self.elapsed_s, 2),
        }

    def save(self, path: Path) -> None:
        """Serialise to a JSON sidecar file (optional, for debug)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TranscriptionResult:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            input_path=Path(data["input_path"]),
            output_path=Path(data["output_path"]),
            detected_language=data["detected_language"],
            segments=[SRTSegment.from_dict(s) for s in data["segments"]],
            config=TranscriptionConfig.from_dict(data["config"]),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
        )
