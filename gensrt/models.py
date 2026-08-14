"""Core data models for GenSRT.

All models are immutable (frozen) dataclasses with JSON round-trip support.
``TranscriptionConfig`` is the single source of truth for configuration
defaults — ``config.py`` derives ``BUILTIN_DEFAULTS`` from its field values
automatically so the two never drift apart.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
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

    The first four fields are the subtitle itself.  Everything after them is
    *diagnostics* — what the model reported about its own output, and where
    the segment sat in the chunk plan.  All of it is optional and defaults to
    ``None``, so a segment built by hand (a test, a loaded SRT file) is still
    valid.

    Decoder metrics
    ---------------
    These come straight from faster-whisper's ``Segment`` and describe how the
    decode went.  Whisper generates text one token at a time, assigning each a
    probability; these are summaries of that process.

        avg_logprob
            Mean log-probability per token.  Always negative; closer to zero
            means the model was more confident.  Roughly: -0.2 is a confident
            decode, -1.0 is shaky, below -1.5 is usually noise.  This is the
            single most useful signal for separating real speech from
            hallucinated filler, because a model that is inventing text is
            typically unsure while doing it.

        compression_ratio
            Length of the text divided by its gzip-compressed length.  Normal
            prose sits near 1.2-2.0.  Repetitive text compresses extremely
            well, so a runaway loop ("ആരോപിച്ചു ആരോപിച്ചു ആരോപിച്ചു…") pushes
            this up sharply.  faster-whisper treats anything above 2.4 as a
            failed decode by default.

        no_speech_prob
            The model's own estimate that this audio contains no speech at
            all.  High values on a segment that nonetheless produced text
            indicate the model wrote words over silence or noise.

        temperature
            The sampling temperature the *successful* decode ran at.  0.0
            means the first attempt passed the quality gates.  Anything higher
            means earlier attempts failed and were retried — so a non-zero
            value is itself evidence the decoder struggled here.

    Chunk position
    --------------
    Only populated by the chunked engine.  This is information GenSRT has that
    a general-purpose cleaner does not: "chunk-tail fragment" is a structural
    claim, and these fields let it be tested directly rather than inferred
    from duration.

        chunk_index       1-based chunk this segment came from
        chunk_position    1-based position within that chunk
        chunk_n_segments  how many segments that chunk emitted in total
    """

    index: int
    start: float
    end: float
    text: str

    # Decoder metrics (see class docstring).
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None
    temperature: float | None = None

    # Chunk provenance (chunked engine only).
    chunk_index: int | None = None
    chunk_position: int | None = None
    chunk_n_segments: int | None = None

    @property
    def duration(self) -> float:
        """Display duration in seconds, as the model timed it."""
        return max(0.0, self.end - self.start)

    @property
    def is_chunk_tail(self) -> bool:
        """True when this was the last segment its chunk emitted.

        The chunk-tail hypothesis is that spurious short fragments cluster
        here — the decoder, having reached the end of the audio it was given,
        emits one more low-confidence token group before stopping.  Unknown
        provenance returns False rather than guessing.
        """
        if self.chunk_position is None or self.chunk_n_segments is None:
            return False
        return self.chunk_position == self.chunk_n_segments

    @property
    def is_chunk_sole(self) -> bool:
        """True when this segment was the *only* output of its chunk.

        Distinguished from :attr:`is_chunk_tail` because a sole segment is
        both head and tail, and lumping the two together would inflate any
        measurement of tail behaviour.
        """
        return self.chunk_n_segments == 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SRTSegment:
        """Build from a dict, ignoring keys this version does not know.

        Tolerant on purpose: segment dicts are written to disk by the
        diagnostics dump and read back by analysis code, and a field added or
        removed between versions should not break that round trip.
        """
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ── Transcription configuration ───────────────────────────────────────────

@dataclass(frozen=True)
class TranscriptionConfig:
    """All knobs that control a transcription run.

    This dataclass is the *single source of truth* for built-in defaults.
    ``config.py`` derives ``BUILTIN_DEFAULTS`` from these field values
    automatically — no manual sync required.
    """

    # ASR engine selection.
    #
    #   "auto"      route by model name (see gensrt.asr.factory)
    #   "chunked"   split at silence and decode each chunk separately
    #   "longform"  hand the whole file to Whisper in one call
    #
    # "auto" is right almost always. The override exists because the routing
    # is a heuristic about how a model was trained, and a heuristic that
    # cannot be overridden is a guess the user has to live with. Forcing
    # "chunked" on a multilingual model is a legitimate experiment: it trades
    # Whisper's cross-window context for tighter timestamps and bounded
    # hallucination, which may be the better trade on some material.
    asr_engine: str = "auto"

    # Model
    model: str = "large-v3-turbo"
    # "auto" probes the hardware; "cuda" / "cpu" are honoured as explicit
    # requests (see operations.build_transcription_config).  Through v1.2.1
    # this defaulted to "cuda" and was overwritten by the probe regardless,
    # which made the field unusable as an override.
    device: str = "auto"
    # "auto" means "derive from the resolved device" — CTranslate2 is asked
    # what the device actually supports rather than assuming float16.
    compute_type: str = "auto"
    gpu_id: int = 0

    # Language
    source_language: str = "auto"

    # VAD
    vad_enabled: bool = True
    vad_threshold: float = 0.5          # speech probability threshold (0–1)
    vad_min_speech_ms: int = 250        # minimum speech segment duration
    vad_min_silence_ms: int = 2000      # minimum silence gap that splits segments
    vad_speech_pad_ms: int = 200        # padding before/after detected speech (faster-whisper default: 400)

    # SRT output
    max_subtitle_duration_s: float = 3.0    # cap subtitle display time; 0 = no cap
    # Floor subtitle display time; 0 = no floor.
    #
    # NOTE FOR MEASUREMENT WORK: a non-zero floor rewrites the end timestamp
    # of every shorter cue, collapsing their true durations to this exact
    # value in the written SRT.  Set to 0 before characterising a model's
    # timestamp behaviour, or measure on the pre-build_srt segments.
    min_subtitle_duration_s: float = 1.0
    max_line_chars: int = 42                # soft per-line wrap budget
    max_lines: int = 2                      # preferred lines per cue (never truncates)

    # Diagnostics.  When set, the chunked ASR engine exports every chunk's
    # audio plus a decode-telemetry manifest under
    # <debug_chunk_dir>/<audio stem>/.  Off by default; costs nothing unset.
    debug_chunk_dir: str = ""
    # When set, the raw ASR segment table is written to this directory as
    # <audio stem>.segments.csv — before post-processing and before
    # translation.  See gensrt/segment_dump.py.
    dump_segments_dir: str = ""

    # Translation
    translation_engine: str = "google"
    translate: bool = True
    target_language: str = "en"     # ISO 639-1; non-en only supported by 'google'

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
