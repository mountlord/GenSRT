"""Project-wide constants and default values.

All magic numbers and default configuration values live here.
Import from this module rather than hardcoding values elsewhere.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Media file extensions used for directory scanning heuristics.
# Actual processing capability is governed by FFmpeg — this list is only
# used to filter files when walking directories.
# ---------------------------------------------------------------------------
MEDIA_EXTENSIONS: frozenset[str] = frozenset({
    # Video containers
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".vob",
    ".ogv", ".3gp", ".3g2", ".divx", ".f4v", ".rm", ".rmvb",
    # Audio containers
    ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".m4a",
    ".wma", ".aiff", ".aif", ".ape", ".mka",
})

# ---------------------------------------------------------------------------
# Whisper model
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = "large-v3-turbo"
DEFAULT_DEVICE: str = "cuda"
DEFAULT_COMPUTE_TYPE: str = "float16"
DEFAULT_SOURCE_LANGUAGE: str = "auto"

# ---------------------------------------------------------------------------
# VAD
# ---------------------------------------------------------------------------
VAD_ENABLED_DEFAULT: bool = True

# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------
DEFAULT_TRANSLATION_ENGINE: str = "google"

# NLLB model ID on HuggingFace
NLLB_MODEL_ID: str = "facebook/nllb-200-distilled-600M"

# MarianMT model IDs per language
MARIAN_MODELS: dict[str, str] = {
    "ja": "Helsinki-NLP/opus-mt-ja-en",
    "ko": "Helsinki-NLP/opus-mt-ko-en",
    "ml": "Helsinki-NLP/opus-mt-ml-en",
}

# NLLB language codes (flores-200 format)
NLLB_LANGUAGE_CODES: dict[str, str] = {
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ml": "mal_Mlym",
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "zh": "zho_Hans",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "it": "ita_Latn",
}

# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------
AUDIO_SAMPLE_RATE: int = 16_000
AUDIO_CHANNELS: int = 1
AUDIO_FORMAT: str = "wav"

# ---------------------------------------------------------------------------
# GPU probe
# ---------------------------------------------------------------------------
DEFAULT_GPU_ID: int = 0

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
# Number of real phases reported by run_pipeline(): extract → transcribe →
# translate → write.  Used as the denominator for progress(current, total)
# callbacks.  Both pipeline.py and server.py import this so the value cannot
# drift between them.
PIPELINE_PHASES: int = 4

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
SERVER_HOST: str = "127.0.0.1"
SERVER_PORT_RANGE: tuple[int, int] = (5100, 5200)   # scan for a free port
