"""OCR language registry, model fetch, and engine construction.

One detector serves every language; one recognition model per language is
downloaded on demand.  The models are small enough (3-11 MB) that the fetch
is unremarkable — unlike NLLB's 650 MB, this needs no up-front ceremony and
happens lazily on first use of a given language.

All PP-OCR models here are Apache-2.0, so unlike the NLLB weights there is
no license rider to surface to the user.

Choosing the Japanese model
---------------------------
Two Japanese heads were compared on real frames (78 crops, 2026-09-12):

* ``japan_rec_crnn`` (v1-era, 3.6 MB) — read the same on-screen logo three
  different ways across three frames.
* ``japan_PP-OCRv3_rec`` (10 MB) — read it identically each time, and got
  the studio name right where v1 produced noise.

Self-consistency across repeated on-screen text is the useful metric when
no ground truth is available: a recognizer that reads one logo three ways
is wrong at least twice.  v3 won on that, and was faster (4 ms vs 9 ms per
crop).  Its one observed weakness — full-width digits — is handled by
``normalize_text``.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from gensrt.exceptions import ConfigError
from gensrt.ocr.base import TextDetector, TextRecognizer
from gensrt.ocr.ppocr_onnx import OCRError, PPOCRDetector, PPOCRRecognizer

logger = logging.getLogger(__name__)


class _Lang:
    """A language's recognition model: where to get it, how to run it."""

    __slots__ = ("code", "label", "repo", "filename", "rec_height", "size_mb", "note")

    def __init__(self, code, label, repo, filename, rec_height, size_mb, note=""):
        self.code = code
        self.label = label
        self.repo = repo
        self.filename = filename
        self.rec_height = rec_height
        self.size_mb = size_mb
        self.note = note


#: Supported OCR languages, keyed by the same ISO 639-1 codes used for
#: source/target language everywhere else in GenSRT.
#:
#: ``ch_PP-OCRv4_rec`` covers Chinese AND English in one model — that is how
#: PaddleOCR ships it, and it is also the strongest model in the family, so
#: "zh" and "en" deliberately point at the same file (downloaded once).
OCR_LANGUAGES: dict[str, _Lang] = {
    "ja": _Lang(
        "ja", "Japanese",
        "growdle/p2s-ocr", "japan_PP-OCRv3_rec_infer.onnx", 48, 10.1,
        "PP-OCRv3 Japanese. Chosen over the v1-era model on measured "
        "self-consistency; see module docstring.",
    ),
    "zh": _Lang(
        "zh", "Chinese (+ English)",
        "SWHL/RapidOCR", "PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx", 48, 10.9,
        "PP-OCRv4 Chinese/English — the strongest model in the family.",
    ),
    "en": _Lang(
        "en", "English",
        "SWHL/RapidOCR", "PP-OCRv4/ch_PP-OCRv4_rec_infer.onnx", 48, 10.9,
        "Shares the Chinese model, which handles Latin script.",
    ),
    "ko": _Lang(
        "ko", "Korean",
        "SWHL/RapidOCR", "PP-OCRv1/korean_mobile_v2.0_rec_infer.onnx", 32, 3.3,
        "v2-era Korean. Untested on real material — verify before relying on it.",
    ),
}

#: Fallback when the configured language has no OCR model.
DEFAULT_OCR_LANGUAGE = "ja"

_detector_lock = threading.Lock()
_detector: TextDetector | None = None
_recognizers: dict[str, TextRecognizer] = {}


def available_languages() -> list[dict]:
    """Registry contents, for the GUI's language picker and self-check."""
    return [
        {
            "code": lang.code,
            "label": lang.label,
            "size_mb": lang.size_mb,
            "present": is_model_present(lang.code),
            "note": lang.note,
        }
        for lang in OCR_LANGUAGES.values()
    ]


def resolve_language(code: str | None) -> _Lang:
    """Map a language code to its registry entry.

    Raises:
        ConfigError: If the language has no OCR model, listing the ones that
            do — the caller asked for something specific and silently reading
            the wrong script would be worse than stopping.
    """
    key = (code or "").strip().lower()
    if key in OCR_LANGUAGES:
        return OCR_LANGUAGES[key]
    raise ConfigError(
        f"No OCR model for language {code!r}. Available: "
        f"{', '.join(sorted(OCR_LANGUAGES))}. "
        f"Set \"ocr_language\" in gensrt-config.json."
    )


def model_path_for(code: str) -> Path:
    """Where this language's recognition model lives (or will live).

    Under ``models/ocr/`` — a subdirectory rather than the top level,
    because ``models/`` is browsed by users looking for Whisper models and
    a dozen small ONNX files scattered among them is noise.
    """
    from gensrt.model_paths import models_dir

    lang = resolve_language(code)
    return models_dir() / "ocr" / Path(lang.filename).name


def is_model_present(code: str) -> bool:
    try:
        return model_path_for(code).is_file()
    except ConfigError:
        return False


def ensure_model(code: str, *, status=None) -> Path:
    """Fetch this language's recognition model if it is not already on disk.

    Unlike the NLLB fetch, this is lazy: the models are 3-11 MB, so pulling
    one at the moment OCR is first used costs a second or two and needs no
    up-front warning.

    Raises:
        OCRError: If the model is absent and cannot be downloaded.
    """
    lang = resolve_language(code)
    target = model_path_for(code)
    if target.is_file():
        return target

    message = (
        f"Downloading OCR model for {lang.label} "
        f"(~{lang.size_mb:.0f} MB, one-time)"
    )
    logger.info("%s: %s", message, lang.repo)
    if callable(status):
        status(f"{message}…")

    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id=lang.repo, filename=lang.filename)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Copy out of the HF cache into models/ocr/ so the file survives a
        # cache purge and is visible where users expect models to live.
        target.write_bytes(Path(cached).read_bytes())
    except Exception as exc:
        raise OCRError(
            f"Could not download the {lang.label} OCR model "
            f"({lang.repo} :: {lang.filename}): {exc}\n\n"
            f"If this machine is offline, fetch the file on another machine "
            f"and place it at:\n  {target}"
        ) from exc

    logger.info("OCR model ready: %s", target)
    return target


def get_detector() -> TextDetector:
    """The shared, language-agnostic text detector (constructed once)."""
    global _detector
    if _detector is None:
        with _detector_lock:
            if _detector is None:
                _detector = PPOCRDetector()
    return _detector


def get_recognizer(code: str, *, status=None) -> TextRecognizer:
    """The recognizer for *code*, downloading its model if needed.

    Instances are cached per language: loading an ONNX session costs more
    than running it, and the OCR endpoint is called repeatedly as the user
    pauses on different frames.
    """
    lang = resolve_language(code)
    with _detector_lock:
        cached = _recognizers.get(lang.code)
    if cached is not None:
        return cached

    path = ensure_model(lang.code, status=status)
    recognizer = PPOCRRecognizer(
        path, rec_height=lang.rec_height, label=f"ppocr-{lang.code}"
    )
    with _detector_lock:
        _recognizers[lang.code] = recognizer
    return recognizer
