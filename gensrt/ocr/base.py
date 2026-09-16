"""OCR interfaces: detection and recognition are separate, pluggable stages.

Why two stages rather than one "OCR engine"
-------------------------------------------
Text recognition splits cleanly into *finding* text and *reading* it, and the
two halves have opposite properties:

* **Detection** is language-agnostic.  The same DB network finds Japanese,
  Korean and English boxes equally well, because it is looking for
  text-shaped pixels, not for characters.  One model serves every language,
  forever.

* **Recognition** is language-specific and wildly variable.  PP-OCR's
  Japanese head is a 10 MB CTC model; kha-white's manga-ocr is a 400 MB
  autoregressive sequence model.  Which one is *right* depends on the
  language and on how much download the user will tolerate.

Fusing them into one engine class would mean re-implementing detection for
every recognition model added, and would make "use the good Japanese reader
but keep the cheap detector" unexpressible.  So the pipeline is:

    frame ──▶ TextDetector ──▶ quads ──▶ crops ──▶ TextRecognizer ──▶ text

and the factory picks each half independently.

Measured on real material (2026-09-12, 78 crops from a Japanese title):
PP-OCR detection found every visible text region with no errors, and
PP-OCRv3 Japanese recognition ran at 4 ms per crop on CPU.  That speed is
why recognition is worth keeping cheap: it leaves whole-video OCR (burned-in
subtitle extraction) open as a future feature, which a 400 MB
seconds-per-crop model would foreclose.
"""

from __future__ import annotations

import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TextRegion:
    """One detected piece of text on one frame.

    Attributes:
        index:      1-based position in reading order (top-to-bottom,
                    then left-to-right).  Stable within a single detection
                    pass, and what the UI labels boxes with.
        quad:       Four ``(x, y)`` corner points in source-image pixels.
                    A quadrilateral rather than a rectangle because text is
                    rarely perfectly level; the crop is perspective-corrected
                    from these points.
        text:       Recognised text; empty until a recognizer fills it in.
        confidence: Recognizer's own score in ``[0, 1]``, or ``None`` when
                    the model does not produce one.
        crop_png:   PNG bytes of the perspective-corrected crop.  Carried
                    because the UI shows the crop beside the text — for a
                    user who cannot read the script, the crop is the only
                    way to judge whether the reading is right.
    """

    index: int
    quad: list[tuple[float, float]]
    text: str = ""
    confidence: float | None = None
    crop_png: bytes = b""

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Axis-aligned ``(x, y, w, h)`` around the quad, for UI overlays."""
        xs = [p[0] for p in self.quad]
        ys = [p[1] for p in self.quad]
        return (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    def to_dict(self, *, include_crop: bool = True) -> dict:
        """JSON-safe form for the ``/api/ocr`` response."""
        import base64

        out = {
            "index": self.index,
            "quad": [[float(x), float(y)] for x, y in self.quad],
            "bbox": [float(v) for v in self.bbox],
            "text": self.text,
            "confidence": self.confidence,
        }
        if include_crop and self.crop_png:
            out["crop"] = (
                "data:image/png;base64,"
                + base64.b64encode(self.crop_png).decode("ascii")
            )
        return out


class TextDetector(ABC):
    """Finds text regions in an image. Language-agnostic by design."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def detect(self, image_bgr) -> list[TextRegion]:
        """Return regions in reading order, with ``text`` still empty.

        Args:
            image_bgr: HxWx3 uint8 numpy array, BGR channel order (OpenCV's
                convention, which is what every stage here speaks).
        """


class TextRecognizer(ABC):
    """Reads text out of a cropped, upright text line."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def recognize(self, crop_bgr) -> tuple[str, float | None]:
        """Return ``(text, confidence)`` for one crop."""

    def recognize_batch(self, crops: list) -> list[tuple[str, float | None]]:
        """Read several crops.

        The default loops.  Recognizers with a real batched path (CTC models
        can pad to a common width and run one session call) should override
        it; at 4 ms per crop the loop is not currently the bottleneck.
        """
        return [self.recognize(c) for c in crops]


def normalize_text(text: str) -> str:
    """Tidy raw recognizer output for use as subtitle text.

    NFKC folds the full-width forms PP-OCR's Japanese head emits for digits
    and Latin letters (``０ｉ０ｏ５４`` → ``0i0o54``) into their ASCII
    equivalents.  Observed on real frames: the v3 Japanese model reads
    product codes as full-width even when the source is plainly half-width,
    and full-width digits would flow through translation and into the SRT.

    NFKC deliberately does NOT touch kana or kanji, so Japanese text itself
    is unchanged — the half-width katakana it does normalise (``ｱ`` → ``ア``)
    is a correction too.
    """
    return unicodedata.normalize("NFKC", text).strip()
