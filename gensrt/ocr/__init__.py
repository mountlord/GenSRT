"""On-screen text recognition for paused video frames.

Detection and recognition are separate pluggable stages — see
:mod:`gensrt.ocr.base` for why. The entry point most callers want is
:func:`read_frame`, which runs the whole pipeline on one image.
"""

from __future__ import annotations

import logging

from gensrt.ocr.base import TextRegion, normalize_text
from gensrt.ocr.ppocr_onnx import OCRError

logger = logging.getLogger(__name__)

__all__ = ["TextRegion", "OCRError", "read_frame", "normalize_text"]


def read_frame(
    image_bgr,
    language: str,
    *,
    max_regions: int = 40,
    min_confidence: float = 0.0,
    include_crops: bool = True,
    status=None,
) -> list[TextRegion]:
    """Detect and read every text region in one frame.

    Args:
        image_bgr:      HxWx3 uint8 BGR image (OpenCV convention).
        language:       ISO 639-1 code; must be in the OCR registry.
        max_regions:    Cap on regions returned. A busy frame can detect
                        dozens of boxes — watermarks, UI chrome, timestamps —
                        and the UI has to present them all to a human.
        min_confidence: Drop readings below this score. Default 0 keeps
                        everything and lets the UI decide, which is the right
                        default while nobody knows what the scores look like
                        on real material.
        include_crops:  Attach PNG crops. The UI needs them (a user who
                        cannot read the script judges by the crop); other
                        callers do not.
        status:         Optional ``(str) -> None`` progress callback.

    Returns:
        Regions in reading order, ``text`` filled in.

    Raises:
        OCRError:    Missing dependency, missing model, or a failed download.
        ConfigError: Unknown language.
    """
    from gensrt.ocr.factory import get_detector, get_recognizer
    from gensrt.ocr.ppocr_onnx import crop_region, encode_png

    # Recognizer first: it is the stage that may download, and failing
    # before spending detection time gives a faster, clearer error.
    recognizer = get_recognizer(language, status=status)
    detector = get_detector()

    regions = detector.detect(image_bgr)
    if not regions:
        logger.info("OCR: no text detected")
        return []

    truncated = len(regions) > max_regions
    regions = regions[:max_regions]
    if truncated:
        logger.info("OCR: %d regions detected, showing first %d",
                    len(regions), max_regions)

    out: list[TextRegion] = []
    for region in regions:
        crop = crop_region(image_bgr, region.quad)
        text, confidence = recognizer.recognize(crop)
        region.text = normalize_text(text)
        region.confidence = confidence
        if include_crops:
            region.crop_png = encode_png(crop)
        # An empty reading means detection found something text-shaped that
        # the recognizer could not read — a logo, a texture, a face. Dropping
        # it is right: an empty row in the picker is pure noise.
        if not region.text:
            continue
        if confidence is not None and confidence < min_confidence:
            continue
        out.append(region)

    # Renumber so the UI's labels are contiguous after filtering.
    for i, region in enumerate(out, start=1):
        region.index = i

    logger.info("OCR: %d region(s) read (%s)", len(out), language)
    return out
