"""PP-OCR detection and recognition on ONNX Runtime.

Detection goes through RapidOCR, which packages PaddleOCR's DB detector
along with the contour/unclip post-processing that turns a probability map
into quadrilaterals.  Recognition is run directly on ``onnxruntime`` here
instead: the CTC decode is twenty lines, it removes any ambiguity about
which model RapidOCR actually loaded, and it lets a per-language model be
swapped in by path without going through RapidOCR's own config plumbing.

Both halves of this arrangement were validated on real material before any
of it was written into GenSRT — 78 crops from a Japanese title, detection
finding every visible region with no failures, recognition at 4 ms a crop.

Character dictionaries
----------------------
RapidOCR-converted recognition models carry their character list *inside*
the ONNX file, under the ``character`` metadata key, newline-separated.
That is a genuine convenience: no dictionary file to locate, download or
keep in sync with the model.  Third-party conversions may lack it, so a
sidecar ``<model>.dict.txt`` is honoured as a fallback.

PaddleOCR's CTC convention: index 0 is the blank symbol, the characters
follow, and a space is appended at the end.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from gensrt.exceptions import GenSRTError
from gensrt.ocr.base import TextDetector, TextRecognizer, TextRegion

logger = logging.getLogger(__name__)


class OCRError(GenSRTError):
    """OCR could not be performed."""


# ── Detection ─────────────────────────────────────────────────────────────

#: How detection resizes its input. RapidOCR's own default is
#: ``limit_type="min", limit_side_len=736``: scale so the SHORT side is at
#: least 736. That is sensible for pages and photographs and pathological
#: for subtitle bands, which are short and very wide — it UPSCALES them.
#:
#: Measured, on a real 1920x178 subtitle band:
#:     min / 736   -> detection input 7932x736 (5.8 MPix), 0.94 s/frame
#:     max / 1280  -> detection input 1280x119 (0.15 MPix), 0.34 s/frame
#: identical boxes found on every frame tested.
#:
#: It gets worse as the band gets thinner: a 3840x220 region becomes
#: 12832x736 (9.4 MPix) under the default, so tightening the region — the
#: obvious optimisation — makes detection SLOWER, not faster.
#:
#: Capping the LONG side instead bounds the work regardless of shape. This
#: costs nothing in recognition quality: detection only locates boxes, and
#: the crops fed to the recogniser are taken from the ORIGINAL full-
#: resolution frame.
DEFAULT_DET_LIMIT_TYPE = "max"
DEFAULT_DET_LIMIT_SIDE_LEN = 1280


class PPOCRDetector(TextDetector):
    """PaddleOCR DB text detection via RapidOCR, recognition disabled.

    RapidOCR ships its detection model inside the wheel (~4.5 MB), so this
    stage needs no download at all — unlike the language-specific
    recognition models.

    Args:
        limit_side_len: Longest side the detector's input is scaled to.
            Lower is faster and coarser; 1280 kept detection identical to
            the default on real subtitle frames.
        limit_type: ``"max"`` caps the long side (see the module constants
            for why that matters here); ``"min"`` restores RapidOCR's own
            behaviour if a region ever needs it.
    """

    def __init__(self, limit_side_len: int | None = None,
                 limit_type: str | None = None) -> None:
        self._limit_side_len = int(limit_side_len or DEFAULT_DET_LIMIT_SIDE_LEN)
        self._limit_type = (limit_type or DEFAULT_DET_LIMIT_TYPE).lower()
        self._engine = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return "ppocr-det"

    def _load(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as exc:
                raise OCRError(
                    "OCR needs the 'rapidocr-onnxruntime' package, which is "
                    "not installed. Reinstall GenSRT, or "
                    "`pip install rapidocr-onnxruntime==1.4.4`."
                ) from exc
            self._engine = RapidOCR(
                det_limit_type=self._limit_type,
                det_limit_side_len=self._limit_side_len,
            )
            logger.info(
                "PP-OCR detector loaded (det input capped at %d px, %s side)",
                self._limit_side_len, self._limit_type,
            )
            return self._engine

    def detect(self, image_bgr) -> list[TextRegion]:
        import numpy as np

        engine = self._load()
        # use_rec=False returns boxes only. RapidOCR's return shape has
        # differed across its 1.x line, so unwrap defensively rather than
        # trusting one version's contract.
        result = engine(image_bgr, use_det=True, use_cls=False, use_rec=False)
        if isinstance(result, tuple):
            result = result[0]
        if not result:
            return []

        quads = []
        for item in result:
            box = item
            # With recognition on, an item is [box, text, score]; guard for it
            # so a future default change cannot silently produce nonsense.
            if (isinstance(item, (list, tuple)) and len(item) == 3
                    and isinstance(item[1], str)):
                box = item[0]
            arr = np.asarray(box, dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] == 4:
                quads.append(arr)

        # Reading order: top to bottom, then left to right. Users refer to
        # regions by number, so the numbering must match how a person scans
        # the frame.
        quads.sort(key=lambda q: (float(q[:, 1].min()), float(q[:, 0].min())))
        return [
            TextRegion(index=i, quad=[(float(x), float(y)) for x, y in q])
            for i, q in enumerate(quads, start=1)
        ]


def crop_region(image_bgr, quad) -> "object":
    """Perspective-correct a detected quad into an upright crop.

    PaddleOCR's ``get_rotate_crop_image``.  A plain bounding-box crop of a
    tilted line includes wedges of background and feeds skewed glyphs to the
    recognizer; warping the quad to a rectangle does not.

    Tall-and-narrow crops are rotated upright: that shape means either
    vertical text or a rotated line, and PP-OCR recognition expects a
    horizontal line either way.
    """
    import cv2
    import numpy as np

    pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    width = int(max(
        np.linalg.norm(pts[0] - pts[1]), np.linalg.norm(pts[2] - pts[3])
    ))
    height = int(max(
        np.linalg.norm(pts[0] - pts[3]), np.linalg.norm(pts[1] - pts[2])
    ))
    width, height = max(width, 1), max(height, 1)

    dst = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    crop = cv2.warpPerspective(
        image_bgr, matrix, (width, height),
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC,
    )
    if height / width >= 1.5:
        crop = np.rot90(crop)
    return np.ascontiguousarray(crop)


def encode_png(image_bgr) -> bytes:
    """PNG-encode an image for transport to the UI."""
    import cv2

    ok, buf = cv2.imencode(".png", image_bgr)
    return buf.tobytes() if ok else b""


# ── Recognition ───────────────────────────────────────────────────────────

class PPOCRRecognizer(TextRecognizer):
    """A PP-OCR CTC recognition head run directly on onnxruntime.

    Args:
        model_path: ONNX recognition model.
        rec_height: The model's fixed input height — 32 for v1-era CRNN
            models, 48 for v3/v4.  This is not discoverable from the graph
            (the dimension is dynamic) and getting it wrong yields garbage
            text rather than an error, so it is carried in the language
            registry alongside the model reference.
        label: Name reported to the UI and logs.
    """

    #: Recognition is CPU-only on purpose.  A single frame's worth of crops
    #: costs a few hundred milliseconds; onnxruntime-gpu would add ~200 MB
    #: to the installer and a second consumer of cuDNN, whose version is
    #: already pinned differently between the default and Pascal builds.
    _PROVIDERS = ("CPUExecutionProvider",)

    def __init__(self, model_path: Path, rec_height: int = 48,
                 label: str = "ppocr-rec") -> None:
        self._model_path = Path(model_path)
        self._rec_height = int(rec_height)
        self._label = label
        self._session = None
        self._input_name = ""
        self._charset: list[str] = []
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._label

    def _load(self):
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:
                return
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise OCRError(
                    "OCR needs the 'onnxruntime' package, which is not "
                    "installed. Reinstall GenSRT, or `pip install onnxruntime`."
                ) from exc

            if not self._model_path.is_file():
                raise OCRError(f"OCR model not found: {self._model_path}")

            session = ort.InferenceSession(
                str(self._model_path), providers=list(self._PROVIDERS)
            )
            self._charset = self._load_charset(session)
            self._input_name = session.get_inputs()[0].name
            self._session = session
            logger.info(
                "OCR recognizer loaded: %s (%d chars, input height %d)",
                self._model_path.name, len(self._charset), self._rec_height,
            )

    def _load_charset(self, session) -> list[str]:
        meta = session.get_modelmeta().custom_metadata_map
        chars: list[str] | None = None
        if "character" in meta:
            chars = meta["character"].split("\n")
        else:
            sidecar = self._model_path.with_suffix(".dict.txt")
            if sidecar.is_file():
                chars = sidecar.read_text(encoding="utf-8").splitlines()
        if not chars:
            raise OCRError(
                f"{self._model_path.name} carries no embedded character "
                f"dictionary and no sidecar was found. Place the model's "
                f"character list at {self._model_path.with_suffix('.dict.txt')} "
                f"(one character per line)."
            )
        # index 0 is CTC blank; trailing space is PaddleOCR convention.
        return ["<blank>", *chars, " "]

    def recognize(self, crop_bgr) -> tuple[str, float | None]:
        import cv2
        import numpy as np

        self._load()

        height, width = crop_bgr.shape[:2]
        if height < 2 or width < 2:
            return "", None

        target_w = max(16, int(round(width * self._rec_height / max(height, 1))))
        img = cv2.resize(
            crop_bgr, (target_w, self._rec_height), interpolation=cv2.INTER_LINEAR
        )
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5                     # PP-OCR normalisation
        img = img.transpose(2, 0, 1)[None, ...]     # NHWC -> NCHW

        probs = self._session.run(None, {self._input_name: img})[0][0]

        # Greedy CTC: argmax per timestep, collapse runs, drop blanks.
        ids = probs.argmax(axis=-1)
        scores = probs.max(axis=-1)
        chars: list[str] = []
        kept: list[float] = []
        previous = -1
        for step, idx in enumerate(ids):
            idx = int(idx)
            if idx != previous and idx != 0 and idx < len(self._charset):
                chars.append(self._charset[idx])
                kept.append(float(scores[step]))
            previous = idx

        text = "".join(chars).strip()
        confidence = float(sum(kept) / len(kept)) if kept else None
        return text, confidence
