"""OCR backend: registry, decoding, pipeline wiring, and the HTTP surface.

What is worth pinning mechanically, and what is not
---------------------------------------------------
Recognition *quality* is not testable here — it needs real frames, real
models and a human judgement, and that comparison was made before this code
existed (78 crops from a Japanese title; PP-OCRv3 beat the v1-era model on
self-consistency and speed).  What CAN regress silently, and is therefore
pinned below:

* the language registry — a code the GUI offers but the registry lacks is a
  runtime failure at the worst moment;
* CTC decoding — blank handling and repeat collapsing produce *wrong text*
  rather than errors when they drift;
* NFKC normalisation — the observed full-width-digit quirk of the Japanese
  model;
* the "detected but unreadable" filter, which keeps empty rows out of the
  picker;
* the API contract the frontend will code against.

Every test here runs without downloading a model or importing onnxruntime.
"""

from __future__ import annotations

import base64
import json

import pytest

from gensrt.exceptions import ConfigError
from gensrt.models import TranscriptionConfig
from gensrt.ocr.base import TextRegion, normalize_text
from gensrt.ocr.factory import (
    DEFAULT_OCR_LANGUAGE,
    OCR_LANGUAGES,
    available_languages,
    model_path_for,
    resolve_language,
)

np = pytest.importorskip("numpy")


# ── Language registry ─────────────────────────────────────────────────────

def test_japanese_is_the_default_and_is_registered():
    assert DEFAULT_OCR_LANGUAGE == "ja"
    assert resolve_language("ja").label == "Japanese"


def test_japanese_uses_the_v3_model_at_height_48():
    """The v1-era model read one logo three different ways across three
    frames; v3 read it identically each time. rec_height is not discoverable
    from the ONNX graph and a wrong value yields garbage, not an error."""
    ja = resolve_language("ja")
    assert "PP-OCRv3" in ja.filename
    assert ja.rec_height == 48


def test_chinese_and_english_share_one_model():
    """PaddleOCR ships Chinese and Latin in a single model; pointing both
    codes at it means one download serves both."""
    assert resolve_language("zh").filename == resolve_language("en").filename


def test_config_default_language_is_in_the_registry():
    assert TranscriptionConfig().ocr_language in OCR_LANGUAGES


def test_unknown_language_lists_the_alternatives():
    with pytest.raises(ConfigError) as exc:
        resolve_language("xx")
    message = str(exc.value)
    assert "ocr_language" in message
    for code in OCR_LANGUAGES:
        assert code in message


@pytest.mark.parametrize("code", ["JA", " ja ", "Ja"])
def test_language_lookup_is_forgiving_about_case_and_space(code):
    assert resolve_language(code).code == "ja"


def test_available_languages_shape_matches_what_the_gui_needs():
    langs = available_languages()
    assert langs
    for entry in langs:
        assert set(entry) == {"code", "label", "size_mb", "present", "note"}
        assert isinstance(entry["present"], bool)


def test_models_live_in_an_ocr_subdirectory():
    """models/ is browsed by users looking for Whisper models; a dozen small
    ONNX files loose among them is noise."""
    assert model_path_for("ja").parent.name == "ocr"


# ── Text normalisation ────────────────────────────────────────────────────

def test_nfkc_folds_the_fullwidth_digits_the_japanese_model_emits():
    """Observed on real frames: the v3 head reads a plainly half-width
    product code as full-width. Left alone it flows into the SRT."""
    assert normalize_text("０ｉ０ｏ５４") == "0i0o54"


def test_normalisation_leaves_japanese_alone():
    for text in ("快感潮ダ漏れ", "アイポケ", "日本語のテキスト"):
        assert normalize_text(text) == text


def test_normalisation_repairs_halfwidth_katakana():
    assert normalize_text("ｱｲﾎﾟｹ") == "アイポケ"


def test_normalisation_trims():
    assert normalize_text("  text  ") == "text"


# ── TextRegion ────────────────────────────────────────────────────────────

def test_bbox_is_the_axis_aligned_hull_of_a_tilted_quad():
    region = TextRegion(1, [(10, 20), (110, 25), (108, 55), (8, 50)])
    x, y, w, h = region.bbox
    assert (x, y) == (8, 20)
    assert (w, h) == (102, 35)


def test_to_dict_is_json_serialisable_and_embeds_the_crop():
    region = TextRegion(1, [(0, 0), (10, 0), (10, 5), (0, 5)],
                        text="テスト", confidence=0.91, crop_png=b"\x89PNG-fake")
    payload = region.to_dict()
    json.dumps(payload)                     # must not raise
    assert payload["text"] == "テスト"
    assert payload["crop"].startswith("data:image/png;base64,")
    assert base64.b64decode(payload["crop"].split(",", 1)[1]) == b"\x89PNG-fake"


def test_to_dict_can_omit_the_crop():
    region = TextRegion(1, [(0, 0), (1, 0), (1, 1), (0, 1)], crop_png=b"x")
    assert "crop" not in region.to_dict(include_crop=False)


# ── CTC decoding ──────────────────────────────────────────────────────────

class _FakeSession:
    """Stands in for an onnxruntime session with a tiny known vocabulary."""

    def __init__(self, probs, charset=("あ", "い", "う")):
        self._probs = probs
        self._charset = charset

    class _Meta:
        def __init__(self, chars):
            self.custom_metadata_map = {"character": "\n".join(chars)}

    class _Input:
        name = "x"

    def get_modelmeta(self):
        return self._Meta(self._charset)

    def get_inputs(self):
        return [self._Input()]

    def run(self, _outputs, _feed):
        return [self._probs[None, ...]]


def _recognizer_with(probs, charset=("あ", "い", "う"), tmp_path=None):
    from gensrt.ocr.ppocr_onnx import PPOCRRecognizer

    model = (tmp_path / "fake_rec.onnx")
    model.write_bytes(b"not-a-real-model")
    rec = PPOCRRecognizer(model, rec_height=48, label="fake")
    session = _FakeSession(np.asarray(probs, dtype=np.float32), charset)
    rec._session = session
    rec._input_name = "x"
    rec._charset = ["<blank>", *charset, " "]
    return rec


def _onehot(sequence, classes=5):
    """Build a [T, C] probability matrix from a list of class indices."""
    out = np.full((len(sequence), classes), 0.01, dtype=np.float32)
    for t, c in enumerate(sequence):
        out[t, c] = 0.95
    return out


def test_ctc_collapses_repeats_and_drops_blanks(tmp_path):
    # blank=0, あ=1, い=2 -> "あい", not "ああいい"
    rec = _recognizer_with(_onehot([1, 1, 0, 2, 2]), tmp_path=tmp_path)
    text, confidence = rec.recognize(np.zeros((20, 60, 3), dtype=np.uint8))
    assert text == "あい"
    assert 0.9 < confidence <= 1.0


def test_ctc_keeps_a_repeat_separated_by_a_blank(tmp_path):
    """あ-blank-あ is a genuine double character, not a decoding artifact."""
    rec = _recognizer_with(_onehot([1, 0, 1]), tmp_path=tmp_path)
    assert rec.recognize(np.zeros((20, 60, 3), dtype=np.uint8))[0] == "ああ"


def test_all_blank_yields_empty_text_and_no_confidence(tmp_path):
    rec = _recognizer_with(_onehot([0, 0, 0]), tmp_path=tmp_path)
    text, confidence = rec.recognize(np.zeros((20, 60, 3), dtype=np.uint8))
    assert text == ""
    assert confidence is None


def test_degenerate_crop_is_not_sent_to_the_model(tmp_path):
    rec = _recognizer_with(_onehot([1]), tmp_path=tmp_path)
    assert rec.recognize(np.zeros((1, 1, 3), dtype=np.uint8)) == ("", None)


def test_missing_dictionary_names_the_sidecar_path(tmp_path):
    """A third-party conversion without embedded metadata must say exactly
    what to provide, not fail obscurely at decode time."""
    from gensrt.ocr.ppocr_onnx import OCRError, PPOCRRecognizer

    class _NoMeta(_FakeSession):
        def get_modelmeta(self):
            class _M:
                custom_metadata_map = {}
            return _M()

    model = tmp_path / "third_party.onnx"
    model.write_bytes(b"x")
    rec = PPOCRRecognizer(model)
    with pytest.raises(OCRError) as exc:
        rec._load_charset(_NoMeta(np.zeros((1, 5), dtype=np.float32)))
    assert "third_party.dict.txt" in str(exc.value)


def test_sidecar_dictionary_is_honoured(tmp_path):
    from gensrt.ocr.ppocr_onnx import PPOCRRecognizer

    model = tmp_path / "m.onnx"
    model.write_bytes(b"x")
    (tmp_path / "m.dict.txt").write_text("か\nき\nく\n", encoding="utf-8")

    class _NoMeta(_FakeSession):
        def get_modelmeta(self):
            class _M:
                custom_metadata_map = {}
            return _M()

    charset = PPOCRRecognizer(model)._load_charset(
        _NoMeta(np.zeros((1, 5), dtype=np.float32))
    )
    assert charset[0] == "<blank>"
    assert charset[1:4] == ["か", "き", "く"]


# ── Crop geometry ─────────────────────────────────────────────────────────

def test_tall_narrow_crops_are_stood_upright():
    """A tall-and-narrow box is vertical text or a rotated line; PP-OCR
    recognition expects a horizontal line either way."""
    pytest.importorskip("cv2")
    from gensrt.ocr.ppocr_onnx import crop_region

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = crop_region(image, [(50, 20), (80, 20), (80, 160), (50, 160)])
    assert crop.shape[1] > crop.shape[0]        # wider than tall after rotation


def test_horizontal_crops_are_left_alone():
    pytest.importorskip("cv2")
    from gensrt.ocr.ppocr_onnx import crop_region

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = crop_region(image, [(10, 40), (170, 40), (170, 75), (10, 75)])
    assert crop.shape[1] > crop.shape[0]


# ── Pipeline ──────────────────────────────────────────────────────────────

class _StubDetector:
    name = "stub-det"

    def __init__(self, count):
        self._count = count

    def detect(self, _image):
        return [
            TextRegion(i, [(0, i * 10), (30, i * 10), (30, i * 10 + 8), (0, i * 10 + 8)])
            for i in range(1, self._count + 1)
        ]


class _StubRecognizer:
    name = "stub-rec"

    def __init__(self, answers):
        self._answers = list(answers)

    def recognize(self, _crop):
        return self._answers.pop(0) if self._answers else ("", None)


def _patch_pipeline(monkeypatch, detector, recognizer):
    monkeypatch.setattr("gensrt.ocr.factory.get_detector", lambda **_kw: detector)
    monkeypatch.setattr("gensrt.ocr.factory.get_recognizer",
                        lambda code, status=None: recognizer)
    monkeypatch.setattr("gensrt.ocr.ppocr_onnx.crop_region",
                        lambda img, quad: np.zeros((10, 40, 3), dtype=np.uint8))
    monkeypatch.setattr("gensrt.ocr.ppocr_onnx.encode_png", lambda img: b"png")


def test_unreadable_regions_are_dropped_and_numbering_stays_contiguous(monkeypatch):
    """Detection finds text-shaped things that are not text — logos,
    textures, faces. An empty row in the picker is pure noise."""
    from gensrt.ocr import read_frame

    _patch_pipeline(monkeypatch, _StubDetector(4),
                    _StubRecognizer([("あ", 0.9), ("", None), ("い", 0.8), ("   ", 0.7)]))
    regions = read_frame(np.zeros((100, 100, 3), dtype=np.uint8), "ja")
    assert [r.text for r in regions] == ["あ", "い"]
    assert [r.index for r in regions] == [1, 2]


def test_pipeline_applies_nfkc(monkeypatch):
    from gensrt.ocr import read_frame

    _patch_pipeline(monkeypatch, _StubDetector(1), _StubRecognizer([("０１２", 0.9)]))
    assert read_frame(np.zeros((10, 10, 3), dtype=np.uint8), "ja")[0].text == "012"


def test_min_confidence_filters(monkeypatch):
    from gensrt.ocr import read_frame

    _patch_pipeline(monkeypatch, _StubDetector(3),
                    _StubRecognizer([("a", 0.9), ("b", 0.2), ("c", 0.8)]))
    regions = read_frame(np.zeros((10, 10, 3), dtype=np.uint8), "ja",
                         min_confidence=0.5)
    assert [r.text for r in regions] == ["a", "c"]


def test_max_regions_caps_a_busy_frame(monkeypatch):
    from gensrt.ocr import read_frame

    _patch_pipeline(monkeypatch, _StubDetector(50),
                    _StubRecognizer([("x", 0.9)] * 50))
    assert len(read_frame(np.zeros((10, 10, 3), dtype=np.uint8), "ja",
                          max_regions=5)) == 5


def test_no_detections_returns_empty_not_error(monkeypatch):
    from gensrt.ocr import read_frame

    _patch_pipeline(monkeypatch, _StubDetector(0), _StubRecognizer([]))
    assert read_frame(np.zeros((10, 10, 3), dtype=np.uint8), "ja") == []


def test_crops_can_be_skipped(monkeypatch):
    from gensrt.ocr import read_frame

    _patch_pipeline(monkeypatch, _StubDetector(1), _StubRecognizer([("あ", 0.9)]))
    regions = read_frame(np.zeros((10, 10, 3), dtype=np.uint8), "ja",
                         include_crops=False)
    assert regions[0].crop_png == b""


# ── HTTP surface ──────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_ocr_translator_cache():
    """The picker's translation engine is cached in module-global state so
    the NLLB model is not reloaded per request. That cache is shared across
    tests too, so clear it around each one — otherwise a test that patches
    get_engine silently gets a previous test's engine instead."""
    from gensrt.api import ocr

    ocr._ocr_translator_cache.clear()
    yield
    ocr._ocr_translator_cache.clear()


@pytest.fixture()
def client():
    from gensrt.server import app

    app.config["TESTING"] = True
    return app.test_client()


def test_languages_endpoint_lists_the_registry(client):
    body = client.get("/api/ocr/languages").get_json()
    assert body["default"] == "ja"
    assert {lang["code"] for lang in body["languages"]} == set(OCR_LANGUAGES)


def test_missing_image_is_a_400(client):
    assert client.post("/api/ocr", json={}).status_code == 400


def test_bad_base64_is_a_400(client):
    resp = client.post("/api/ocr", json={"image": "data:image/png;base64,!!!"})
    assert resp.status_code == 400
    assert "base64" in resp.get_json()["error"]


def test_undecodable_image_is_a_400(client):
    pytest.importorskip("cv2")
    payload = base64.b64encode(b"definitely not a png").decode()
    resp = client.post("/api/ocr", json={"image": payload})
    assert resp.status_code == 400


def test_unknown_language_is_a_422_not_a_500(client, monkeypatch):
    """The caller asked for something specific; that is their mistake to
    fix, and the frontend needs to tell them apart from a server fault."""
    pytest.importorskip("cv2")
    import cv2

    ok, buf = cv2.imencode(".png", np.zeros((20, 60, 3), dtype=np.uint8))
    assert ok
    resp = client.post("/api/ocr", json={
        "image": base64.b64encode(buf.tobytes()).decode(),
        "language": "xx",
    })
    assert resp.status_code == 422
    assert "xx" in resp.get_json()["error"]


def test_successful_read_returns_the_documented_shape(client, monkeypatch):
    pytest.importorskip("cv2")
    import cv2

    def _fake_read_frame(image, language, **kwargs):
        return [TextRegion(1, [(0, 0), (10, 0), (10, 5), (0, 5)],
                           text="アイポケ", confidence=0.93, crop_png=b"png")]

    monkeypatch.setattr("gensrt.ocr.read_frame", _fake_read_frame)
    ok, buf = cv2.imencode(".png", np.zeros((20, 60, 3), dtype=np.uint8))
    resp = client.post("/api/ocr", json={
        "image": "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode(),
        "language": "ja",
    })
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["language"] == "ja"
    region = body["regions"][0]
    assert region["text"] == "アイポケ"
    assert region["confidence"] == pytest.approx(0.93)
    assert region["crop"].startswith("data:image/png;base64,")
    assert len(region["quad"]) == 4 and len(region["bbox"]) == 4


# ── Interactive translation policy and engine caching ────────────────────

@pytest.mark.parametrize("cfg,expected,why", [
    ({"translation_engine": "nllb"}, "nllb", "auto follows the main engine"),
    ({"translation_engine": "madlad"}, "madlad", "auto follows the main engine"),
    ({"translation_engine": "none"}, "none", "translation off"),
    ({"translation_engine": "nllb", "ocr_translation_engine": "madlad"}, "madlad",
     "explicit override wins"),
    ({}, "nllb", "empty config falls back to the built-in default"),
])
def test_ocr_engine_choice_follows_the_configured_engine(cfg, expected, why):
    from gensrt.api.ocr import _resolve_ocr_engine_key

    assert _resolve_ocr_engine_key(cfg) == expected, why


def test_translation_engine_is_built_once_and_reused(monkeypatch):
    """The NLLB model was being pushed to the GPU on every request — 2-3s
    each time for a model that was resident a moment earlier."""
    from gensrt.api import ocr

    built = []

    def _count(key, config=None):
        built.append(key)
        return object()

    monkeypatch.setattr("gensrt.translation.factory.get_engine", _count)
    monkeypatch.setattr(ocr, "_ocr_translator_cache", {})

    cfg = {"translation_engine": "nllb", "translation_model": "m", "device": "cuda"}
    first = ocr._get_ocr_translator(cfg, "nllb")
    second = ocr._get_ocr_translator(cfg, "nllb")
    assert first is second
    assert built == ["nllb"]


def test_changing_the_model_builds_a_new_engine(monkeypatch):
    """Cache keys include the settings that shape the engine, so a config
    change produces a fresh entry rather than a stale one."""
    from gensrt.api import ocr

    monkeypatch.setattr("gensrt.translation.factory.get_engine",
                        lambda key, config=None: object())
    monkeypatch.setattr(ocr, "_ocr_translator_cache", {})

    a = ocr._get_ocr_translator({"translation_engine": "nllb",
                                    "translation_model": "one"}, "nllb")
    b = ocr._get_ocr_translator({"translation_engine": "nllb",
                                    "translation_model": "two"}, "nllb")
    assert a is not b


def test_translation_success_attaches_english(client, monkeypatch):
    """The success path — untested until an OCR run came back with no
    translations at all and nothing in the UI to say why."""
    pytest.importorskip("cv2")
    import cv2

    monkeypatch.setattr(
        "gensrt.ocr.read_frame",
        lambda image, language, **kw: [
            TextRegion(1, [(0, 0), (1, 0), (1, 1), (0, 1)], text="ついて", confidence=0.9)
        ],
    )

    class _Engine:
        name = "fake"

        def translate_batch(self, texts, src, tgt):
            assert (src, tgt) == ("ja", "en")
            return [f"about {t}" for t in texts]

    monkeypatch.setattr("gensrt.translation.factory.get_engine",
                        lambda key, config=None: _Engine())
    ok, buf = cv2.imencode(".png", np.zeros((20, 60, 3), dtype=np.uint8))
    body = client.post("/api/ocr", json={
        "image": base64.b64encode(buf.tobytes()).decode(),
        "language": "ja", "translate": True, "target_language": "en",
    }).get_json()
    region = body["regions"][0]
    assert region["translation"] == "about ついて"
    assert "translation_error" not in region


def test_translation_is_skipped_when_not_requested(client, monkeypatch):
    pytest.importorskip("cv2")
    import cv2

    monkeypatch.setattr(
        "gensrt.ocr.read_frame",
        lambda image, language, **kw: [
            TextRegion(1, [(0, 0), (1, 0), (1, 1), (0, 1)], text="ついて")
        ],
    )

    def _must_not_be_called(*_a, **_k):
        raise AssertionError("translation engine built without translate=True")

    monkeypatch.setattr("gensrt.translation.factory.get_engine", _must_not_be_called)
    ok, buf = cv2.imencode(".png", np.zeros((20, 60, 3), dtype=np.uint8))
    body = client.post("/api/ocr", json={
        "image": base64.b64encode(buf.tobytes()).decode(), "language": "ja",
    }).get_json()
    assert "translation" not in body["regions"][0]


def test_translation_failure_preserves_the_ocr_result(client, monkeypatch):
    """A dead translation engine must not lose the source text — the crop
    and the original are still what the user needs."""
    pytest.importorskip("cv2")
    import cv2

    monkeypatch.setattr(
        "gensrt.ocr.read_frame",
        lambda image, language, **kw: [
            TextRegion(1, [(0, 0), (1, 0), (1, 1), (0, 1)], text="快感", confidence=0.8)
        ],
    )

    def _boom(*_a, **_k):
        raise RuntimeError("no engine")

    monkeypatch.setattr("gensrt.translation.factory.get_engine", _boom)
    ok, buf = cv2.imencode(".png", np.zeros((20, 60, 3), dtype=np.uint8))
    body = client.post("/api/ocr", json={
        "image": base64.b64encode(buf.tobytes()).decode(),
        "language": "ja",
        "translate": True,
    }).get_json()
    assert body["regions"][0]["text"] == "快感"
    assert "no engine" in body["regions"][0]["translation_error"]
