"""The GUI config validator must keep pace with the config dataclass.

Why this file exists
--------------------
server.py validates config saves against a hand-maintained table,
``_CONFIG_VALIDATORS``. TranscriptionConfig grew ten fields between v1.2.7
and the OCR work — NLLB, MADLAD, chunk sizes, every OCR setting — and none
of them were added to that table. The GUI rejected all of them with
"unknown configuration key", so none could be changed from Config at all;
the only way to set them was to hand-edit gensrt-config.json.

Nothing caught it because the table and the dataclass were only ever
compared by eye. These tests compare them mechanically.
"""

from __future__ import annotations

import dataclasses

import pytest

from gensrt.models import TranscriptionConfig
from gensrt.server import _CONFIG_VALIDATORS, _validate_config_patch


def test_every_config_field_is_saveable_from_the_gui():
    """Any field on TranscriptionConfig must have a validator.

    If this fails, you added a config field and the GUI cannot save it.
    Add it to _CONFIG_VALIDATORS in server.py.
    """
    fields = {f.name for f in dataclasses.fields(TranscriptionConfig)}
    missing = sorted(fields - set(_CONFIG_VALIDATORS))
    assert not missing, (
        f"These config fields have no validator, so the GUI rejects them "
        f"with 'unknown configuration key': {missing}"
    )


def test_translation_engine_choices_match_the_factory():
    """The validator must accept every engine the factory can build.

    These were hand-copied and went stale the moment NLLB shipped.
    """
    from gensrt.translation.factory import ENGINE_KEYS

    for key in ENGINE_KEYS:
        sanitized, errors = _validate_config_patch({"translation_engine": key})
        assert not errors, f"factory offers {key!r} but the validator rejects it"
        assert sanitized["translation_engine"] == key


def test_translation_fallback_choices_match_the_factory():
    from gensrt.translation.factory import FALLBACK_KEYS

    for key in FALLBACK_KEYS:
        _sanitized, errors = _validate_config_patch({"translation_fallback": key})
        assert not errors, f"factory offers {key!r} but the validator rejects it"


@pytest.mark.parametrize("patch", [
    {"translation_engine": "madlad"},
    {"translation_engine": "nllb"},
    {"translation_fallback": "madlad"},
    {"translation_model": "mijuanlo/nllb-200-distilled-600M-ct2-int8"},
    {"madlad_model": "olob0/madlad400-3b-mt-ct2-int8_float16"},
    {"max_chunk_s": 6.0},
    {"min_chunk_s": 2.0},
    {"ocr_language": "zh"},
    {"ocr_translation_engine": "auto"},
    {"ocr_det_limit_side_len": 1280},
    {"ocr_max_regions": 40},
    {"ocr_min_confidence": 0.0},
])
def test_settings_the_gui_previously_could_not_save(patch):
    """One case per field that returned 'unknown configuration key'."""
    _sanitized, errors = _validate_config_patch(patch)
    assert not errors, errors


@pytest.mark.parametrize("patch,key", [
    ({"translation_engine": "not-an-engine"}, "translation_engine"),
    ({"translation_fallback": "not-a-fallback"}, "translation_fallback"),
    ({"ocr_translation_engine": "nonsense"}, "ocr_translation_engine"),
    ({"max_chunk_s": 0.0}, "max_chunk_s"),
    ({"ocr_det_limit_side_len": 10}, "ocr_det_limit_side_len"),
    ({"ocr_min_confidence": 5.0}, "ocr_min_confidence"),
    ({"completely_made_up": 1}, "completely_made_up"),
])
def test_bad_values_are_still_rejected(patch, key):
    """Widening the table must not have made it permissive."""
    _sanitized, errors = _validate_config_patch(patch)
    assert key in errors


def test_integer_fields_are_coerced():
    """Int-typed keys come back as int for clean JSON, as the others do."""
    sanitized, errors = _validate_config_patch(
        {"ocr_max_regions": 40, "ocr_det_limit_side_len": 1280})
    assert not errors
    assert isinstance(sanitized["ocr_max_regions"], int)
    assert isinstance(sanitized["ocr_det_limit_side_len"], int)


def test_model_refs_may_be_empty_meaning_use_the_default():
    """A config that never set these rendered a blank box and then refused
    to save it, with hand-editing the JSON as the only way out."""
    for key in ("translation_model", "madlad_model"):
        _sanitized, errors = _validate_config_patch({key: ""})
        assert not errors, f"{key} rejected an empty value: {errors}"


def test_model_refs_still_reject_non_strings():
    for key in ("translation_model", "madlad_model"):
        _sanitized, errors = _validate_config_patch({key: 42})
        assert key in errors
