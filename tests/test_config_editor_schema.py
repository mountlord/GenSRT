"""The GUI config editor's schema (templates/js/config.js) against the
backend, checked mechanically.

Two failures this class of test would have caught: the ``profile`` field
declared with a type the renderer has no branch for, so it fell through to
the numeric input (2026-09-30); and config fields added to the dataclass
that never reached the schema, so the GUI could not set them.  The
frontend has no test runner, so the schema is read as text here.
"""

from __future__ import annotations

import re
from pathlib import Path

from gensrt.config import BUILTIN_DEFAULTS

_JS = (Path(__file__).parent.parent / "gensrt" / "templates" / "js" / "config.js").read_text(encoding="utf-8")
_SCHEMA = _JS[_JS.index("const configSchema"):_JS.index("\n};", _JS.index("const configSchema"))]

#: Config keys the editor deliberately does not show.  Diagnostics dirs
#: and output naming are per run (CLI flags); the OCR settings wait for the
#: settings revamp's tabbed panel.  Shrink this list, never grow it
#: without a reason in the commit.
GUI_ABSENT = frozenset({
    "debug_chunk_dir", "dump_segments_dir", "heuristics_report_dir",
    "output_filename", "recurse",
    "ocr_det_limit_side_len", "ocr_language", "ocr_max_regions",
    "ocr_min_confidence", "ocr_translation_engine",
})


def _schema_fields() -> dict[str, str]:
    """field key → declared type."""
    out = {}
    for m in re.finditer(r"^\s{4}'([a-z_0-9]+)': \{(.*?)^\s{4}\},", _SCHEMA, re.M | re.S):
        key, body = m.group(1), m.group(2)
        t = re.search(r"type: '([a-z-]+)'", body)
        out[key] = t.group(1) if t else "?"
    return out


def test_every_schema_type_has_a_renderer_branch():
    # 'number' is the renderer's final else-branch, which is exactly why an
    # unknown type silently becomes a numeric input.
    handled = set(re.findall(r"fieldDef\.type === '([a-z-]+)'", _JS)) | {"number"}
    for key, typ in _schema_fields().items():
        assert typ in handled, f"{key}: type {typ!r} has no branch in renderConfigEditor — it would render as a number"


def test_every_config_field_is_in_the_gui_or_deliberately_absent():
    fields = set(_schema_fields())
    missing = sorted(set(BUILTIN_DEFAULTS) - fields - GUI_ABSENT)
    assert not missing, f"config fields with no GUI entry (add to config.js or to GUI_ABSENT with a reason): {missing}"
    unknown = sorted(fields - set(BUILTIN_DEFAULTS))
    assert not unknown, f"GUI schema names keys that are not config keys: {unknown}"
    stale = sorted(GUI_ABSENT & fields)
    assert not stale, f"listed as absent but present in the GUI: {stale}"


def test_profile_field_is_a_picker_fed_by_the_profiles_route():
    assert _schema_fields()["profile"] == "select-dynamic"
    assert "source: 'profiles'" in _SCHEMA
    assert "/api/profiles" in _JS
