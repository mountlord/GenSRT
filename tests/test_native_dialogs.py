"""The file filters handed to pywebview's create_file_dialog must satisfy
its parser (webview/util.py parse_file_type): description of [\\w ] only,
then "(*.ext;*.ext)".  "MPEG-4 video (*.mp4)" — with a hyphen — raised
ValueError inside the Save MP4 dialog and the button looked dead."""

from __future__ import annotations

import re
from pathlib import Path

# pywebview 5.x parse_file_type, verbatim.
_VALID = re.compile(r'^([\w ]+)\((\*(?:\.(?:\w+|\*))*(?:;\*(?:\.(?:\w+|\*))*)*)\)$')


def test_every_file_filter_in_server_py_parses():
    src = (Path(__file__).parent.parent / "gensrt" / "server.py").read_text(encoding="utf-8")
    filters = re.findall(r'"([^"]*\(\*[^"]*\))"', src)
    assert filters, "no file filters found — did the dialogs move?"
    bad = [f for f in filters if not _VALID.match(f)]
    assert not bad, f"invalid pywebview file filters: {bad}"
