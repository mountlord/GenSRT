"""word_timestamps must stay off.

Both engines requested it and neither read the result. The cost was an extra
alignment pass per chunk, and on a distil-architecture model
(kotoba-whisper-v2.0: 32 encoder layers, 2 decoder layers) the alignment call
reads out of bounds and segfaults — killing the process with no traceback and
no chance for any handler to run.

Pinned by inspecting the source rather than by behaviour: the failure it
guards against cannot be caught, so it has to be prevented from being
reintroduced.
"""

from __future__ import annotations

import inspect

import pytest

from gensrt.asr import monolingual_whisper, multilingual_whisper


@pytest.mark.parametrize("module", [monolingual_whisper, multilingual_whisper])
def test_word_timestamps_not_requested(module):
    src = inspect.getsource(module)
    assert "word_timestamps=True" not in src


@pytest.mark.parametrize("module", [monolingual_whisper, multilingual_whisper])
def test_reason_is_recorded_next_to_the_call(module):
    """A bare absence invites someone to add it back as an improvement."""
    src = inspect.getsource(module)
    assert "word_timestamps" in src          # the explanatory comment
    assert "align" in src


def test_nothing_consumes_word_level_output():
    """If this ever fails, word timestamps have a real consumer and the
    trade-off needs revisiting rather than the flag simply being restored."""
    from pathlib import Path

    import gensrt

    root = Path(gensrt.__file__).parent
    hits = []
    for py in root.rglob("*.py"):
        for n, line in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
            if ".words" in line and not line.lstrip().startswith("#"):
                hits.append(f"{py.name}:{n}")
    assert not hits, f"word-level output is used at {hits}"
