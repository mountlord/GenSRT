"""Before/after corpus for the heuristics — every guard is a regression test.

The rules in gensrt/heuristics.py operate on source-language text the
maintainer cannot read and on machine-translation output whose edge cases
arrive one report at a time ("Can rest assured", "Know, like this.",
"I'm Lena Kotama, 22" — four guards in two days).  Each guard used to be
a fix driven by a report; now it is a line in ``tests/corpus/`` and the
suite fails if a later change undoes it.

Two files, one JSON object per line, ``ensure_ascii=False`` so the
source text is readable in a diff:

``subject_rule.jsonl``   {"lang", "source", "mt", "expect", "note"}
    ``source`` is the pre-translation line, ``mt`` the translator's
    English, ``expect`` the text after the subject rule — or ``null`` when
    the rule must leave it alone.

``post_asr_rules.jsonl`` {"lang", "mode"?, "cues", "expect", "note"}
    ``cues`` and ``expect`` are ``[start, end, text]`` triples; the whole
    pre-translation stage runs (drop / hallucination / junk lists, density,
    collapse with the span cap) under the SHIPPED rules; ``mode`` names a
    heuristics mode ("soft-speech") or is absent.

To add a case: append a line.  To add a guard: add the case first, watch
it fail, then change the rule.  The ``note`` is for the next reader; keep
it to why, not what.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gensrt.heuristics import (
    BUILTIN_HEURISTICS,
    Heuristics,
    run_heuristics,
    strip_subjects,
)
from gensrt.models import SRTSegment

CORPUS = Path(__file__).parent / "corpus"


def _load(name: str) -> list[dict]:
    rows = []
    for n, line in enumerate((CORPUS / name).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            row["_line"] = n
            rows.append(row)
    return rows


def _ids(rows: list[dict], key: str) -> list[str]:
    return [f"L{r['_line']}:{r['lang']}:{r[key] if isinstance(r[key], str) else r['note']}"[:60]
            for r in rows]


_SUBJECT = _load("subject_rule.jsonl")
_POST_ASR = _load("post_asr_rules.jsonl")
_SHIPPED = Heuristics(raw=BUILTIN_HEURISTICS)


@pytest.mark.parametrize("row", _SUBJECT, ids=_ids(_SUBJECT, "mt"))
def test_subject_rule_corpus(row):
    rules = _SHIPPED.subject_for(row["lang"], "en")
    seg = SRTSegment(index=1, start=0.0, end=1.0, text=row["mt"])
    out, n, _examples = strip_subjects([row["source"]], [seg], rules)
    got = out[0].text
    want = row["expect"] if row["expect"] is not None else row["mt"]
    assert got == want, f"line {row['_line']}: {row['note']!r}\n  mt={row['mt']!r}\n  got={got!r}\n  want={want!r}"
    assert n == (0 if row["expect"] is None else 1)


@pytest.mark.parametrize("row", _POST_ASR, ids=_ids(_POST_ASR, "note"))
def test_post_asr_rules_corpus(row):
    cues = [SRTSegment(index=i + 1, start=float(s), end=float(e), text=t)
            for i, (s, e, t) in enumerate(row["cues"])]
    out, _stats = run_heuristics(cues, _SHIPPED, row["lang"], row.get("mode"))
    got = [[s.start, s.end, s.text] for s in out]
    want = [[float(s), float(e), t] for s, e, t in row["expect"]]
    assert got == want, f"line {row['_line']}: {row['note']!r}\n  got={got}\n  want={want}"
    assert [s.index for s in out] == list(range(1, len(out) + 1)), "re-indexed 1..n"


def test_corpus_files_are_well_formed():
    for row in _SUBJECT:
        assert set(row) >= {"lang", "source", "mt", "expect", "note"}, row
    for row in _POST_ASR:
        assert set(row) >= {"lang", "cues", "expect", "note"}, row
        for triple in row["cues"] + row["expect"]:
            assert len(triple) == 3 and triple[0] <= triple[1], triple
