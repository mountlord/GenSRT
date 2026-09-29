"""Post-processing shared by the offline engines' detokenisers.

NLLB-200's training corpora carry un-normalised HTML entities and
tokeniser-style spacing, and the model reproduces them on some lines:
``I &apos;m going to massage from the neckline .`` was measured on 181 of
1,702 cues in one file, spread evenly through it.  MADLAD has not shown it,
but its output passes through the same function as a guard.  Everything
here is a plain string fix; nothing changes the words.
"""

from __future__ import annotations

import html
import re

# " ." → "."   " ," → ","   " ?" → "?"   " !" → "!"   " :" → ":"   " ;" → ";"
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?;:%)\]}])")
# "( " → "("
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")
# "I 'm" / "don 't" / "it ’s" → "I'm" / "don't" / "it’s"
_SPACE_BEFORE_CLITIC = re.compile(r"\s+(['’](?:s|m|re|ve|ll|d|t)\b)", re.IGNORECASE)
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


def clean_mt_output(text: str) -> str:
    """Unescape HTML entities and undo tokeniser spacing in *text*."""
    if not text:
        return text
    out = html.unescape(text)
    out = _SPACE_BEFORE_CLITIC.sub(r"\1", out)
    out = _SPACE_BEFORE_PUNCT.sub(r"\1", out)
    out = _SPACE_AFTER_OPEN.sub(r"\1", out)
    out = _MULTI_SPACE.sub(" ", out)
    return out.strip()
