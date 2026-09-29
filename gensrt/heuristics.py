"""Post-ASR heuristics, driven by a user-editable rules file.

Why a separate file and not config fields
-----------------------------------------
These rules are lists of strings that depend on the language and on the
material — what counts as a throwaway interjection in one genre is content
in another — and they will grow as more patterns are catalogued.  A JSON
file next to ``gensrt-config.json`` keeps them editable without touching
code, keeps the config editor from sprouting a dozen list fields, and lets
a user carry their own rules between releases.

Rules live in ``gensrt-heuristics.json``, discovered the same way the config
file is (next to the executable, then the working directory).  If the file
is absent the built-in defaults below apply; ``gensrt --init-heuristics``
writes them out for editing.

The first rule: interjection collapse
-------------------------------------
Measured on a 152-minute file with VAD-free chunking: 1,713 segments, of
which 763 were short vocalisations transcribed as words — ``ああ`` ×221,
``ごめん`` ×131, ``はい`` ×91, ``うん`` ×68 — with zero hallucination markers.
That is the model faithfully rendering real audio, not inventing text, and
five consecutive ``ああ`` cues across 0.4 seconds are not something a viewer
needs five of.

``collapse_interjections`` merges consecutive identical short cues within a
time window into one cue spanning the run.  Nothing is invented and nothing
that differs from its neighbour is removed.

Be clear about what it can and cannot do.  On that file it removes ~137 of
the 763: the interjections mostly ALTERNATE (``ああ, はい, ごめん, ああ``)
rather than repeat, and alternating cues are each real audio.  Removing
those is the ``drop`` list's job (1,713 → 1,329 on the same file with the
pure vocalisations listed), and that stays the user's decision.  A separate ``drop`` list exists
for strings a user never wants to see at all; it is empty by default because
in this material ``気持ちいい`` is content and ``ああ`` is noise, and only the
user can say which.

Rules two to four, and why telemetry is not one of them
--------------------------------------------------------
Measured on a second file (1,292 segments): the decoder's own confidence
does NOT separate hallucinated fillers from dialogue.  Long dialogue sat at
avg_logprob p50 −0.37; ``ごめん`` (110×) at −0.57, ``ごちそう`` (21×) at −0.61
— lower, but the distributions overlap so heavily that any threshold which
removes most fillers also removes the quiet real lines this mode exists to
recover.  no_speech_prob never exceeded 0.22 and compression_ratio never
exceeded 2.4.  The model is confident it heard "sorry" over the moaning.
So hallucination stays list- and pattern-based:

``hallucination`` (per language) — real words the model produces as its
"safe guess" over vocalisation with no speech: ``ごめん`` / ``すごい`` /
``ごちそう`` on this material, confirmed by watching the video, not by
telemetry.  Removed outright, reported separately from ``drop`` so a user
can move a string between the two lists without losing the distinction.

``density`` — a short cue repeated N times inside a window with no dialogue
between the repeats.  Real apologies do not arrive five in a row; a run of
five ``ごめん`` in thirty seconds is vocalisation being lexicalised.  This
catches strings the lists do not name, and leaves an isolated ``ごめん``
alone.

``subject`` — a post-TRANSLATION text rule.  Japanese, Malayalam and Korean
drop the subject when context supplies it; the translator has no context
beyond one chunk, so it guesses — and "I" for "you" is the guess a viewer
notices most.  The fix is not a better guess but no guess: when the source
line contains no explicit subject word, a leading English subject pronoun
(``I``, ``You``, ``I'm``, ``It's``…) is stripped and the line re-capitalised
— "I'm sorry." → "Sorry."  Only the sentence-initial pronoun is touched, and
never when the source itself names a subject.

Still deliberately NOT here: repetition-loop removal for LONG identical
lines (compression_ratio is its signal — zero instances so far).
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from gensrt.models import SRTSegment

logger = logging.getLogger(__name__)

DEFAULT_HEURISTICS_NAME = "gensrt-heuristics.json"

#: Shipped defaults.  Japanese-leaning because that is where the data came
#: from; the ``max_chars`` rule is language-agnostic and does most of the work.
BUILTIN_HEURISTICS: dict[str, Any] = {
    "_comment": (
        "GenSRT post-processing rules. Edit freely; unknown keys are ignored. "
        "Delete this file to return to the built-in defaults."
    ),
    "interjections": {
        "enabled": True,
        # Any cue whose text is at most this many characters is treated as an
        # interjection for collapse purposes, whatever it says.
        "max_chars": 3,
        # Consecutive identical interjections closer than this many seconds
        # merge into one cue spanning the run.  6, not 3: with 5-8 s chunks
        # each chunk yields at most one interjection, so identical neighbours
        # are 4-6 s apart (measured p50 4.0 s, p90 5.7 s).
        "window_s": 6.0,
        # A collapsed run is shown for at most this long.  Merging 20 "うん"
        # over 10 s into one cue spanning the run put "Yeah." on screen for
        # ten seconds (39 such cues in one file); the viewer needs it for a
        # couple of seconds, not for the whole run.
        "max_span_s": 3.0,
        # Language-specific lists, applied by the DETECTED language so a user
        # who cannot read the source script never has to type in it.  The
        # ja lists come from measurement on real material; the others are
        # empty scaffolds because nothing has been measured for them yet —
        # the run report shows what each file actually produces.
        "languages": {
            "ja": {
                "enabled": True,
                # Real words used as vocalisation: collapse runs, never drop.
                "always_collapse": ["気持ちいい"],
                # Non-lexical vocalisation only — carries no dialogue.
                # はい / うん are deliberately NOT here: they are words.
                "drop": ["ああ", "あっ", "うっ", "お", "ん", "はあ", "あ"],
                # Real words the model produces over vocalisation with NO
                # speech ("I'm sorry", "That's amazing", "Gochiso" over
                # silence — confirmed by watching).  Removed outright.  Move
                # a string to always_collapse if it is content in your files.
                # ご視聴ありがとうございました: Whisper's best-known Japanese
                # silence hallucination ("Thank you for watching"), seen at
                # 01:02:52 of a file with no such line.  Matching ignores
                # trailing punctuation, so the 。 variant is covered.
                "hallucination": ["ごめん", "すごい", "ごちそう",
                                  "ご視聴ありがとうございました"],
                # Substrings: a cue CONTAINING one of these is a hallucination.
                # For the sign-off phrases Whisper learned from subtitled
                # video, which arrive in variants ("…ました", "…ました！").
                "hallucination_contains": ["ご視聴"],
            },
            # Hangul packs a syllable per character: three characters is a
            # whole word (예산이 "the budget" was removed as a dense run of an
            # interjection).  Interjections are one or two blocks: 어 응 아 네
            # 진짜.  max_chars here overrides the global value.
            "ko": {"enabled": True, "max_chars": 2,
                   "always_collapse": [], "drop": [], "hallucination": [],
                   # A 6-hour Korean stream produced "See you in the next
                   # video." 37 times, "Thank you for watching." and a
                   # subtitle credit — Whisper's Korean YouTube sign-offs,
                   # over silence.  These are the sources behind them.
                   "hallucination_contains": ["다음 영상에서", "시청해주셔서 감사",
                                              "시청해 주셔서 감사", "구독과 좋아요"]},
            "ml": {"enabled": True, "always_collapse": [], "drop": [], "hallucination": [],
                   "hallucination_contains": []},
        },
        # Lists applied to EVERY language, on top of the per-language ones.
        "always_collapse": [],
        "drop": [],
        "hallucination": [],
        "hallucination_contains": [],
    },
    "density": {
        "enabled": True,
        # A short cue (interjection-class: short, or on a list) whose text
        # occurs at least min_count times within window_s seconds, with no
        # non-interjection cue between the repeats, is vocalisation being
        # lexicalised: every occurrence in that run is removed.  Isolated
        # occurrences are untouched, whatever the string.
        "min_count": 4,
        "window_s": 30.0,
    },
    "subject": {
        "enabled": True,
        # Pro-drop languages: the translator invents a subject the speaker
        # never said, and guesses wrong ("I" for "you").  When the SOURCE
        # line contains none of the explicit subject words below, the
        # leading English subject pronoun is stripped instead of guessed:
        # "I'm sorry." → "Sorry."   Only applied when translating to English.
        "languages": {
            "ja": {
                "enabled": True,
                "explicit_subjects": [
                    "私", "わたし", "僕", "ぼく", "俺", "おれ", "あたし", "自分",
                    "あなた", "貴方", "君", "きみ", "お前", "おまえ", "あんた",
                    "彼", "彼女", "うち", "みんな", "皆",
                ],
            },
            "ml": {
                "enabled": True,
                "explicit_subjects": [
                    "ഞാൻ", "ഞങ്ങൾ", "നമ്മൾ", "നമുക്ക്", "എനിക്ക്", "എന്റെ",
                    "നീ", "നിങ്ങൾ", "നിനക്ക്", "താൻ", "തനിക്ക്",
                    "അവൻ", "അവൾ", "അവർ", "അയാൾ", "അദ്ദേഹം", "അത്",
                ],
            },
            "ko": {
                "enabled": True,
                "explicit_subjects": [
                    "나", "내가", "저", "제가", "우리", "저희",
                    "너", "네가", "당신", "니가", "그", "그녀", "그들",
                ],
            },
        },
        # English forms removed when they open a line.  Contractions that
        # carry tense ('ll, 'd, 've) are left alone on purpose.
        # First and second person, plus "it" and "we": the guesses a viewer
        # notices ("I" for "you").  He / She / They are NOT stripped by
        # default — a third-person subject the translator produced usually
        # came from something in the sentence, and "He's a legend." → "A
        # legend." reads worse than a possibly-wrong pronoun.  Add them here
        # if your material says otherwise.
        "en_strip": ["I", "You", "We", "It", "I'm", "You're", "We're", "It's"],
    },
}


def _strset(values: Any) -> frozenset[str]:
    if not isinstance(values, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(s).strip() for s in values if str(s).strip())


@dataclass(frozen=True)
class InterjectionRules:
    enabled: bool = True
    max_chars: int = 3
    window_s: float = 6.0
    max_span_s: float = 3.0
    always_collapse: frozenset[str] = field(default_factory=frozenset)
    drop: frozenset[str] = field(default_factory=frozenset)
    hallucination: frozenset[str] = field(default_factory=frozenset)
    #: Substrings; a cue containing any of them is a hallucination.
    hallucination_contains: frozenset[str] = field(default_factory=frozenset)
    #: Which language section, if any, these effective lists came from.
    language: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any], language: str | None = None) -> "InterjectionRules":
        """Effective rules for *language*: global lists plus that language's.

        A flat file with only top-level ``always_collapse`` / ``drop`` (the
        first release's shape) still works — those are the global lists.
        """
        collapse = set(_strset(d.get("always_collapse", [])))
        drop = set(_strset(d.get("drop", [])))
        halluc = set(_strset(d.get("hallucination", [])))
        contains = set(_strset(d.get("hallucination_contains", [])))
        used = None
        langs = d.get("languages") or {}
        if language and isinstance(langs, dict):
            code = language.strip().lower()
            # "ja-JP" resolves to the "ja" section; report the key that matched.
            for key in (code, code.split("-")[0]):
                section = langs.get(key)
                if isinstance(section, dict):
                    if section.get("enabled", True):
                        if section.get("max_chars") is not None:
                            d = {**d, "max_chars": section["max_chars"]}
                        collapse |= _strset(section.get("always_collapse", []))
                        drop |= _strset(section.get("drop", []))
                        halluc |= _strset(section.get("hallucination", []))
                        contains |= _strset(section.get("hallucination_contains", []))
                        used = key
                    break
        return cls(
            enabled=bool(d.get("enabled", True)),
            max_chars=max(0, int(d.get("max_chars", 3) or 0)),
            window_s=max(0.0, float(d.get("window_s", 6.0) or 0.0)),
            max_span_s=max(0.0, float(d.get("max_span_s", 3.0) or 0.0)),
            always_collapse=frozenset(collapse),
            drop=frozenset(drop),
            hallucination=frozenset(halluc),
            hallucination_contains=frozenset(contains),
            language=used,
        )


@dataclass(frozen=True)
class DensityRules:
    enabled: bool = True
    min_count: int = 4
    window_s: float = 30.0

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DensityRules":
        return cls(
            enabled=bool(d.get("enabled", True)),
            min_count=max(2, int(d.get("min_count", 4) or 0)),
            window_s=max(0.0, float(d.get("window_s", 30.0) or 0.0)),
        )


@dataclass(frozen=True)
class SubjectRules:
    enabled: bool = False
    explicit_subjects: frozenset[str] = field(default_factory=frozenset)
    en_strip: frozenset[str] = field(default_factory=frozenset)   # lower-cased
    language: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any], language: str | None) -> "SubjectRules":
        """Effective rules for *language*; disabled unless it has a section."""
        if not d.get("enabled", True) or not language:
            return cls()
        langs = d.get("languages") or {}
        code = language.strip().lower()
        for key in (code, code.split("-")[0]):
            section = langs.get(key) if isinstance(langs, dict) else None
            if isinstance(section, dict):
                if not section.get("enabled", True):
                    return cls()
                strip = _strset(d.get("en_strip", BUILTIN_HEURISTICS["subject"]["en_strip"]))
                return cls(
                    enabled=True,
                    explicit_subjects=_strset(section.get("explicit_subjects", [])),
                    en_strip=frozenset(w.lower() for w in strip),
                    language=key,
                )
        return cls()


@dataclass(frozen=True)
class Heuristics:
    """Loaded rules file.  ``for_language`` resolves the effective rules."""
    raw: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    def for_language(self, language: str | None) -> InterjectionRules:
        inter = self.raw.get("interjections") or {}
        if not isinstance(inter, dict):
            inter = {}
        return InterjectionRules.from_dict(inter, language)

    def density(self) -> DensityRules:
        d = self.raw.get("density") or {}
        return DensityRules.from_dict(d if isinstance(d, dict) else {})

    def subject_for(self, language: str | None, target_language: str | None) -> SubjectRules:
        """Subject stripping is English-only for now: no other target has rules."""
        if (target_language or "").strip().lower() not in ("en", "english"):
            return SubjectRules()
        d = self.raw.get("subject") or {}
        return SubjectRules.from_dict(d if isinstance(d, dict) else {}, language)

    # Kept for the first release's callers and tests.
    @property
    def interjections(self) -> InterjectionRules:
        return self.for_language(None)


def _find_heuristics_file() -> Path | None:
    """Same search order as the config file: next to the exe, then cwd."""
    import sys

    for p in (Path(sys.argv[0]).resolve().parent / DEFAULT_HEURISTICS_NAME,
              Path.cwd() / DEFAULT_HEURISTICS_NAME):
        if p.is_file():
            return p
    return None


def load_heuristics(path: Path | None = None) -> Heuristics:
    """Load rules from *path*, the discovered file, or the built-in defaults.

    A malformed file is reported and the defaults are used — a bad edit to
    the rules must not take the whole pipeline down.
    """
    path = path or _find_heuristics_file()
    data: dict[str, Any] = BUILTIN_HEURISTICS
    if path is not None:
        try:
            loaded = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("top level must be a JSON object")
            data = {**BUILTIN_HEURISTICS, **loaded}
            logger.debug("Heuristics loaded: %s", path)
        except Exception as exc:
            logger.warning(
                "Could not read %s (%s) — using built-in heuristics.", path, exc
            )
            path = None
    return Heuristics(raw=data, source_path=path)


def write_default_heuristics(path: Path | None = None) -> Path:
    """Write the built-in rules out for editing (``--init-heuristics``)."""
    from gensrt.model_paths import sidecar_dir

    path = path or (sidecar_dir() / DEFAULT_HEURISTICS_NAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(BUILTIN_HEURISTICS, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


# ── Rule: interjection collapse ───────────────────────────────────────────

_TRAILING_PUNCT = "。．.!！?？、,…"


def _key(text: str) -> str:
    """List-membership key: trailing punctuation ignored ("ごめん。" is ごめん)."""
    return (text or "").strip().rstrip(_TRAILING_PUNCT).strip()


def _is_junk(text: str) -> bool:
    """Text the decoder could not render: contains the replacement character."""
    return "\ufffd" in text


def _is_hallucination(text: str, rules: InterjectionRules) -> bool:
    if _key(text) in rules.hallucination:
        return True
    return any(sub in text for sub in rules.hallucination_contains)


def _is_interjection(text: str, rules: InterjectionRules) -> bool:
    return len(text) <= rules.max_chars or _key(text) in rules.always_collapse


def collapse_interjections(
    segments: list[SRTSegment], rules: InterjectionRules
) -> tuple[list[SRTSegment], dict[str, int]]:
    """Merge runs of identical short cues; drop listed strings outright.

    Returns the new segment list (re-indexed) and a small stats dict so the
    pipeline can log what happened — a rule that silently removes a third of
    the cues should say so.

    Behaviour, precisely:
      * A cue whose whole text is in ``rules.drop`` is removed.
      * A cue that is an interjection (short, or in ``always_collapse``) and
        has the SAME text as the previous kept cue, starting within
        ``window_s`` of that cue's end, is merged into it: the kept cue's end
        extends to cover it.  Confidence fields keep the first cue's values.
      * Everything else passes through untouched.  Two consecutive identical
        LONG lines are left alone — that is a repetition-loop signal and a
        different rule's job.
    """
    stats = {"in": len(segments), "dropped": 0, "collapsed": 0, "out": 0}
    if not rules.enabled or not segments:
        stats["out"] = len(segments)
        return list(segments), stats

    out: list[SRTSegment] = []
    run_end = 0.0          # true end of the current run, before the span cap
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            out.append(seg)
            continue
        if _key(text) in rules.drop:
            stats["dropped"] += 1
            continue
        if out and _is_interjection(text, rules):
            prev = out[-1]
            if (prev.text or "").strip() == text and seg.start - run_end <= rules.window_s:
                run_end = max(run_end, seg.end)
                shown_end = run_end
                if rules.max_span_s > 0:
                    shown_end = min(run_end, prev.start + rules.max_span_s)
                out[-1] = replace(prev, end=max(prev.end, shown_end))
                stats["collapsed"] += 1
                continue
        out.append(seg)
        run_end = seg.end

    out = [replace(s, index=i + 1) for i, s in enumerate(out)]
    stats["out"] = len(out)
    return out, stats


@dataclass
class HeuristicsStats:
    """What each rule did, per source string — feeds the log line and report."""
    cues_in: int = 0
    cues_out: int = 0
    collapsed: int = 0
    #: reason → {source text → occurrences removed}; reasons are
    #: "drop", "hallucination", "density".
    dropped: dict[str, "Counter[str]"] = field(default_factory=dict)
    subject_stripped: int = 0
    subject_examples: list[tuple[str, str]] = field(default_factory=list)

    def dropped_total(self, reason: str | None = None) -> int:
        if reason:
            return sum(self.dropped.get(reason, Counter()).values())
        return sum(sum(c.values()) for c in self.dropped.values())

    def reason_for(self, text: str) -> str | None:
        for reason, c in self.dropped.items():
            if c.get(text):
                return reason
        return None


# ── Rule: density ─────────────────────────────────────────────────────────

def drop_dense_runs(
    segments: list[SRTSegment], inter: InterjectionRules, density: DensityRules
) -> tuple[list[SRTSegment], "Counter[str]"]:
    """Remove interjection-class cues that recur densely with no dialogue between.

    The segment list is cut into stretches at every non-interjection cue.
    Inside a stretch, the occurrences of each distinct text are scanned with
    a sliding window of ``density.window_s``: any window holding at least
    ``density.min_count`` of them marks all of those for removal.  A text
    that appears three times over two minutes survives; five ``ごめん`` in
    thirty seconds do not.  Interjections of OTHER texts inside the window
    neither block nor count — ``ごめん ああ ごめん ああ ごめん`` is still a run.
    """
    removed: Counter[str] = Counter()
    if not density.enabled or not segments or density.min_count < 2:
        return list(segments), removed

    def is_inter(seg: SRTSegment) -> bool:
        t = (seg.text or "").strip()
        return bool(t) and _is_interjection(t, inter)

    doomed: set[int] = set()
    stretch: list[int] = []

    def flush() -> None:
        by_text: dict[str, list[int]] = defaultdict(list)
        for i in stretch:
            by_text[(segments[i].text or "").strip()].append(i)
        for text, idxs in by_text.items():
            n = len(idxs)
            if n < density.min_count:
                continue
            starts = [segments[i].start for i in idxs]
            for a in range(n):
                b = a
                while b + 1 < n and starts[b + 1] - starts[a] <= density.window_s:
                    b += 1
                if b - a + 1 >= density.min_count:
                    doomed.update(idxs[a:b + 1])
        stretch.clear()

    for i, seg in enumerate(segments):
        if is_inter(seg):
            stretch.append(i)
        else:
            flush()
    flush()

    out = []
    for i, seg in enumerate(segments):
        if i in doomed:
            removed[(seg.text or "").strip()] += 1
        else:
            out.append(seg)
    return out, removed


# ── Rule: subject stripping (post-translation, English target) ────────────

# Leading word, with an optional clitic that may arrive as an ASCII or curly
# apostrophe or as an HTML entity, sometimes space-separated ("I &apos;m").
_EN_LEAD = re.compile(
    r"^(?P<pre>[\s\"'“‘(\[-]*)"
    r"(?P<word>[A-Za-z]+(?:\s*(?:'|’|&apos;|&#39;)[a-z]+)?)"
    r"(?P<rest>.*)$",
    re.S,
)

#: If the remainder opens with one of these, the line is left alone: "Can
#: rest assured." / "Is characterized by…" / "Have chosen…" are not English.
#: Negative contractions ("don't know", "can't") and content verbs read fine
#: without a subject, so they are not here.
_EN_KEEP_IF_NEXT = frozenset(
    "am is are was were be been being have has had do does did "
    "can could will would shall should may might must "
    # Stative / discourse verbs that read as a different sentence without
    # their subject: "You know, like this." → "Know, like this."; "I know
    # a little…" → "Know a little…"; "You see," → "See,".
    "know knew knows mean meant means see saw bet suppose".split()
)


def strip_leading_subject(text: str, rules: SubjectRules) -> str | None:
    """Return *text* without its leading subject pronoun, or None if unchanged.

    "I'm sorry." → "Sorry."   "You did it!" → None (auxiliary next)
    "It's a huge one." → "A huge one."   "I'm Lena, 22." → None (a name)
    Only the first word is considered; the remainder must still contain a
    word, must not open with a bare auxiliary, and its first letter is
    capitalised.
    """
    m = _EN_LEAD.match(text or "")
    if not m:
        return None
    word = re.sub(r"\s+", "", m.group("word")).replace("’", "'")
    word = word.replace("&apos;", "'").replace("&#39;", "'").lower()
    if word not in rules.en_strip:
        return None
    rest = m.group("rest").lstrip(" ,")
    nxt = re.match(r"[A-Za-z]+(?:\s*(?:'|’|&apos;|&#39;)t\b)?", rest)
    if not nxt or nxt.group(0).lower() in _EN_KEEP_IF_NEXT:
        return None
    # A capitalised word or a number after the pronoun is a name, a
    # nationality, an age: "I'm Lena Kotama, 22." must keep its "I'm".
    # ("I" itself is capitalised but is never the second word here.)
    if rest[0].isupper() or rest[0].isdigit():
        return None
    rest = rest[0].upper() + rest[1:]
    return m.group("pre") + rest


def strip_subjects(
    source_texts: list[str], segments: list[SRTSegment], rules: SubjectRules
) -> tuple[list[SRTSegment], int, list[tuple[str, str]]]:
    """Apply :func:`strip_leading_subject` where the source line has no subject.

    *source_texts* are the pre-translation texts, positionally aligned with
    *segments* (translation is positional).  A segment whose text still
    equals its source (translation failed or was skipped) is left alone.
    """
    if not rules.enabled or len(source_texts) != len(segments):
        return list(segments), 0, []
    out, n, examples = [], 0, []
    for src, seg in zip(source_texts, segments):
        text = seg.text or ""
        src = (src or "").strip()
        if not src or text.strip() == src or any(w in src for w in rules.explicit_subjects):
            out.append(seg)
            continue
        new = strip_leading_subject(text, rules)
        if new is None:
            out.append(seg)
            continue
        n += 1
        if len(examples) < 8:
            examples.append((text.strip(), new.strip()))
        out.append(replace(seg, text=new))
    return out, n, examples


# ── Orchestration ─────────────────────────────────────────────────────────

def run_heuristics(
    segments: list[SRTSegment],
    heuristics: Heuristics | None = None,
    language: str | None = None,
) -> tuple[list[SRTSegment], HeuristicsStats]:
    """Run the pre-translation rules for *language*, in this order:

    1. ``drop`` and ``hallucination`` lists (whole-text match) — removed.
    2. ``density`` — dense runs of any interjection-class text removed.
    3. ``collapse`` — surviving runs of identical short cues merged.

    Density runs before collapse because collapse would hide the counts it
    needs; the lists run first so a string the user has already decided on
    never reaches the density heuristic.
    """
    h = heuristics or load_heuristics()
    rules = h.for_language(language)
    stats = HeuristicsStats(cues_in=len(segments), cues_out=len(segments))
    if not rules.enabled or not segments:
        return list(segments), stats

    kept: list[SRTSegment] = []
    listed: dict[str, Counter[str]] = {"drop": Counter(), "hallucination": Counter(),
                                       "junk": Counter()}
    for seg in segments:
        text = (seg.text or "").strip()
        if _is_junk(text):
            # U+FFFD: the decoder emitted bytes that are not text (seen as a
            # zero-duration loop of "1\ufffdgn" ×6).  Not a word in any
            # language; no list needed.
            listed["junk"][text] += 1
        elif _key(text) in rules.drop:
            listed["drop"][text] += 1
        elif _is_hallucination(text, rules):
            listed["hallucination"][text] += 1
        else:
            kept.append(seg)
    stats.dropped.update({k: v for k, v in listed.items() if v})

    kept, dense = drop_dense_runs(kept, rules, h.density())
    if dense:
        stats.dropped["density"] = dense

    # collapse_interjections re-indexes; its drop pass finds nothing left.
    result, st = collapse_interjections(kept, replace(rules, drop=frozenset()))
    stats.collapsed = st["collapsed"]
    stats.cues_out = len(result)

    if stats.dropped_total() or stats.collapsed:
        logger.info(
            "Heuristics [%s]: %d cues → %d (%d collapsed; dropped: %d listed, "
            "%d hallucination, %d dense, %d junk)%s",
            rules.language or "global", stats.cues_in, stats.cues_out, stats.collapsed,
            stats.dropped_total("drop"), stats.dropped_total("hallucination"),
            stats.dropped_total("density"), stats.dropped_total("junk"),
            f" [{h.source_path.name}]" if h.source_path else " [built-in rules]",
        )
    return result, stats


def apply_heuristics(
    segments: list[SRTSegment],
    heuristics: Heuristics | None = None,
    language: str | None = None,
) -> list[SRTSegment]:
    """Segments only; see :func:`run_heuristics` for the stats."""
    return run_heuristics(segments, heuristics, language)[0]


# ── Run report: what the rules acted on, in the user's language ───────────
#
# The rules match SOURCE-language text, which a user who cannot read that
# script has no way to write.  The report closes that gap: for every short
# string the model produced it shows the count, the share of the file, what
# it translates to, and whether the current rules dropped it — enough to
# decide "keep" or "drop" without reading a character of the source.
#
# Mechanics only here.  The GUI panel that renders it and ticks strings into
# the drop list lands with the settings revamp.

@dataclass(frozen=True)
class ReportRow:
    source: str
    count: int
    share: float                 # fraction of all raw cues
    translation: str | None      # most common translation seen, if any
    dropped: int                 # occurrences removed by the current rules
    collapsed: int               # cues merged into a neighbour
    reason: str | None = None    # "drop" | "hallucination" | "density"


@dataclass(frozen=True)
class HeuristicsReport:
    language: str | None
    raw_cues: int
    final_cues: int
    rows: list[ReportRow]
    source_path: Path | None
    subject_stripped: int = 0
    subject_examples: list[tuple[str, str]] = field(default_factory=list)
    #: Full lines (any length) the model produced repeatedly.  The short-
    #: string table cannot see "다음 영상에서 만나요" ×37; this can.
    repeats: list[ReportRow] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "raw_cues": self.raw_cues,
            "final_cues": self.final_cues,
            "rules_file": str(self.source_path) if self.source_path else None,
            "rows": [r.__dict__ for r in self.rows],
            "subject_stripped": self.subject_stripped,
            "subject_examples": [list(p) for p in self.subject_examples],
            "repeats": [r.__dict__ for r in self.repeats],
        }

    def write(self, directory: Path, stem: str) -> tuple[Path, Path]:
        """Write ``<stem>.heuristics.txt`` (all rows) and ``.json`` to *directory*."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        txt = directory / f"{stem}.heuristics.txt"
        js = directory / f"{stem}.heuristics.json"
        txt.write_text(self.as_text(limit=len(self.rows) or 1) + "\n", encoding="utf-8")
        js.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
        return txt, js

    def as_text(self, limit: int = 15) -> str:
        head = (f"Heuristics report ({self.language or 'global'} rules"
                f"{', ' + self.source_path.name if self.source_path else ', built-in'}): "
                f"{self.raw_cues} raw cues → {self.final_cues}\n")
        lines = [head, f"  {'source':<12}{'cues':>6}{'share':>7}   {'means':<24} action"]
        for r in self.rows[:limit]:
            if r.dropped:
                why = {"drop": "drop list", "hallucination": "hallucination",
                       "density": "dense run", "junk": "junk"}.get(r.reason or "", r.reason or "")
                action = f"DROPPED {r.dropped}/{r.count} ({why})" if r.dropped < r.count \
                    else f"DROPPED ({why})"
            elif r.collapsed:
                action = f"collapsed x{r.collapsed}"
            else:
                action = "kept"
            lines.append(f"  {r.source:<12}{r.count:>6}{100 * r.share:>6.1f}%   "
                         f"{(r.translation or '?')[:24]:<24} {action}")
        if self.repeats:
            lines.append("\n  Repeated lines, any length (a sign-off or credit that recurs "
                         "over silence is a hallucination — add a piece of it to "
                         "hallucination_contains):")
            for r in self.repeats[:limit]:
                why = {"drop": "drop list", "hallucination": "hallucination",
                       "density": "dense run", "junk": "junk"}.get(r.reason or "", "")
                action = f"DROPPED ({why})" if r.dropped else "kept"
                lines.append(f"  {r.count:>4}x  {r.source[:28]:<28}  {(r.translation or '?')[:30]:<30} {action}")
        if self.subject_stripped:
            lines.append(f"\n  Subject pronouns stripped (source names no subject): "
                         f"{self.subject_stripped}")
            for before, after in self.subject_examples[:5]:
                lines.append(f"    {before[:34]!r:<36} → {after[:30]!r}")
        return "\n".join(lines)


def build_report(
    raw_segments: list[SRTSegment],
    final_segments: list[SRTSegment],
    rules: InterjectionRules,
    *,
    max_chars: int = 4,
    translate=None,
    source_path: Path | None = None,
    stats: HeuristicsStats | None = None,
) -> HeuristicsReport:
    """Pair the model's raw short strings with what they became.

    Args:
        raw_segments:   The model's output BEFORE heuristics ran.
        final_segments: The segments AFTER heuristics and translation.
        rules:          The rules that were applied.
        max_chars:      Report strings up to this length (one wider than the
                        collapse threshold, so borderline words show too).
        translate:      Optional ``(list[str]) -> list[str]``.  Strings the
                        rules DROPPED never reached the translator, so their
                        meaning is unknown unless fetched here; it is a
                        handful of short strings, so it is cheap.
        stats:          From :func:`run_heuristics`; gives per-string removal
                        counts and reasons.  Without it, only the ``drop`` /
                        ``hallucination`` list membership is known.
    """
    counts: Counter[str] = Counter()
    for s in raw_segments:
        t = (s.text or "").strip()
        if t and len(t) <= max_chars:
            counts[t] += 1
    # Translation seen for each source string, matched on start time.
    raw_at = {round(s.start, 1): (s.text or "").strip() for s in raw_segments}
    # Repeated full lines, any length, three or more times.
    full: Counter[str] = Counter((s.text or "").strip() for s in raw_segments)
    repeat_rows: list[ReportRow] = []
    for t, n in full.most_common():
        if n < 3 or not t or len(t) <= max_chars:
            continue
        if stats is not None:
            reason = stats.reason_for(t)
            dropped = stats.dropped[reason].get(t, 0) if reason else 0
        else:
            reason = "hallucination" if _is_hallucination(t, rules) else None
            dropped = n if reason else 0
        tr = None
        for s in final_segments:
            src = raw_at.get(round(s.start, 1))
            if src == t and (s.text or "").strip() != t:
                tr = (s.text or "").strip()
                break
        if tr is None and translate:
            try:
                tr = translate([t])[0]
            except Exception:
                tr = None
        repeat_rows.append(ReportRow(t, n, n / max(1, len(raw_segments)), tr, dropped, 0, reason))

    n_subj = stats.subject_stripped if stats else 0
    subj_ex = list(stats.subject_examples) if stats else []
    if not counts:
        return HeuristicsReport(rules.language, len(raw_segments), len(final_segments),
                                [], source_path, n_subj, subj_ex, repeat_rows)

    # Translation seen for each source string, matched on start time.
    seen: dict[str, Counter[str]] = defaultdict(Counter)
    for s in final_segments:
        src = raw_at.get(round(s.start, 1))
        if src in counts and (s.text or "").strip() and (s.text or "").strip() != src:
            seen[src][(s.text or "").strip()] += 1

    # Collapsed counts: raw occurrences minus final occurrences of the source
    # text (approximate but honest — collapse keeps one per run).
    final_src_counts: Counter[str] = Counter()
    for s in final_segments:
        src = raw_at.get(round(s.start, 1))
        if src in counts:
            final_src_counts[src] += 1

    missing = [t for t in counts if t not in seen]
    if translate and missing:
        try:
            for src_text, tr in zip(missing, translate(missing)):
                if tr and tr.strip():
                    seen[src_text][tr.strip()] += 1
        except Exception as exc:          # a report must never fail the run
            logger.debug("Report translation skipped: %s", exc)

    total = max(1, len(raw_segments))
    rows = []
    for t, n in counts.most_common():
        if stats is not None:
            reason = stats.reason_for(t)
            dropped = stats.dropped[reason].get(t, 0) if reason else 0
        else:
            reason = ("drop" if t in rules.drop
                      else "hallucination" if t in rules.hallucination else None)
            dropped = n if reason else 0
        collapsed = max(0, n - dropped - final_src_counts.get(t, 0))
        best = seen[t].most_common(1)[0][0] if seen.get(t) else None
        rows.append(ReportRow(t, n, n / total, best, dropped, collapsed, reason))
    return HeuristicsReport(rules.language, len(raw_segments), len(final_segments),
                            rows, source_path, n_subj, subj_ex, repeat_rows)
