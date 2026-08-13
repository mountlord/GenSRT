#!/usr/bin/env python3
"""Evaluate candidate cleanup rules against hand-labelled cues.

Drop B.  Reads a segment dump (``--dump-segments``) plus a list of cue numbers
a human judged spurious, and scores candidate rules against that judgement.

Why bother
----------
The v1.3 cleanup feature was designed around "cues with exactly 1.00s
duration".  That value turned out to be produced by GenSRT's own minimum
duration floor, not by the model — so the rule was measuring an artifact of
its own pipeline.  The lesson is not "pick a better constant"; it is that a
rule nobody measured is a rule nobody can defend.

This script exists so the thresholds that ship are ones we can show working,
on real broadcast audio, against a native reader's judgement.

The two numbers it reports
--------------------------
**Precision** — of the cues this rule would remove, what fraction were
actually spurious?  Low precision means the rule destroys real subtitles.
This is the number to protect: a viewer notices a missing sentence far more
than a surviving fragment.

**Recall** — of the spurious cues, what fraction does this rule catch?  Low
recall means fragments survive into the output.  Annoying, but recoverable —
the user can still delete them by hand.

They trade off, so both are reported along with F-beta.  Beta below 1 weights
precision above recall; the default of 0.5 encodes "deleting real speech is
about four times worse than leaving a fragment in", which suits a tool whose
stated philosophy is to show artifacts rather than hide them.

Usage
-----
    gensrt --input clip.mp4 --model <model> --dump-segments ./eval

    python -m gensrt.evaluate_cleanup \\
        --segments ./eval/clip.segments.csv \\
        --labels   ./eval/clip.flagged.txt

The labels file is one spurious cue number per line.  Blank lines and lines
beginning with ``#`` are ignored, so it can carry notes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

# Weights precision over recall.  See the module docstring.
DEFAULT_BETA = 0.5


@dataclass
class Row:
    """One segment from the dump, with its numeric fields already parsed."""

    index: int
    start_s: float
    duration_s: float
    avg_logprob: float | None
    compression_ratio: float | None
    no_speech_prob: float | None
    temperature: float | None
    is_chunk_tail: bool
    is_chunk_sole: bool
    chars: int
    text: str


def load_segments(path: Path) -> list[Row]:
    def f(value):
        value = (value or "").strip()
        return float(value) if value else None

    rows: list[Row] = []
    with Path(path).open(encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            rows.append(Row(
                index=int(raw["index"]),
                start_s=float(raw["start_s"] or 0.0),
                duration_s=float(raw["duration_s"] or 0.0),
                avg_logprob=f(raw.get("avg_logprob")),
                compression_ratio=f(raw.get("compression_ratio")),
                no_speech_prob=f(raw.get("no_speech_prob")),
                temperature=f(raw.get("temperature")),
                is_chunk_tail=(raw.get("is_chunk_tail") or "0") == "1",
                is_chunk_sole=(raw.get("is_chunk_sole") or "0") == "1",
                chars=int(raw.get("chars") or 0),
                text=raw.get("text", ""),
            ))
    return rows


def transfer_labels_from_srt(
    old_srt: Path,
    flagged_indices: set[int],
    rows: list[Row],
    tolerance_s: float = 0.75,
) -> tuple[set[int], list[int]]:
    """Carry hand-labelled cue numbers from an older SRT onto a fresh dump.

    Cue numbers are not stable across runs.  Two transcriptions of the same
    file with the same model and settings can differ — a repeat run of one
    clip produced 49 cues once and 50 the next time — and a single extra cue
    early in the file shifts every index after it.  So a list of flagged cue
    numbers made against last month's SRT cannot be trusted against today's
    dump.

    Rather than making you re-label, this matches on content: a flagged cue
    from the old SRT is located in the new dump by identical text within
    *tolerance_s* of the same timestamp.  Text equality is required because
    the fragments are short and repetitive — several cues may read
    ``ആരോപിച്ചു`` — so time alone would mismatch, and text alone would be
    ambiguous.

    Args:
        old_srt:         The SRT the labelling was done against.
        flagged_indices: Cue numbers flagged in that file.
        rows:            Segments from the current dump.
        tolerance_s:     Timestamp slack.

    Returns:
        ``(matched_dump_indices, unmatched_old_indices)``.  Anything
        unmatched is reported rather than dropped silently — a low match rate
        means the two runs diverged too much to compare and you should
        re-label instead.
    """
    try:
        import srt as srt_lib
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("The 'srt' package is required for --labels-srt") from exc

    subtitles = list(srt_lib.parse(Path(old_srt).read_text(encoding="utf-8-sig")))
    by_index = {sub.index: sub for sub in subtitles}

    matched: set[int] = set()
    unmatched: list[int] = []

    for old_index in sorted(flagged_indices):
        sub = by_index.get(old_index)
        if sub is None:
            unmatched.append(old_index)
            continue

        # SRT cue text may be wrapped across lines; the dump is unwrapped.
        wanted = " ".join(sub.content.split())
        start_s = sub.start.total_seconds()

        candidates = [
            r for r in rows
            if " ".join(r.text.split()) == wanted
            and abs(r.start_s - start_s) <= tolerance_s
            and r.index not in matched
        ]
        if candidates:
            best = min(candidates, key=lambda r: abs(r.start_s - start_s))
            matched.add(best.index)
        else:
            unmatched.append(old_index)

    return matched, unmatched


def load_labels(path: Path) -> set[int]:
    labels: set[int] = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.replace(",", " ").split():
            labels.add(int(part))
    return labels


@dataclass
class Score:
    name: str
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def fbeta(self, beta: float = DEFAULT_BETA) -> float:
        p, r = self.precision, self.recall
        if p == 0 and r == 0:
            return 0.0
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r)


def score_rule(name, rule, rows: list[Row], labels: set[int]) -> Score:
    tp = fp = fn = tn = 0
    for row in rows:
        flagged = bool(rule(row))
        spurious = row.index in labels
        if flagged and spurious:
            tp += 1
        elif flagged and not spurious:
            fp += 1
        elif not flagged and spurious:
            fn += 1
        else:
            tn += 1
    return Score(name, tp, fp, fn, tn)


def candidate_rules(rows: list[Row]) -> list[tuple[str, object]]:
    """Rules to score, from the crudest baseline upward.

    Ordering matters for reading the output: each family adds one signal, so
    the gain from adding it is visible as the difference between adjacent
    blocks.  A rule that does not beat the duration baseline is not worth the
    code it takes to implement.
    """
    rules: list[tuple[str, object]] = []

    # Baseline: duration alone.  This is what the original v1.3 design used,
    # via the "exactly 1.00s" proxy.  Everything else has to beat it.
    for t in (0.35, 0.5, 0.75, 1.0, 1.25, 1.6):
        rules.append((f"duration < {t}s", lambda r, t=t: r.duration_s < t))

    # Confidence alone.  avg_logprob is per-token mean log-probability, so it
    # is length-normalised and directly comparable across cues.
    for t in (-0.8, -1.0, -1.2, -1.5):
        rules.append((f"logprob < {t}", lambda r, t=t: (
            r.avg_logprob is not None and r.avg_logprob < t)))

    # The WhisperJAV gate, sign-corrected: short segments judged MORE harshly,
    # not less.  Their code subtracts the margin, which makes short segments
    # harder to drop; that suits their domain and is backwards for ours.
    for base in (-0.8, -1.0):
        for margin in (0.3, 0.5):
            for window in (1.0, 1.6):
                rules.append((
                    f"logprob < {base}{margin:+.1f} if dur<={window}s",
                    lambda r, b=base, m=margin, w=window: (
                        r.avg_logprob is not None
                        and r.avg_logprob < (b + m if r.duration_s <= w else b)
                    ),
                ))

    # Chunk position — a structural signal a general-purpose cleaner does not
    # have.  If the fragments really are chunk-tail artifacts, this should
    # show up as a large precision gain over duration alone.
    for t in (0.75, 1.0):
        rules.append((f"tail & duration < {t}s", lambda r, t=t: (
            r.is_chunk_tail and not r.is_chunk_sole and r.duration_s < t)))
    rules.append(("tail & logprob < -1.0", lambda r: (
        r.is_chunk_tail and not r.is_chunk_sole
        and r.avg_logprob is not None and r.avg_logprob < -1.0)))

    # Combined.
    for t in (0.75, 1.0):
        rules.append((f"tail & dur<{t}s & logprob<-0.8", lambda r, t=t: (
            r.is_chunk_tail and not r.is_chunk_sole and r.duration_s < t
            and r.avg_logprob is not None and r.avg_logprob < -0.8)))

    # Repetition and non-zero temperature, as secondary signals.
    rules.append(("compression_ratio > 2.4", lambda r: (
        r.compression_ratio is not None and r.compression_ratio > 2.4)))
    rules.append(("temperature > 0", lambda r: (
        r.temperature is not None and r.temperature > 0.0)))

    return rules


def describe_corpus(rows: list[Row], labels: set[int]) -> None:
    print(f"segments        : {len(rows)}")
    print(f"labelled spurious: {len(labels & {r.index for r in rows})}")

    missing = labels - {r.index for r in rows}
    if missing:
        print(f"WARNING: {len(missing)} labelled cue(s) not in the dump: "
              f"{sorted(missing)[:10]}")

    with_lp = [r for r in rows if r.avg_logprob is not None]
    print(f"with avg_logprob : {len(with_lp)}/{len(rows)}")
    if not with_lp:
        print("\nNo decoder metrics in this dump — only duration rules can be "
              "scored. Re-run the transcription with a faster-whisper version "
              "that exposes avg_logprob.")

    spurious = [r for r in rows if r.index in labels]
    genuine = [r for r in rows if r.index not in labels]

    def summarize(label, group, attr):
        vals = sorted(
            v for v in (getattr(r, attr) for r in group) if v is not None
        )
        if not vals:
            return
        print(f"  {label:<10} n={len(vals):<4} "
              f"min={vals[0]:>7.3f}  median={vals[len(vals)//2]:>7.3f}  "
              f"max={vals[-1]:>7.3f}")

    for attr in ("duration_s", "avg_logprob", "compression_ratio"):
        print(f"\n{attr}:")
        summarize("spurious", spurious, attr)
        summarize("genuine", genuine, attr)

    if spurious:
        tails = sum(1 for r in spurious if r.is_chunk_tail and not r.is_chunk_sole)
        print(f"\nchunk-tail hypothesis: {tails}/{len(spurious)} spurious cues "
              f"are non-sole chunk tails")
        g_tails = sum(1 for r in genuine if r.is_chunk_tail and not r.is_chunk_sole)
        print(f"                       {g_tails}/{len(genuine)} genuine cues "
              f"are too (the false-positive pool)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Score cleanup rules against hand-labelled cues.")
    ap.add_argument("--segments", required=True, type=Path,
                    help="CSV from gensrt --dump-segments")
    ap.add_argument("--labels", required=True, type=Path,
                    help="One spurious cue number per line")
    ap.add_argument("--labels-srt", type=Path, default=None,
                    help="The SRT those cue numbers were labelled against. "
                         "Give this when the labels come from an older run: "
                         "cue numbers are not stable between runs, so labels "
                         "are matched onto the current dump by text and "
                         "timestamp instead of by number.")
    ap.add_argument("--match-tolerance", type=float, default=0.75,
                    help="Timestamp slack for --labels-srt (default 0.75s)")
    ap.add_argument("--beta", type=float, default=DEFAULT_BETA,
                    help=f"F-beta weight; <1 favours precision "
                         f"(default {DEFAULT_BETA})")
    ap.add_argument("--show-errors", action="store_true",
                    help="List the mistakes made by the best rule")
    args = ap.parse_args(argv)

    rows = load_segments(args.segments)
    labels = load_labels(args.labels)

    if args.labels_srt:
        labels, unmatched = transfer_labels_from_srt(
            args.labels_srt, labels, rows, args.match_tolerance
        )
        print(f"Transferred {len(labels)} label(s) from {args.labels_srt.name} "
              f"by text + timestamp.")
        if unmatched:
            print(f"WARNING: {len(unmatched)} flagged cue(s) had no match in "
                  f"the current dump: {unmatched[:15]}")
            print("         A low match rate means the two runs diverged too "
                  "much to compare; re-label against the current run instead.")
        print()

    if not rows:
        print("No segments in dump.", file=sys.stderr)
        return 1
    if not labels:
        print("No labels provided.", file=sys.stderr)
        return 1

    print("=" * 78)
    print("  Corpus")
    print("=" * 78)
    describe_corpus(rows, labels)

    scores = [score_rule(n, r, rows, labels) for n, r in candidate_rules(rows)]
    scores.sort(key=lambda s: s.fbeta(args.beta), reverse=True)

    print()
    print("=" * 78)
    print(f"  Rules, ranked by F{args.beta} (precision-weighted)")
    print("=" * 78)
    print(f"{'rule':<38} {'prec':>6} {'recall':>7} "
          f"{'F' + str(args.beta):>6} {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 78)
    for s in scores:
        print(f"{s.name:<38} {s.precision:>6.2f} {s.recall:>7.2f} "
              f"{s.fbeta(args.beta):>6.2f} {s.tp:>4} {s.fp:>4} {s.fn:>4}")

    if args.show_errors and scores:
        best = scores[0]
        rule = dict(candidate_rules(rows))[best.name]
        print()
        print("=" * 78)
        print(f"  Mistakes made by: {best.name}")
        print("=" * 78)
        print("\nFALSE POSITIVES (real subtitles this rule would delete):")
        for r in rows:
            if rule(r) and r.index not in labels:
                print(f"  #{r.index:<4} {r.duration_s:>5.2f}s  "
                      f"lp={r.avg_logprob}  {r.text[:50]}")
        print("\nFALSE NEGATIVES (spurious cues it would miss):")
        for r in rows:
            if not rule(r) and r.index in labels:
                print(f"  #{r.index:<4} {r.duration_s:>5.2f}s  "
                      f"lp={r.avg_logprob}  {r.text[:50]}")

    print()
    print("Read precision first. A rule that deletes real speech is worse "
          "than one that leaves fragments in.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
