# GenSRT Investigations

This document captures investigation arcs — open questions, hypotheses,
evaluations, and resolutions — for technical decisions in GenSRT.  It's
the long-form companion to the release plan (`V12_PLAN.md`) and the
public-facing README.

Investigations are numbered with an issue prefix (`I-`) followed by a
sequential identifier.  Closed leads (evaluated but not pursued) get
`CL-`.  Each entry has a status: `OPEN`, `RESOLVED SUCCESS`,
`RESOLVED FAILURE`, `DEFERRED`, or `CLOSED LEAD`.

---

## v1.2 investigations

### I-1: wav2vec2 forced alignment for vegam timestamps

**Status:** `RESOLVED FAILURE`
**Closed:** v1.2 development (mid-cycle)

**Hypothesis:** vegam transcribes Malayalam audio fluently but with poor
word-level timing.  A wav2vec2-based forced-alignment pass could produce
accurate timestamps over vegam's text output, yielding a best-of-both
pipeline.

**Method:** Loaded `gvs/wav2vec2-large-xlsr-malayalam` via torchaudio
2.10.0+cu128 in the GenSRT venv.  Used
`torchaudio.functional.forced_align` to align vegam's transcribed text
against the source audio.

**Result:** The alignment mechanically succeeded — wav2vec2 produced
timestamps for every word vegam emitted.  But native-reader verification
revealed the deeper problem: vegam's text on long-form audio is partially
fabricated (the dropped-speech failure mode that became I-2).  Aligning
fabricated text to real audio produces accurate timestamps for inaccurate
content.  The technique works only when the upstream transcription is
correct.

**Lesson:** Alignment can't rescue a corrupted upstream transcription.
The right problem to attack was vegam's dropping behavior itself, not
its timestamps.  This led directly to I-2.

---

### I-2: vegam chunked inference

**Status:** `RESOLVED SUCCESS`
**Closed:** v1.2 development (late cycle)
**Ships in:** v1.2 (Drop I.15)

> **Correction (v1.2.1):** the training-distribution hypothesis below was
> found to be mechanistically incorrect.  The chunking *method* and *results*
> in this entry stand and ship unchanged — but the actual root cause of the
> cutoff is a CTranslate2 decode-length cap, not vegam's training corpus.  See
> the correction at the end of this entry and **I-7** for the real mechanism.

**Hypothesis:** vegam was fine-tuned on Mozilla Common Voice 11.0
Malayalam, a corpus of 5-15 second isolated phrases.  When called on
long-form audio, vegam produces ~6-8 seconds of transcription per
inference call regardless of input length, then emits an effective
"done" token and silently drops the rest.  Slicing audio into chunks
that match vegam's training distribution should eliminate the drops.

**Method:** Multi-phase investigation.

1. **Confirm the failure mode.**  13-config parameter sweep
   (`no_speech_threshold`, `logprob_threshold`, `suppress_eos`,
   `beam_size`, `condition_on_previous_text`, etc.) on a 197-second
   Malayalam news clip — all 13 configurations produced byte-identical
   output, confirming the issue is in vegam's learned output
   distribution, not in any tunable inference parameter.

2. **Develop the chunking algorithm.**  Iterated on `chunked_vegam_*`
   test scripts:
   - First attempt: fixed-size chunks (5s, 6s, 8s, 10s).  Worked but
     produced mid-character truncation (the `�` issue) at ~19% of
     chunks on the 6s configuration.
   - Refined attempt: silent-boundary chunking with progressive VAD
     sweep.  Tier 0 (default thresholds) → Tier 1 (moderate strictness)
     → Tier 2 (aggressive 0.8 threshold).  Prefer reliable tiers when
     available; fall back to energy-min RMS detection only when no
     detected silence exists in the lookahead range.

3. **Validate qualitatively.**  Two Malayalam news clips with native-
   reader review.  First clip (FIFA World Cup news, 197s): 49 cues,
   86% silence cuts, 8.2% mid-character truncation, "very high
   accuracy, almost perfect timing."  Second clip — MalayalamNews-2.mp4 (274.0s):
   66 cues, similar density, "beats realtime closed captioning by a
   mile."

**Result:** Algorithm proven on Malayalam news audio.  Integrated into
GenSRT as `gensrt/asr/_silence_chunking.py` plus the
`MonolingualWhisperEngine` in `gensrt/asr/monolingual_whisper.py`.
Auto-engaged for vegam variants and any custom Whisper model added
via the GUI.

**Remaining limitations (documented, not addressed in v1.2):**

- **Category B — hallucinated repetition.**  vegam occasionally
  emits a phrase from earlier in the audio at a chunk tail.
  Detectable as tail-substring overlap with the previous cue's
  text.  Cleanup deferred to v1.3 (see future work below).
- **Category C — genuine mid-content drops.**  When vegam runs out
  of content to transcribe within a chunk before the chunk ends, no
  amount of chunking helps.  Comparable to live broadcast closed
  captioning's known behavior on fast/dense speech.

**Correction — actual root cause.**  The hypothesis that opened this
investigation (Common-Voice short-phrase training → learned "stop after a
phrase" behavior) is wrong.  The cutoff is a structural decode-length cap in
CTranslate2, diagnosed by Kavya Manohar after v1.2 shipped and documented in
full under **I-7**.  Notably, the 13-config byte-identical result in Method
step 1 above was already evidence *for* a structural cap (a learned behavior
could be nudged by sampling parameters; a hard length cap cannot) — we
mis-read it at the time.  The chunking pipeline is the correct fix either
way; only the explanation changed.

---

### I-3a: Malayalam-monolingual IndicConformer

**Status:** `DEFERRED`
**Closed:** v1.2 development (late cycle)

**Hypothesis:** AI4Bharat's IndicConformer family includes both a
600m-multilingual variant (tested earlier in the project) and
language-specific monolingual variants.  A Malayalam-monolingual
IndicConformer might outperform vegam-chunked on dense speech without
the phrase-distribution constraint that motivated chunked inference.

**Method preparation:** Standalone test script
(`monolingual_indicconformer_test.py`) written.

**Decision:** Deferred without running.  The chunked-vegam pipeline
proved sufficient for v1.2's quality bar (native-reader verdict:
"beats realtime CC by a mile").  Adding IndicConformer requires the
NeMo dependency chain (separate venv discipline; torchcodec conflict
issues we worked around in I-1).  The cost-benefit favors shipping
v1.2 with chunked vegam now over deferring release for a marginal
quality improvement.

**Re-open trigger:** If forum users report quality complaints on
content classes where chunked vegam struggles (very fast speech,
heavy code-switching), revisit IndicConformer as an alternative
engine routed through the existing ASR factory.

---

## v1.2.1 investigations

### I-7: CTranslate2 decode-length cap as the root cause of fine-tune long-form cutoff

**Status:** `RESOLVED SUCCESS` (root cause) — one `OPEN` sub-question (R-MFT 1.00s emission rate)
**Closed:** v1.2.1 development
**Relationship to I-2:** supersedes I-2's *causal hypothesis* only; I-2's
chunking method and results are unaffected.

**Background.**  I-2 shipped a working chunking pipeline on the hypothesis
that vegam's ~7-second cutoff came from its Common Voice training
distribution.  The pipeline works; the mechanism was wrong.

**Correct root cause (diagnosis: Kavya Manohar, vegam maintainer, Adalat
AI).**  The cutoff is structural to the inference stack, not the training
data:

- CTranslate2's Whisper decoder caps the total token sequence at 448
  tokens.  With Whisper's timestamp and language/task tokens interleaved
  into that budget, the effective text-token allowance is roughly half —
  about 224 tokens (`max_length = min(total_max_length / 2,
  total_max_length − prompt_length)`).
- Whisper's BPE tokenizer is highly inefficient on Indic scripts: a single
  Malayalam grapheme can expand to several tokens.  Dense Malayalam
  consumes the ~224-token text budget in roughly 7 seconds of speech, at
  which point the decoder reaches its length limit and stops — producing
  the "transcribe a few seconds, silently drop the rest" behavior.

This accounts for an I-2 observation the original hypothesis didn't explain
cleanly: the 13-config parameter sweep produced byte-identical output.  A
learned "stop after a phrase" behavior could in principle be nudged by
sampling parameters; a hard decode-length cap cannot — which is exactly what
byte-identical output across 13 configs demonstrates.

**Why chunking is the correct fix (under the right model).**  Each chunk is
an independent inference call with a fresh 224-token budget.  Keeping every
silent-boundary chunk under ~7 seconds keeps it under the cap, so no chunk
hits the length limit mid-content.  Same fix; the reason it works is the
per-call decode-length reset, not a match to training-phrase length.

**GenSRT's empirical contribution (the citable part).**
- Characterization of the cutoff on real broadcast content (not synthetic
  or benchmark audio): the ~7-second boundary, the mid-character truncation
  (`�`) rate, and cue-density figures from the I-2 corpus (two Malayalam
  news clips, native-reader validated).
- The 13-config byte-identical parameter sweep establishing that the cutoff
  is not inference-parameter-tunable — the evidence distinguishing a
  structural cap from a learned behavior.
- The silent-boundary chunking workaround
  (`gensrt/asr/_silence_chunking.py`).

**Subsequent observations — adalat-ai R-MFT (v1.2.1 recommended Malayalam
model).**  Adalat AI's `whisper-medium-ml-rmft`, converted to CT2 and added
in v1.2.1, sits under the same 224-token cap but stops more cleanly at the
boundary.  Figures below are *Adalat AI's measurements using GenSRT* on the
197s clip — attributed as theirs pending independent reproduction on the
GenSRT side before being cited as GenSRT's own:
- Mid-character truncation (`�`) rate ~10% (vegam) → ~2.6% (R-MFT).
- Runtime ~296s (vegam) → ~154s (R-MFT) on an RTX 3060 Ti, same clip and
  chunking plan.
- Quality markedly better than vegam on English code-switching, entity
  names, and place names (native-reader comparison).

**RESOLVED sub-question — short chunk-tail fragments.**  Characterised in
full on 2026-08-10; the earlier "roughly a third of cues, exactly 1.00s"
description was partly a measurement artifact and is superseded below.

*Corpus:* MalayalamNews.mp4 (197.3s, 43 chunks) and MalayalamNews-2.mp4
(274.0s, 58 chunks), both Asianet Malayalam news.  Six R-MFT runs across two
machines (RTX 3060 Ti, RTX 3060 Laptop), chunk plan identical in every run.

**Rate.**  40.4%–43.4% of emitted cues are sub-second, stable across clips,
machines and repeat runs.  vegam on the same clips: 9.1%–10.6%.  The ratio is
therefore ~4:1, close to the ~5:1 previously estimated.

**True durations.**  min 0.020s, **median 0.060s**, max 0.620s.  Not ~1.00s.
The earlier figure was produced by GenSRT's own `min_subtitle_duration_s: 1`
floor rewriting every shorter cue to exactly `start + 1.000s`; the observation
had been made on written SRT files, downstream of that floor.  This is a
correction of roughly 17x in magnitude and it changes what the artifact *is*:
at 60ms these are not short utterances but timestamp emissions with a
collapsed span.

**Physical implausibility.**  Fragments carry a median 74–88 characters per
second against 14–16 for the same models' normal cues.  ആയിരുന്നു (9
characters, 4–5 syllables) appears with a 0.020s span — roughly 20x faster
than Malayalam is spoken.  No human utterance is being transcribed here.

**Bimodality.**  Across 240 cues on two clips, nothing falls between 0.62s and
1.25s.  Any threshold in that band separates the two populations completely.

**Structure — the mechanism.**  Chunk shapes over 101 chunks:

| shape | chunks |
|---|---|
| (N) | 44 |
| (N, F) | 36 |
| (N, F, F) | 19 |
| (N, F, F, F) | 1 |
| (N, N, F) | 1 |

Every chunk begins with exactly one normal cue.  Fragments only ever trail.
No chunk starts with a fragment, and only once in 101 chunks does normal
speech follow one.  `is_chunk_tail` fires on **zero of 220 normal cues**
across three runs and two clips.  The chunk-tail framing is confirmed
structurally, not inferred from duration.

**Not caused by chunk boundaries.**  Native-reader listening to the exported
chunk audio confirmed every boundary falls in real silence with no word
sliced, on both clips.  The most obvious competing explanation is eliminated.

**All spurious.**  Native-reader adjudication of the full fragment population
on MalayalamNews-2 (43 cues) and of the one contested case on MalayalamNews
(മത്സരം) found every fragment spurious.  In each contested case the fragment's
text already appeared in its own chunk's first cue — the decoder re-emitting a
word it had just transcribed.

**Reproducibility, and what it points to.**  Two identical R-MFT runs on GPU
(`float16`) differ by one cue.  All 59 real cues are byte-identical between
runs; every difference is inside the fragment population.  Transcription is
reproducible, the fragments are not.  vegam by contrast differs in 8 real cues
between identical runs, which follows from its temperature-fallback behaviour
(see I-9).

On **CPU with `int8`**, two runs of MalayalamNews-2 were byte-identical —
101 cues, 42 floored, 41 clamped, and all nine duration-histogram buckets
matching exactly.  So the residual GPU variation is precision-dependent, not
inherent to the model: it is consistent with cuBLAS kernel selection under
`float16`, which integer arithmetic does not have the same freedom to vary.
This also gives a practical answer for anyone needing reproducible output —
run `int8`, on CPU if necessary — at roughly 15x the wall time (2448.8s vs
162.2s on comparable hardware).

`int8` and `float16` produce the same cue structure and near-identical text.
Where they differ, it is in passages the model is least confident about: on
this corpus, English sports commentary that a Malayalam-only fine-tune renders
phonetically.  Both precisions are wrong there, differently.  The precision
does not cause those errors; it moves them.

**Detection.**  Duration alone separates the populations at 100% precision and
100% recall on both clips against native-reader labels.  `avg_logprob`
performs worse (4 false positives on clip 2).  `compression_ratio` is
*inverted* for this artifact — median 0.79 for fragments against 1.70 for
normal cues, because gzip does not compress a nine-character string — so
repetition-based detectors do not apply.  `is_chunk_tail` gives 100% precision
at 70–76% recall and is best used as corroboration rather than as the primary
signal.

*Remaining open:* whether decode parameters influence the emission rate.  This
is now a lower priority: duration filtering resolves the practical problem at
100% precision, so the parameter sweep would refine understanding rather than
unblock the cleanup feature.  Any such sweep must run against raw
`SRTSegment` output or with `min_subtitle_duration_s: 0`, and must establish a
noise floor first — repeat runs vary by ~0.5–1.6 percentage points.

*Instrumentation added (v1.2.2–v1.2.5), so this stays measurable:*
`run_pipeline` logs the raw duration histogram before post-processing;
`build_srt` logs whenever the floor fires and what the true durations were;
`SRTSegment` carries `avg_logprob`, `compression_ratio`, `no_speech_prob`,
`temperature` and chunk provenance; `--dump-segments` writes the raw segment
table to CSV; `--debug-chunks` exports per-chunk audio and telemetry.

*Contribution split:* root-cause diagnosis of the underlying decode-length cap
credited to Kavya Manohar / Adalat AI.  The empirical characterisation above —
rate, true durations, chunk-shape structure, detection evaluation and the
measurement correction — is GenSRT's.

**Citation note.**  Adalat AI has asked to cite GenSRT's investigation of the
token-limit issue in a forthcoming technical report; GenSRT reciprocates.
Contribution split for citation: root-cause diagnosis — Kavya Manohar /
Adalat AI; empirical characterization on real broadcast content and the
chunking workaround — GenSRT.

---

### Closed leads (evaluated, not pursued)

**CL-1: AI4Bharat Canary-MahaDhwani.**  Fine-tuned Canary-1B-flash on
the MahaDhwani Malayalam corpus.  Scored OIWER 16.0 on Malayalam (best
in published benchmarks) but not publicly released — only the
encoder weights are available.  Cannot use as a transcription model
without the decoder.

**CL-2: indic-seamless.**  Meta's SeamlessM4T variant for Indic
languages.  License is CC-BY-NC-4.0 (no commercial use); we cannot
bundle.  Also a translation-not-transcription model — its
"transcribe" mode is an off-label use of source-language target
translation.  Closed on license alone.

**CL-3: kurianbenoy/malayalam-whisperX.**  Whisper + wav2vec2
alignment combo published by the same author as
`Indic-Subtitler`.  Thin repo with no published benchmarks; uses
the same Whisper backbone with the same hallucination failure mode
that I-1 demonstrated.  The wav2vec2 alignment model itself was
considered separately under I-1 and found unhelpful for the same
reasons.

---

## Future investigations

### I-9: vegam's slow chunks are temperature fallback, and it costs reproducibility

**Status:** `RESOLVED` — mechanism confirmed from decoder telemetry.

**Observation.**  On MalayalamNews.mp4 with vegam, 10 of 43 chunks consumed
78% of decode time — 15–32s each against a median of 1.8s.  The slow set was
*identical* across repeated runs, so the cause is a property of the audio
content rather than scheduling noise.  On MalayalamNews-2.mp4 the effect is
larger still: 20 of 58 chunks, 88% of decode time.

**Mechanism.**  faster-whisper re-decodes from scratch at successively higher
temperatures (0.0, 0.2, 0.4, 0.6, 0.8, 1.0) whenever a decode fails its
`compression_ratio_threshold` (2.4) or `log_prob_threshold` (-1.0) check.  Six
passes at `beam_size=5` accounts for the observed ~10x per-chunk penalty.
Per-segment telemetry confirms it directly: slow chunks report temperatures up
to **1.0** with compression ratios of 2.05–2.20, sitting right at the
rejection threshold.  R-MFT on the same clips and the same chunk plan reaches
a maximum temperature of 0.2 and produces **zero** slow chunks.

Caveat: 13 of 20 slow chunks report temperature 0.0, so fallback explains most
of the cost but not all.  What makes those chunks expensive is unresolved.

**Consequence 1 — speed.**  Same machine (RTX 3060 Laptop), same clip, same
58-chunk plan: vegam 514.7s, R-MFT 162.2s.  **3.2x.**  On an RTX 3060 Ti with
MalayalamNews.mp4 and translation enabled: vegam 280.2s, R-MFT 128.9s
(2.2x).  R-MFT is *not* faster per chunk — its median chunk decode is slightly
slower — the entire difference is the absence of the pathology.

**Consequence 2 — reproducibility.**  Sampling above temperature 0 is
stochastic by construction, so any chunk that enters fallback is a coin toss.
Two identical vegam runs on the same machine produced 66 cues both times but
differed in **8 real cues**, with different words at identical timestamps.
R-MFT differed by one cue, entirely within the fragment population, with all
59 real cues byte-identical.

**Consequence 3 — output quality.**  A blind native-reader comparison (the
listener did not know which SRT came from which model) found R-MFT better on
both detection and accuracy.  vegam's errors sit inside full-length cues —
including an outright repetition loop, `പി ചെയ്യ് പീ ചെയ്യ് പീ ചെയ്യ്`, in a
4.66s cue — and are not removable by post-processing.  R-MFT's errors are
predominantly the sub-second fragments of I-7, which duration filtering
removes at 100% precision.

**Interpretation.**  One property with four visible consequences.  If R-MFT's
fine-tuning suppresses repetition, that keeps `compression_ratio` below the
rejection threshold, which prevents the fallback ladder, which yields the
speed advantage *and* the reproducibility advantage as side effects, while
the repetition suppression itself is the quality advantage.  This is
GenSRT-side inference from external measurement; whether the fine-tuning was
in fact aimed at repetition is a question for Adalat AI.

**Bearing on I-2.**  I-2 records a 13-config byte-identical parameter sweep on
vegam.  Given that vegam output is *not* reproducible run-to-run on this
corpus, that result warrants re-examination before it is cited — either the
sweep hit a hard structural truncation where output was stable regardless, or
it ran on a numerically different path.  Flagged rather than resolved.

**Caveats.**  Two clips, one listener, six adjudicated divergences.  Strong
enough to inform a model recommendation; not a benchmark.  Timings are for
Malayalam broadcast news on consumer NVIDIA hardware; audio that does not
trigger fallback would narrow the gap considerably.

---

### I-10: WhisperJAV post-processing modules — evaluated, not adopted

**Status:** `RESOLVED NEGATIVE` — mechanisms sound, no target in this corpus.

WhisperJAV (github.com/meizhong986/WhisperJAV, MIT) contains two modules that
appeared to map onto GenSRT's open cleanup problems.  Both were read at source
and evaluated against the I-7 corpus.  Neither was adopted.

**`segment_filters.py` — short-segment confidence gate.**  Drops segments
below an `avg_logprob` threshold, with a different threshold for short
segments.  Three findings:

1. The margin runs the opposite way to what GenSRT needs.  The code computes
   `threshold - margin` with a positive margin, which on a negative threshold
   makes short segments *harder* to filter.  Defensible for their domain,
   where short utterances are genuinely low-confidence but real; inverted for
   ours.
2. It performs worse than duration.  `avg_logprob` separated cleanly on one
   clip but produced 4 false positives on the other.  Duration alone: 100%
   precision and recall on both.
3. GenSRT has a better signal that a general-purpose cleaner cannot have.
   `is_chunk_tail` derives from GenSRT's own chunk plan and fires on zero of
   220 normal cues.

**`cross_subtitle_processor.py` — consecutive-duplicate merge.**  Requires
three or more consecutive near-duplicate cues (`DEDUP_THRESHOLD = 3`).  Across
four segment dumps this corpus contains **zero** such runs; the chunk-tail
fragments occur in runs of one to three *following* a normal cue, so pairs at
most.  Its whole-file high-density phrase detector matches on
`\p{Hiragana}|\p{Katakana}|[一-龯々ヶ]` and returns an empty list on
Malayalam.  Its `_merge_subtitle_group` also re-imposes a 1.0s minimum
duration — the exact artifact corrected in I-7.

**No repetition to clean.**  Across all four dumps, maximum
`compression_ratio` on a real cue is 2.33 and **no cue exceeds
faster-whisper's 2.4 repetition threshold**.  R-MFT does not produce the
within-cue repetition these modules target.

**Qualifier.**  This is an R-MFT result.  vegam *does* produce repetition —
compression ratios of 2.05–2.20 on its slow chunks, and an outright loop in
one adjudicated cue (I-9).  The modules are aimed at a real phenomenon; it is
one R-MFT appears to have addressed upstream.  Since R-MFT is the recommended
Malayalam model, adopting them would mean carrying code and an MIT attribution
obligation for a case GenSRT does not hit.

**What was taken.**  One idea, not code: WhisperJAV runs its cleaners as
individually-toggleable stages, each emitting a modifications log recording
every change with its type, the original text and a confidence.  That is the
honest-signal philosophy applied to cleanup, and it is the natural data model
for the planned chunk-tail cleanup UI — the user sees each proposed removal
and accepts or rejects it rather than the tool quietly sanitising.  Adopted as
a design pattern; no source carried over, so no attribution rides along.

---

### I-4: Hallucination-repetition post-processor

**Status:** `OPEN — v1.3 candidate`

**Hypothesis:** vegam (and likely other fine-tunes) emit phrases
from earlier in the audio at chunk tails as a "fallback" when the
model has nothing genuine to transcribe.  A post-processor could
detect these by checking each cue's text for tail substrings (≥4
characters) that appear in the previous cue's text, and strip them.

**Success criteria:** On the v1.2 test corpus (two Malayalam news
clips), the cleanup reduces visible repetition without removing
legitimate emphatic repetition.  Native-reader validation required.

**Open question:** Whether this generalizes across languages, or
whether each language has different repetition patterns that need
language-specific tuning.

**Blocker:** Cross-language native-reader validation (Tamil, Hindi,
Bengali, Kannada at minimum) before shipping a horizontal cleanup
pass that affects all models.

---

### I-5: whisperX-style alignment with reliable upstream

**Status:** `OPEN — speculative, v1.3+`

**Hypothesis:** I-1 failed because vegam's text was unreliable.
If we had a reliable upstream transcription (chunked vegam looks
qualitatively reliable now), wav2vec2 forced alignment could
produce sub-cue word-level timing that's currently approximated
by faster-whisper's chunk-relative timestamps.

**Question for the future:** Is word-level alignment useful enough
to justify the torchaudio dependency overhead?  Most subtitle
consumers display cue-level timing, not word-level.  The use case
would be karaoke-style highlighting or sub-cue refinement — both
nice-to-haves rather than core needs.

---

## Conventions

- An investigation gets an entry as soon as we have a hypothesis
  worth recording, not after it resolves.
- Closed leads document what we evaluated and rejected, so we
  don't re-evaluate the same ground twice.
- Every resolution names the test method and success criteria, so
  the resolution is reproducible from this document alone.
- Native-reader verification is required for any quality claim
  involving non-Latin scripts.  Benchmark numbers (OIWER, WER) are
  necessary but not sufficient.

