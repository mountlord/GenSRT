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
   accuracy, almost perfect timing."  Second clip (general news, 268s):
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

