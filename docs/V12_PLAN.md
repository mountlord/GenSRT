# GenSRT v1.2 Release Plan

This document records the v1.2 release plan and its execution status.
It's the long-form companion to `INVESTIGATIONS.md` (technical
investigations) and the user-facing README's "What's new" section.

---

## v1.2 release theme

**Better Indic-language transcription via fine-tuned-model support.**

v1.1 shipped pluggable Whisper models (a user could already select
vegam in the GUI), but fine-tuned models dropped content on long-form
audio and were practically unusable for anything longer than a single
phrase.  v1.2's job is to make those models actually work.

---

## Scope decisions

### In scope (shipped)

- **ASR engine factory.**  Pluggable engine layer mirroring the
  existing translation factory pattern.  Two engines:
  `MultilingualWhisperEngine` (v1.1 behavior) and
  `MonolingualWhisperEngine` (silent-boundary chunking).
- **Silent-boundary chunked inference.**  Algorithm proven during
  I-2.  Auto-engaged for vegam and any custom Whisper model.
- **Vegam auto-routing.**  No user opt-in required.  Selecting a
  vegam variant via the footer Model dropdown automatically routes
  through chunked inference.
- **README + user_guide.html updates.**  Documents chunked
  inference, the `�` character convention, and credits SMC + the
  Manohar/Pillai/Sherly paper.

### Explicitly deferred to v1.3 or later

- **Hallucination-repetition cleanup.**  Vegam emits phrases from
  earlier in the audio at chunk tails.  Detectable in
  post-processing but needs cross-language native-reader
  validation before shipping a horizontal cleanup.  See `I-4` in
  `INVESTIGATIONS.md`.
- **Per-model chunked-inference toggle.**  Currently all custom
  Whisper models route to monolingual; vegam is hardcoded to
  always-chunked and not overridable.  If a future user reports a
  fine-tuned Whisper model that doesn't need chunking, add a
  checkbox to the Add Model modal.  Schema extension needed in
  `gensrt-known-models.json`.
- **IndicConformer ASR engine.**  Evaluated during v1.2 (`I-3a`).
  Sufficient quality from chunked vegam meant deferring the
  IndicConformer integration work.
- **`--debug-chunking` diagnostic flag.**  Standalone-script
  outputs (chunks.json, per-chunk WAVs in `_temp/`) are not
  preserved in production.  The one-line health summary in the
  INFO log carries the signal.  Add a flag to surface the full
  detail if support cases need it.

---

## Drop history

The v1.2 development arc was structured as discrete drops, each
with a clear scope and test plan.  Listed in chronological order:

- **Drop I.13** — ffmpeg/ffprobe bundling via gyan.dev essentials
  build.  Verified working on a clean Windows install with no
  pre-existing ffmpeg.
- **Drop I.14** — README/USER_GUIDE polish.  Documentation pass
  before forum launch planning.
- **Drop I.14.5** — Replaced USER_GUIDE.md and UserGuide.docx with
  a self-contained `user_guide.html` (embedded CSS, base64
  images, Known Limitations section).  Pack-gensrt.ps1 updated.
- **Drop I.14.7** — Pluggable Whisper model directory.  Exposed
  faster-whisper's ability to load arbitrary HF model IDs via a
  "Custom model" GUI/CLI field.  ~50 LOC.  (Originally scoped
  with verification of HF Whisper-format compatibility; shipped
  with the verified subset of CTranslate2-compatible models.)
- **Drop I.15** — ASR engine factory + silent-boundary chunking
  integration.  The core v1.2 deliverable.  `gensrt/asr/`
  package created; `pipeline.py` modified to dispatch through the
  factory.  No config schema changes, no CLI flag changes.
- **Drop I.15.1** — VTT track cleanup on New Project / empty
  state.  Two-line bug: `renderLinks()` returned early on empty
  chapters before calling `_refreshSubtitleTrack()`, and that
  function didn't fully clear cues from the TextTrack object
  when called with empty VTT.  Fixed both.
- **Drop I.16** — Release documentation.  README + user_guide.html
  updates, `docs/INVESTIGATIONS.md`, this file.

---

## Architecture decisions

### Engine naming

User-facing labels: **Whisper (Multilingual)** for the built-in
path, **Whisper (Monolingual)** for the chunked path.

Reasoning: matches published ML literature vocabulary (AI4Bharat
uses the same terminology for their model variants).  Sets up
clean naming for future engines — e.g. "IndicConformer
(Monolingual)" if I-3a ever ships.  "Monolingual" describes the
typical model class using the engine, not what the engine
mechanically does (which is chunked inference); the user-facing
label is the model-class name, while internal routing in the
factory decodes the actual mechanical behavior.

### Routing rules

Implemented in `gensrt/asr/factory.py`:

- **Model in `BUILTIN_RECOMMENDED`** (tiny..large-v3-turbo) →
  `MultilingualWhisperEngine` (no chunking, v1.1 behavior
  preserved exactly).
- **Model name matches `ALWAYS_CHUNKED_MODEL_PREFIXES`** (vegam
  variants from both `smcproject/` and `kurianbenoy/`
  namespaces) → `MonolingualWhisperEngine` (chunked, not
  user-overridable).
- **Any other custom user model** → `MonolingualWhisperEngine`
  (chunked by default; most community fine-tunes are Common-Voice-
  trained and need chunking).

### VAD parameter handling

User-configured VAD parameters (`vad_threshold`,
`vad_min_speech_ms`, `vad_min_silence_ms`, `vad_speech_pad_ms`)
are honored only for the **outer VAD pass** in
`MonolingualWhisperEngine` — i.e. the speech-region detection
that feeds the chunking algorithm.  The **inner progressive
sweep** (9 configs of threshold + min_silence + speech_pad) is
algorithm-controlled and not user-tunable; exposing it would let
users break chunking quality without realising it.

For `MultilingualWhisperEngine`, all four VAD parameters pass
through to faster-whisper's internal VAD unchanged.

### Production diagnostics

Default production output: SRT + VTT.  Plus one INFO log line per
chunked-inference run with: chunk count, % silence cuts, %
energy-min, chunking ceiling check.

Test-script-level diagnostics (`chunks.json`, per-chunk WAVs,
energy-min provenance tracking) are NOT preserved in production.
The one-line summary is sufficient for "is the pipeline healthy"
signal.  Full diagnostics can be added via a future
`--debug-chunking` flag if support cases require it.

### Backward compatibility

- No config schema changes.  Existing `gensrt-config.json` files
  work unchanged.
- No known-models schema changes.  Existing
  `gensrt-known-models.json` files (flat list of strings) work
  unchanged.
- No CLI flag changes.  Every command that worked in v1.1 works
  in v1.2 identically.
- Built-in Whisper users see no behavior change.  Vegam users
  see chunked inference auto-engage with no action required.

---

## Quality validation

### Native-reader testing

Two Malayalam news clips reviewed by native Malayalam reader (project
maintainer) with side-by-side audio playback and SRT reading:

**Clip 1 — MalayalamNews.mp4 (197.3s, dense news anchor speech):**

- 49 cues
- 86% chunks cut at detected silence
- 8.2% chunks contain `�` (mid-character truncation)
- Native-reader verdict: "very high accuracy", "almost perfect
  timing", "no other model produces anything close"

**Clip 2 — MalayalamNews-2.mp4 (274.0s, watched for the
first time during verification):**

- 66 cues
- ~10% chunks contain `�`
- Native-reader verdict on closed-captioning comparison: "beats
  realtime closed captioning by a mile" (vs broadcast CC the
  maintainer watches nightly)

### Pre-release checklist

- [x] Drop I.15.1 VTT fix applied and tested
- [x] `Pack-gensrt.ps1` excludes for `torchaudio` and `silero_vad`
      removed
- [x] `requirements-cuda.txt` includes `torchaudio==2.10.0`
- [x] README updated with chunked inference, � note, SMC credit
- [x] user_guide.html updated with the same
- [ ] Fresh-venv pip install verification
- [ ] Pack-gensrt fresh build
- [ ] Smoke test of packaged `.exe` on Malayalam clip from
      a separate machine (clean install verification)
- [ ] Git tag v1.2
- [ ] GitHub release page with release notes

---

## Outreach plan (post-release)

- **Elizabeth Sherly** (head of VRCLC at Digital University Kerala
  and co-author of arxiv 2409.02449).  Phone contact established.
  Framing: "Windows-native, no-WSL2-required version of the ASR
  stack you've been writing about, using SMC's vegam with the
  chunked-inference fix that addresses the Common-Voice phrase-
  distribution dropping problem."
- **Forum launch** to doom9, videohelp, r/subtitles.  Held until
  v1.2 since the Malayalam capability is the differentiator vs
  existing tools.
- **SMC / Kavya Manohar** likely via Sherly as forward.

