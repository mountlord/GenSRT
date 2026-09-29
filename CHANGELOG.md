# Changelog

## v1.3.0 — in development

### Removed
- **The separate Pascal installer** (`-Variant pascal`, `requirements-cuda-pascal.txt`, the second venv). It existed because cuDNN ≥ 9.11 refuses compute-capability-6.x GPUs; CTranslate2 ≥ 4.6.3 no longer routes Whisper's convolutions through cuDNN, so the standard CUDA build runs on GTX 10-series / Tesla P4-P40-P100 unchanged (verified on a Tesla P4 with cuDNN 9.24, int8). `ctranslate2>=4.6.3` is now pinned and the packager refuses to build with an older one.
- **Google GTX translation engine** (`translation_engine: "google"`), its MyMemory per-cue fallback, and the `translation_fallback` config field / `--translation-fallback` flag. The unofficial endpoint blocks IPs that translate at subtitle volumes, and the block was observed to survive an IP change and persist for months. Both remaining engines (`nllb`, `madlad`) run offline; the default is now `nllb`. A leftover `"google"` in an older config produces an explanatory `ConfigError`; a leftover `translation_fallback` key is ignored on load.

### Added
- **MADLAD-400 offline translation engine** (`translation_engine: "madlad"`, `madlad_model`): Apache-2.0 weights, ~2.9 GB, roughly twice NLLB's per-cue cost. Offered beside NLLB because which reads better depends on the material.
- **OCR**: Read Frame (paused-frame text → cue) and Extract Subtitles (burned-in subtitles across a whole video → SRT, with translation) via RapidOCR/PP-OCR on onnxruntime. `--extract-subtitles` on the CLI.
- **Fixed-window chunk mode** (`chunk_mode: "fixed"` / `--chunk-mode fixed`): decode the whole file in 5–8 s windows with no outer voice detector, for soft speech and speech under other vocal sounds.
- **Onset snapping in fixed-window mode** (`snap_onsets`, `--no-snap-onsets` to disable): Whisper stamps a lone short utterance `0.00→2.00` wherever it sits in the window, so cues appeared 2–4 s early (560 of 1,293 on one file). A low-threshold silero pass now finds the audible onset in each chunk and moves such cues to it, duration preserved.
- `heuristics_report_dir` / `--heuristics-report-dir DIR`: write the heuristics report to `<name>.heuristics.txt` and `.json` per file, all rows.
- Heuristics: `hallucination_contains` per language (substring match, for sign-off phrases that arrive in variants) — `ja` ships `ご視聴`, `ko` ships the "see you in the next video" / "thanks for watching" / "like and subscribe" / "Korean subtitles by" sources; the report gains a **Repeated lines** table for full sentences that recur, with translations, which the short-string table could not show.
- **Post-ASR heuristics** (`gensrt-heuristics.json`, `--init-heuristics`, `--heuristics-report`): per-language interjection collapse, drop and hallucination lists, a density rule, and post-translation subject-pronoun stripping for pro-drop source languages when the target is English.
- Configuration editor can save every config field (the validator now derives its choices from the engine factory instead of a hand-kept table).

### Internal
- `gensrt/server.py` (2,400 lines, 21 routes, all state) is split into `gensrt/api/` — one Flask blueprint per concern (`jobs`, `config`, `media`, `ocr`) plus the shared operation gate (`_state`) and path validation (`_paths`); `server.py` keeps the app, the JSON error handler and `launch_server`. URLs are unchanged; the names tests and the CLI imported from `server` are re-exported.

### Fixed
- **`.ts` recordings no longer play in the review window** — not a GenSRT change: WebView2's bundled FFmpeg build lost the MPEG-TS demuxer in an Edge update (`DEMUXER_ERROR_COULD_NOT_OPEN: FFmpegDemuxer: open context failed`, on a stream ffprobe scores 100). `gensrt/remux.py` now rewraps `.ts`/`.m2ts`/`.mts` into MP4 (`-c copy`, `+faststart`) on first open, cached under `<temp>/gensrt_remux` keyed by path+size+mtime with a 20 GB oldest-first cap; `POST /api/media/prepare` + `GET /api/media/status` drive a progress overlay in the player and `/api/media` serves the rewrap with Range support. The video element reports its `MediaError` in the video area (and the console) instead of staying silently black; a file dropped onto the desktop window goes through the path flow rather than a blob URL, so a `.ts` drop no longer flashes a cannot-play dialog while the rewrap runs. **Save MP4** in the header saves the rewrap of the open recording to a path of your choosing (`POST /api/media/export`).
- **Out-of-memory handling on small GPUs** (measured on a Tesla P4, 8 GB, MADLAD-400 3B + large-v3-turbo). A translation batch that hit CUDA OOM used to write the file *untranslated* behind a one-line warning; the engines now halve the token batch down to 128 and then reload on CPU before giving up, and a file that still goes out untranslated is an ERROR in the log, a status line, a `translation_error` on the result, a `NOT TRANSLATED` line in the CLI summary and a flagged dialog in the GUI. Whisper's loader, on CUDA OOM, now releases the resident translation engine and retries the GPU once before falling back to CPU. Translation runs in 64-cue slices so a retry redoes a slice, not the file; the engine remembers the batch size that held; the Whisper model is released explicitly (not at the next GC pass) and the run logs GPU memory after each release.
- A registered monolingual model given a different source language (kotoba-whisper with `--source-lang ko`) now stops with a `ConfigError` before any work, instead of emitting Japanese for Korean audio and translating it. kotoba-whisper v1.0/v2.0 are now in the monolingual registry (Japanese), so they also get chunked inference and the language lock automatically.
- NLLB output sometimes carried HTML entities and tokeniser spacing from its training corpora (`I &apos;m going … .`); both offline engines' output is now normalised (`html.unescape`, punctuation and clitic spacing).

## v1.2.7 — 2026-08-24

### Added
- **NLLB-200 offline translation engine** on CTranslate2 (`translation_engine: "nllb"` / `--translation-engine nllb`): fully offline, any mapped language pair, GPU-accelerated (`int8_float16` on CUDA, `int8` on CPU). The model (~650 MB) downloads once, automatically, at the start of the first run that needs it, into the `models/` directory. Zero new dependencies — CTranslate2 runs it, `tokenizers` (already present via faster-whisper) reads its tokenizer, `huggingface_hub` (likewise) downloads it. **Note: the NLLB model weights are CC-BY-NC-4.0 (non-commercial use only)** — the engine logs this at every load; see README, "Offline translation (NLLB)".
- `translation_fallback` config field / `--translation-fallback` flag: what happens when a Google GTX batch fails outright — `nllb` (translate offline; default), `mymemory` (the old behaviour), or `none` (keep the source text).
- `translation_model` config field: which NLLB conversion to use — a HuggingFace repo ID, a folder name under `models/`, or a full path.
- `--self-check` now reports whether the NLLB model is on disk.
- `max_chunk_s` / `min_chunk_s` config fields (`--max-chunk-s` / `--min-chunk-s`): the chunked-inference sizes, previously hardcoded, are now tunable. Validated before any audio work (0 < min < max ≤ 30s).
- **Add button** in the SRT Lines toolbar: creates the first line of a from-scratch subtitle file, prefilled from the playhead. Deliberately enabled only while the list is empty — once any line exists, Split's free-form times already place a new line anywhere (including gaps), and a second insertion affordance would only duplicate it.

### Changed
- Google GTX rate-limit handling: HTTP 429/503 responses back off on a longer ladder (2 s, 8 s) and honour `Retry-After` (capped at 30 s) instead of retrying at 0.25 s — retrying a rate limiter that fast only deepens the hole the IP is in. Batch requests are additionally paced 0.4 s apart so a 3,000-cue file no longer presents the burst signature that provokes throttling.
- Translation failure logging no longer floods: the first failed batch logs a WARNING with its cause, subsequent failures log at DEBUG, and the run ends with one summary WARNING carrying the totals and the likely diagnosis. Previously a throttled IP produced one WARNING per batch — ~80 for a typical long recording.
- If NLLB is configured as the fallback but its model cannot be fetched (offline machine), the run warns once and continues with `translation_fallback: "none"` rather than failing. NLLB as the *primary* engine still fails loudly when unavailable — you asked for it by name.

### Fixed
- **Chunked inference no longer discards short utterances.** Speech regions briefer than `min_chunk_s` (2s) were silently dropped — deleting every short exclamation from the transcript, measured at ~4 minutes of speech lost in a 10-minute sparse-dialogue excerpt. Such regions are now transcribed whole (`short_region` in the chunk-plan log, which reports how many were kept). `min_chunk_s` now governs where cuts may land, not which speech exists.
- A translation batch that failed after Google's own retries always fell back to MyMemory, whose output quality is not usable for subtitles and whose per-cue round-trips added ~50 s per failed batch. Failure handling is now configurable and defaults to an engine that produces usable output offline.

## v1.2.6 — 2026-08-14

### Added
- `models\` directory beside the executable for locally-converted CTranslate2 models; a bare folder name resolves against it.
- Conversion guidance now prints a ready-to-run `ct2-transformers-converter` command with the real output path filled in.
- Error messages are selectable and have a copy button.
- Patch distribution for CUDA installs (`Create-gensrt-patch.ps1`, `tools/make_patch.py`), so an update is ~13 MB rather than a full re-download.

### Changed
- Model validation reports HuggingFace 401 responses accurately: private, gated and nonexistent repositories are indistinguishable from outside, and GenSRT no longer claims otherwise.
- Certificate-verification failures explain the likely cause and what to try.
- A misconfigured `REQUESTS_CA_BUNDLE` / `CURL_CA_BUNDLE` / `SSL_CERT_FILE` is named explicitly, with the offending values.
- `gensrt-config.json` and `gensrt-known-models.json` are created beside the executable rather than in the working directory. Existing files are not moved.

### Fixed
- Text selection was disabled at the pywebview window level (`text_select=False`), below anything CSS could override.
- Local models were only searched for beside the executable, so a model in a project folder was invisible when running from source.
- Multi-line messages collapsed into a single paragraph.


## v1.2.5 — 2026-08-12

### Added
- CPU-only installer variant (`Pack-gensrt.ps1 -Variant cpu`), alongside the CUDA build.
- `--target-language` CLI flag. Translation to non-English targets now works.
- `--self-check` verifies an installation is complete: imports every module, runs the bundled FFmpeg, loads the CUDA libraries by name, and tests HTTPS to HuggingFace. The build refuses to produce an installer that fails it.
- `--dump-segments` and `--debug-chunks` export per-cue decoder telemetry and per-chunk audio.
- Cue numbers in the cue list.
- WebVTT output alongside SRT.

### Changed
- **PyTorch removed.** GPU detection uses CTranslate2 instead. Substantially smaller downloads for every user.
- **Offline translation engines (NLLB-200, MarianMT) removed.** Both could only produce English and required PyTorch. Old configs naming them get an explanation.
- Custom-model validation now checks for CTranslate2 format, not just that the repo exists.
- Model validation uses the same TLS trust store as model downloads.
- Default subtitle line length is 42 characters (was 84).
- `device` defaults to `auto` and is honoured when set explicitly.
- Recommended Malayalam model is now `adalat-ai/ct2-whisper-medium-ml-rmft`.

### Fixed
- Subtitle text past two lines was silently discarded.
- Cues could overlap each other after the minimum-duration floor was applied.
- GUI footer was clipped at 125%+ display scaling ([#1](https://github.com/mountlord/GenSRT/issues/1), reported by @moob158).
- `"device": "cpu"` was ignored — the GPU probe overwrote it unconditionally.
- Burn-in failed on filenames containing `[`, `]`, `,`, `;` or `'`, silently.
- Dropping a language-variant SRT (`clip.ml.srt`) loaded `clip.srt` instead.
- Google translation batching broke on non-English targets: the batch delimiter was itself being translated, leaving ~93% of cues untranslated.
- `pyproject.toml` declared MIT; GenSRT is AGPL-3.0.
- `python -m gensrt` ran the CLI on import and always exited 0.
- Chunk diagnostics were written to a directory named after the temp audio file.

### Known limitations
- Fine-tuned models emit very short spurious cues at chunk boundaries (median 60 ms). Cleanup is candidate work for v1.3.
- A `�` at the end of a line means the model stopped generating mid-character. The text before it is valid.
- Monolingual fine-tunes render other-language passages phonetically rather than skipping them.
- Cue boundaries within a chunk are Whisper's own and occasionally split mid-word.

---

## v1.2.1 and earlier

See the [release history](https://github.com/mountlord/GenSRT/releases).
