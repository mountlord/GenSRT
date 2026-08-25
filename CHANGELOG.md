# Changelog

## v1.2.7 — 2026-08-24

### Added
- **NLLB-200 offline translation engine** on CTranslate2 (`translation_engine: "nllb"` / `--translation-engine nllb`): fully offline, any mapped language pair, GPU-accelerated (`int8_float16` on CUDA, `int8` on CPU). The model (~650 MB) downloads once, automatically, at the start of the first run that needs it, into the `models/` directory. Zero new dependencies — CTranslate2 runs it, `tokenizers` (already present via faster-whisper) reads its tokenizer, `huggingface_hub` (likewise) downloads it. **Note: the NLLB model weights are CC-BY-NC-4.0 (non-commercial use only)** — the engine logs this at every load; see README, "Offline translation (NLLB)".
- `translation_fallback` config field / `--translation-fallback` flag: what happens when a Google GTX batch fails outright — `nllb` (translate offline; default), `mymemory` (the old behaviour), or `none` (keep the source text).
- `translation_model` config field: which NLLB conversion to use — a HuggingFace repo ID, a folder name under `models/`, or a full path.
- `--self-check` now reports whether the NLLB model is on disk.
- **Add button** in the SRT Lines toolbar: creates the first line of a from-scratch subtitle file, prefilled from the playhead. Deliberately enabled only while the list is empty — once any line exists, Split's free-form times already place a new line anywhere (including gaps), and a second insertion affordance would only duplicate it.

### Changed
- Google GTX rate-limit handling: HTTP 429/503 responses back off on a longer ladder (2 s, 8 s) and honour `Retry-After` (capped at 30 s) instead of retrying at 0.25 s — retrying a rate limiter that fast only deepens the hole the IP is in. Batch requests are additionally paced 0.4 s apart so a 3,000-cue file no longer presents the burst signature that provokes throttling.
- Translation failure logging no longer floods: the first failed batch logs a WARNING with its cause, subsequent failures log at DEBUG, and the run ends with one summary WARNING carrying the totals and the likely diagnosis. Previously a throttled IP produced one WARNING per batch — ~80 for a typical long recording.
- If NLLB is configured as the fallback but its model cannot be fetched (offline machine), the run warns once and continues with `translation_fallback: "none"` rather than failing. NLLB as the *primary* engine still fails loudly when unavailable — you asked for it by name.

### Fixed
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
