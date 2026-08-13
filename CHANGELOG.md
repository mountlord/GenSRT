# Changelog

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
