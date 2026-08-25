<p align="center">
  <img src="docs/LogoTwo.png" alt="GenSRT — Video subtitles in any language" width="800"/>
</p>

<p align="center">
  <a href="https://github.com/mountlord/GenSRT/releases"><img src="https://img.shields.io/github/v/release/mountlord/GenSRT" alt="Latest release"/></a>
  <img src="https://img.shields.io/badge/platform-Windows%2010%20%7C%2011-blue" alt="Platform"/>
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76b900" alt="NVIDIA CUDA"/>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL%20v3-blue" alt="AGPL v3 License"/></a>
</p>

<p align="center">
  <b>GPU-accelerated subtitle generation for Windows.</b><br/>
  Transcribe video to SRT using OpenAI Whisper, translate any-to-any — online or fully offline — edit cues in a built-in player.
</p>

---

## What it does

GenSRT generates SRT subtitle files from video using GPU-accelerated speech recognition, with built-in editing and any-to-any translation. The target use cases are serious subtitle work — content creators, fan subtitlers, accessibility teams, researchers working with non-English media.

- **Transcribe.** Drop a video file, pick a language (or auto-detect), generate an SRT. Built-in support for OpenAI Whisper sizes (`tiny` through `large-v3-turbo`) plus any HuggingFace-compatible faster-whisper model — including community fine-tunes for specific languages.
- **Translate.** Translate the generated SRT to any of 100+ languages — via Google Translate, or fully offline with NLLB-200 running on your own GPU/CPU. Translation preserves original timestamps, and what happens when Google rate-limits you is configurable (offline fallback by default).
- **Edit.** Built-in player with live subtitle display. Split, merge, delete, and edit cues with immediate feedback in the player. Save to disk as SRT and WebVTT in one operation.

<p align="center">
  <img src="docs/gensrt-subtitles-playing.png" alt="GenSRT transcribing Malayalam audio and translating to English subtitles" width="900"/>
</p>

<p align="center">
  <sub>
  <i>Screenshot: Malayalam audio from an Asianet News clip on the US–Iran truce, transcribed with <code>smcproject/vegam-whisper-medium-ml-int8_float16</code> using v1.2's chunked inference and translated to English. The "Circ Circ Circ…" row in the cue list is a known fine-tuned-Whisper hallucination GenSRT displays as-is — see <a href="#known-limitations">Known Limitations</a>.</i>
  </sub>
</p>

## What's new in v1.2.7

- **Offline translation via NLLB-200 on CTranslate2** — no network, no rate limits, one-time ~650 MB model download, zero new dependencies. Non-commercial model license; see [Offline translation (NLLB)](#offline-translation-nllb).
- **Configurable Google failure handling** (`translation_fallback`: `nllb` / `mymemory` / `none`) — failed batches no longer silently degrade to MyMemory, and a run with failures logs one summary line instead of one warning per batch.
- **Rate-limit-aware Google GTX** — HTTP 429/503 now back off on a longer ladder (honouring `Retry-After`), and batch requests are paced to avoid provoking the throttle in the first place.
- **Chunked inference on any model, with tunable sizes** — force `--asr-engine chunked` (or `longform`) on any model, and tune `max_chunk_s` / `min_chunk_s`. See [Chunked vs. long-form inference](#chunked-vs-long-form-inference).
- **Short utterances are no longer lost** — the chunker silently discarded speech regions under 2 seconds; on one test film that was 47% of everything the voice detector found. They are now transcribed whole.
- **Add button in the SRT editor** — start a subtitle file from scratch. Available only while the list is empty; once lines exist, Split places a new line anywhere, including gaps.

## What's new in v1.2.6

**Models you converted yourself have a home.** Put a CTranslate2 model in `models\` beside `gensrt.exe` and enter just the folder name. When GenSRT meets a PyTorch model it now hands you the conversion command with your real path already in it.

**Error messages can be copied.** They could not be before — pywebview disables text selection at the window level, below where CSS reaches — so reporting one meant taking a screenshot. Now selectable, with a copy button.

**A ~13 MB patch** is available for anyone already on v1.2.5, instead of the full download. Seven files changed between the two releases.

## What's new in v1.2.5

**Two installers.** A CPU-only build alongside the CUDA one. GenSRT no longer bundles PyTorch — transcription runs on CTranslate2 end to end, and PyTorch was only ever there to answer "is a GPU present?", which CTranslate2 answers itself. The CPU build carries no GPU libraries at all. If bandwidth is expensive where you are, that is the point.

**Translate to any language.** `--target-language` now exists on the command line, and translation to non-English targets actually works — a batching bug meant the delimiter GenSRT used to separate cues was itself being translated.

**Acknowledgments.** GUI clipping at 125–150% display scaling reported by [@moob158](https://github.com/mountlord/GenSRT/issues/1) — fixed.

### Chunked inference (v1.2)

**Chunked inference for fine-tuned Whisper models.** Community fine-tunes like SMC's `vegam-whisper-medium-ml` (for Malayalam) were practically unusable on long-form audio — they would transcribe the first 6-8 seconds and silently drop the rest. v1.2 solves this with silent-boundary chunked inference: audio is sliced along naturally-detected pauses, each chunk is transcribed independently, and the results are assembled into a single SRT with original timestamps preserved.

For Malayalam users with vegam, this produces 2-3× more transcribed content than running the same model without chunking. The chunked path engages automatically — no configuration required.

See the [v1.2 release notes](https://github.com/mountlord/GenSRT/releases/tag/v1.2.0) for the full changelog.

## Quick start

1. Download the installer from the [latest release](https://github.com/mountlord/GenSRT/releases) and run it. It's a 7z self-extracting installer — pick a folder, and it'll unpack GenSRT there.

   | Download | For | Size |
   |---|---|---|
   | `gensrt-install.exe` | NVIDIA GPU machines | larger |
   | `gensrt-install-cpu.exe` | everything else — Intel/AMD, integrated graphics, older laptops | **much smaller** |

   If you're unsure, or you're on a metered or slow connection, take the CPU build. It runs everywhere; it's just slower. You can always add the CUDA build later.
2. Run `gensrt.exe` from the install folder. The GUI opens.
3. Drop a video file onto the player. The Model selector in the footer defaults to `large-v3-turbo` (works well for English and most European languages).
4. Click **Generate SRT**. The model auto-downloads on first use (~1-2 GB depending on the model), then transcription begins.
5. When done, the right-pane cue list populates and subtitles display in the in-player overlay during playback.

> **First-run note:** the first time you generate an SRT with a given model, the model auto-downloads to your HuggingFace cache (~1-2 GB). The download is one-time. On CPU-only machines, transcription itself runs to completion but takes substantially longer than on a CUDA GPU — see [Requirements](#requirements) for typical timings.

**For Malayalam:** select `adalat-ai/ct2-whisper-medium-ml-rmft` from the Model dropdown. In our testing it was more accurate than the alternatives and roughly 3× faster — see [Malayalam models](#malayalam-models) below. `smcproject/vegam-whisper-medium-ml-int8_float16` also works well. Chunked inference runs automatically for both.

**For other Indic or less-common languages:** start with `large-v3-turbo`. If results are poor, search HuggingFace for community fine-tunes and add the repo path via **New…** in the Model dropdown.

## Malayalam models

Two community fine-tunes work well, and they trade off differently. Both are medium-sized Whisper fine-tunes, both use GenSRT's chunked inference automatically.

| | `adalat-ai/ct2-whisper-medium-ml-rmft` | `smcproject/vegam-whisper-medium-ml-int8_float16` |
|---|---|---|
| Accuracy | better in our testing | good |
| Speed | **~3× faster** | slower |
| Repeat runs | near-identical | text varies between runs |
| Short spurious cues | ~40% of cues | ~10% of cues |

We recommend **R-MFT** (the Adalat AI model). On a blind comparison — the listener didn't know which output came from which model — it was better on both detection and accuracy on a 4.5-minute Asianet news clip.

The trade-off is real but one-sided: R-MFT emits about four times as many very short spurious cues (median 60 milliseconds — too brief to read during playback, but present in the SRT). vegam emits fewer of those, but its mistakes sit *inside* full-length cues, including occasional repetition loops, which no post-processing can remove without removing real speech.

**Caveat on the comparison.** One clip, one native-speaker listener, six adjudicated differences. Enough to inform a recommendation; not a benchmark. Both models are actively maintained and your audio may differ from ours. The methodology and full measurements are in [INVESTIGATIONS.md](docs/INVESTIGATIONS.md) (entries I-7 and I-9).

## Chunked vs. long-form inference

GenSRT can feed audio to a Whisper model in two ways, and you can force either one on **any** model:

- **`chunked`** — audio is sliced at detected silences and each piece is decoded independently, then reassembled with original timestamps. This is what makes community fine-tunes (vegam, R-MFT, kotoba and similar) usable on long files at all: their converted checkpoints stop emitting text after the first few seconds when handed a whole recording. Chunking also gives tighter timestamps and bounds how far a hallucination can spread — at the cost of Whisper's cross-window context. Speech regions shorter than `min_chunk_s` are transcribed whole; nothing the voice detector finds is discarded.
- **`longform`** — the whole file goes to the model in one pass, the way stock Whisper is designed to run. On general multilingual models (`large-v3`, `large-v3-turbo`…) this keeps the model's cross-window context on continuous speech.
- **`auto`** (default) — picks per model: known fine-tune patterns get `chunked` (they need it), general multilingual models get `longform`.

Two sizes control the chunker, tunable in the config, on the CLI, or in the settings editor (⚙️):

- `max_chunk_s` (default 6.0) — the longest allowed chunk. 6s suits fine-tuned models' truncation limit (~7s of dense speech); values up to 30 (Whisper's window) trade that safety for more context per decode.
- `min_chunk_s` (default 2.0) — the smallest chunk a cut may produce. Cut placement only: shorter speech regions are still transcribed, as single whole chunks.

Where to set the engine: `--asr-engine chunked` / `longform` / `auto` on the CLI, `"asr_engine"` in `gensrt-config.json`, or the Configuration editor (⚙️) in the GUI.

> **Quiet or breathy speech?** The voice detector's defaults are tuned for clear dialogue and will skip soft speech entirely. Lower the threshold and widen the padding — e.g. `--vad-threshold 0.20 --vad-speech-pad-ms 300 --vad-min-speech-ms 150` recovered ~30% more speech on quiet test material.

## Offline translation (NLLB)

Since v1.2.7 GenSRT can translate **fully offline** using [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M) (No Language Left Behind, Meta AI) running on CTranslate2 — the same runtime that runs Whisper. One multilingual model covers every language in GenSRT's dropdowns, in any direction, with no API, no network dependency and no rate limits. The model (~650 MB) downloads once, automatically, at the start of the first run that needs it, into the same `models/` directory your Whisper models use.

By default NLLB is the **fallback**: Google GTX remains the primary translator, and any batch Google fails — most commonly HTTP 429 once the endpoint starts rate-limiting your IP, which sustained heavy use (nightly batches of long recordings) reliably provokes — is translated offline instead. To skip Google entirely, set `"translation_engine": "nllb"` in `gensrt-config.json` or pass `--translation-engine nllb`.

### License notice — read this if your use is commercial

The NLLB-200 model **weights** are licensed **[CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/)** by Meta: **non-commercial use only**. This restriction travels with the weights regardless of who converted or hosts them, and it applies to *your use of the model*, not to GenSRT itself (which remains AGPL-3.0 and does not bundle the weights — they download from HuggingFace on first use). GenSRT logs a reminder of this every time the NLLB engine loads.

GenSRT does not interpret what counts as "commercial" — that is Meta's license text. If your subtitling work is commercial, opt out:

```json
"translation_fallback": "none"
```

(or `"mymemory"`), and leave `translation_engine` on `"google"`. With `"none"`, batches that Google fails keep their source text and the run summarises how many were affected. We are exploring a permissively-licensed offline alternative (M2M100, MIT) for a future release.

### Choosing the model

`translation_model` in the config selects which NLLB conversion to use — a HuggingFace repo ID (downloaded once into `models/`), a folder name under `models/`, or a full path. The default is a community int8 conversion of the official distilled-600M checkpoint; it loads as `int8_float16` on CUDA and `int8` on CPU, mirroring how GenSRT loads Whisper models.

## Requirements

- **Windows 10 or Windows 11**
- **Recommended:** NVIDIA GPU with CUDA support (~2 GB VRAM is enough for vegam; ~4 GB for `large-v3-turbo`)
- **Also works on CPU** (Intel/AMD, including integrated graphics like Intel Arc) — GenSRT falls back automatically when no CUDA GPU is detected
- Internet connection for first-run model download

### Download size

GenSRT does not bundle PyTorch. Transcription runs on CTranslate2 end to end, and PyTorch was only ever there to answer one question — "is a GPU present?" — which CTranslate2 answers itself, more accurately and for free.

The practical effect is that the CPU-only build carries no GPU libraries at all, and the CUDA build carries only the two NVIDIA libraries CTranslate2 actually uses. If you're somewhere bandwidth is expensive, that difference is the point of shipping two builds.

Forcing CPU mode on a machine that has a GPU is a supported configuration — set `"device": "cpu"` in `gensrt-config.json`, or pick **cpu** in the Config panel. That's also the workaround if your CUDA install is broken: GenSRT will fall back to CPU on its own with a warning, but setting it explicitly skips the failed attempt.

### A note on speed

Model choice matters as much as hardware. On the same machine, same clip and same chunk plan, `adalat-ai/ct2-whisper-medium-ml-rmft` completed in 162s where `smcproject/vegam-whisper-medium-ml-int8_float16` took 515s — not because it decodes faster per chunk, but because it avoids a decoder-retry path the other model triggers on about a third of chunks.

GPU vs. CPU is also a substantial difference. Measured on the same 4.5-minute Malayalam news clip with `adalat-ai/ct2-whisper-medium-ml-rmft`:

| Hardware | Compute | Time | vs. clip length |
|---|---|---|---|
| RTX 3060 Laptop (CUDA) | float16 | **2.7 min** | 0.6× |
| AMD Ryzen 5 PRO 4650GE, 6-core (CPU) | int8 | **41 min** | **9×** |

**Plan for roughly nine times the length of your audio on CPU.** A 5-minute clip takes about 45 minutes; a 45-minute episode takes most of a working day. That is workable for short clips or an overnight batch, and painful if you were expecting minutes.

Two notes on that number. It came from a competent 6-core desktop CPU, so a 4-core laptop will be slower — treat 9× as a floor rather than an average. And CPU timings drift with thermals: two identical runs on the same machine differed by 7%, the second being slower because the machine was already warm.

Output quality is comparable. int8 and float16 produce the same cue structure and near-identical text, with occasional small wording differences in passages where the model is least confident. int8 on CPU is also fully deterministic — repeated runs produce byte-identical output, which float16 on GPU does not quite manage.

If your machine has 8 GB of RAM, prefer medium-sized models. A medium model in int8 leaves little headroom once Python, Flask and the browser view are loaded, and a larger one may push the machine into swapping — which on a run already measured in tens of minutes is worth avoiding.

## Features

- **Use models you converted yourself** — drop a CTranslate2 model into `models\` next to `gensrt.exe` and enter the folder name
- **Translate to any language** — `--target-language ko` (or `ja`, `hi`, `fr`…), or pick a target in the Config panel. Google Translate online, or NLLB-200 fully offline.
- **Plug-in any HuggingFace Whisper model** — add custom faster-whisper-compatible models via the GUI's Model selector or the `--model` CLI argument.
- **Chunked or long-form inference on any model** — forceable and tunable; see [Chunked vs. long-form inference](#chunked-vs-long-form-inference).
- **WebVTT alongside SRT** — every generation writes both `.srt` and `.vtt` so the output works in HTML5 `<video>` elements natively.
- **Live in-player subtitle display** while editing — Add, Split, Merge, Delete, and text edits show in the player immediately.
- **Burn-in subtitles** — bake subtitles into a copy of the video with one click; runs in the background.
- **Bundled ffmpeg** — no separate install required on target machines.
- **Any-to-any translation** — Korean → Malayalam, Japanese → Tamil, Spanish → Hindi, all supported via Google Translate.
- **Plex / Jellyfin / Kodi compatible** filename suffixes for SRT output.
- **Self-contained `user_guide.html`** shipped alongside the executable.

## Documentation

- **User guide:** `user_guide.html` shipped alongside the executable, covering full GUI workflow and CLI usage.
- **Architecture and decisions:** [`docs/V12_PLAN.md`](docs/V12_PLAN.md) — release plan, scope decisions, architectural choices.
- **Investigation history:** [`docs/INVESTIGATIONS.md`](docs/INVESTIGATIONS.md) — technical investigations, including evaluation of alternative ASR engines (IndicConformer) and forced-alignment approaches (wav2vec2).

## Known limitations

- Fine-tuned Whisper models emit very short spurious cues at chunk boundaries — typically a word the model has just transcribed, re-emitted with a near-zero timestamp span (median 60 ms). They are too brief to read during playback but appear in the SRT and matter when editing or translating. Rate is ~40% of cues for R-MFT and ~10% for vegam. Every one we have checked against audio was spurious. A one-click cleanup is candidate work for v1.3; until then they are straightforward to delete in the editor. Fully characterised in [INVESTIGATIONS.md](docs/INVESTIGATIONS.md) I-7.
- **A monolingual fine-tune transcribes only its training language.** Passages in another language are not skipped — they are rendered phonetically into the training language, so English commentary in a Malayalam broadcast comes out as plausible-looking but meaningless Malayalam text. Nothing currently flags this. These are full-length cues, so no duration filter removes them; a native speaker reading the output will spot them immediately, and a non-speaker will not.
- Cue boundaries *within* a chunk are Whisper's own, and it will occasionally split mid-word. GenSRT's chunk boundaries themselves are cut at detected silence and have been verified by ear not to slice words.
- Whisper's tokenizer can stop generating mid-character on Indic scripts; a `�` at the end of a subtitle line is GenSRT signaling this honestly rather than masking it. The text *before* the `�` is accurate.
- Built-in Whisper models struggle with fast, dense speech in Indian languages (news broadcasts with English code-switching). Use a fine-tuned model where available; verify against audio before publishing where one isn't.

See the user guide's "Known Limitations" section for the complete list.

## Acknowledgments

GenSRT's chunked inference path was developed against **[vegam-whisper-medium-ml](https://huggingface.co/smcproject/vegam-whisper-medium-ml)** from **[Swathanthra Malayalam Computing (SMC)](https://smc.org.in/)**. Kavya Manohar, Leena G Pillai, and Elizabeth Sherly's analysis of Indic-script ASR evaluation pitfalls ([arxiv 2409.02449](https://arxiv.org/abs/2409.02449)) shaped how we think about quality measurement for these models. AI4Bharat's OIWER benchmark ([arxiv 2603.00941](https://arxiv.org/abs/2603.00941)) provides the most rigorous published Malayalam ASR comparison.

## Built with

[OpenAI Whisper](https://github.com/openai/whisper) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [CTranslate2](https://github.com/OpenNMT/CTranslate2) · [Flask](https://flask.palletsprojects.com/) · [pywebview](https://pywebview.flowrl.com/) · [ffmpeg](https://ffmpeg.org/)

## License

GenSRT is licensed under the **GNU Affero General Public License v3.0** — see [LICENSE](LICENSE). You can redistribute and modify GenSRT under AGPL terms. If you build derivative works or run modified versions as a network-accessible service, the AGPL terms apply, including the obligation to make source code of your modified version available to its users.

For commercial licensing options (e.g. proprietary integration), [contact the maintainer](https://github.com/mountlord/GenSRT/issues/new).

<p align="center">
  <br/>
  <img src="docs/LogoOne.png" alt="" width="80"/>
</p>
