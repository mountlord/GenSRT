# GenSRT

GPU-accelerated subtitle (SRT) generation using OpenAI Whisper, with any-to-any
translation via Google Translate (or X-to-English via Meta NLLB-200 / Helsinki
MarianMT). Ships with both a desktop GUI for editing and a CLI for batch jobs.

> **New here?** Looking for help using the desktop app? Open `user_guide.html`
> (included in every GenSRT install — it opens in your browser). This README
> is the reference for the CLI, configuration knobs, and tuning parameters.

---

## Requirements

- Python 3.10+ (only needed if installing from source — see *Installation*)
- NVIDIA GPU with CUDA 12.8 (CPU fallback available, significantly slower)
- Internet connection for Google translation (first Whisper model download also requires internet)

FFmpeg is bundled — no separate install needed.

---

## Installation

```powershell
# 1. Install CUDA torch FIRST (prevents pip pulling the CPU build)
pip install -r requirements-cuda.txt

# 2. Install remaining dependencies
pip install -r requirements.txt

# 3. Install GenSRT in editable mode
pip install -e .
```

---

## Two ways to run it

GenSRT has a GUI for interactive editing and a CLI for batch jobs. Same engine
underneath; same config file; same output format.

```powershell
# GUI — no arguments launches the desktop app
gensrt

# CLI — any --input argument runs headless
gensrt --input video.mkv
```

---

## GUI workflow

The GUI opens a desktop window (pywebview) with a video player on the left and
an SRT editor on the right.

**Loading content**

- Drag a video onto the player — if a matching `.srt` exists next to it, it loads automatically into the right pane
- Drag an SRT onto the editor — if a matching video exists next to it, that loads automatically into the player
- Use the *Browse* button for a native file picker (same behaviour as drag-drop)

**Footer controls (per-job overrides)**

| Selector | Effect |
|---|---|
| Source language | What language the audio is in (`auto` to detect) |
| Target language | What language to translate to (English unless changed) |
| Translation engine | Google / NLLB / Marian / none |
| VAD on/off | Toggle silence filtering for this run |
| Generate SRT | Runs the pipeline on the loaded video |

The footer pre-fills from your saved defaults and acts as a per-job override.
Changes here don't touch the config file.

**Editing**

The editor's video player shows your current subtitles as you edit — every
Split, Merge, Delete, or text change is reflected in the player immediately.
No need to save and re-open in a separate player to check your work.

- Click any row in the right pane → seeks the video to that line and pauses
- Check a row's checkbox + click *Split* → opens the split editor with two halves
- Check two adjacent rows + click *Merge* → combines them into one (texts joined newline-separated)
- Click *Edit* on a row → opens a modal to fix text or timestamps
- *Delete* removes the selected line(s)
- *Save* writes back to the loaded `.srt` (and the matching `.vtt`)
- *Save As* writes to a chosen path (next Save will go to that path)

**Burn SRT** — bakes the loaded subtitles into a copy of the video, producing
a standalone `.mp4` with the subtitles permanently visible (no separate
subtitle track needed). The button is in the footer; it's disabled until
both a video and an SRT are loaded. Burn runs ffmpeg in the background and
returns control to the app immediately — you can keep working while it
finishes. Output: `<video>_subbed.mp4` next to the source; subsequent burns
auto-version to `_subbed_1.mp4`, `_subbed_2.mp4`, etc.

**Config modal**

The ⚙ button opens persistent defaults. Anything saved here flows into the
footer on the next session and the next config save also syncs the footer
immediately. The Config modal exposes the same knobs documented in *Tuning
Parameters* below.

---

## CLI quick start

```powershell
# Transcribe a single file
gensrt --input video.mkv

# Batch process a directory
gensrt --input D:\Videos\

# Recurse into subdirectories
gensrt --input D:\Videos\ --recurse

# Keep subtitles in source language (no translation)
gensrt --input video.mkv --no-translate

# Use a specific output folder
gensrt --input video.mkv --output D:\Subtitles\
```

---

## Translation

```powershell
--translation-engine ENGINE    # Default: google
--no-translate                 # Output in source language
--source-language LANG         # Default: auto  |  e.g. ja, ko, ml
```

| Engine | Key | Targets supported | Internet | Notes |
|---|---|---|---|---|
| Google Translate (GTX) | `google` | **Any → any** | Required | Default. Best quality for Malayalam and other non-English targets |
| Meta NLLB-200 | `nllb` | X → English only | After model download | Offline. Single model for all source languages |
| Helsinki MarianMT | `marian` | X → English only | After model download | Offline. Per-language source models |
| None (passthrough) | `none` | (no translation) | — | Output in source language |

**Engine policy:** non-English target languages are supported by Google only.
Picking Marian or NLLB with a non-English target produces an immediate error
("Target language 'ml' is only supported by the 'google' translation engine…")
— no model loads, no audio extracts. Switch to Google or set the target back
to English.

```powershell
# Default: Google, source auto-detected, translated to English
gensrt --input video.mkv

# Korean source → Malayalam subtitles (set target via config file, see below)
gensrt --input video.mkv --source-language ko

# Offline translation (X → English only)
gensrt --input video.mkv --translation-engine nllb

# No translation (keep source language)
gensrt --input video.mkv --no-translate
```

The target language for CLI runs is set via `gensrt-config.json` — see the
*Configuration File* section. The GUI has a target-language selector in the
footer.

---

## Output file naming

GenSRT writes two files per transcription, both Plex / Jellyfin / Kodi compatible:

| Target language | SRT file | VTT file (companion) |
|---|---|---|
| English (default) | `video.srt` | `video.vtt` |
| Malayalam | `video.ml.srt` | `video.ml.vtt` |
| Korean | `video.ko.srt` | `video.ko.vtt` |
| Japanese | `video.ja.srt` | `video.ja.vtt` |
| (any other ISO 639-1) | `video.<lang>.srt` | `video.<lang>.vtt` |

The `.vtt` is identical content to the `.srt` in WebVTT format, suitable for
HTML5 `<video>` elements as a `<track>` source — useful if you're embedding
the video in a webpage or working with players that prefer VTT.

English stays unsuffixed for backward compatibility with existing libraries.
Multiple language SRTs coexist next to the same video and are recognised as
separate subtitle tracks by media servers like Jellyfin and Kodi.

When `--output-filename` is set, that name is used verbatim (no language
suffix is inserted). When both `--output` and `--output-filename` are set,
`--output-filename` wins and `--output` is ignored with a warning.

**Sidecar discovery on load:** drop a video and GenSRT looks for a sidecar
SRT next to it. `video.srt` is preferred when present; otherwise the first
`video.<lang>.srt` found is loaded. To load a specific language variant when
multiple exist, drag that file directly instead of the video.

---

## Tuning Parameters

GenSRT exposes several knobs for improving subtitle quality.
All parameters can be set via CLI flags, the Config modal in the GUI, or
persisted in `gensrt-config.json`.

### Subtitle Display Duration

**Problem:** Some subtitles hang on screen for minutes.
This happens when Whisper assigns a long end timestamp to a segment — common
in content with long pauses or music between dialogue.

```powershell
--max-subtitle-duration SEC   # Default: 3.0  |  0 = no cap
```

| Value | Effect |
|---|---|
| `3.0` (default) | Tight cap — good for fast dialogue, prevents subtitles from hanging |
| `5.0` | Moderate cap |
| `10.0` | Looser cap — good for slow/deliberate speech |
| `0` | Disabled — use Whisper's raw timestamps |

---

### Voice Activity Detection (VAD)

VAD runs inside Whisper on the GPU and removes silence before transcription.
It dramatically reduces processing time and hallucinated subtitles on silent
sections.

**Problem: Missing speech** — VAD is filtering out real dialogue.
Lower the threshold or reduce the minimum silence gap.

**Problem: Too many hallucinations** — VAD is not filtering enough.
Raise the threshold.

```powershell
--no-vad                       # Disable VAD entirely
--vad-threshold FLOAT          # Default: 0.50  |  Range: 0.0–1.0
--vad-min-speech-ms MS         # Default: 250
--vad-min-silence-ms MS        # Default: 2000
--vad-speech-pad-ms MS         # Default: 200  (faster-whisper library default: 400)
```

#### `--vad-threshold`
Speech probability threshold. A window is classified as speech only if
Whisper's VAD model assigns it a probability above this value.

| Value | Effect |
|---|---|
| `0.3` | More permissive — captures quiet or accented speech, more false positives |
| `0.5` | Default — good balance |
| `0.7` | More aggressive filtering — fewer hallucinations, may miss soft speech |

#### `--vad-min-speech-ms`
Minimum duration of a speech window to be kept. Short utterances below this
threshold are discarded.

| Value | Effect |
|---|---|
| `100` | Keeps very short utterances (single words, exclamations) |
| `250` | Default |
| `500` | Only keeps sustained speech — good for filtering out brief noise |

#### `--vad-min-silence-ms`
Minimum silence gap that causes VAD to split into separate segments.
Also controls how much silence is required before a speech region ends.

| Value | Effect |
|---|---|
| `500` | Splits on short pauses — more, shorter segments |
| `2000` | Default — splits on 2-second pauses |
| `4000` | Fewer splits — longer segments, joins sentences spoken with short pauses |

#### `--vad-speech-pad-ms`
Padding (in milliseconds) that faster-whisper's VAD adds before and after
each detected speech region. This **directly shifts subtitle start
timestamps**: higher padding = subtitle appears earlier on screen than the
speaker starts.

The faster-whisper library default is `400` ms, which causes subtitles to
appear visibly before the speaker. GenSRT ships a tighter `200` ms default.

| Value | Effect |
|---|---|
| `100` | Tightest alignment — minor risk of clipping the first syllable |
| `200` | Default — balanced |
| `400` | faster-whisper library default — subtitles appear noticeably early |

---

### Model Selection

```powershell
--model NAME    # Default: large-v3-turbo
```

| Model | Speed | Accuracy | VRAM |
|---|---|---|---|
| `tiny` | Fastest | Lowest | ~1 GB |
| `base` | Very fast | Low | ~1 GB |
| `small` | Fast | Moderate | ~2 GB |
| `medium` | Moderate | Good | ~5 GB |
| `large-v3` | Slow | Best | ~10 GB |
| `large-v3-turbo` | Fast | Very good | ~6 GB |

`large-v3-turbo` is the recommended default — it is nearly as accurate as
`large-v3` at roughly 3× the speed. For Korean content where accuracy matters
more than speed, `large-v3` may produce better results.

---

### Compute Type

```powershell
--compute-type TYPE    # Default: float16
```

| Type | Device | Speed | Memory |
|---|---|---|---|
| `float16` | GPU | Fastest | Highest |
| `int8_float16` | GPU | Fast | Lower |
| `int8` | CPU | Slow | Lowest |

GenSRT automatically falls back through this list if the GPU does not support
the requested compute type.

---

## Configuration File

All knobs can be persisted in `gensrt-config.json` so you don't repeat them
on every run.

```powershell
# Generate a default config file
gensrt --init-config

# Show the fully resolved config for the current run
gensrt --dump-config
```

Example `gensrt-config.json` for Korean → Malayalam workflow:

```json
{
  "model": "large-v3-turbo",
  "source_language": "ko",
  "target_language": "ml",
  "translate": true,
  "translation_engine": "google",
  "vad_enabled": true,
  "vad_threshold": 0.4,
  "vad_min_silence_ms": 1000,
  "max_subtitle_duration_s": 8.0
}
```

Precedence order (highest wins):

  **GUI footer overrides > CLI flags > config file > built-in defaults**

The Config modal in the GUI writes to the same file. Saving from the modal
also syncs the footer selectors immediately — no page refresh needed.

---

## First Run — Model Download

On the first run, Whisper downloads the model weights (~800 MB for
`large-v3-turbo`) to:

```
%USERPROFILE%\.cache\huggingface\hub\
```

Subsequent runs use the cached weights and start immediately.

---

## Building a Standalone Executable

```powershell
# From the project root with venv activated
.\Pack-gensrt.ps1
```

Output: `.\dist\gensrt\gensrt.exe`

Before the first pack, drop `ffmpeg.exe` and `ffprobe.exe` (from the
gyan.dev "essentials" build at https://www.gyan.dev/ffmpeg/builds/) into
`gensrt\bin\`. The pack script verifies they're present and fails loudly
if either is missing.

The packaged executable does not bundle the Whisper model weights — they're
downloaded on first run as above. CUDA 12.8 must be installed on the
target machine.

---

## Known Limitations

GenSRT is a draft tool for serious subtitle work. The AI does the heavy
lifting; you polish the result. A few things to know:

- **Indic-language source audio** — Whisper struggles with fast, dense
  speech in Indian languages, particularly news broadcasts with English
  code-switching. Output may contain hallucinated content that reads
  fluently but doesn't match the audio. For these cases, generate a draft
  and verify against the audio before publishing. (Better Indic
  transcription is planned for version 1.2.)
- **Code-switching audio** — recordings that mix multiple languages
  (e.g., English-heavy Hindi conversations) may produce inconsistent
  script output.
- **Music-heavy content** — Whisper occasionally hallucinates subtitles
  over long musical passages. Use the editor to remove them.
- **Speaker overlap** — overlapping speech from multiple speakers is
  transcribed as a single segment without speaker attribution.
- **Long videos** — for videos over 60 minutes, monitor VRAM usage;
  consider splitting first if memory becomes an issue.

---

## Roadmap

- **IndicConformer ASR engine** for Indian-language source audio (v1.2) —
  alternative ASR engine that handles fast, dense Indic speech significantly
  better than Whisper, with VAD-based chunking and per-region timestamp
  assembly. See *Known Limitations* above for the current state.
- **`--target-language` CLI flag** — currently target language is set via the
  config file or the GUI footer. A direct flag would round out the CLI
  surface for batch jobs that want per-language output without writing a
  config.

---

## What's new

- WebVTT output written alongside every SRT — `.vtt` works in HTML5
  `<video>` elements natively as a `<track>` source
- Live in-player subtitle display while editing — every Split, Merge,
  Delete, or text edit shows in the player immediately
- Burn-in subtitles — bake subtitles into a copy of the video with one
  click, runs in the background
- Bundled ffmpeg — no separate install required on target machines
- Any-to-any translation via Google Translate (Korean → Malayalam,
  Japanese → Tamil, etc.)
- Plex / Jellyfin / Kodi compatible filename suffixes
- Self-contained `user_guide.html` shipped alongside the executable
- Documented known limitations for transparency about what the tool
  does and doesn't do well
