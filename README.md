# GenSRT

GPU-accelerated subtitle (SRT) generation using OpenAI Whisper, with optional
translation to English via Google Translate.

---

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA 12.8 (CPU fallback available, significantly slower)
- FFmpeg on system PATH
- Internet connection for Google translation (first Whisper model download also requires internet)

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

## Quick Start

```powershell
# Transcribe a single file (GUI launches if no --input given)
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

## Tuning Parameters

GenSRT exposes several knobs for improving subtitle quality.
All parameters can be set via CLI flags or persisted in `gensrt-config.json`.

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

```powershell
# Cap at 7 seconds
gensrt --input video.mkv --max-subtitle-duration 7.0
```

---

### Voice Activity Detection (VAD)

VAD runs inside Whisper on the GPU and removes silence before transcription.
It dramatically reduces processing time and hallucinated subtitles on silent sections.

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

```powershell
# Missing speech in a quiet video
gensrt --input video.mkv --vad-threshold 0.3

# Too many hallucinations in a noisy video
gensrt --input video.mkv --vad-threshold 0.7
```

#### `--vad-min-speech-ms`
Minimum duration of a speech window to be kept. Short utterances below this
threshold are discarded.

| Value | Effect |
|---|---|
| `100` | Keeps very short utterances (single words, exclamations) |
| `250` | Default |
| `500` | Only keeps sustained speech — good for filtering out brief noise |

```powershell
# Keep short single-word utterances
gensrt --input video.mkv --vad-min-speech-ms 100
```

#### `--vad-min-silence-ms`
Minimum silence gap that causes VAD to split into separate segments.
Also controls how much silence is required before a speech region ends.

| Value | Effect |
|---|---|
| `500` | Splits on short pauses — more, shorter segments |
| `2000` | Default — splits on 2-second pauses |
| `4000` | Fewer splits — longer segments, joins sentences spoken with short pauses |

```powershell
# Speaker pauses briefly mid-sentence — join them
gensrt --input video.mkv --vad-min-silence-ms 500

# Lots of missing speech — reduce the silence requirement
gensrt --input video.mkv --vad-min-silence-ms 1000
```

#### `--vad-speech-pad-ms`
Padding (in milliseconds) that faster-whisper's VAD adds before and after each
detected speech region. This **directly shifts subtitle start timestamps**:
higher padding = subtitle appears earlier on screen than the speaker starts.

The faster-whisper library default is `400` ms, which causes subtitles to appear
visibly before the speaker. GenSRT ships a tighter `200` ms default.

| Value | Effect |
|---|---|
| `100` | Tightest alignment — minor risk of clipping the first syllable |
| `200` | Default — balanced |
| `400` | faster-whisper library default — subtitles appear noticeably early |

```powershell
# Subtitles still appear too early — tighten further
gensrt --input video.mkv --vad-speech-pad-ms 100

# First syllable gets clipped — loosen
gensrt --input video.mkv --vad-speech-pad-ms 300
```

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
`large-v3` at roughly 3× the speed. For Korean content where accuracy
matters more than speed, `large-v3` may produce better results.

```powershell
# Higher accuracy for Korean
gensrt --input video.mkv --model large-v3

# Fast pass on easy content
gensrt --input video.mkv --model medium
```

---

### Translation Engine

```powershell
--translation-engine ENGINE    # Default: google
--no-translate                 # Output in source language
--source-language LANG         # Default: auto  |  e.g. ja, ko, ml
```

| Engine | Key | Requires internet | Notes |
|---|---|---|---|
| Google Translate (GTX) | `google` | Yes | Default. Best quality, especially for Malayalam |
| Meta NLLB-200 | `nllb` | No (after download) | Offline. Single model for all languages |
| Helsinki MarianMT | `marian` | No (after download) | Offline. Per-language models |
| None (passthrough) | `none` | No | Output in source language |

```powershell
# Offline translation
gensrt --input video.mkv --translation-engine nllb

# No translation (keep Japanese)
gensrt --input video.mkv --no-translate

# Explicit source language (faster than auto-detect)
gensrt --input video.mkv --source-language ja
```

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

All CLI parameters can be persisted in `gensrt-config.json` so you do not
need to repeat them on every run.

```powershell
# Generate a default config file
gensrt --init-config

# Show the fully resolved config for the current run
gensrt --dump-config
```

Example `gensrt-config.json`:
```json
{
  "model": "large-v3-turbo",
  "translation_engine": "google",
  "vad_enabled": true,
  "vad_threshold": 0.4,
  "vad_min_silence_ms": 1000,
  "max_subtitle_duration_s": 8.0
}
```

Precedence order (highest wins): **CLI flags > config file > built-in defaults**

---

## Output File Naming

| Input | Output |
|---|---|
| `video.mkv` | `video.srt` (same folder) |
| `--output /subs` | `/subs/video.srt` |
| `--output-filename custom.srt` | `custom.srt` (same folder as input) |

When both `--output` and `--output-filename` are set, `--output-filename` wins
and `--output` is ignored with a warning.

---

## First Run — Model Download

On the first run, Whisper will download the model weights (~800 MB for
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

The packaged executable does not bundle the Whisper model weights — they
are downloaded on first run as above. FFmpeg and CUDA 12.8 must be
installed on the target machine.
