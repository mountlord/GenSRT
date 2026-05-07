"""VAD is handled entirely by faster-whisper's built-in filter.

Silero VAD was removed — it added 100-200s of overhead on a 108-minute
file with no quality benefit over faster-whisper's own VAD, which runs
on the same GPU as the model.

VAD is configured via TranscriptionConfig fields:
    vad_enabled        — passed as vad_filter=True to faster-whisper
    vad_threshold      — speech probability threshold
    vad_min_speech_ms  — minimum speech segment duration
    vad_min_silence_ms — minimum silence gap that splits segments

See gensrt/pipeline.py _run_whisper() for the implementation.
"""
