"""MADLAD-400 offline translation engine on CTranslate2.

Why this exists alongside NLLB
------------------------------
:mod:`gensrt.translation.nllb_ct2` solved the network problem — no API, no
rate limit, no PyTorch.  It did not solve the *licence* problem: the NLLB
weights are CC-BY-NC-4.0, so commercial users have to opt out of the
feature entirely, and the README carries a rider saying so.

MADLAD-400 (Google) is **Apache-2.0**, covers 400+ languages, and is a T5
model — an architecture CTranslate2 runs natively, exactly like NLLB.  So
it is the same deal with no licence rider, at the cost of disk:

    NLLB-200-600M    ~651 MB    CC-BY-NC-4.0   12 ms/cue   (measured, 3060 Ti)
    MADLAD-400-3B  ~2,976 MB    Apache-2.0     25 ms/cue   (measured, same)

Neither is strictly better.  MADLAD is five times the download and twice
the per-cue cost; whether it is worth that is a judgement about output
quality on the user's own material, which is why GenSRT offers both rather
than picking for them.

Dependencies — still zero new ones
----------------------------------
Same three as NLLB: ``ctranslate2``, ``tokenizers``, ``huggingface_hub``,
all already present.  The default repository ships ``tokenizer.json``,
which ``tokenizers`` reads directly.  A conversion that ships only a
sentencepiece model works too, if the ``sentencepiece`` package happens to
be installed — but that is a fallback, not a requirement.

Tokenisation recipe — the part that fails silently if you get it wrong
----------------------------------------------------------------------
MADLAD takes the TARGET language as a ``<2xx>`` token and infers the
source language itself (so, unlike NLLB, there is no source-language
argument and no FLORES mapping).  That token must be placed INSIDE the
source string BEFORE tokenising:

    encode(f"<2{target}> {text}")                      # correct
    ["<2en>", *encode(text)]                           # WRONG

Prepending it to the token list instead loses the word-boundary piece that
belongs in front of it, and the translation comes back mangled — with no
error raised anywhere.  If MADLAD output ever looks like fluent nonsense,
this line is the first place to look.

The model
---------
Default: ``olob0/madlad400-3b-mt-ct2-int8_float16`` — a pre-converted
CTranslate2 export of ``google/madlad400-3b-mt``.

Loads on the same device/compute ladder as NLLB and Whisper:
``int8_float16`` on CUDA, ``int8`` on CUDA for pre-Volta cards, ``int8``
on CPU.  Configurable via ``madlad_model``, which is deliberately a
SEPARATE setting from ``translation_model`` (that one names an NLLB repo;
pointing the two engines at one field would hand each the other's model).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from gensrt.exceptions import TranslationError
from gensrt.translation._clean import clean_mt_output
from gensrt.translation._oom import SLICE_CUES, translate_with_oom_ladder
from gensrt.translation.base import TranslationEngine

logger = logging.getLogger(__name__)

#: Default model reference.  A HuggingFace repo ID, a bare directory name
#: under ``models/``, or a full path — resolved by :func:`model_dir_for`.
DEFAULT_MODEL = "olob0/madlad400-3b-mt-ct2-int8_float16"

#: Logged once per load.  Unlike NLLB's, this notice is good news: it
#: exists so a user comparing the two engines can see the difference
#: without going to the README.
LICENSE_NOTICE = (
    "MADLAD-400 weights are licensed Apache-2.0 by Google — commercial use "
    "permitted. (The NLLB alternative is CC-BY-NC-4.0, non-commercial only.)"
)

#: Approximate one-time download size, for status messages.
DOWNLOAD_SIZE_HINT = "~2.9 GB"

#: Files worth downloading.  Everything else in the repo is noise, and
#: restricting the patterns protects against a repo that later grows
#: unrelated large files.
_DOWNLOAD_PATTERNS = (
    "model.bin",
    "config.json",
    "generation_config.json",
    "shared_vocabulary*",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "sentencepiece.model",
)

#: Tokens that are markup rather than text.
_SPECIAL_TOKENS = frozenset({"<s>", "</s>", "<pad>", "<unk>"})


def target_token(iso: str) -> str:
    """``"en"`` → ``"<2en>"`` — MADLAD's target-language marker.

    MADLAD uses plain language codes rather than FLORES-200 identifiers, so
    no mapping table is needed; the code GenSRT already carries is the code
    MADLAD wants. Codes it does not know produce poor output rather than an
    error, which is why this validates shape only.

    Raises:
        TranslationError: For an empty or ``auto`` target. Unlike the
            SOURCE language (which MADLAD infers), the target must be
            concrete — there is no "translate into whatever" mode.
    """
    code = (iso or "").strip().lower()
    if not code or code == "auto":
        raise TranslationError(
            "madlad",
            f"MADLAD needs a concrete target language and received {iso!r}. "
            f"MADLAD infers the SOURCE language on its own, but the target "
            f"must be named.",
        )
    return f"<2{code}>"


# ── Model location & download ─────────────────────────────────────────────

def model_dir_for(ref: str | None = None) -> Path:
    """Where the MADLAD model for *ref* lives (or will live) on disk.

    Same convention as Whisper and NLLB models — see
    :func:`gensrt.translation.nllb_ct2.model_dir_for`.
    """
    from gensrt.model_paths import (
        model_search_dirs,
        models_dir,
        normalize_model_ref,
    )

    ref = normalize_model_ref(ref or DEFAULT_MODEL) or DEFAULT_MODEL
    candidate = Path(ref)

    if (candidate.is_absolute() or any(s in ref for s in ("\\", "/"))) \
            and candidate.is_dir():
        return candidate

    leaf = ref.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    for root in model_search_dirs():
        local = root / leaf
        if local.is_dir():
            return local
    return models_dir() / leaf


def is_model_present(ref: str | None = None) -> bool:
    """Whether the model directory holds the files the engine needs."""
    d = model_dir_for(ref)
    return (d / "model.bin").is_file() and (
        (d / "tokenizer.json").is_file()
        or (d / "spiece.model").is_file()
        or (d / "sentencepiece.model").is_file()
    )


def ensure_model(ref: str | None = None, *, status=None) -> Path:
    """Make sure the MADLAD model is on disk; download it if it is not.

    Called from the pipeline before any transcription work starts, so the
    one-time ~2.9 GB fetch happens in the same interactive moment as a
    first-time Whisper download — never lazily mid-job.

    Raises:
        TranslationError: If the model is absent and cannot be downloaded,
            with the cause and the manual alternative spelled out.
    """
    ref = ref or DEFAULT_MODEL
    target = model_dir_for(ref)
    if is_model_present(ref):
        logger.debug("MADLAD model present: %s", target)
        return target

    if "/" not in ref.replace("\\", "/"):
        raise TranslationError(
            "madlad",
            f"MADLAD model {ref!r} was not found under any models directory "
            f"and is not a HuggingFace repo ID, so it cannot be downloaded. "
            f"Expected it at: {target}",
        )

    msg = (
        f"Downloading MADLAD translation model ({DOWNLOAD_SIZE_HINT}, "
        f"one-time): {ref} → {target}"
    )
    logger.info("%s", msg)
    logger.info("%s", LICENSE_NOTICE)
    if callable(status):
        status(f"Downloading translation model ({DOWNLOAD_SIZE_HINT}, one-time)…")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=ref,
            local_dir=str(target),
            allow_patterns=list(_DOWNLOAD_PATTERNS),
        )
    except Exception as exc:
        raise TranslationError(
            "madlad",
            f"Could not download the MADLAD model {ref!r}: {exc}\n\n"
            f"If this machine is offline or HuggingFace is unreachable, "
            f"download the repository on another machine and place its "
            f"contents in:\n  {target}\n"
            f"(the directory must contain model.bin and tokenizer.json).",
        ) from exc

    if not is_model_present(ref):
        raise TranslationError(
            "madlad",
            f"Download of {ref!r} completed but {target} does not contain "
            f"the expected files (model.bin plus tokenizer.json or a "
            f"sentencepiece model). The repository layout may have changed "
            f"— check https://huggingface.co/{ref}",
        )

    logger.info("MADLAD model ready: %s", target)
    return target


# ── The engine ────────────────────────────────────────────────────────────

class MADLADCT2Engine(TranslationEngine):
    """Offline translation via MADLAD-400 on CTranslate2.

    Loading is lazy and happens once per instance, and the device ladder
    matches NLLB's and Whisper's: honour an explicit request, otherwise
    probe; degrade CUDA→CPU loudly rather than failing the run.
    """

    #: Subtitle cues are short, so a modest beam costs little.
    _BEAM_SIZE = 4

    #: CTranslate2 token-count batch cap.
    _MAX_BATCH_TOKENS = 1024

    #: MADLAD will happily repeat itself on very short or noisy input —
    #: exactly what an OCR fragment or a half-heard cue looks like. Blocking
    #: repeated trigrams is the model card's own recommendation.
    _NO_REPEAT_NGRAM = 3

    def __init__(self, config=None) -> None:
        self._model_ref: str = (
            getattr(config, "madlad_model", None) or DEFAULT_MODEL
        )
        self._requested_device: str = (
            getattr(config, "device", None) or "auto"
        ).strip().lower()
        self._translator = None      # ctranslate2.Translator, once loaded
        # The token batch cap that last held on this device.  Starts at the
        # class default; the OOM ladder lowers it and it stays lowered, so a
        # small card walks the ladder once per process, not once per file.
        self._batch_tokens: int = self._MAX_BATCH_TOKENS
        self._tokenizer = None       # tokenizers.Tokenizer, once loaded
        self._sp = None              # sentencepiece fallback, if ever used
        self._lock = threading.Lock()

    # -- TranslationEngine interface --------------------------------------

    @property
    def name(self) -> str:
        return "madlad"

    def is_available(self) -> bool:
        try:
            import ctranslate2  # noqa: F401
        except ImportError:  # pragma: no cover — ct2 is a hard dependency
            return False
        return True

    def translate(self, text: str, source_language: str, target_language: str = "en") -> str:
        return self.translate_batch([text], source_language, target_language)[0]

    def translate_batch(
        self, texts: list[str], source_language: str, target_language: str = "en"
    ) -> list[str]:
        """Translate *texts* into *target_language*.

        ``source_language`` is accepted for interface compatibility and
        deliberately IGNORED: MADLAD infers the source itself. That is why
        this engine works on mixed-language input where NLLB would need the
        language named per cue.
        """
        if not texts:
            return []

        tgt = target_token(target_language or "en")

        self._load()

        # Empty cues pass through untouched and never reach the model — an
        # empty source sequence invites the decoder to invent something.
        work_indices = [i for i, t in enumerate(texts) if t.strip()]
        results = list(texts)
        if not work_indices:
            return results

        sources = [self._encode(texts[i], tgt) for i in work_indices]

        for start in range(0, len(sources), SLICE_CUES):
            chunk = sources[start:start + SLICE_CUES]

            def _run(max_batch: int, _chunk=chunk):
                return self._translator.translate_batch(
                    _chunk,
                    batch_type="tokens",
                    max_batch_size=max_batch,
                    beam_size=self._BEAM_SIZE,
                    no_repeat_ngram_size=self._NO_REPEAT_NGRAM,
                )

            translations, self._batch_tokens = translate_with_oom_ladder(
                _run, max_batch_tokens=self._batch_tokens,
                move_to_cpu=self._move_to_cpu, engine_name="MADLAD",
            )
            for i, tr in zip(work_indices[start:start + SLICE_CUES], translations):
                results[i] = self._decode(tr.hypotheses[0])
        return results

    def _move_to_cpu(self) -> None:
        """Drop the GPU translator and reload on CPU (the OOM ladder's last rung)."""
        import gc

        import ctranslate2

        with self._lock:
            model_dir = ensure_model(self._model_ref)
            self._translator = None
            gc.collect()
            self._translator = ctranslate2.Translator(str(model_dir), device="cpu", compute_type="int8")
            logger.info("MADLAD translator reloaded: %s (device=cpu, compute=int8)", model_dir.name)

    # -- Loading ----------------------------------------------------------

    def _load(self) -> None:
        """Load the translator and tokenizer, once, thread-safely."""
        if self._translator is not None:
            return
        with self._lock:
            if self._translator is not None:  # lost the race, work is done
                return

            model_dir = ensure_model(self._model_ref)
            logger.info("%s", LICENSE_NOTICE)

            self._load_tokenizer(model_dir)

            import ctranslate2

            device = self._requested_device
            if device in ("", "auto"):
                try:
                    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
                except Exception:
                    device = "cpu"

            # Same ladder as NLLB, including the cuda/int8 rung for pre-Volta
            # cards (Tesla P4, GTX 10-series) that cannot run int8_float16.
            if device == "cpu":
                attempts = [("cpu", "int8")]
            else:
                attempts = [
                    (device, "int8_float16"),
                    (device, "int8"),
                    ("cpu", "int8"),
                ]
            last_exc: Exception | None = None
            for dev, compute in attempts:
                try:
                    self._translator = ctranslate2.Translator(
                        str(model_dir), device=dev, compute_type=compute
                    )
                except Exception as exc:
                    last_exc = exc
                    logger.debug(
                        "MADLAD load failed (device=%r, compute=%r): %s",
                        dev, compute, exc,
                    )
                    continue
                if dev != device:
                    logger.warning(
                        "MADLAD: GPU unavailable — translating on CPU. This "
                        "is a 3B model; expect it to be slow. Cause: %s",
                        last_exc,
                    )
                elif (dev, compute) != attempts[0][:2] and dev == "cuda":
                    logger.info(
                        "MADLAD: this GPU has no efficient fp16 path — "
                        "using int8 on CUDA (pre-Volta card, e.g. "
                        "GTX 10-series / Tesla P4)."
                    )
                logger.info(
                    "MADLAD translator loaded: %s (device=%s, compute=%s)",
                    model_dir.name, dev, compute,
                )
                return

            raise TranslationError(
                "madlad",
                f"Failed to load MADLAD model from {model_dir} on any of "
                f"{[f'{d}/{c}' for d, c in attempts]}: {last_exc}",
            )

    def _load_tokenizer(self, model_dir: Path) -> None:
        """Prefer ``tokenizer.json`` via ``tokenizers`` (already a GenSRT
        dependency); fall back to sentencepiece only if that is all the
        conversion shipped."""
        tok_json = model_dir / "tokenizer.json"
        if tok_json.is_file():
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(str(tok_json))
            logger.debug("MADLAD tokenizer: tokenizer.json")
            return

        spm_file = next(
            (p for p in (model_dir / "spiece.model",
                         model_dir / "sentencepiece.model") if p.is_file()),
            None,
        )
        if spm_file is not None:
            try:
                import sentencepiece as spm
            except ImportError as exc:
                raise TranslationError(
                    "madlad",
                    f"{model_dir} ships only {spm_file.name}, and the "
                    f"'sentencepiece' package is not installed. Either use a "
                    f"conversion that includes tokenizer.json (the default "
                    f"repository does) or `pip install sentencepiece`.",
                ) from exc
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(str(spm_file))
            logger.debug("MADLAD tokenizer: %s", spm_file.name)
            return

        raise TranslationError(
            "madlad",
            f"No tokenizer found in {model_dir} (need tokenizer.json or a "
            f"sentencepiece model).",
        )

    # -- Token plumbing ---------------------------------------------------

    def _encode(self, text: str, tgt_token: str) -> list[str]:
        """``<2xx> text`` tokenised as ONE string.

        The target token goes inside the string on purpose — see the module
        docstring. Prepending it to the returned list instead produces
        mangled output and raises nothing.
        """
        prompt = f"{tgt_token} {text}"
        if self._tokenizer is not None:
            return self._tokenizer.encode(prompt, add_special_tokens=True).tokens
        return self._sp.encode(prompt, out_type=str)

    def _decode(self, tokens: list[str]) -> str:
        """Hypothesis tokens → text, dropping special tokens."""
        pieces = [t for t in tokens if t not in _SPECIAL_TOKENS]
        if self._tokenizer is not None:
            ids = [self._tokenizer.token_to_id(p) for p in pieces]
            ids = [i for i in ids if i is not None]
            return clean_mt_output(self._tokenizer.decode(ids, skip_special_tokens=True))
        return clean_mt_output(self._sp.decode(pieces))
