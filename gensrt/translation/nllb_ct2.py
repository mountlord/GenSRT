"""NLLB-200 offline translation engine on CTranslate2.

Why this exists
---------------
Translation via the unofficial Google GTX endpoint is rate-limited by IP,
and a nightly batch of multi-hour recordings — thousands of cues per file —
is exactly the traffic pattern that gets an IP throttled for hours at a
time.  Once that happens, every batch fails, the per-batch fallback fires
thousands of times, and a 74-minute run spends 50 of those minutes waiting
on a fallback service whose output quality was never acceptable anyway.

The durable fix is to not need the network at all.  NLLB-200 (No Language
Left Behind, Meta AI) is a single multilingual model covering 200 languages
— including all three of GenSRT's primary use cases (Malayalam, Korean,
Japanese) and every language in the GUI's target dropdown — and it runs on
CTranslate2, the same runtime GenSRT already ships for Whisper.  No new
runtime, no PyTorch, no API, no rate limit.

Dependencies — deliberately zero new ones
-----------------------------------------
* ``ctranslate2``  — already here; it runs Whisper.
* ``tokenizers``   — already here; it arrives with faster-whisper and is
                     already named in the PyInstaller hiddenimports.
* ``huggingface_hub`` — already here; faster-whisper downloads through it.

The model repository ships a ``tokenizer.json`` (HuggingFace fast-tokenizer
serialisation), which the ``tokenizers`` package loads directly — no
``transformers``, no ``sentencepiece``, no torch.

License — read this before shipping changes here
------------------------------------------------
The NLLB-200 *weights* are licensed **CC-BY-NC-4.0** by Meta:
non-commercial use only.  That restriction travels with the weights
regardless of who converted or hosts them, and it binds the *user of the
weights*, not GenSRT (which remains AGPL-3.0).  GenSRT therefore:

* logs :data:`LICENSE_NOTICE` every time the engine loads — the notice
  must exist at the point of use, not only in a README nobody reads;
* documents the restriction and the opt-outs in README.md
  ("Offline translation (NLLB)");
* keeps ``translation_fallback: "none"`` and ``"mymemory"`` available for
  users whose work is commercial.

The notice states what the license says and nothing more.  What counts as
"commercial" is Meta's license text, Meta's ambiguity — GenSRT does not
interpret it.

The model
---------
Default: ``mijuanlo/nllb-200-distilled-600M-ct2-int8`` — a pre-converted
int8 CTranslate2 export of ``facebook/nllb-200-distilled-600M``.

* ~651 MB on disk (int8 storage), one-time download into ``models/``.
* Loads as ``int8_float16`` on CUDA and ``int8`` on CPU — the same compute
  ladder GenSRT already uses for Whisper models, and exactly what the
  repository's own model card recommends.
* Contains everything needed: ``model.bin``, ``shared_vocabulary.json``,
  ``config.json``, ``tokenizer.json`` and ``sentencepiece.bpe.model``.

The repo is configurable (``translation_model`` in gensrt-config.json), so
switching to a self-published, verified conversion later — or pointing at a
locally-converted directory under ``models/`` — is a config change, not a
code change.  A third-party HuggingFace repo can vanish or be re-uploaded;
pinning our own copy is planned follow-up work.

Tokenisation recipe
-------------------
NLLB was trained with the source sequence

    [src_lang] piece piece … ["</s>"]

and decodes behind a target-language prefix ``[tgt_lang]``.  This matches
both the CTranslate2 documentation example and the model card of the
default repository.  Language identifiers are FLORES-200 codes
(``kor_Hang``, ``mal_Mlym``…), so the ISO 639-1 codes GenSRT uses
everywhere else are mapped through :data:`ISO_TO_FLORES`.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

from gensrt.exceptions import TranslationError
from gensrt.translation.base import TranslationEngine

logger = logging.getLogger(__name__)

#: Default model reference.  A HuggingFace repo ID, a bare directory name
#: under ``models/``, or a full path — resolved by :func:`model_dir_for`.
DEFAULT_MODEL = "mijuanlo/nllb-200-distilled-600M-ct2-int8"

#: Logged at every engine load and at download time.  See module docstring.
LICENSE_NOTICE = (
    "NLLB-200 weights are licensed CC-BY-NC-4.0 by Meta — non-commercial "
    "use only. Commercial users: set translation_fallback (and, if set as "
    "the primary engine, translation_engine) to another value. "
    "See README, 'Offline translation (NLLB)'."
)

#: Approximate one-time download size, for status messages.
DOWNLOAD_SIZE_HINT = "~650 MB"

#: Files worth downloading from the model repo.  Everything else (README,
#: .gitattributes) is noise; restricting the patterns also protects against
#: a repo that later grows unrelated large files.
_DOWNLOAD_PATTERNS = (
    "model.bin",
    "config.json",
    "generation_config.json",
    "shared_vocabulary*",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)

# ── ISO 639-1 → FLORES-200 ────────────────────────────────────────────────
# Covers every language offered in the GUI source/target dropdowns, plus
# the handful Whisper commonly detects beyond them.  NLLB supports 200
# languages; extend here as needed — the FLORES code list is at
# https://github.com/facebookresearch/flores/blob/main/flores200/README.md
ISO_TO_FLORES: dict[str, str] = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "hi": "hin_Deva",
    "ml": "mal_Mlym",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "kn": "kan_Knda",
    "bn": "ben_Beng",
    "mr": "mar_Deva",
    "gu": "guj_Gujr",
    "pa": "pan_Guru",
    "ar": "arb_Arab",
    "tr": "tur_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "pl": "pol_Latn",
    "nl": "nld_Latn",
    "sv": "swe_Latn",
    "da": "dan_Latn",
    "no": "nob_Latn",
    "fi": "fin_Latn",
    "cs": "ces_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "hu": "hun_Latn",
    "ro": "ron_Latn",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "fa": "pes_Arab",
    "tl": "tgl_Latn",
}

#: Tokens that are markup rather than text.  Stripped from hypotheses
#: before detokenisation; a FLORES language token is also stripped when it
#: leads the hypothesis (CTranslate2 echoes the target prefix back).
_SPECIAL_TOKENS = frozenset({"<s>", "</s>", "<pad>", "<unk>"})

_FLORES_CODES = frozenset(ISO_TO_FLORES.values())


def flores_code(iso: str) -> str:
    """Map an ISO 639-1 code (or a FLORES code, passed through) to FLORES-200.

    Raises:
        TranslationError: For ``auto``/empty (NLLB must be told the source
            language — the pipeline always has the detected language by
            translation time, so hitting this means a wiring bug) and for
            languages not in :data:`ISO_TO_FLORES`.
    """
    code = (iso or "").strip()
    if code in _FLORES_CODES:
        return code
    code = code.lower()
    if not code or code == "auto":
        raise TranslationError(
            "nllb",
            "NLLB requires a concrete source language and received "
            f"{iso!r}. The pipeline passes the language Whisper detected, "
            "so this indicates a bug — please report it.",
        )
    mapped = ISO_TO_FLORES.get(code)
    if mapped is None:
        raise TranslationError(
            "nllb",
            f"No FLORES-200 mapping for language code {iso!r}. NLLB-200 "
            f"supports 200 languages, but GenSRT only maps the codes it "
            f"offers in its own UI ({', '.join(sorted(ISO_TO_FLORES))}). "
            f"The mapping table is gensrt/translation/nllb_ct2.py — adding "
            f"a language is a one-line change.",
        )
    return mapped


# ── Model location & download ─────────────────────────────────────────────

def model_dir_for(ref: str | None = None) -> Path:
    """Where the NLLB model for *ref* lives (or will live) on disk.

    Follows the same convention as Whisper models — the ``models``
    directory beside the executable (see :mod:`gensrt.model_paths`):

    * A reference containing a path separator that names an existing
      directory is used as given.
    * A bare name found under any conventional models directory wins.
    * Otherwise: ``models/<leaf-of-ref>`` — the download target for a
      HuggingFace repo ID.
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
    """Whether the model directory exists and holds the files the engine needs."""
    d = model_dir_for(ref)
    return (
        (d / "model.bin").is_file()
        and (
            (d / "tokenizer.json").is_file()
            or (d / "sentencepiece.bpe.model").is_file()
        )
    )


def ensure_model(ref: str | None = None, *, status=None) -> Path:
    """Make sure the NLLB model is on disk; download it if it is not.

    Called from the pipeline *before* any transcription work starts, so the
    one-time fetch happens in the same run — and the same interactive
    moment — as a first-time Whisper model download, never lazily in the
    middle of an unattended job.

    Args:
        ref:    Model reference (repo ID / bare name / path).  ``None``
                uses :data:`DEFAULT_MODEL`.
        status: Optional ``(str) -> None`` callback for GUI/CLI progress.

    Returns:
        The model directory.

    Raises:
        TranslationError: If the model is absent and cannot be downloaded
            (offline, blocked, out of disk…), with the cause and the manual
            alternative spelled out.
    """
    ref = ref or DEFAULT_MODEL
    target = model_dir_for(ref)
    if is_model_present(ref):
        logger.debug("NLLB model present: %s", target)
        return target

    if "/" not in ref.replace("\\", "/"):
        # A bare name that is not on disk cannot be downloaded — bare names
        # are the *local directory* convention, not repo IDs.
        raise TranslationError(
            "nllb",
            f"NLLB model {ref!r} was not found under any models directory "
            f"and is not a HuggingFace repo ID, so it cannot be downloaded. "
            f"Expected it at: {target}",
        )

    msg = (
        f"Downloading NLLB translation model ({DOWNLOAD_SIZE_HINT}, "
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
            "nllb",
            f"Could not download the NLLB model {ref!r}: {exc}\n\n"
            f"If this machine is offline or HuggingFace is unreachable, "
            f"download the repository on another machine and place its "
            f"contents in:\n  {target}\n"
            f"(the directory must contain model.bin and tokenizer.json).",
        ) from exc

    if not is_model_present(ref):
        raise TranslationError(
            "nllb",
            f"Download of {ref!r} completed but {target} does not contain "
            f"the expected files (model.bin plus tokenizer.json or "
            f"sentencepiece.bpe.model). The repository layout may have "
            f"changed — check https://huggingface.co/{ref}",
        )

    logger.info("NLLB model ready: %s", target)
    return target


# ── The engine ────────────────────────────────────────────────────────────

class NLLBCT2Engine(TranslationEngine):
    """Offline translation via NLLB-200 on CTranslate2.

    Loading is lazy and happens once per instance: as the *primary* engine
    the model loads on the first batch; as a *fallback* it loads only if
    Google actually fails, so the healthy path pays nothing.

    Device selection mirrors the Whisper loader's philosophy: honour an
    explicit request, otherwise probe; degrade CUDA→CPU with a loud warning
    rather than failing the run.  Compute types follow the model card:
    ``int8_float16`` on CUDA, ``int8`` on CPU.
    """

    #: Beam size from the default repository's model card.  Translation
    #: decodes are short (subtitle cues), so the quality/speed trade of a
    #: modest beam is cheap.
    _BEAM_SIZE = 4

    #: CTranslate2 token-count batch cap, as recommended by the model card.
    _MAX_BATCH_TOKENS = 1024

    def __init__(self, config=None) -> None:
        self._model_ref: str = (
            getattr(config, "translation_model", None) or DEFAULT_MODEL
        )
        self._requested_device: str = (
            getattr(config, "device", None) or "auto"
        ).strip().lower()
        self._translator = None      # ctranslate2.Translator, once loaded
        self._tokenizer = None       # tokenizers.Tokenizer, once loaded
        self._sp = None              # sentencepiece fallback, if ever used
        self._lock = threading.Lock()

    # -- TranslationEngine interface --------------------------------------

    @property
    def name(self) -> str:
        return "nllb"

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
        if not texts:
            return []

        src = flores_code(source_language)
        tgt = flores_code(target_language or "en")

        self._load()

        # Empty/whitespace cues pass through untouched, and are excluded
        # from the model call entirely — an empty source sequence invites
        # the decoder to invent something.
        work_indices = [i for i, t in enumerate(texts) if t.strip()]
        results = list(texts)
        if not work_indices:
            return results

        sources = [self._encode(texts[i], src) for i in work_indices]
        target_prefix = [[tgt]] * len(sources)

        translations = self._translator.translate_batch(
            sources,
            target_prefix=target_prefix,
            batch_type="tokens",
            max_batch_size=self._MAX_BATCH_TOKENS,
            beam_size=self._BEAM_SIZE,
        )

        for i, tr in zip(work_indices, translations):
            results[i] = self._decode(tr.hypotheses[0])
        return results

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

            attempts = [device] if device == "cpu" else [device, "cpu"]
            last_exc: Exception | None = None
            for dev in attempts:
                compute = "int8_float16" if dev == "cuda" else "int8"
                try:
                    self._translator = ctranslate2.Translator(
                        str(model_dir), device=dev, compute_type=compute
                    )
                except Exception as exc:
                    last_exc = exc
                    logger.debug(
                        "NLLB load failed (device=%r, compute=%r): %s",
                        dev, compute, exc,
                    )
                    continue
                if dev != device:
                    logger.warning(
                        "NLLB: GPU unavailable — translating on CPU "
                        "(slower). Cause: %s", last_exc,
                    )
                logger.info(
                    "NLLB translator loaded: %s (device=%s, compute=%s)",
                    model_dir.name, dev, compute,
                )
                return

            raise TranslationError(
                "nllb",
                f"Failed to load NLLB model from {model_dir} on any of "
                f"{attempts}: {last_exc}",
            )

    def _load_tokenizer(self, model_dir: Path) -> None:
        """Prefer ``tokenizer.json`` via ``tokenizers`` (already a GenSRT
        dependency); fall back to raw sentencepiece if only the ``.bpe``
        model is present (a user's own minimal conversion, for example)."""
        tok_json = model_dir / "tokenizer.json"
        if tok_json.is_file():
            from tokenizers import Tokenizer

            self._tokenizer = Tokenizer.from_file(str(tok_json))
            return

        spm_file = model_dir / "sentencepiece.bpe.model"
        if spm_file.is_file():
            try:
                import sentencepiece as spm
            except ImportError as exc:
                raise TranslationError(
                    "nllb",
                    f"{model_dir} has only sentencepiece.bpe.model, and the "
                    f"'sentencepiece' package is not installed. Either add "
                    f"tokenizer.json to the model directory (present in the "
                    f"default repository) or `pip install sentencepiece`.",
                ) from exc
            self._sp = spm.SentencePieceProcessor()
            self._sp.Load(str(spm_file))
            return

        raise TranslationError(
            "nllb",
            f"No tokenizer found in {model_dir} (need tokenizer.json or "
            f"sentencepiece.bpe.model).",
        )

    # -- Token plumbing ---------------------------------------------------

    def _encode(self, text: str, src_flores: str) -> list[str]:
        """``[src_lang] pieces… </s>`` — the sequence NLLB was trained on."""
        if self._tokenizer is not None:
            pieces = self._tokenizer.encode(text, add_special_tokens=False).tokens
        else:
            pieces = self._sp.EncodeAsPieces(text)
        return [src_flores, *pieces, "</s>"]

    def _decode(self, tokens: list[str]) -> str:
        """Hypothesis tokens → text, dropping language and special tokens."""
        pieces = [
            t for t in tokens
            if t not in _SPECIAL_TOKENS and t not in _FLORES_CODES
        ]
        if self._tokenizer is not None:
            ids = [self._tokenizer.token_to_id(p) for p in pieces]
            ids = [i for i in ids if i is not None]
            return self._tokenizer.decode(ids, skip_special_tokens=True).strip()
        return self._sp.DecodePieces(pieces).strip()
