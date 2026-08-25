"""Google GTX translation engine.

Makes direct HTTP calls to the unofficial Google Translate GTX endpoint —
exactly the same approach used by the gTranslate browser extension.
No API key, no ``googletrans`` package required.

Endpoint:
    https://translate.googleapis.com/translate_a/single
    ?client=gtx&sl={from}&tl={to}&dt=t&q={text}

Batching:
    Multiple subtitle segments are joined with a ``\\n{{SPLIT}}\\n`` glue
    string, sent as a single request, and split apart on the response.
    This mirrors the extension's ``googleGlueBatchTranslate`` approach and
    reduces network round-trips from N to ~1 per batch.

Fallback:
    What happens when a GTX batch fails is configurable
    (``translation_fallback`` in gensrt-config.json, wired in by the
    factory):

        nllb      translate the failed batch offline via NLLB-200 (default)
        mymemory  per-segment retry via MyMemory (free, no key, low quality)
        none      keep the source text for the failed batch

    Failures are counted and summarised once at the end of the batch run —
    a throttled IP fails *every* batch, and eighty identical WARNING lines
    communicate nothing that one summary line does not.

Rate limiting:
    The GTX endpoint throttles by IP, and repeated heavy bursts (a nightly
    batch of multi-hour files) can leave an IP throttled for hours — at
    which point in-run retries cannot help, only the fallback can.  To
    avoid *becoming* throttled, requests are paced (`_BATCH_DELAY_S`
    between batch requests) and HTTP 429/503 responses back off on their
    own, much longer, ladder — honouring Retry-After when Google sends it.
"""

from __future__ import annotations

import logging
import re
import time
from urllib.parse import quote

import requests

from gensrt.exceptions import TranslationError
from gensrt.translation.base import TranslationEngine

logger = logging.getLogger(__name__)

_GTX_URL = "https://translate.googleapis.com/translate_a/single"
_MYMEMORY_URL = "https://api.mymemory.translated.net/get"

# Budget the URL-encoded byte count of the q parameter, not the Unicode
# code point count of the input.  Google's GTX endpoint returns 400 Bad
# Request when the URL exceeds ~6–8 KB; we budget to 5.5 KB to leave
# headroom for the URL prefix, other params, and any padding Google
# might add on its side.
#
# Why bytes, not characters:
#   - Each ASCII character URL-encodes 1:1 (e.g. "abc" stays "abc").
#   - Each 2-byte UTF-8 character (Latin diacritics, Cyrillic, Greek)
#     expands to 6 URL-encoded characters (two %XX sequences).
#   - Each 3-byte UTF-8 character (Malayalam, Hindi, Tamil, Bengali, all
#     Indic scripts, most CJK) expands to 9 URL-encoded characters.
#   - Each 4-byte UTF-8 character (emoji, rare CJK) expands to 12.
#
# A character-count budget that works for ASCII content (the gTranslate
# browser-extension use case) sends 9× too much for Indic-script content
# (the GenSRT subtitle use case).  Budgeting by URL-encoded byte count
# handles every script class with one rule.
#
# Practical batch sizes under a 5500-byte budget:
#     ASCII:        ~5500 chars/batch
#     2-byte UTF-8: ~1800 chars/batch
#     3-byte UTF-8:  ~610 chars/batch  (Malayalam, Hindi, Tamil, ...)
#     4-byte UTF-8:  ~450 chars/batch  (emoji-heavy)
_BATCH_URL_BUDGET = 5500
_BATCH_MAX_ITEMS = 40
# Batch delimiter.
#
# This used to be "{{SPLIT}}", which worked only because the target was always
# English: SPLIT is an English word, so Google left it alone. Translating
# Malayalam to Korean turned every delimiter into {{분할}}, the split found one
# part instead of eighteen, and 93% of cues came back untranslated.
#
# The marker is now purely non-alphabetic, so there is no word for a
# translator to translate. The regex tolerates whitespace changes and a
# varying run length, because MT engines do reflow punctuation.
#
# Even so, no marker can be *guaranteed* to survive an opaque translation
# service — so the mismatch path below falls back to per-item translation
# rather than trusting the marker.
_GLUE = "\n@@@\n"
_GLUE_RE = re.compile(r"\s*@{2,}\s*")
# Pre-computed at module load — the glue is constant ASCII + control chars,
# so its URL-encoded byte count never changes.
_GLUE_URL_BYTES = len(quote(_GLUE, safe=""))

# Pause between requests on the per-cue path. Small enough to be invisible on
# a 100-cue file (~5s total), large enough not to look like a burst.
_PER_ITEM_DELAY_S = 0.05

# Pause between *batch* requests. A 3,000-cue file makes ~80 batch requests;
# firing them back-to-back is exactly the burst signature that gets an IP
# throttled, after which no in-run behaviour can recover. 0.4s costs ~30s on
# that file and keeps the request rate boring.
_BATCH_DELAY_S = 0.4

_MAX_RETRIES = 3
_RETRY_BASE_S = 0.25
# HTTP 429/503 get their own, much longer, backoff ladder. The short ladder
# above is sized for transient network hiccups; a rate limiter that answered
# 429 at t=0 will still answer 429 at t=0.25s, so retrying that fast merely
# adds to the request count being held against the IP. Retry-After, when
# Google sends it, overrides these (capped at _RATE_LIMIT_MAX_WAIT_S).
_RATE_LIMIT_DELAYS_S = (2.0, 8.0)
_RATE_LIMIT_MAX_WAIT_S = 30.0
_TIMEOUT_S = 15.0


class GoogleGTXEngine(TranslationEngine):
    """Translation via the unofficial Google Translate GTX endpoint.

    Args:
        fallback: What to do when a batch fails outright — ``"nllb"``,
            ``"mymemory"`` or ``"none"``.  Defaults to ``"mymemory"`` so
            that direct construction (tests, scripts) behaves exactly as it
            did before v1.2.7; the factory passes the configured value,
            whose *config* default is ``"nllb"``.
        fallback_engine_factory: Zero-arg callable returning a
            :class:`TranslationEngine`, used when ``fallback="nllb"``.
            Injected (rather than imported here) so this module never
            depends on the NLLB engine, and so tests can substitute a fake.
            Called lazily on the first failed batch and never on the happy
            path.
    """

    #: Set when a batch delimiter fails to survive translation. Once that
    #: happens for a given target language it will happen for every batch, so
    #: the remaining batches go straight to the per-cue path instead of
    #: wasting a failed request each.
    _glue_disabled: bool = False

    def __init__(self, fallback: str = "mymemory",
                 fallback_engine_factory=None) -> None:
        self._fallback = (fallback or "mymemory").lower()
        self._fallback_engine_factory = fallback_engine_factory
        self._fallback_engine: TranslationEngine | None = None

    def is_available(self) -> bool:
        return True   # pure HTTP — no package deps beyond requests

    @property
    def name(self) -> str:
        return "google-gtx"

    # ── Single text ────────────────────────────────────────────────────────

    def translate(self, text: str, source_language: str, target_language: str = "en") -> str:
        if not text.strip():
            return text
        results = self.translate_batch([text], source_language, target_language)
        return results[0]

    # ── Batch (primary interface used by pipeline) ─────────────────────────

    def translate_batch(self, texts: list[str], source_language: str, target_language: str = "en") -> list[str]:
        """Translate *texts* to *target_language* using GTX glue-string batching.

        Splits the list into chunks respecting ``_BATCH_MAX_CHARS`` /
        ``_BATCH_MAX_ITEMS``, sends each chunk as a single GTX request,
        falls back to MyMemory per-item if GTX fails for a chunk.
        """
        if not texts:
            return []

        # Reset per call: whether glue survives depends on the target
        # language, and a single engine instance may be reused across files.
        self._glue_disabled = False

        src = source_language if source_language not in ("auto", "") else "auto"
        tgt = target_language or "en"
        results: list[str] = [""] * len(texts)

        chunks = _make_chunks(texts)
        failed_batches = 0
        last_cause: Exception | None = None

        for n, (chunk_indices, chunk_texts) in enumerate(chunks):
            if n and not self._glue_disabled:
                # Pace the batch requests; the per-cue path paces itself.
                time.sleep(_BATCH_DELAY_S)
            try:
                if self._glue_disabled:
                    translated = self._translate_each(chunk_texts, src, tgt)
                else:
                    translated = self._gtx_glue_batch(chunk_texts, src, tgt)
            except Exception as exc:
                failed_batches += 1
                last_cause = exc
                # The first failure is worth a WARNING with its cause; the
                # remaining ones are the same story (a throttled IP fails
                # every batch identically) and go to DEBUG. One summary line
                # at the end carries the totals.
                log = logger.warning if failed_batches == 1 else logger.debug
                log(
                    "[google-gtx] Batch failed (%s) — handling via %r.",
                    exc, self._fallback,
                )
                translated = self._handle_failed_batch(chunk_texts, src, tgt)

            for idx, text in zip(chunk_indices, translated):
                results[idx] = text

        if failed_batches:
            logger.warning(
                "[google-gtx] %d of %d batches failed (last cause: %s) — "
                "handled via %r. If this recurs on every run, the GTX "
                "endpoint is likely rate-limiting this IP; consider "
                "translation_engine \"nllb\" to skip Google entirely.",
                failed_batches, len(chunks), last_cause, self._fallback,
            )

        return results

    # ── Failed-batch handling ──────────────────────────────────────────────

    def _handle_failed_batch(
        self, chunk_texts: list[str], src: str, tgt: str
    ) -> list[str]:
        """Apply the configured fallback to one failed batch.

        Every path returns a list the same length as *chunk_texts* and never
        raises: a failed batch degrades, it does not abort the file.
        """
        if self._fallback == "none":
            return list(chunk_texts)

        if self._fallback == "nllb":
            engine = self._get_fallback_engine()
            if engine is not None:
                try:
                    return engine.translate_batch(chunk_texts, src, tgt)
                except Exception as exc:
                    logger.warning(
                        "[google-gtx] NLLB fallback failed (%s) — keeping "
                        "originals for this batch.", exc,
                    )
            return list(chunk_texts)

        # "mymemory" — the pre-v1.2.7 behaviour.
        return [self._mymemory_single(t, src, tgt) for t in chunk_texts]

    def _get_fallback_engine(self):
        """Construct the injected fallback engine once; None on failure."""
        if self._fallback_engine is not None:
            return self._fallback_engine
        if self._fallback_engine_factory is None:
            logger.warning(
                "[google-gtx] fallback is 'nllb' but no fallback engine was "
                "injected — keeping originals. (Engines built by the factory "
                "always have one; this is reachable only via direct "
                "construction.)"
            )
            return None
        try:
            self._fallback_engine = self._fallback_engine_factory()
        except Exception as exc:
            logger.warning(
                "[google-gtx] Could not initialise the NLLB fallback (%s) — "
                "keeping originals for failed batches.", exc,
            )
            self._fallback_engine_factory = None   # do not retry every batch
            return None
        return self._fallback_engine

    # ── GTX glue-string request ────────────────────────────────────────────

    def _gtx_glue_batch(self, texts: list[str], src: str, tgt: str = "en") -> list[str]:
        """Translate *texts* via a single GTX request using glue strings."""
        if len(texts) == 1:
            return [self._gtx_single(texts[0], src, tgt)]

        combined = _GLUE.join(texts)
        params = {
            "client": "gtx",
            "sl": src,
            "tl": tgt,
            "dt": "t",
            "q": combined,
        }
        data = self._fetch_gtx(params)
        full = "".join(seg[0] for seg in data[0] if seg[0])

        parts = _GLUE_RE.split(full)

        if len(parts) != len(texts):
            # The delimiter did not survive. Padding the tail with originals
            # (what this did before v1.2.5) is the worst option available: it
            # silently emits untranslated text, and when the service merges
            # everything into one blob it leaves ~95% of cues untranslated
            # behind a single log line no GUI user sees. Worse, a *partial*
            # split shifts every later cue onto the wrong translation, which
            # looks plausible and is harder to notice than no translation.
            #
            # Translate the batch one cue at a time instead: slower, correct.
            logger.warning(
                "[google-gtx] Batch delimiter did not survive translation to "
                "%r (expected %d parts, got %d). Falling back to per-cue "
                "translation for the rest of this job — slower, but correct.",
                tgt, len(texts), len(parts),
            )
            self._glue_disabled = True
            return self._translate_each(texts, src, tgt)

        return parts

    def _translate_each(self, texts: list[str], src: str, tgt: str) -> list[str]:
        """Translate one cue per request.

        The correct-but-slow path, used when batching cannot be trusted. A
        short pause between requests keeps a 100-cue file from looking like a
        burst to the endpoint; the retry/back-off in _fetch_gtx handles the
        rate limiting that still gets through.
        """
        out: list[str] = []
        for i, text in enumerate(texts):
            if i:
                time.sleep(_PER_ITEM_DELAY_S)
            try:
                out.append(self._gtx_single(text, src, tgt))
            except Exception as exc:
                logger.warning(
                    "[google-gtx] Per-cue translation failed (%s) — "
                    "keeping the original for this cue.", exc
                )
                out.append(text)
        return out

    def _gtx_single(self, text: str, src: str, tgt: str = "en") -> str:
        """Single-text GTX request."""
        params = {"client": "gtx", "sl": src, "tl": tgt, "dt": "t", "q": text}
        data = self._fetch_gtx(params)
        return "".join(seg[0] for seg in data[0] if seg[0])

    def _fetch_gtx(self, params: dict) -> list:
        """GET the GTX endpoint with retry/back-off. Returns parsed JSON.

        Two backoff ladders: transient errors (timeouts, 5xx other than
        503, malformed responses) retry quickly on ``_RETRY_BASE_S``;
        rate-limit responses (429/503) wait on ``_RATE_LIMIT_DELAYS_S`` or
        the server's own Retry-After, because a limiter that just said no
        will keep saying no for a while, and rapid retries only deepen the
        hole the IP is in.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            rate_limited = False
            retry_after_s: float | None = None
            try:
                resp = requests.get(
                    _GTX_URL, params=params, timeout=_TIMEOUT_S
                )
                if resp.status_code in (429, 503):
                    rate_limited = True
                    retry_after_s = _parse_retry_after(
                        resp.headers.get("Retry-After")
                    )
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                if not data or not data[0]:
                    raise ValueError("Empty GTX response")
                return data
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
                    if rate_limited:
                        if retry_after_s is not None:
                            delay = min(retry_after_s, _RATE_LIMIT_MAX_WAIT_S)
                        else:
                            ladder = _RATE_LIMIT_DELAYS_S
                            delay = ladder[min(attempt, len(ladder)) - 1]
                    else:
                        delay = _RETRY_BASE_S * (2 ** (attempt - 1))
                    logger.debug(
                        "[google-gtx] Attempt %d/%d failed (%s) — retry in %.1fs",
                        attempt, _MAX_RETRIES, exc, delay,
                    )
                    time.sleep(delay)

        raise TranslationError("google", f"GTX failed after {_MAX_RETRIES} attempts: {last_exc}")

    # ── MyMemory fallback ──────────────────────────────────────────────────

    def _mymemory_single(self, text: str, src: str, tgt: str = "en") -> str:
        """Translate a single text via MyMemory (fallback)."""
        if not text.strip():
            return text
        lang_pair = f"{src}|{tgt}"
        try:
            resp = requests.get(
                _MYMEMORY_URL,
                params={"q": text, "langpair": lang_pair},
                timeout=_TIMEOUT_S,
            )
            resp.raise_for_status()
            data = resp.json()
            translated = data.get("responseData", {}).get("translatedText", "")
            if translated:
                return translated
        except Exception as exc:
            logger.debug("[google-gtx] MyMemory fallback failed: %s", exc)
        return text   # return original on complete failure


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header. Seconds form only; HTTP-date is rare
    enough from this endpoint that falling back to the ladder is fine."""
    if not value:
        return None
    try:
        secs = float(value.strip())
    except (TypeError, ValueError):
        return None
    return secs if secs >= 0 else None


# ── Chunk helpers ──────────────────────────────────────────────────────────

def _encoded_byte_count(text: str) -> int:
    """URL-encoded byte length of *text* — what Google's URL parser sees.

    ASCII characters stay at 1 byte; non-ASCII characters expand to 3
    URL-encoded characters per UTF-8 byte.  Returns 0 for empty input.
    """
    if not text:
        return 0
    return len(quote(text, safe=""))


def _make_chunks(
    texts: list[str],
) -> list[tuple[list[int], list[str]]]:
    """Split *texts* into (indices, texts) chunks within batch limits.

    Two caps apply per batch:
        * URL-encoded byte budget (:data:`_BATCH_URL_BUDGET`) — protects
          against Google's GTX URL length limit.  This is the cap that
          actually matters for Indic scripts.
        * Item count cap (:data:`_BATCH_MAX_ITEMS`) — keeps the glue-split
          alignment on Google's side reliable.  For very short ASCII
          texts the byte budget would happily put hundreds in one batch,
          but the larger the batch the higher the chance Google
          re-orders or merges some of our glue strings.

    A single text larger than the byte budget is still allowed through
    in a batch of its own; the GTX request will likely fail, and the
    per-chunk fallback in :meth:`translate_batch` will hand it to
    MyMemory.  This is the correct behavior — we'd rather fail one
    long cue gracefully than refuse to translate it at all.
    """
    chunks: list[tuple[list[int], list[str]]] = []
    cur_indices: list[int] = []
    cur_texts: list[str] = []
    cur_url_bytes = 0

    for i, text in enumerate(texts):
        text_bytes = _encoded_byte_count(text)
        # Every item after the first in a batch is preceded by a glue.
        # Account for that when checking the budget for this candidate.
        glue_bytes = _GLUE_URL_BYTES if cur_texts else 0

        if cur_texts and (
            len(cur_texts) >= _BATCH_MAX_ITEMS
            or cur_url_bytes + glue_bytes + text_bytes > _BATCH_URL_BUDGET
        ):
            chunks.append((cur_indices, cur_texts))
            cur_indices, cur_texts = [], []
            cur_url_bytes = 0
            glue_bytes = 0  # first item of new batch has no preceding glue

        cur_indices.append(i)
        cur_texts.append(text)
        cur_url_bytes += glue_bytes + text_bytes

    if cur_texts:
        chunks.append((cur_indices, cur_texts))

    if logger.isEnabledFor(logging.DEBUG):
        sizes = [len(t) for _i, t in chunks]
        url_bytes = []
        for _i, ts in chunks:
            n = sum(_encoded_byte_count(x) for x in ts)
            n += max(0, len(ts) - 1) * _GLUE_URL_BYTES
            url_bytes.append(n)
        logger.debug(
            "[google-gtx] _make_chunks: %d texts → %d batches "
            "(items per batch: %s, URL bytes per batch: %s)",
            len(texts), len(chunks), sizes, url_bytes,
        )

    return chunks

