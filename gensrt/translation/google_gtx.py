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
    If Google fails, individual segments are retried via MyMemory (free,
    no key required, lower quality but reliable).
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
_GLUE = "\n{{SPLIT}}\n"
_GLUE_RE = re.compile(r"\s*\{\{SPLIT\}\}\s*", re.IGNORECASE)
# Pre-computed at module load — the glue is constant ASCII + control chars,
# so its URL-encoded byte count never changes.
_GLUE_URL_BYTES = len(quote(_GLUE, safe=""))

_MAX_RETRIES = 3
_RETRY_BASE_S = 0.25
_TIMEOUT_S = 15.0


class GoogleGTXEngine(TranslationEngine):
    """Translation via the unofficial Google Translate GTX endpoint."""

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

        src = source_language if source_language not in ("auto", "") else "auto"
        tgt = target_language or "en"
        results: list[str] = [""] * len(texts)

        for chunk_indices, chunk_texts in _make_chunks(texts):
            try:
                translated = self._gtx_glue_batch(chunk_texts, src, tgt)
            except Exception as exc:
                logger.warning(
                    "[google-gtx] Batch failed (%s) — falling back to MyMemory.", exc
                )
                translated = [
                    self._mymemory_single(t, src, tgt) for t in chunk_texts
                ]

            for idx, text in zip(chunk_indices, translated):
                results[idx] = text

        return results

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
            logger.warning(
                "[google-gtx] Glue split mismatch: expected %d, got %d — "
                "padding with originals.",
                len(texts), len(parts),
            )
            while len(parts) < len(texts):
                parts.append(texts[len(parts)])
            parts = parts[: len(texts)]

        return parts

    def _gtx_single(self, text: str, src: str, tgt: str = "en") -> str:
        """Single-text GTX request."""
        params = {"client": "gtx", "sl": src, "tl": tgt, "dt": "t", "q": text}
        data = self._fetch_gtx(params)
        return "".join(seg[0] for seg in data[0] if seg[0])

    def _fetch_gtx(self, params: dict) -> list:
        """GET the GTX endpoint with retry/back-off. Returns parsed JSON."""
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.get(
                    _GTX_URL, params=params, timeout=_TIMEOUT_S
                )
                if resp.status_code in (429, 503):
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                if not data or not data[0]:
                    raise ValueError("Empty GTX response")
                return data
            except Exception as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES:
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

