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

import requests

from gensrt.exceptions import TranslationError
from gensrt.translation.base import TranslationEngine

logger = logging.getLogger(__name__)

_GTX_URL = "https://translate.googleapis.com/translate_a/single"
_MYMEMORY_URL = "https://api.mymemory.translated.net/get"

# Same limits as the browser extension
_BATCH_MAX_CHARS = 4000
_BATCH_MAX_ITEMS = 40
_GLUE = "\n{{SPLIT}}\n"
_GLUE_RE = re.compile(r"\s*\{\{SPLIT\}\}\s*", re.IGNORECASE)

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

    def translate(self, text: str, source_language: str) -> str:
        if not text.strip():
            return text
        results = self.translate_batch([text], source_language)
        return results[0]

    # ── Batch (primary interface used by pipeline) ─────────────────────────

    def translate_batch(self, texts: list[str], source_language: str) -> list[str]:
        """Translate *texts* to English using GTX glue-string batching.

        Splits the list into chunks respecting ``_BATCH_MAX_CHARS`` /
        ``_BATCH_MAX_ITEMS``, sends each chunk as a single GTX request,
        falls back to MyMemory per-item if GTX fails for a chunk.
        """
        if not texts:
            return []

        src = source_language if source_language not in ("auto", "") else "auto"
        results: list[str] = [""] * len(texts)

        for chunk_indices, chunk_texts in _make_chunks(texts):
            try:
                translated = self._gtx_glue_batch(chunk_texts, src)
            except Exception as exc:
                logger.warning(
                    "[google-gtx] Batch failed (%s) — falling back to MyMemory.", exc
                )
                translated = [
                    self._mymemory_single(t, src) for t in chunk_texts
                ]

            for idx, text in zip(chunk_indices, translated):
                results[idx] = text

        return results

    # ── GTX glue-string request ────────────────────────────────────────────

    def _gtx_glue_batch(self, texts: list[str], src: str) -> list[str]:
        """Translate *texts* via a single GTX request using glue strings."""
        if len(texts) == 1:
            return [self._gtx_single(texts[0], src)]

        combined = _GLUE.join(texts)
        params = {
            "client": "gtx",
            "sl": src,
            "tl": "en",
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

    def _gtx_single(self, text: str, src: str) -> str:
        """Single-text GTX request."""
        params = {"client": "gtx", "sl": src, "tl": "en", "dt": "t", "q": text}
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

    def _mymemory_single(self, text: str, src: str) -> str:
        """Translate a single text via MyMemory (fallback)."""
        if not text.strip():
            return text
        lang_pair = f"{src}|en"
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

def _make_chunks(
    texts: list[str],
) -> list[tuple[list[int], list[str]]]:
    """Split *texts* into (indices, texts) chunks within batch limits."""
    chunks: list[tuple[list[int], list[str]]] = []
    cur_indices: list[int] = []
    cur_texts: list[str] = []
    cur_chars = 0

    for i, text in enumerate(texts):
        n = len(text)
        if cur_texts and (
            len(cur_texts) >= _BATCH_MAX_ITEMS
            or cur_chars + n + len(_GLUE) > _BATCH_MAX_CHARS
        ):
            chunks.append((cur_indices, cur_texts))
            cur_indices, cur_texts, cur_chars = [], [], 0

        cur_indices.append(i)
        cur_texts.append(text)
        cur_chars += n + len(_GLUE)

    if cur_texts:
        chunks.append((cur_indices, cur_texts))

    return chunks

