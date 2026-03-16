# data_pipeline/kg/extractor_batch.py
"""
FinDKG-Lite Batch Extractors — V3

Thay đổi chính so với V2:
  - Chunk-based text: gộp headlines + content cho mỗi (ticker, date), chia chunk
    3000 chars với overlap 200 chars, extract từng chunk rồi merge+dedup triples
  - Fix lỗi 2: _all_tickers được truyền đúng qua cột _all_tickers sau explode
  - Fix lỗi 7: detect_primary_ticker dùng title weight × 3 so với body weight × 1
  - Fix lỗi strict-mode tier 2: effective_min nâng lên 0.75
  - Bỏ KMeans entity resolution khỏi pipeline này (sẽ dùng alias dict ở Stage B)

Fixes từ prompts.py v2 (applied vào V3):
  - min_relevance default: 0.30 → 0.50 (aligned với prompt threshold)
  - min_confidence default: 0.35 → 0.65 (aligned với prompt threshold)
  - rescore_triples_for_ticker: price_impact_score được zero-out cho cross-ticker triples
    (Bug fix: score được tính cho primary_ticker; direction không đáng tin cho ticker khác)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types

from .prompts import (
    FINDKG_LITE_SYSTEM_PROMPT,
    FINDKG_LITE_USER_PROMPT,
    FEW_SHOT_EXAMPLES,
    VALID_ENTITY_TYPES,
    VALID_RELATIONS,
)

MODEL_ID = "gemini-2.0-flash"

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "subject": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "type": {"type": "STRING", "enum": VALID_ENTITY_TYPES},
                },
                "required": ["name", "type"],
            },
            "relation":            {"type": "STRING", "enum": VALID_RELATIONS},
            "object": {
                "type": "OBJECT",
                "properties": {
                    "name": {"type": "STRING"},
                    "type": {"type": "STRING", "enum": VALID_ENTITY_TYPES},
                },
                "required": ["name", "type"],
            },
            "confidence":          {"type": "NUMBER"},
            "price_impact_score":  {"type": "NUMBER"},
            "relevance_to_ticker": {"type": "NUMBER"},
            "reasoning":           {"type": "STRING"},
        },
        "required": [
            "subject", "relation", "object",
            "confidence", "price_impact_score", "relevance_to_ticker",
        ],
    },
}

_GEN_CONFIG_DICT = {
    "response_mime_type": "application/json",
    "response_schema":    RESPONSE_SCHEMA,
    "temperature":        0.1,
    "max_output_tokens":  2048,
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _parse_tickers(val: Any) -> List[str]:
    """Parse ticker column into list. Supports "AAPL,GOOGL" | ["AAPL"] | "AAPL"."""
    if isinstance(val, list):
        return [t.strip().upper() for t in val if isinstance(t, str) and t.strip()]
    if isinstance(val, str):
        return [t.strip().upper() for t in val.split(",") if t.strip()]
    return []


def _build_few_shot_str() -> str:
    parts = []
    for ex in FEW_SHOT_EXAMPLES:
        parts.append(
            f"TICKER: {ex['ticker']}\n"
            f"ARTICLE: {ex['input']}\n"
            f"OUTPUT: {json.dumps(ex['output'], ensure_ascii=False)}"
        )
    return "\n\n---\n\n".join(parts)


_FEW_SHOT_STR = _build_few_shot_str()


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK-BASED TEXT SPLITTING
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_SIZE    = 3000
CHUNK_OVERLAP = 200


def _split_at_sentence_boundary(text: str, max_chars: int) -> int:
    if len(text) <= max_chars:
        return len(text)
    window = text[:max_chars]
    for sep in (". ", "! ", "? ", "\n", " "):
        pos = window.rfind(sep)
        if pos > max_chars * 0.6:
            return pos + len(sep)
    return max_chars


def split_text_chunks(text: str, chunk_size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start  = 0
    while start < len(text):
        end   = _split_at_sentence_boundary(text[start:], chunk_size)
        chunk = text[start: start + end].strip()
        if chunk:
            chunks.append(chunk)
        if start + end >= len(text):
            break
        start = start + end - overlap
        if start < 0:
            start = 0
    return chunks if chunks else [text[:chunk_size]]


def build_combined_text(titles: List[str], contents: List[str]) -> str:
    headline_block = "\n".join(f"- {t}" for t in titles if t and t.strip())
    content_block  = "\n\n---\n\n".join(c for c in contents if c and c.strip())
    parts = []
    if headline_block:
        parts.append(f"HEADLINES:\n{headline_block}")
    if content_block:
        parts.append(f"ARTICLES:\n{content_block}")
    return "\n\n".join(parts)


def build_user_prompt(text: str, ticker: str, news_date: str,
                      sector: Optional[str] = None) -> str:
    class _SafeDict(dict):
        def __missing__(self, key): return ""
    user_part = FINDKG_LITE_USER_PROMPT.format_map(_SafeDict(
        ticker=ticker,
        news_date=news_date,
        news_text=text,
        sector="",
    ))
    return (
        f"EXAMPLES (study these carefully):\n\n{_FEW_SHOT_STR}\n\n"
        f"{'='*60}\n\n"
        f"NOW EXTRACT FROM THIS NEW ARTICLE:\n\n{user_part}"
    )


def _filter_and_clamp(raw: Any, min_relevance: float,
                      min_confidence: float) -> List[Dict]:
    if not isinstance(raw, list):
        return []
    out = []
    for t in raw:
        if not isinstance(t, dict):
            continue
        rel  = float(t.get("relevance_to_ticker", 0))
        conf = float(t.get("confidence", 0))
        if rel < min_relevance or conf < min_confidence:
            continue
        t["confidence"]          = max(0.0, min(1.0, conf))
        t["price_impact_score"]  = max(-1.0, min(1.0, float(t.get("price_impact_score", 0.0))))
        t["relevance_to_ticker"] = max(0.0, min(1.0, rel))
        out.append(t)
    return out


def dedup_triples(triples: List[Dict]) -> List[Dict]:
    seen, out = set(), []
    for t in triples:
        key = (
            t.get("subject", {}).get("name", ""),
            t.get("relation", ""),
            t.get("object",  {}).get("name", ""),
        )
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY TICKER DETECTION  (title weight × 3)
# ─────────────────────────────────────────────────────────────────────────────

TICKER_NAME_MAP: Dict[str, List[str]] = {
    "TSLA":  ["Tesla", "TSLA"],
    "AAPL":  ["Apple", "AAPL"],
    "AMZN":  ["Amazon", "AMZN"],
    "MSFT":  ["Microsoft", "MSFT"],
    "GOOGL": ["Google", "Alphabet", "GOOGL"],
    "GOOG":  ["Google", "Alphabet", "GOOG"],
    "META":  ["Meta", "Facebook", "META"],
    "BA":    ["Boeing", "BA"],
    "JPM":   ["JPMorgan", "JP Morgan", "JPM"],
    "WMT":   ["Walmart", "WMT"],
    "NVDA":  ["Nvidia", "NVDA"],
    "NFLX":  ["Netflix", "NFLX"],
    "INTC":  ["Intel", "INTC"],
    "AMD":   ["AMD", "Advanced Micro"],
    "RIVN":  ["Rivian", "RIVN"],
    "LCID":  ["Lucid", "LCID"],
    "GM":    ["General Motors", "GM"],
    "F":     ["Ford", "F Motor"],
}

TITLE_WEIGHT = 3


def detect_primary_ticker(title: str, content: str, tickers: List[str]) -> str:
    if not tickers:
        return ""
    if len(tickers) == 1:
        return tickers[0]

    title_upper   = (title   or "").upper()
    content_upper = (content or "").upper()

    counts = {}
    for t in tickers:
        score = 0
        for name in TICKER_NAME_MAP.get(t, [t]):
            n_up = name.upper()
            score += title_upper.count(n_up) * TITLE_WEIGHT
            score += content_upper.count(n_up)
        counts[t] = score

    best = max(counts.values())
    if best == 0:
        return tickers[0]
    for t in tickers:
        if counts[t] == best:
            return t


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TICKER RESCORE
# ─────────────────────────────────────────────────────────────────────────────

def _ticker_mentioned_in_text(ticker: str, text_upper: str) -> bool:
    for name in TICKER_NAME_MAP.get(ticker, [ticker]):
        name_upper = name.upper()
        if len(name) <= 3:
            pattern = r'\b' + re.escape(name_upper) + r'\b'
            if re.search(pattern, text_upper):
                return True
        else:
            if name_upper in text_upper:
                return True
    return False


def _ticker_mentioned_in_triple(ticker: str, triple: Dict) -> bool:
    target_lower = ticker.lower()
    subj_name    = triple.get("subject", {}).get("name", "").lower()
    obj_name     = triple.get("object",  {}).get("name", "").lower()
    if target_lower in subj_name or target_lower in obj_name:
        return True
    for name in TICKER_NAME_MAP.get(ticker.upper(), []):
        name_lower = name.lower()
        if name_lower in subj_name or name_lower in obj_name:
            return True
    return False


def rescore_triples_for_ticker(
    triples: List[Dict],
    primary_ticker: str,
    target_ticker: str,
    min_relevance: float = 0.50,
    article_text: str = "",
    all_article_tickers: Optional[List[str]] = None,
) -> List[Dict]:
    """
    3-tier rescore + price_impact zeroing cho cross-ticker triples.

    Tier 1 — target có trong text: normal min_relevance
    Tier 2 — target KHÔNG có, nhưng ticker khác có: effective_min = 0.75
             → strict filter nếu triple không đề cập target trong subj/obj
    Tier 3 — không ticker nào có (pure macro): effective_min = max(0.50, min_relevance)

    price_impact_score (BUG FIX):
      Luôn zero-out cho cross-ticker triples. Lý do: LLM tính score này
      từ góc nhìn primary_ticker; direction không đáng tin cho target_ticker.
      Ví dụ: "Apple ANNOUNCES headset" → +0.50 cho AAPL, nhưng cho MSFT
      đây là competitive threat → phải âm, không phải +0.50.
      Triple vẫn giữ nguyên structure value cho GATv2.
    """
    if primary_ticker.upper() == target_ticker.upper():
        return triples

    text_upper     = (article_text or "").upper()
    target_in_text = _ticker_mentioned_in_text(target_ticker, text_upper)
    others_in_text = any(
        _ticker_mentioned_in_text(t, text_upper)
        for t in (all_article_tickers or [])
        if t.upper() != target_ticker.upper()
    )

    if target_in_text:
        strict_mode   = False
        effective_min = min_relevance
    elif others_in_text:
        strict_mode   = True
        effective_min = 0.75
    else:
        strict_mode   = False
        effective_min = max(0.50, min_relevance)

    out = []
    for t in triples:
        t2       = dict(t)
        mentions = _ticker_mentioned_in_triple(target_ticker, t)
        orig_rel = float(t.get("relevance_to_ticker", 0.0))

        if strict_mode:
            if not mentions:
                continue
            t2["relevance_to_ticker"] = min(1.0, orig_rel * 1.1)
        else:
            t2["relevance_to_ticker"] = (
                min(1.0, orig_rel * 1.1) if mentions else orig_rel * 0.4
            )

        # ── BUG FIX: zero out price_impact cho mọi cross-ticker triple ──────
        t2["price_impact_score"] = 0.0

        if t2["relevance_to_ticker"] >= effective_min:
            out.append(t2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC CONCURRENT EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class AsyncConcurrentExtractor:
    """
    Gửi N chunks đồng thời qua asyncio.gather() + semaphore.
    Phù hợp cho corpus < 500 articles hoặc test.

    Threshold defaults aligned với prompts.py:
      min_relevance  = 0.50  (was 0.30)
      min_confidence = 0.65  (was 0.35)
    """

    def __init__(
        self,
        api_key:        Optional[str] = None,
        model:          str   = MODEL_ID,
        temperature:    float = 0.1,
        min_relevance:  float = 0.50,   # ← was 0.30
        min_confidence: float = 0.65,   # ← was 0.35
        max_concurrent: int   = 5,
    ):
        self.min_relevance  = min_relevance
        self.min_confidence = min_confidence
        self.max_concurrent = max_concurrent
        self.model          = model
        self.temperature    = temperature

        _key = api_key or os.getenv("GEMINI_API_KEY")
        if not _key:
            raise RuntimeError("Missing GEMINI_API_KEY.")
        self.client = genai.Client(api_key=_key)
        self._gen_config = types.GenerateContentConfig(
            system_instruction=FINDKG_LITE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=temperature,
            max_output_tokens=2048,
        )

    async def _extract_one_async(
        self, idx: int, article: Dict[str, str], semaphore: asyncio.Semaphore,
    ) -> Tuple[int, List[Dict]]:
        async with semaphore:
            prompt = build_user_prompt(
                text=article.get("text", ""),
                ticker=article.get("ticker", "UNKNOWN"),
                news_date=article.get("date", ""),
            )
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model,
                    contents=prompt,
                    config=self._gen_config,
                )
                raw = json.loads(response.text)
                return idx, _filter_and_clamp(raw, self.min_relevance, self.min_confidence)
            except Exception as e:
                print(f"  Async extract error idx={idx} ticker={article.get('ticker')}: {e}")
                return idx, []

    def extract_batch(self, articles: List[Dict[str, str]]) -> List[List[Dict]]:
        """
        articles: list of {text, ticker, date}
        Trả về list[list[triple]] — index i = triples cho articles[i].
        """
        if not articles:
            return []

        async def _run_all():
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [self._extract_one_async(i, art, semaphore)
                     for i, art in enumerate(articles)]
            return await asyncio.gather(*tasks)

        print(f"  AsyncConcurrentExtractor: {len(articles)} chunks, "
              f"max_concurrent={self.max_concurrent}")
        t0 = time.time()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    pairs = pool.submit(asyncio.run, _run_all()).result()
            else:
                pairs = loop.run_until_complete(_run_all())
        except RuntimeError:
            pairs = asyncio.run(_run_all())

        elapsed = time.time() - t0
        ok = sum(1 for _, t in pairs if t is not None)
        print(f"  Done in {elapsed:.1f}s  ({ok}/{len(articles)} succeeded)")

        results: List[List[Dict]] = [[] for _ in articles]
        for idx, triples in pairs:
            results[idx] = triples or []
        return results


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI BATCH API EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class GeminiBatchAPIExtractor:
    """
    50% cost, async job — dùng cho corpus > 500 chunks.

    Threshold defaults aligned với prompts.py:
      min_relevance  = 0.50  (was 0.30)
      min_confidence = 0.65  (was 0.35)
    """

    def __init__(
        self,
        api_key:            Optional[str] = None,
        model:              str   = MODEL_ID,
        min_relevance:      float = 0.50,   # ← was 0.30
        min_confidence:     float = 0.65,   # ← was 0.35
        poll_interval_secs: int   = 30,
        max_wait_secs:      int   = 86400,
        display_name:       str   = "findkg-lite-v3",
    ):
        self.min_relevance  = min_relevance
        self.min_confidence = min_confidence
        self.poll_interval  = poll_interval_secs
        self.max_wait       = max_wait_secs
        self.display_name   = display_name
        self.model          = f"models/{model}"

        _key = api_key or os.getenv("GEMINI_API_KEY")
        if not _key:
            raise RuntimeError("Missing GEMINI_API_KEY.")
        self.client = genai.Client(api_key=_key)

    def extract_batch(self, articles: List[Dict[str, str]]) -> List[List[Dict]]:
        if not articles:
            return []

        inline_requests = []
        for i, art in enumerate(articles):
            prompt = build_user_prompt(
                text=art.get("text", ""),
                ticker=art.get("ticker", "UNKNOWN"),
                news_date=art.get("date", ""),
            )
            inline_requests.append({
                "key": str(i),
                "request": {
                    "contents": [{"parts": [{"text": prompt}], "role": "user"}],
                    "system_instruction": {"parts": [{"text": FINDKG_LITE_SYSTEM_PROMPT}]},
                    "generation_config": _GEN_CONFIG_DICT,
                },
            })

        print(f"  GeminiBatchAPIExtractor: submitting {len(inline_requests)} chunks")
        batch_job = self.client.batches.create(
            model=self.model,
            src=inline_requests,
            config={"display_name": self.display_name},
        )
        job_name = batch_job.name
        print(f"  Job: {job_name}  State: {batch_job.state.name}")

        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
        elapsed  = 0
        while elapsed < self.max_wait:
            time.sleep(self.poll_interval)
            elapsed += self.poll_interval
            batch_job = self.client.batches.get(name=job_name)
            state = batch_job.state.name
            print(f"  [{elapsed:5d}s] {state}")
            if state in terminal:
                break

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            print(f"Batch job failed: {batch_job.state.name}")
            return [[] for _ in articles]

        results: List[List[Dict]] = [[] for _ in articles]
        inlined = getattr(batch_job.response, "inlined_responses", None) or []
        for resp in inlined:
            try:
                idx = int(resp.key)
            except (ValueError, AttributeError):
                continue
            if 0 <= idx < len(articles):
                try:
                    raw_text = resp.response.candidates[0].content.parts[0].text
                    results[idx] = _filter_and_clamp(
                        json.loads(raw_text), self.min_relevance, self.min_confidence
                    )
                except Exception:
                    results[idx] = []
        return results