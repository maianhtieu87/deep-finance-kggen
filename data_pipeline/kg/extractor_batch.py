# data_pipeline/kg/extractor_batch.py
"""
FinDKG-Lite Batch Extractors

Hai extractor, dùng cho hai use case khác nhau:

AsyncConcurrentExtractor  — N articles gửi đồng thời qua asyncio.gather()
  - Cost  : standard rate (1.0x)
  - Latency: ~max(single_call_latency)  ← phù hợp test / corpus nhỏ
  - Dùng khi: test, corpus < 500 articles, cần kết quả ngay

GeminiBatchAPIExtractor   — 1 job submit, Gemini server xử lý async
  - Cost  : 50% off (standard × 0.5)
  - Latency: 30s poll, thực tế 1–30 phút ← phù hợp production
  - Dùng khi: corpus lớn (>500 articles), không cần kết quả ngay

CHỌN EXTRACTOR:
  n < 500   → AsyncConcurrentExtractor(max_concurrent=5~15)
  n >= 500  → GeminiBatchAPIExtractor(display_name="...")

MULTI-TICKER FLOW:
  - Mỗi unique article chỉ được gọi 1 lần (key = sha1 của text)
  - primary_ticker = ticker xuất hiện nhiều nhất trong content
  - rescore_triples_for_ticker() điều chỉnh relevance khi fan-out
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import hashlib
import re
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from .prompts import (
    FINDKG_LITE_SYSTEM_PROMPT,
    FINDKG_LITE_USER_PROMPT,
    FEW_SHOT_EXAMPLES,
    VALID_ENTITY_TYPES,
    VALID_RELATIONS,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

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

# dict format for GeminiBatchAPIExtractor inline_requests
_GEN_CONFIG_DICT = {
    "response_mime_type": "application/json",
    "response_schema":    RESPONSE_SCHEMA,
    "temperature":        0.1,
    "max_output_tokens":  2048,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _build_few_shot_str() -> str:
    parts = []
    for ex in FEW_SHOT_EXAMPLES:
        parts.append(
            f"TICKER: {ex['ticker']}\n"
            f"ARTICLE: {ex['input']}\n"
            f"OUTPUT: {json.dumps(ex['output'], ensure_ascii=False)}"
        )
    return "\n\n---\n\n".join(parts)


_FEW_SHOT_STR = _build_few_shot_str()   # built once at import time


def build_user_prompt(text: str, ticker: str, news_date: str,
                      sector: Optional[str] = None) -> str:
    """Build complete user prompt for 1 article. sector param kept for compat, unused."""
    # format_map with fallback default so missing keys (e.g. {sector} in old prompts.py)
    # are silently replaced with empty string instead of raising KeyError.
    class _SafeDict(dict):
        def __missing__(self, key): return ""
    user_part = FINDKG_LITE_USER_PROMPT.format_map(_SafeDict(
        ticker=ticker,
        news_date=news_date,
        news_text=text[:3500],
        sector="",
    ))
    return (
        f"EXAMPLES (study these carefully):\n\n{_FEW_SHOT_STR}\n\n"
        f"{'='*60}\n\n"
        f"NOW EXTRACT FROM THIS NEW ARTICLE:\n\n{user_part}"
    )


def _filter_and_clamp(raw: Any, min_relevance: float,
                      min_confidence: float) -> List[Dict]:
    """Apply threshold filter + clamp numeric fields."""
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
        t["price_impact_score"]  = max(-1.0, min(1.0,
                                       float(t.get("price_impact_score", 0.0))))
        t["relevance_to_ticker"] = max(0.0, min(1.0, rel))
        out.append(t)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-TICKER RESCORE
# ─────────────────────────────────────────────────────────────────────────────

# Company name variants for mention detection in rescore
_TICKER_NAME_MAP: Dict[str, List[str]] = {
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
    "AMD":   ["AMD"],
    "RIVN":  ["Rivian", "RIVN"],
}


def _ticker_mentioned(ticker: str, text_upper: str) -> bool:
    """Check if ticker or any of its company name variants appear in text."""
    for name in _TICKER_NAME_MAP.get(ticker, [ticker]):
        if name.upper() in text_upper:
            return True
    return False


def rescore_triples_for_ticker(
    triples: List[Dict],
    primary_ticker: str,
    target_ticker: str,
    min_relevance: float = 0.30,
    article_text: str = "",
    all_article_tickers: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Adjust relevance_to_ticker when fan-out to a different ticker.

    2-tier logic:
      Tier 1 — target ticker IS mentioned in article text:
               → boost × 1.1 for triples mentioning it in subj/obj
               → decay × 0.4 for others (macro context still relevant)

      Tier 2 — target ticker NOT mentioned, but OTHER tickers ARE:
               → strict mode: only keep triples that explicitly mention target
               → raises effective min_relevance to 0.60 to block noise
               → Example: GOOGL/MSFT in an Apple+Tesla article get dropped

      Tier 3 — no ticker mentioned at all (pure macro article):
               → normal × 0.4 decay (sector-level relevance)
    """
    if primary_ticker.upper() == target_ticker.upper():
        return triples

    target_lower = target_ticker.lower()
    text_upper   = (article_text or "").upper()

    # Determine context
    target_in_text = _ticker_mentioned(target_ticker, text_upper)
    others_in_text = any(
        _ticker_mentioned(t, text_upper)
        for t in (all_article_tickers or [])
        if t.upper() != target_ticker.upper()
    )

    # Strict mode: other tickers explicitly present but NOT this target
    strict_mode     = others_in_text and not target_in_text
    effective_min   = 0.60 if strict_mode else min_relevance

    out = []
    for t in triples:
        t2        = dict(t)
        subj_name = t.get("subject", {}).get("name", "").lower()
        obj_name  = t.get("object",  {}).get("name", "").lower()
        mentions  = target_lower in subj_name or target_lower in obj_name

        orig_rel = float(t.get("relevance_to_ticker", 0.0))
        t2["relevance_to_ticker"] = (
            min(1.0, orig_rel * 1.1) if mentions else orig_rel * 0.4
        )
        if t2["relevance_to_ticker"] >= effective_min:
            out.append(t2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ASYNC CONCURRENT EXTRACTOR  (test / small corpus)
# ─────────────────────────────────────────────────────────────────────────────

class AsyncConcurrentExtractor:
    """
    Gửi N articles đồng thời qua asyncio.gather() + semaphore.

    - Cost    : standard rate (1.0x)
    - Latency : ~max(single_call_latency), thường 3–10s
    - Best for: test scripts, corpus < 500 articles

    CÁCH DÙNG:
        extractor = AsyncConcurrentExtractor(max_concurrent=5)
        results = extractor.extract_batch(articles)
        # results[i] = List[RichTriple] cho articles[i]
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL_ID,
        temperature: float = 0.1,
        min_relevance: float = 0.30,
        min_confidence: float = 0.35,
        max_concurrent: int = 5,
        # Free tier: 5 | Paid tier: 10-15
        # Gemini Flash RPM: 15 (free) / 2000 (paid Tier 1)
    ):
        self.min_relevance  = min_relevance
        self.min_confidence = min_confidence
        self.max_concurrent = max_concurrent
        self.model          = model
        self.temperature    = temperature

        _key = api_key or os.getenv("GEMINI_API_KEY")
        if not _key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Set: export GEMINI_API_KEY='your_key'"
            )
        self.client = genai.Client(api_key=_key)

        self._gen_config = types.GenerateContentConfig(
            system_instruction=FINDKG_LITE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=temperature,
            max_output_tokens=2048,
        )

    async def _extract_one_async(
        self,
        idx: int,
        article: Dict[str, str],
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, List[Dict]]:
        """Extract one article, respecting the concurrency semaphore."""
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
                triples = _filter_and_clamp(raw, self.min_relevance, self.min_confidence)
                return idx, triples
            except Exception as e:
                print(f"⚠️  Async extract error idx={idx} "
                      f"ticker={article.get('ticker')}: {e}")
                return idx, []

    def extract_batch(self, articles: List[Dict[str, str]]) -> List[List[Dict]]:
        """
        Extract all articles concurrently. Blocking call (runs event loop internally).

        Args:
            articles: list of {text, ticker, date}

        Returns:
            List[List[RichTriple]] — index i = triples for articles[i]
        """
        if not articles:
            return []

        async def _run_all():
            semaphore = asyncio.Semaphore(self.max_concurrent)
            tasks = [
                self._extract_one_async(i, art, semaphore)
                for i, art in enumerate(articles)
            ]
            return await asyncio.gather(*tasks)

        print(f"📤 AsyncConcurrentExtractor: {len(articles)} articles, "
              f"max_concurrent={self.max_concurrent}")
        t0 = time.time()

        pairs = asyncio.run(_run_all())

        elapsed = time.time() - t0
        ok = sum(1 for _, t in pairs if t is not None)
        print(f"✅ Done in {elapsed:.1f}s  ({ok}/{len(articles)} succeeded)")

        results: List[List[Dict]] = [[] for _ in articles]
        for idx, triples in pairs:
            results[idx] = triples or []
        return results


# ─────────────────────────────────────────────────────────────────────────────
# GEMINI BATCH API EXTRACTOR  (production / large corpus)
# ─────────────────────────────────────────────────────────────────────────────

class GeminiBatchAPIExtractor:
    """
    Xử lý nhiều articles qua Gemini Batch API chính thức.

    - Cost    : 50% so với standard rate
    - Latency : 30s poll interval, thực tế 1–30 phút
    - Best for: corpus lớn (>500 articles), production runs

    CÁCH DÙNG:
        extractor = GeminiBatchAPIExtractor(display_name="daily-2024-03-15")
        results = extractor.extract_batch(articles)
        # results[i] = List[RichTriple] cho articles[i]
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL_ID,
        min_relevance: float = 0.30,
        min_confidence: float = 0.35,
        poll_interval_secs: int = 30,
        max_wait_secs: int = 86400,  # 24h max
        display_name: str = "findkg-lite-v2",
    ):
        self.min_relevance  = min_relevance
        self.min_confidence = min_confidence
        self.poll_interval  = poll_interval_secs
        self.max_wait       = max_wait_secs
        self.display_name   = display_name
        self.model          = f"models/{model}"

        _key = api_key or os.getenv("GEMINI_API_KEY")
        if not _key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. Set: export GEMINI_API_KEY='your_key'"
            )
        self.client = genai.Client(api_key=_key)

    def extract_batch(self, articles: List[Dict[str, str]]) -> List[List[Dict]]:
        """
        Submit batch job, poll until done, return results in input order.

        Args:
            articles: list of {text, ticker, date}

        Returns:
            List[List[RichTriple]] — index i = triples for articles[i]
            Returns [[]] for articles with parse errors.
        """
        if not articles:
            return []

        # Build inline requests
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
                    "contents": [{
                        "parts": [{"text": prompt}],
                        "role": "user",
                    }],
                    "system_instruction": {
                        "parts": [{"text": FINDKG_LITE_SYSTEM_PROMPT}]
                    },
                    "generation_config": _GEN_CONFIG_DICT,
                },
            })

        print(f"📤 GeminiBatchAPIExtractor: submitting {len(inline_requests)} requests "
              f"[display_name={self.display_name}]")
        batch_job = self.client.batches.create(
            model=self.model,
            src=inline_requests,
            config={"display_name": self.display_name},
        )
        job_name = batch_job.name
        print(f"   Job: {job_name}  |  State: {batch_job.state.name}")

        # Poll
        terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
        elapsed  = 0
        while elapsed < self.max_wait:
            time.sleep(self.poll_interval)
            elapsed += self.poll_interval
            batch_job = self.client.batches.get(name=job_name)
            state = batch_job.state.name
            print(f"   [{elapsed:5d}s] {state}")
            if state in terminal:
                break

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            print(f"❌ Batch job did not succeed. Final state: {batch_job.state.name}")
            if hasattr(batch_job, "error") and batch_job.error:
                print(f"   Error: {batch_job.error}")
            return [[] for _ in articles]

        # Parse responses
        results: List[List[Dict]] = [[] for _ in articles]
        inlined = getattr(batch_job.response, "inlined_responses", None) or []
        parsed_ok = 0
        for resp in inlined:
            try:
                idx = int(resp.key)
            except (ValueError, AttributeError):
                continue
            if idx < 0 or idx >= len(articles):
                continue
            try:
                raw_text = resp.response.candidates[0].content.parts[0].text
                raw      = json.loads(raw_text)
                results[idx] = _filter_and_clamp(raw, self.min_relevance,
                                                  self.min_confidence)
                parsed_ok += 1
            except Exception as e:
                print(f"⚠️  Parse error key={resp.key}: {e}")
                results[idx] = []

        print(f"✅ Batch done. Parsed {parsed_ok}/{len(articles)} responses.")
        return results