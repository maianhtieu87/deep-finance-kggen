# data_pipeline/kg/extractor.py
import os
import json
import time
import random
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types

from .prompts import (
    FINDKG_LITE_SYSTEM_PROMPT,
    FINDKG_LITE_USER_PROMPT,
    FEW_SHOT_EXAMPLES,
    VALID_ENTITY_TYPES,
    VALID_RELATIONS,
)

_ENTITY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "name": {"type": "STRING"},
        "type": {"type": "STRING", "enum": VALID_ENTITY_TYPES},
    },
    "required": ["name", "type"],
}

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "subject":             _ENTITY_SCHEMA,
            "relation":            {"type": "STRING", "enum": VALID_RELATIONS},
            "object":              _ENTITY_SCHEMA,
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


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL EXTRACTOR  (1 article per call — fallback / unit test only)
# For production use AsyncConcurrentExtractor or GeminiBatchAPIExtractor.
# ─────────────────────────────────────────────────────────────────────────────

class FinDKGLiteExtractor:
    """
    Sequential KG extractor. 1 API call per article.
    Prefer AsyncConcurrentExtractor for batches.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        min_relevance: float = 0.30,
        min_confidence: float = 0.35,
        max_retries: int = 3,
        backoff_base: float = 10.0,
    ):
        self.min_relevance  = min_relevance
        self.min_confidence = min_confidence
        self.max_retries    = max_retries
        self.backoff_base   = backoff_base
        self.model_name     = model

        _api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not _api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY. "
                "Set environment variable: export GEMINI_API_KEY='your_key'"
            )

        self.client = genai.Client(api_key=_api_key)
        self._gen_config = types.GenerateContentConfig(
            system_instruction=FINDKG_LITE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=temperature,
            max_output_tokens=2048,
        )

    # ── Prompt assembly ───────────────────────────────────────────────────────

    def _build_few_shot_str(self) -> str:
        parts = []
        for ex in FEW_SHOT_EXAMPLES:
            parts.append(
                f"TICKER: {ex['ticker']}\n"
                f"ARTICLE: {ex['input']}\n"
                f"OUTPUT: {json.dumps(ex['output'], ensure_ascii=False)}"
            )
        return "\n\n---\n\n".join(parts)

    def _build_prompt(self, text: str, ticker: str, news_date: str) -> str:
        """Build complete prompt. sector removed — no longer in USER_PROMPT template."""
        few_shot = self._build_few_shot_str()
        class _SafeDict(dict):
            def __missing__(self, key): return ""
        user_part = FINDKG_LITE_USER_PROMPT.format_map(_SafeDict(
            ticker=ticker,
            news_date=news_date,
            news_text=text[:3500],
            sector="",
        ))
        return (
            f"EXAMPLES (study these carefully):\n\n{few_shot}\n\n"
            f"{'='*60}\n\n"
            f"NOW EXTRACT FROM THIS NEW ARTICLE:\n\n{user_part}"
        )

    # ── Extraction ────────────────────────────────────────────────────────────

    def extract(
        self,
        text: str,
        ticker: str,
        news_date: str = "",
        sector: Optional[str] = None,   # kept for backward compat, ignored
    ) -> List[Dict[str, Any]]:
        if not text or not text.strip():
            return []

        prompt = self._build_prompt(text, ticker, news_date)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self._gen_config,
                )
                raw = json.loads(response.text)
                if not isinstance(raw, list):
                    return []
                return self._filter_and_clamp(raw)

            except Exception as e:
                last_err = e
                wait = self.backoff_base * (2 ** attempt) + random.uniform(0, 2)
                print(f"⚠️  Gemini error (attempt {attempt+1}/{self.max_retries}): {e}")
                print(f"    Retry in {wait:.1f}s ...")
                time.sleep(wait)

        print(f"❌ Extraction failed after {self.max_retries} retries: {last_err}")
        return []

    def _filter_and_clamp(self, raw: list) -> List[Dict]:
        filtered = []
        for t in raw:
            if not isinstance(t, dict):
                continue
            rel  = float(t.get("relevance_to_ticker", 0))
            conf = float(t.get("confidence", 0))
            if rel < self.min_relevance or conf < self.min_confidence:
                continue
            t["confidence"]          = max(0.0, min(1.0, conf))
            t["price_impact_score"]  = max(-1.0, min(1.0, float(t.get("price_impact_score", 0.0))))
            t["relevance_to_ticker"] = max(0.0, min(1.0, rel))
            filtered.append(t)
        return filtered


# ─────────────────────────────────────────────────────────────────────────────
# BACKWARD COMPAT — legacy cache migration
# ─────────────────────────────────────────────────────────────────────────────

def upgrade_legacy_triple(t: Any) -> Dict[str, Any]:
    if isinstance(t, dict):
        return t
    if isinstance(t, (list, tuple)) and len(t) == 3:
        s, p, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
        return {
            "subject":             {"name": s, "type": "COMP"},
            "relation":            "RELATES_TO",
            "object":              {"name": o, "type": "CONCEPT"},
            "confidence":          0.5,
            "price_impact_score":  0.0,
            "relevance_to_ticker": 0.5,
            "reasoning":           f"Legacy triple: {s} — {p} — {o}",
        }
    return None


def upgrade_legacy_cache_file(cache_path: str) -> bool:
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        triples = obj.get("triples", [])
        if not triples or isinstance(triples[0], dict):
            return False
        new_triples = [upgrade_legacy_triple(t) for t in triples]
        new_triples = [t for t in new_triples if t is not None]
        obj["triples"] = new_triples
        obj["_format_version"] = "v2"
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        return True
    except Exception:
        return False