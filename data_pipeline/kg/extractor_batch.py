# data_pipeline/kg/extractor_batch.py — V5
"""
V5 changes vs V4.1:

  **ticker_aliases integration:**
    - TICKER_NAME_MAP, ALL_TICKER_NAMES_PATTERN, normalize_entity_name
      now imported from configs.ticker_aliases (Single Source of Truth)
    - Removed: hardcoded TICKER_NAME_MAP, _NAME_TO_TICKER, _ALL_TICKER_NAMES_PATTERN
    - Removed: _normalize_subject_for_dedup() — replaced by normalize_entity_name()
    - _normalize_object_for_dedup() uses ALL_TICKER_NAMES_PATTERN for ALL tickers
    - smart_dedup_triples() uses normalize_entity_name() for subject key

  **P0 — No chunking (full-article extraction):**
    - prepare_article_text() replaces split_text_chunks() as default
    - Gemini Flash 2.0 handles 1M tokens; finance articles rarely exceed 15K chars
    - Fallback chunking available via GlobalConfig.KG_ENABLE_CHUNKING=True

  **P1 — Improved analyst cap logic:**
    - limit_signals_per_source() caps by (firm_name, action_type)
    - Separate caps: PT actions + rating actions
    - Prevents bypass where "Hold rating" didn't contain "price target"

  Carried over from V3.6:
  - tag_triples_source(), LIMIT_RELS_PERSON includes CUTS,
    monetary normalization, filter_triples_for_ticker() mention-only.
"""
from __future__ import annotations
import asyncio, json, os, re, time, hashlib
from typing import Any, Dict, List, Optional, Tuple
from google import genai
from google.genai import types
from configs.config import GlobalConfig
from configs.ticker_aliases import (
    TICKER_NAME_MAP,
    ALL_TICKER_NAMES_PATTERN,
    normalize_entity_name,
)
from .prompts import (
    FINDKG_LITE_SYSTEM_PROMPT, FINDKG_LITE_USER_PROMPT, FEW_SHOT_EXAMPLES,
    VALID_ENTITY_TYPES, VALID_RELATIONS,
)

MODEL_ID = "gemini-2.0-flash"

RESPONSE_SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "subject": {"type": "OBJECT",
                "properties": {"name": {"type": "STRING"}, "type": {"type": "STRING", "enum": VALID_ENTITY_TYPES}},
                "required": ["name", "type"]},
            "relation": {"type": "STRING", "enum": VALID_RELATIONS},
            "object": {"type": "OBJECT",
                "properties": {"name": {"type": "STRING"}, "type": {"type": "STRING", "enum": VALID_ENTITY_TYPES}},
                "required": ["name", "type"]},
            "confidence": {"type": "NUMBER"},
            "price_impact_score": {"type": "NUMBER"},
            "relevance_to_ticker": {"type": "NUMBER"},
            "article_index": {"type": "INTEGER"},
            "reasoning": {"type": "STRING"},
        },
        "required": ["subject", "relation", "object", "confidence", "price_impact_score", "relevance_to_ticker", "article_index"],
    },
}

_GEN_CONFIG_DICT = {
    "response_mime_type": "application/json",
    "response_schema": RESPONSE_SCHEMA,
    "temperature": 0.1,
    "max_output_tokens": 8192,
}

def _norm(s): return re.sub(r"\s+", " ", (s or "")).strip()
def _sha1(s): return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def _parse_tickers(val):
    if isinstance(val, list): return [t.strip().upper() for t in val if isinstance(t, str) and t.strip()]
    if isinstance(val, str):  return [t.strip().upper() for t in val.split(",") if t.strip()]
    return []

def _build_few_shot_str():
    parts = []
    for ex in FEW_SHOT_EXAMPLES:
        parts.append(f"TICKER: {ex['ticker']}\nARTICLE: {ex['input']}\nOUTPUT: {json.dumps(ex['output'], ensure_ascii=False)}")
    return "\n\n---\n\n".join(parts)

_FEW_SHOT_STR = _build_few_shot_str()


# ── ARTICLE TEXT PREPARATION (V5: no-chunk default) ───────────────────────

def prepare_article_text(text: str, max_chars: int = None) -> List[str]:
    """
    Prepare article text for LLM extraction.

    Default: send full article as single piece (no chunking).
    Gemini Flash 2.0 handles 1M tokens (~4M chars).
    Finance articles rarely exceed 15K chars.

    Only truncates if text exceeds max_chars.
    Returns list with single element for API compatibility.
    """
    if max_chars is None:
        max_chars = getattr(GlobalConfig, 'KG_MAX_ARTICLE_CHARS', 15000)

    text = text.strip()
    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    # Truncate at sentence boundary
    truncated = text[:max_chars]
    for sep in (". ", "! ", "? ", "\n"):
        pos = truncated.rfind(sep)
        if pos > max_chars * 0.8:
            truncated = truncated[:pos + len(sep)]
            break

    return [truncated]


# Legacy chunking — only used if GlobalConfig.KG_ENABLE_CHUNKING=True
CHUNK_SIZE    = getattr(GlobalConfig, 'KG_CHUNK_SIZE', 5000)
CHUNK_OVERLAP = getattr(GlobalConfig, 'KG_CHUNK_OVERLAP', 0)

def _split_at_sentence_boundary(text, max_chars):
    if len(text) <= max_chars: return len(text)
    window = text[:max_chars]
    for sep in (". ", "! ", "? ", "\n", " "):
        pos = window.rfind(sep)
        if pos > max_chars * 0.6: return pos + len(sep)
    return max_chars

def split_text_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Legacy chunking — only used if GlobalConfig.KG_ENABLE_CHUNKING=True."""
    text = text.strip()
    if len(text) <= chunk_size: return [text]
    chunks, start = [], 0
    while start < len(text):
        end = _split_at_sentence_boundary(text[start:], chunk_size)
        chunk = text[start: start + end].strip()
        if chunk: chunks.append(chunk)
        if start + end >= len(text): break
        start = start + end - overlap
        if start < 0: start = 0
    return chunks if chunks else [text[:chunk_size]]


def get_article_pieces(text: str) -> List[str]:
    """
    Route to correct text preparation based on config.

    Default: no-chunking (full article).
    Set GlobalConfig.KG_ENABLE_CHUNKING=True for legacy chunking.
    """
    if getattr(GlobalConfig, 'KG_ENABLE_CHUNKING', False):
        return split_text_chunks(text)
    else:
        return prepare_article_text(text)


def build_combined_text(titles, contents):
    h = "\n".join(f"- {t}" for t in titles if t and t.strip())
    c = "\n\n---\n\n".join(x for x in contents if x and x.strip())
    parts = []
    if h: parts.append(f"HEADLINES:\n{h}")
    if c: parts.append(f"ARTICLES:\n{c}")
    return "\n\n".join(parts)

def build_user_prompt(text, ticker, news_date, sector=None):
    class _SafeDict(dict):
        def __missing__(self, key): return ""
    user_part = FINDKG_LITE_USER_PROMPT.format_map(_SafeDict(ticker=ticker, news_date=news_date, news_text=text, sector=""))
    return f"EXAMPLES (study these carefully):\n\n{_FEW_SHOT_STR}\n\n{'='*60}\n\nNOW EXTRACT FROM THIS NEW ARTICLE:\n\n{user_part}"


def build_multi_article_prompt(articles_with_index: List[Tuple[int, str]], ticker: str, news_date: str) -> str:
    """
    Build prompt for multi-article extraction (up to KG_MAX_ARTICLES_PER_CALL articles).

    articles_with_index: [(0, "HEADLINES:\n...\nARTICLES:\n..."), (1, "..."), ...]

    Reuses FINDKG_LITE_USER_PROMPT template with combined article text.
    Appends multi-article instruction for article_index attribution.
    """
    # Build combined text with article markers
    blocks = []
    for idx, text in articles_with_index:
        blocks.append(f"[ARTICLE {idx}]\n{text}")
    combined = "\n\n---\n\n".join(blocks)

    class _SafeDict(dict):
        def __missing__(self, key): return ""

    user_part = FINDKG_LITE_USER_PROMPT.format_map(_SafeDict(
        ticker=ticker, news_date=news_date, news_text=combined, sector=""))

    # Append article_index attribution instruction
    user_part += (
        "\n\n━━━ MULTI-ARTICLE NOTE ━━━\n"
        "The text above contains MULTIPLE articles marked [ARTICLE 0], [ARTICLE 1], etc.\n"
        "For EACH triple, set \"article_index\" to the article number it primarily comes from.\n"
        "Apply extraction rules to EACH article independently.\n"
        "If an article is TYPE D with no financial content, extract 0 triples from it."
    )

    return (
        f"EXAMPLES (study these carefully):\n\n{_FEW_SHOT_STR}\n\n"
        f"{'='*60}\n\n"
        f"NOW EXTRACT FROM THESE {len(articles_with_index)} ARTICLES (same ticker, same date):\n\n"
        f"{user_part}"
    )

def _filter_and_clamp(raw, min_relevance, min_confidence):
    if not isinstance(raw, list): return []
    out = []
    for t in raw:
        if not isinstance(t, dict): continue
        rel, conf = float(t.get("relevance_to_ticker", 0)), float(t.get("confidence", 0))
        if rel < min_relevance or conf < min_confidence: continue
        t["confidence"]          = max(0.0, min(1.0, conf))
        t["price_impact_score"]  = max(-1.0, min(1.0, float(t.get("price_impact_score", 0.0))))
        t["relevance_to_ticker"] = max(0.0, min(1.0, rel))
        out.append(t)
    return out

# ── SOURCE TAGGING ────────────────────────────────────────────────────────────

def tag_triples_source(triples: List[Dict], sha1: str) -> List[Dict]:
    src = sha1[:8]
    for t in triples:
        t["_src"] = src
    return triples

# ── DEDUP ─────────────────────────────────────────────────────────────────────

def dedup_triples(triples):
    seen, out = set(), []
    for t in triples:
        key = (t.get("subject",{}).get("name",""), t.get("relation",""), t.get("object",{}).get("name",""))
        if key not in seen:
            seen.add(key); out.append(t)
    return out

_LEGAL_SUFFIX_RE = re.compile(
    r',?\s*(Inc\.?|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|Limited|Group|Platforms?|Holdings?|Services?|LLC|LLP|PLC|S\.A\.|N\.V\.)\.?\s*$',
    re.IGNORECASE)

def _norm_name_selfloop(name):
    if not name: return ""
    n = name
    for _ in range(3):
        stripped = _LEGAL_SUFFIX_RE.sub('', n).strip()
        if stripped == n.strip():
            break
        n = stripped
    return re.sub(r'\s+', ' ', n).strip().lower()


def _normalize_object_for_dedup(name: str) -> str:
    """
    Canonical form for fuzzy dedup of triple objects.

    V5: Stock % pattern uses ALL_TICKER_NAMES_PATTERN from ticker_aliases
    (matches all 9 tickers, not hardcoded WMT).
    """
    if not name:
        return ""
    n = name.lower()

    # ── Numeric formatting ──────────────────────────────────────────────────
    n = re.sub(r'(\d),(\d)', r'\1\2', n)

    # ── Monetary units ──────────────────────────────────────────────────────
    n = re.sub(r'\$\s*(\d+\.?\d*)\s*billion', r'$\1b', n, flags=re.IGNORECASE)
    n = re.sub(r'\$\s*(\d+\.?\d*)\s*million', r'$\1m', n, flags=re.IGNORECASE)
    n = re.sub(r'\$\s*(\d+\.?\d*)\s*b\b',     r'$\1b', n, flags=re.IGNORECASE)
    n = re.sub(r'\$\s*(\d+\.?\d*)\s*m\b',     r'$\1m', n, flags=re.IGNORECASE)

    # ── Stock price % movements (scalable for ALL tickers) ─────────────────
    n = re.sub(
        r'(?:' + ALL_TICKER_NAMES_PATTERN + r'|shares?|stock)'
        r'\s*(?:shares?|stock|price)?\s*([+-]?\d+\.?\d*)\s*%',
        lambda m: f'stock {round(float(m.group(1)))}pct',
        n, flags=re.IGNORECASE,
    )

    # ── Guidance/outlook strings without specific numbers ──────────────────
    if re.search(r'\b(guid|outlook|forecast)\w*', n) and '%' not in n:
        n = 'guidance'

    # ── Trailing unit words ────────────────────────────────────────────────
    n = re.sub(
        r'\b(units?|shares?|vehicles?|cars?|trucks?|vans?|jobs?|employees?|workers?|people|staff|posts?|items?|pieces?)\s*$',
        '', n, flags=re.IGNORECASE,
    )

    return re.sub(r'\s+', ' ', n).strip()


def smart_dedup_triples(triples):
    """
    Fuzzy dedup: resolve cross-article same-event duplicates.

    V5: Uses normalize_entity_name() from ticker_aliases for subject key.
    "Walmart"/"WMT"/"Walmart Inc." all → "wmt" → same key → merge.
    """
    if not triples: return []

    # Pass 1: exact dedup
    exact_seen, exact_deduped = set(), []
    for t in triples:
        key = (t.get("subject",{}).get("name",""), t.get("relation",""), t.get("object",{}).get("name",""))
        if key not in exact_seen:
            exact_seen.add(key); exact_deduped.append(t)

    # Pass 2: fuzzy dedup with normalized subject + object
    fuzzy = {}
    for t in exact_deduped:
        k = (
            normalize_entity_name(t.get("subject",{}).get("name","")),
            t.get("relation",""),
            _normalize_object_for_dedup(t.get("object",{}).get("name","")),
        )
        fuzzy.setdefault(k, []).append(t)

    result = []
    for _, group in fuzzy.items():
        if len(group) == 1:
            result.append(group[0]); continue
        impacts = [float(t.get("price_impact_score", 0)) for t in group]
        if any(x > 0.05 for x in impacts) and any(x < -0.05 for x in impacts):
            best = max(group, key=lambda t: (float(t.get("confidence",0)), abs(float(t.get("price_impact_score",0)))))
        else:
            best = max(group, key=lambda t: float(t.get("confidence",0)) * float(t.get("relevance_to_ticker",0)))
        result.append(best)
    return result

# ── POST-EXTRACTION QUALITY FILTERS ──────────────────────────────────────────

def fix_regulates_direction(triples):
    """Flip reversed REGULATES (COMP→ORG_REG) and drop semantic self-loops."""
    REGULATOR_TYPES = {"ORG_GOV", "ORG_REG"}
    COMPANY_TYPES   = {"COMP", "PRODUCT", "PERSON"}
    result = []
    for t in triples:
        subj, obj = t.get("subject",{}), t.get("object",{})
        if _norm_name_selfloop(subj.get("name","")) == _norm_name_selfloop(obj.get("name","")): continue
        if t.get("relation") != "REGULATES":
            result.append(t); continue
        st, ot = subj.get("type",""), obj.get("type","")
        if st in COMPANY_TYPES and ot in REGULATOR_TYPES:
            t2 = dict(t); t2["subject"] = dict(obj); t2["object"] = dict(subj); result.append(t2)
        elif st in COMPANY_TYPES and ot in COMPANY_TYPES:
            pass
        else:
            result.append(t)
    return result

def post_filter_triples(triples, min_rel_relates_to=0.75):
    result = []
    for t in triples:
        rel  = t.get("relation","")
        rv   = float(t.get("relevance_to_ticker", 0))
        imp  = float(t.get("price_impact_score", 0))
        ot   = t.get("object",{}).get("type","")
        if rel == "RELATES_TO" and ot == "ECON_IND":
            t2 = dict(t); t2["relation"] = "ANNOUNCES"; result.append(t2); continue
        if rel == "RELATES_TO" and rv < min_rel_relates_to: continue
        if abs(imp) < 0.05 and rv < 0.60: continue
        result.append(t)
    return result


def _is_analyst_action(subj_type: str, rel: str, obj_name: str) -> Tuple[bool, str]:
    """
    Detect analyst firm actions and classify them.

    Returns (is_analyst_action, action_type):
      - (True, "pt")      if object contains "price target" or "$X from/to"
      - (True, "rating")  if object contains rating keywords
      - (True, "other")   if COMP with CUTS/SIGNALS/RAISES but no clear subtype
      - (False, "")       otherwise
    """
    if subj_type != "COMP":
        return False, ""
    if rel not in {"CUTS", "SIGNALS", "RAISES"}:
        return False, ""

    obj_lower = obj_name.lower()

    # Price target detection
    if re.search(r'\bprice\s*target\b', obj_lower):
        return True, "pt"
    if re.search(r'\btarget\s*\$', obj_lower):
        return True, "pt"
    if re.search(r'\$\d+.*(?:from|to|→)', obj_lower):
        return True, "pt"

    # Rating detection
    rating_keywords = (
        r'\b(hold|buy|sell|overweight|underweight|neutral|outperform|'
        r'underperform|equal.?weight|sector.?perform|market.?perform|'
        r'strong.?buy|strong.?sell|accumulate|reduce)\b'
    )
    if re.search(rating_keywords, obj_lower):
        return True, "rating"

    return True, "other"


def limit_signals_per_source(
    triples: List[Dict],
    max_per_person:          int = 2,
    max_per_regulator:       int = 2,
    max_comp_signals:        int = 2,
    max_per_analyst_pt:      int = None,
    max_per_analyst_rating:  int = None,
) -> List[Dict]:
    """
    Cap noise from repeated sources.

    V5: Analyst COMP actions capped by (firm_name, action_type).
    Separate caps for PT actions and rating actions.
    Prevents bypass where "Hold rating" didn't contain "price target".

    Sort by confidence DESC → keep highest-quality triples within each cap.
    """
    if max_per_analyst_pt is None:
        max_per_analyst_pt = getattr(GlobalConfig, 'KG_MAX_PER_ANALYST_FIRM', 1)
    if max_per_analyst_rating is None:
        max_per_analyst_rating = getattr(GlobalConfig, 'KG_MAX_PER_ANALYST_RATING', 1)

    LIMIT_RELS_PERSON    = {"SIGNALS", "RAISES", "ANNOUNCES", "CUTS"}
    LIMIT_RELS_REGULATOR = {"REGULATES"}
    LIMIT_RELS_COMP_SIG  = {"SIGNALS"}
    REGULATOR_TYPES      = {"ORG_GOV", "ORG_REG"}

    indexed_sorted = sorted(
        enumerate(triples),
        key=lambda x: float(x[1].get("confidence", 0)),
        reverse=True,
    )
    person_count:        Dict[str, int] = {}
    reg_count:           Dict[str, int] = {}
    comp_sig_count:      Dict[str, int] = {}
    analyst_pt_count:    Dict[str, int] = {}
    analyst_rating_count:Dict[str, int] = {}
    analyst_other_count: Dict[str, int] = {}
    keep: set = set()

    for orig_i, t in indexed_sorted:
        rel       = t.get("relation", "")
        subj_name = t.get("subject", {}).get("name", "")
        subj_type = t.get("subject", {}).get("type", "")
        obj_name  = t.get("object",  {}).get("name", "")

        # Normalize subject for consistent counting
        subj_key = normalize_entity_name(subj_name)

        # ── PERSON cap ──────────────────────────────────────────────────────
        if rel in LIMIT_RELS_PERSON and subj_type == "PERSON":
            if person_count.get(subj_key, 0) >= max_per_person:
                continue
            person_count[subj_key] = person_count.get(subj_key, 0) + 1

        # ── Regulator cap ───────────────────────────────────────────────────
        elif rel in LIMIT_RELS_REGULATOR and subj_type in REGULATOR_TYPES:
            if reg_count.get(subj_key, 0) >= max_per_regulator:
                continue
            reg_count[subj_key] = reg_count.get(subj_key, 0) + 1

        # ── Analyst COMP actions — capped by (firm, action_type) ───────────
        else:
            is_analyst, action_type = _is_analyst_action(subj_type, rel, obj_name)

            if is_analyst:
                if action_type == "pt":
                    if analyst_pt_count.get(subj_key, 0) >= max_per_analyst_pt:
                        continue
                    analyst_pt_count[subj_key] = analyst_pt_count.get(subj_key, 0) + 1
                elif action_type == "rating":
                    if analyst_rating_count.get(subj_key, 0) >= max_per_analyst_rating:
                        continue
                    analyst_rating_count[subj_key] = analyst_rating_count.get(subj_key, 0) + 1
                else:
                    if analyst_other_count.get(subj_key, 0) >= max_per_analyst_pt:
                        continue
                    analyst_other_count[subj_key] = analyst_other_count.get(subj_key, 0) + 1

            # ── COMP SIGNALS cap (TA price level spam) ───────────────────
            elif rel in LIMIT_RELS_COMP_SIG and subj_type == "COMP":
                if comp_sig_count.get(subj_key, 0) >= max_comp_signals:
                    continue
                comp_sig_count[subj_key] = comp_sig_count.get(subj_key, 0) + 1

        keep.add(orig_i)

    return [t for i, t in enumerate(triples) if i in keep]

def apply_quality_filters(triples: List[Dict]) -> List[Dict]:
    triples = smart_dedup_triples(triples)
    triples = fix_regulates_direction(triples)
    triples = post_filter_triples(triples)
    triples = limit_signals_per_source(triples)
    return triples

# ── TICKER DETECTION & CROSS-TICKER FILTER ───────────────────────────────────

TITLE_WEIGHT = 3

def detect_primary_ticker(title, content, tickers):
    if not tickers: return ""
    if len(tickers) == 1: return tickers[0]
    tu, cu = (title or "").upper(), (content or "").upper()
    counts = {}
    for t in tickers:
        score = 0
        for name in TICKER_NAME_MAP.get(t, [t]):
            n = name.upper()
            score += tu.count(n) * TITLE_WEIGHT + cu.count(n)
        counts[t] = score
    best = max(counts.values())
    if best == 0: return tickers[0]
    for t in tickers:
        if counts[t] == best: return t

def _ticker_mentioned_in_triple(ticker: str, triple: Dict) -> bool:
    tl = ticker.lower()
    sn = triple.get("subject",{}).get("name","").lower()
    on = triple.get("object",{}).get("name","").lower()
    if tl in sn or tl in on: return True
    for name in TICKER_NAME_MAP.get(ticker.upper(), []):
        if name.lower() in sn or name.lower() in on: return True
    return False

def filter_triples_for_ticker(
    triples: List[Dict],
    primary_ticker: str,
    target_ticker: str,
    min_relevance: float = None,
) -> List[Dict]:
    if min_relevance is None:
        min_relevance = GlobalConfig.KG_MIN_RELEVANCE
    if primary_ticker.upper() == target_ticker.upper():
        return triples
    return [
        t for t in triples
        if _ticker_mentioned_in_triple(target_ticker, t)
        and float(t.get("relevance_to_ticker", 0)) >= min_relevance
    ]

# ── ASYNC CONCURRENT EXTRACTOR ────────────────────────────────────────────────

class AsyncConcurrentExtractor:
    def __init__(self, api_key=None, model=MODEL_ID, temperature=0.1, min_relevance=None,
                 min_confidence=None, max_concurrent=None):
        self.min_relevance  = min_relevance  if min_relevance  is not None else GlobalConfig.KG_MIN_RELEVANCE
        self.min_confidence = min_confidence if min_confidence is not None else GlobalConfig.KG_MIN_CONFIDENCE
        self.max_concurrent = max_concurrent if max_concurrent is not None else GlobalConfig.KG_MAX_CONCURRENT
        self.model = model
        _key = api_key or os.getenv("GEMINI_API_KEY")
        if not _key: raise RuntimeError("Missing GEMINI_API_KEY.")
        self.client = genai.Client(api_key=_key)
        self._gen_config = types.GenerateContentConfig(
            system_instruction=FINDKG_LITE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=temperature, max_output_tokens=8192)

    async def _extract_one_async(self, idx, article, semaphore):
        """
        Extract triples from one or multiple articles with retry on 429.

        Supports two modes:
          - Single article: article has "text", "ticker", "date"
            → build_user_prompt() wraps with few-shot
          - Multi-article: article has "_raw_prompt" (pre-built by build_multi_article_prompt)
            → use prompt directly, skip build_user_prompt()

        Retry logic:
          - Max retries from GlobalConfig.KG_ASYNC_MAX_RETRIES (default 3)
          - Backoff: KG_ASYNC_BACKOFF_BASE (default 10) * 2^attempt + jitter
          - Inter-request delay: KG_ASYNC_REQUEST_DELAY (default 1.0s)
          - Only retry on 429/RESOURCE_EXHAUSTED; other errors fail immediately
        """
        max_retries   = getattr(GlobalConfig, 'KG_ASYNC_MAX_RETRIES', 3)
        backoff_base  = getattr(GlobalConfig, 'KG_ASYNC_BACKOFF_BASE', 10.0)
        request_delay = getattr(GlobalConfig, 'KG_ASYNC_REQUEST_DELAY', 1.0)

        async with semaphore:
            # Multi-article: use pre-built prompt; Single-article: build from template
            if article.get("_raw_prompt"):
                prompt = article["_raw_prompt"]
            else:
                prompt = build_user_prompt(
                    article.get("text",""), article.get("ticker","UNKNOWN"), article.get("date",""))

            last_err = None
            for attempt in range(max_retries + 1):  # 0, 1, 2, 3 = 4 total tries
                try:
                    # Inter-request delay to spread out calls
                    if request_delay > 0:
                        await asyncio.sleep(request_delay)

                    response = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model, contents=prompt, config=self._gen_config)
                    return idx, _filter_and_clamp(
                        json.loads(response.text), self.min_relevance, self.min_confidence)

                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

                    if is_rate_limit and attempt < max_retries:
                        import random as _rand
                        wait = backoff_base * (2 ** attempt) + _rand.uniform(0, 2)
                        print(f"  429 idx={idx} attempt={attempt+1}/{max_retries} — retry in {wait:.0f}s")
                        await asyncio.sleep(wait)
                        continue
                    else:
                        # Non-retryable error or max retries exceeded
                        print(f"  Extract FAILED idx={idx}: {e}")
                        return idx, None  # None = real failure (not empty [])

            print(f"  Extract FAILED idx={idx} after {max_retries} retries: {last_err}")
            return idx, None

    def extract_batch(self, articles):
        if not articles: return []
        async def _run_all():
            sem = asyncio.Semaphore(self.max_concurrent)
            return await asyncio.gather(*[self._extract_one_async(i, a, sem) for i, a in enumerate(articles)])
        print(f"  AsyncConcurrentExtractor: {len(articles)} articles, max_concurrent={self.max_concurrent}")
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

        # Count actual successes (None = failed, [] = success with 0 triples)
        n_success = sum(1 for _, t in pairs if t is not None)
        n_failed  = sum(1 for _, t in pairs if t is None)
        if n_failed > 0:
            print(f"  Done in {time.time()-t0:.1f}s  ({n_success}/{len(articles)} succeeded, {n_failed} FAILED)")
        else:
            print(f"  Done in {time.time()-t0:.1f}s  ({n_success}/{len(articles)} succeeded)")

        results = [[] for _ in articles]
        for idx, triples in pairs:
            results[idx] = triples if triples is not None else []
        return results

# ── GEMINI BATCH API EXTRACTOR ────────────────────────────────────────────────

class GeminiBatchAPIExtractor:
    def __init__(self, api_key=None, model=MODEL_ID, min_relevance=None, min_confidence=None,
                 poll_interval_secs=30, max_wait_secs=86400, display_name="findkg-lite-v5"):
        self.min_relevance  = min_relevance  if min_relevance  is not None else GlobalConfig.KG_MIN_RELEVANCE
        self.min_confidence = min_confidence if min_confidence is not None else GlobalConfig.KG_MIN_CONFIDENCE
        self.poll_interval  = poll_interval_secs
        self.max_wait       = max_wait_secs
        self.display_name   = display_name
        self.model = f"models/{model}"
        _key = api_key or os.getenv("GEMINI_API_KEY")
        if not _key: raise RuntimeError("Missing GEMINI_API_KEY.")
        self.client = genai.Client(api_key=_key)

    def extract_batch(self, articles):
        if not articles: return []
        inline_requests = [
            types.InlinedRequest(
                contents=build_user_prompt(
                    a.get("text",""), a.get("ticker","UNKNOWN"), a.get("date","")),
                config=types.GenerateContentConfig(
                    system_instruction=FINDKG_LITE_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    temperature=_GEN_CONFIG_DICT["temperature"],
                    max_output_tokens=_GEN_CONFIG_DICT["max_output_tokens"],
                ),
                metadata={"idx": str(i)},
            )
            for i, a in enumerate(articles)
        ]
        print(f"  GeminiBatchAPIExtractor: submitting {len(inline_requests)} articles")
        batch_job = self.client.batches.create(
            model=self.model, src=inline_requests,
            config=types.CreateBatchJobConfig(displayName=self.display_name))
        print(f"  Job: {batch_job.name}  State: {batch_job.state.name}")
        terminal = {"JOB_STATE_SUCCEEDED","JOB_STATE_FAILED","JOB_STATE_CANCELLED"}
        elapsed = 0
        while elapsed < self.max_wait:
            time.sleep(self.poll_interval); elapsed += self.poll_interval
            batch_job = self.client.batches.get(name=batch_job.name)
            print(f"  [{elapsed:5d}s] {batch_job.state.name}")
            if batch_job.state.name in terminal: break
        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            print(f"Batch job failed: {batch_job.state.name}"); return [[] for _ in articles]
        results = [[] for _ in articles]
        dest = getattr(batch_job, "dest", None)
        if dest:
            for resp in (getattr(dest, "inlined_responses", None) or []):
                try:
                    meta = getattr(resp, "metadata", None) or {}
                    idx = int(meta.get("idx", -1))
                    if 0 <= idx < len(articles):
                        results[idx] = _filter_and_clamp(
                            json.loads(resp.response.candidates[0].content.parts[0].text),
                            self.min_relevance, self.min_confidence)
                except Exception:
                    pass
        return results