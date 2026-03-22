# data_pipeline/kg/extractor_batch.py — V3.6
"""
V3.6 changes vs V3.5:
  Fix A — _normalize_object_for_dedup(): stock price % normalization.
    "WMT shares -9.3%" / "WMT stock -9.2%" / "Shares -9.03%" → same bucket.
    Requires explicit ticker name OR shares/stock keyword to avoid
    matching guidance % strings like "Q2 EPS guidance -8 to -9%".

  Fix B — limit_signals_per_source(): cap analyst COMP price target actions.
    Analyst firms typed COMP (e.g. Goldman Sachs, Stifel) that CUTS/SIGNALS/RAISES
    a "price target" object are capped at max_per_analyst_pt=1 per firm.
    Does NOT affect COMP CUTS for real events (guidance, workforce, costs).

  Fix C — _normalize_object_for_dedup(): guidance string normalization.
    "guidance Q2 and FY" / "Q2 profit guidance" / "FY23 guidance" / "guidance"
    → all collapse to canonical "guidance" when no % sign is present.
    "Q2 EPS guidance -8 to -9%" kept as-is (has %, contains actual data).

V3.5 carried over:
  - tag_triples_source(triples, sha1): _src=sha1[:8] traceability field.
  - P0: LIMIT_RELS_PERSON includes CUTS (analyst PERSON PT cuts capped at 2).
  - P1: monetary unit normalization ($88M → $88m etc).
  - CHUNK_SIZE=5000, CHUNK_OVERLAP=0.
  - filter_triples_for_ticker() mention-only cross-ticker filter.
"""
from __future__ import annotations
import asyncio, json, os, re, time, hashlib
from typing import Any, Dict, List, Optional, Tuple
from google import genai
from google.genai import types
from configs.config import GlobalConfig
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
            "reasoning": {"type": "STRING"},
        },
        "required": ["subject", "relation", "object", "confidence", "price_impact_score", "relevance_to_ticker"],
    },
}

_GEN_CONFIG_DICT = {
    "response_mime_type": "application/json",
    "response_schema": RESPONSE_SCHEMA,
    "temperature": 0.1,
    "max_output_tokens": 2048,
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

CHUNK_SIZE    = 5000
CHUNK_OVERLAP = 0

def _split_at_sentence_boundary(text, max_chars):
    if len(text) <= max_chars: return len(text)
    window = text[:max_chars]
    for sep in (". ", "! ", "? ", "\n", " "):
        pos = window.rfind(sep)
        if pos > max_chars * 0.6: return pos + len(sep)
    return max_chars

def split_text_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
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
    """
    Tag each triple with _src = first 8 chars of the article SHA-1.

    Traceability: given _src='a3f2bc91', open
    data/interim/kg_article_cache/a3f2bc91*.json → read '_meta.full_text'.

    Call AFTER apply_quality_filters() and BEFORE _save_cache():
        deduped = apply_quality_filters(merged)
        deduped = tag_triples_source(deduped, sha1)
        _save_cache(cache_dir, sha1, deduped, ...)
    """
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

def _normalize_object_for_dedup(name: str) -> str:
    """
    Canonical form for fuzzy dedup of triple objects.
    Applies Fix A, B (monetary), C in order — each rule is independent.
    """
    if not name:
        return ""
    n = name.lower()

    # ── Numeric formatting ──────────────────────────────────────────────────
    # Remove thousands comma: 10,000 → 10000
    n = re.sub(r'(\d),(\d)', r'\1\2', n)

    # ── Fix B (monetary units) ──────────────────────────────────────────────
    # $88 million / $88M → $88m  |  $1.2 billion / $1.2B → $1.2b
    n = re.sub(r'\$\s*(\d+\.?\d*)\s*billion', r'$\1b', n, flags=re.IGNORECASE)
    n = re.sub(r'\$\s*(\d+\.?\d*)\s*million', r'$\1m', n, flags=re.IGNORECASE)
    n = re.sub(r'\$\s*(\d+\.?\d*)\s*b\b',     r'$\1b', n, flags=re.IGNORECASE)
    n = re.sub(r'\$\s*(\d+\.?\d*)\s*m\b',     r'$\1m', n, flags=re.IGNORECASE)

    # ── Fix A (stock price % movements) ────────────────────────────────────
    # "WMT shares -9.3%" / "WMT stock -9.2%" / "shares -9.03%" → "wmt stock -9pct"
    # Requires explicit ticker OR shares/stock keyword — avoids matching
    # guidance strings like "Q2 EPS guidance -8 to -9%".
    n = re.sub(
        r'(?:wmt|walmart|shares?|stock)\s*(?:shares?|stock|price)?\s*([+-]?\d+\.?\d*)\s*%',
        lambda m: f'wmt stock {round(float(m.group(1)))}pct',
        n, flags=re.IGNORECASE,
    )

    # ── Fix C (guidance/outlook strings without specific numbers) ───────────
    # "guidance Q2 and FY" / "Q2 profit guidance" / "FY23 guidance" → "guidance"
    # "Q2 EPS guidance -8 to -9%" is kept as-is because it contains '%'.
    if re.search(r'\b(guid|outlook|forecast)\w*', n) and '%' not in n:
        n = 'guidance'

    # ── Trailing workforce unit words ───────────────────────────────────────
    # "10000 workers" / "10000 employees" → "10000"
    n = re.sub(
        r'\b(units?|shares?|vehicles?|cars?|trucks?|vans?|jobs?|employees?|workers?|people|staff|posts?|items?|pieces?)\s*$',
        '', n, flags=re.IGNORECASE,
    )

    return re.sub(r'\s+', ' ', n).strip()

def smart_dedup_triples(triples):
    """Fuzzy dedup: resolve cross-article same-event duplicates."""
    if not triples: return []
    exact_seen, exact_deduped = set(), []
    for t in triples:
        key = (t.get("subject",{}).get("name",""), t.get("relation",""), t.get("object",{}).get("name",""))
        if key not in exact_seen:
            exact_seen.add(key); exact_deduped.append(t)
    fuzzy = {}
    for t in exact_deduped:
        k = (
            t.get("subject",{}).get("name","").lower().strip(),
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

_LEGAL_SUFFIX_RE = re.compile(
    r',?\s*(Inc\.?|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|Limited|Group|Platforms?|Holdings?|Services?|LLC|LLP|PLC|S\.A\.|N\.V\.)\.?\s*$',
    re.IGNORECASE)

def _norm_name_selfloop(name):
    if not name: return ""
    n = _LEGAL_SUFFIX_RE.sub('', name).strip().lower()
    return re.sub(r'\s+', ' ', n).strip()

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
    """
    1. RELATES_TO + ECON_IND → promote to ANNOUNCES.
    2. RELATES_TO at rel < 0.75 → drop.
    3. abs(impact) < 0.05 AND rel < 0.60 → drop.
    """
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

def limit_signals_per_source(
    triples: List[Dict],
    max_per_person:       int = 2,
    max_per_regulator:    int = 2,
    max_comp_signals:     int = 2,
    max_per_analyst_pt:   int = 1,
) -> List[Dict]:
    """
    Cap noise from repeated sources.

    PERSON subjects — max 2 of {SIGNALS, RAISES, ANNOUNCES, CUTS} per person.
      Covers analyst PERSON PT cuts + exec statements.

    ORG_REG/ORG_GOV — max 2 REGULATES per regulatory body.

    COMP subjects, SIGNALS — max 2 SIGNALS per company.
      Prevents TA price-level spam (Market Clubhouse pattern).

    COMP subjects, price target actions (Fix B) — max 1 per analyst firm.
      Triggered when relation in {CUTS, SIGNALS, RAISES} AND object contains
      "price target". Prevents 15 analyst COMP CUTS flooding one day.
      Does NOT affect genuine company events (COMP CUTS guidance/workforce).

    Sort by confidence DESC → keep highest-quality triples within each cap.
    """
    LIMIT_RELS_PERSON    = {"SIGNALS", "RAISES", "ANNOUNCES", "CUTS"}
    LIMIT_RELS_REGULATOR = {"REGULATES"}
    LIMIT_RELS_COMP_SIG  = {"SIGNALS"}
    LIMIT_RELS_ANALYST_PT = {"CUTS", "SIGNALS", "RAISES"}   # Fix B
    REGULATOR_TYPES      = {"ORG_GOV", "ORG_REG"}

    indexed_sorted = sorted(
        enumerate(triples),
        key=lambda x: float(x[1].get("confidence", 0)),
        reverse=True,
    )
    person_count:     Dict[str, int] = {}
    reg_count:        Dict[str, int] = {}
    comp_sig_count:   Dict[str, int] = {}
    analyst_pt_count: Dict[str, int] = {}   # Fix B
    keep: set = set()

    for orig_i, t in indexed_sorted:
        rel       = t.get("relation", "")
        subj_name = t.get("subject", {}).get("name", "")
        subj_type = t.get("subject", {}).get("type", "")
        obj_name  = t.get("object",  {}).get("name", "").lower()

        # ── PERSON cap ──────────────────────────────────────────────────────
        if rel in LIMIT_RELS_PERSON and subj_type == "PERSON":
            if person_count.get(subj_name, 0) >= max_per_person:
                continue
            person_count[subj_name] = person_count.get(subj_name, 0) + 1

        # ── Regulator cap ───────────────────────────────────────────────────
        elif rel in LIMIT_RELS_REGULATOR and subj_type in REGULATOR_TYPES:
            if reg_count.get(subj_name, 0) >= max_per_regulator:
                continue
            reg_count[subj_name] = reg_count.get(subj_name, 0) + 1

        # ── Fix B: analyst COMP price target cap ────────────────────────────
        # Check BEFORE generic COMP SIGNALS cap so the two rules don't conflict.
        # "price target" in object name distinguishes analyst PT actions from
        # real company events (COMP CUTS guidance / COMP ANNOUNCES earnings).
        elif (rel in LIMIT_RELS_ANALYST_PT
              and subj_type == "COMP"
              and bool(re.search(r'\bprice\s*target\b', obj_name))):
            if analyst_pt_count.get(subj_name, 0) >= max_per_analyst_pt:
                continue
            analyst_pt_count[subj_name] = analyst_pt_count.get(subj_name, 0) + 1

        # ── COMP SIGNALS cap (TA price level spam) ───────────────────────────
        elif rel in LIMIT_RELS_COMP_SIG and subj_type == "COMP":
            if comp_sig_count.get(subj_name, 0) >= max_comp_signals:
                continue
            comp_sig_count[subj_name] = comp_sig_count.get(subj_name, 0) + 1

        keep.add(orig_i)

    return [t for i, t in enumerate(triples) if i in keep]

def apply_quality_filters(triples: List[Dict]) -> List[Dict]:
    """
    Full quality chain:
      smart_dedup → fix_regulates_direction → post_filter_triples → limit_signals_per_source

    tag_triples_source() is NOT called here — SHA-1 is only known at the
    call site. Pattern:
        deduped = apply_quality_filters(merged)
        deduped = tag_triples_source(deduped, sha1)
        _save_cache(cache_dir, sha1, deduped, ...)
    """
    triples = smart_dedup_triples(triples)
    triples = fix_regulates_direction(triples)
    triples = post_filter_triples(triples)
    triples = limit_signals_per_source(triples)
    return triples

# ── TICKER DETECTION & CROSS-TICKER FILTER ───────────────────────────────────

TICKER_NAME_MAP: Dict[str, List[str]] = {
    "TSLA": ["Tesla","TSLA"], "AAPL": ["Apple","AAPL"], "AMZN": ["Amazon","AMZN"],
    "MSFT": ["Microsoft","MSFT"], "GOOGL": ["Google","Alphabet","GOOGL"],
    "GOOG": ["Google","Alphabet","GOOG"], "META": ["Meta","Facebook","META"],
    "BA": ["Boeing","BA"], "JPM": ["JPMorgan","JP Morgan","JPM"],
    "WMT": ["Walmart","WMT"], "NVDA": ["Nvidia","NVDA"], "NFLX": ["Netflix","NFLX"],
    "INTC": ["Intel","INTC"], "AMD": ["AMD","Advanced Micro"],
    "RIVN": ["Rivian","RIVN"], "LCID": ["Lucid","LCID"],
    "GM": ["General Motors","GM"], "F": ["Ford","F Motor"],
}
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
    """Check if ticker (or any known name) appears in subject or object name."""
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
    """
    Simplified cross-ticker filter.

    Same ticker  → return triples unchanged.
    Cross-ticker → keep only triples where target company is explicitly
                   mentioned in subject or object name, AND rel >= min_relevance.
    No relevance adjustment, no price_impact zeroing. _src preserved.
    """
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
            temperature=temperature, max_output_tokens=2048)

    async def _extract_one_async(self, idx, article, semaphore):
        async with semaphore:
            prompt = build_user_prompt(
                article.get("text",""), article.get("ticker","UNKNOWN"), article.get("date",""))
            try:
                response = await asyncio.to_thread(
                    self.client.models.generate_content,
                    model=self.model, contents=prompt, config=self._gen_config)
                return idx, _filter_and_clamp(json.loads(response.text), self.min_relevance, self.min_confidence)
            except Exception as e:
                print(f"  Async extract error idx={idx}: {e}"); return idx, []

    def extract_batch(self, articles):
        if not articles: return []
        async def _run_all():
            sem = asyncio.Semaphore(self.max_concurrent)
            return await asyncio.gather(*[self._extract_one_async(i, a, sem) for i, a in enumerate(articles)])
        print(f"  AsyncConcurrentExtractor: {len(articles)} chunks, max_concurrent={self.max_concurrent}")
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
        print(f"  Done in {time.time()-t0:.1f}s  ({sum(1 for _,t in pairs if t is not None)}/{len(articles)} succeeded)")
        results = [[] for _ in articles]
        for idx, triples in pairs: results[idx] = triples or []
        return results

# ── GEMINI BATCH API EXTRACTOR ────────────────────────────────────────────────

class GeminiBatchAPIExtractor:
    def __init__(self, api_key=None, model=MODEL_ID, min_relevance=None, min_confidence=None,
                 poll_interval_secs=30, max_wait_secs=86400, display_name="findkg-lite-v3"):
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
        inline_requests = [{
            "key": str(i),
            "request": {
                "contents": [{"parts": [{"text": build_user_prompt(
                    a.get("text",""), a.get("ticker","UNKNOWN"), a.get("date",""))}], "role": "user"}],
                "system_instruction": {"parts": [{"text": FINDKG_LITE_SYSTEM_PROMPT}]},
                "generation_config": _GEN_CONFIG_DICT,
            },
        } for i, a in enumerate(articles)]
        print(f"  GeminiBatchAPIExtractor: submitting {len(inline_requests)} chunks")
        batch_job = self.client.batches.create(
            model=self.model, src=inline_requests, config={"display_name": self.display_name})
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
        for resp in (getattr(batch_job.response, "inlined_responses", None) or []):
            try:
                idx = int(resp.key)
                if 0 <= idx < len(articles):
                    results[idx] = _filter_and_clamp(
                        json.loads(resp.response.candidates[0].content.parts[0].text),
                        self.min_relevance, self.min_confidence)
            except Exception:
                pass
        return results