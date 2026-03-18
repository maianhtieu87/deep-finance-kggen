# data_pipeline/kg/extractor_batch.py — V3.2
"""
Quality filter chain added in V3.2:
  apply_quality_filters() = smart_dedup -> fix_regulates -> post_filter -> limit_signals
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

CHUNK_SIZE = 3000
CHUNK_OVERLAP = 200

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

def dedup_triples(triples):
    seen, out = set(), []
    for t in triples:
        key = (t.get("subject",{}).get("name",""), t.get("relation",""), t.get("object",{}).get("name",""))
        if key not in seen:
            seen.add(key); out.append(t)
    return out

def _normalize_object_for_dedup(name):
    if not name: return ""
    n = name.lower()
    n = re.sub(r'(\d),(\d)', r'\1\2', n)
    n = re.sub(r'\b(units?|shares?|vehicles?|cars?|trucks?|vans?|jobs?|employees?|workers?|people|staff|posts?|items?|pieces?)\s*$', '', n, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', n).strip()

def smart_dedup_triples(triples):
    if not triples: return []
    exact_seen, exact_deduped = set(), []
    for t in triples:
        key = (t.get("subject",{}).get("name",""), t.get("relation",""), t.get("object",{}).get("name",""))
        if key not in exact_seen:
            exact_seen.add(key); exact_deduped.append(t)
    fuzzy = {}
    for t in exact_deduped:
        k = (t.get("subject",{}).get("name","").lower().strip(), t.get("relation",""),
             _normalize_object_for_dedup(t.get("object",{}).get("name","")))
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

_LEGAL_SUFFIX_RE = re.compile(
    r',?\s*(Inc\.?|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|Limited|Group|Platforms?|Holdings?|Services?|LLC|LLP|PLC|S\.A\.|N\.V\.)\.?\s*$',
    re.IGNORECASE)

def _norm_name_selfloop(name):
    if not name: return ""
    n = _LEGAL_SUFFIX_RE.sub('', name).strip().lower()
    return re.sub(r'\s+', ' ', n).strip()

def fix_regulates_direction(triples):
    """Flip reversed REGULATES (COMP->ORG_REG) and drop semantic self-loops."""
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
    """Promote RELATES_TO+ECON_IND -> ANNOUNCES; drop RELATES_TO noise; drop zero-signal."""
    result = []
    for t in triples:
        rel, rv, imp = t.get("relation",""), float(t.get("relevance_to_ticker",0)), float(t.get("price_impact_score",0))
        ot = t.get("object",{}).get("type","")
        if rel == "RELATES_TO" and ot == "ECON_IND":
            t2 = dict(t); t2["relation"] = "ANNOUNCES"; result.append(t2); continue
        if rel == "RELATES_TO" and rv < min_rel_relates_to: continue
        if abs(imp) < 0.05 and rv < 0.60: continue
        result.append(t)
    return result

def limit_signals_per_source(triples, max_per_person=2, max_per_regulator=2):
    """Cap PERSON SIGNALS/RAISES/ANNOUNCES to max_per_person; ORG REGULATES to max_per_regulator."""
    LIMIT_RELS_P = {"SIGNALS", "RAISES", "ANNOUNCES"}
    LIMIT_RELS_R = {"REGULATES"}
    REG_TYPES    = {"ORG_GOV", "ORG_REG"}
    indexed_sorted = sorted(enumerate(triples), key=lambda x: float(x[1].get("confidence",0)), reverse=True)
    pc, rc, keep = {}, {}, set()
    for orig_i, t in indexed_sorted:
        rel, sn, st = t.get("relation",""), t.get("subject",{}).get("name",""), t.get("subject",{}).get("type","")
        if rel in LIMIT_RELS_P and st == "PERSON":
            if pc.get(sn, 0) >= max_per_person: continue
            pc[sn] = pc.get(sn, 0) + 1
        elif rel in LIMIT_RELS_R and st in REG_TYPES:
            if rc.get(sn, 0) >= max_per_regulator: continue
            rc[sn] = rc.get(sn, 0) + 1
        keep.add(orig_i)
    return [t for i, t in enumerate(triples) if i in keep]

def apply_quality_filters(triples):
    """Full quality chain: smart_dedup -> fix_regulates -> post_filter -> limit_signals."""
    triples = smart_dedup_triples(triples)
    triples = fix_regulates_direction(triples)
    triples = post_filter_triples(triples)
    triples = limit_signals_per_source(triples)
    return triples

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

def _ticker_mentioned_in_text(ticker, text_upper):
    for name in TICKER_NAME_MAP.get(ticker, [ticker]):
        nu = name.upper()
        if len(name) <= 3:
            if re.search(r'\b' + re.escape(nu) + r'\b', text_upper): return True
        else:
            if nu in text_upper: return True
    return False

def _ticker_mentioned_in_triple(ticker, triple):
    tl = ticker.lower()
    sn = triple.get("subject",{}).get("name","").lower()
    on = triple.get("object",{}).get("name","").lower()
    if tl in sn or tl in on: return True
    for name in TICKER_NAME_MAP.get(ticker.upper(), []):
        if name.lower() in sn or name.lower() in on: return True
    return False

def rescore_triples_for_ticker(triples, primary_ticker, target_ticker, min_relevance=None, article_text="", all_article_tickers=None):
    if min_relevance is None: min_relevance = GlobalConfig.KG_MIN_RELEVANCE
    if primary_ticker.upper() == target_ticker.upper(): return triples
    tu = (article_text or "").upper()
    tin = _ticker_mentioned_in_text(target_ticker, tu)
    oin = any(_ticker_mentioned_in_text(t, tu) for t in (all_article_tickers or []) if t.upper() != target_ticker.upper())
    if tin:        strict, emin = False, min_relevance
    elif oin:      strict, emin = True,  0.75
    else:          strict, emin = False, max(0.50, min_relevance)
    out = []
    for t in triples:
        t2 = dict(t)
        mentions = _ticker_mentioned_in_triple(target_ticker, t)
        or_ = float(t.get("relevance_to_ticker", 0.0))
        if strict:
            if not mentions: continue
            t2["relevance_to_ticker"] = min(1.0, or_ * 1.1)
        else:
            t2["relevance_to_ticker"] = min(1.0, or_ * 1.1) if mentions else or_ * 0.4
        t2["price_impact_score"] = 0.0
        if t2["relevance_to_ticker"] >= emin: out.append(t2)
    return out


class AsyncConcurrentExtractor:
    def __init__(self, api_key=None, model=MODEL_ID, temperature=0.1, min_relevance=None, min_confidence=None, max_concurrent=None):
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
            prompt = build_user_prompt(article.get("text",""), article.get("ticker","UNKNOWN"), article.get("date",""))
            try:
                response = await asyncio.to_thread(self.client.models.generate_content, model=self.model, contents=prompt, config=self._gen_config)
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


class GeminiBatchAPIExtractor:
    def __init__(self, api_key=None, model=MODEL_ID, min_relevance=None, min_confidence=None, poll_interval_secs=30, max_wait_secs=86400, display_name="findkg-lite-v3"):
        self.min_relevance  = min_relevance  if min_relevance  is not None else GlobalConfig.KG_MIN_RELEVANCE
        self.min_confidence = min_confidence if min_confidence is not None else GlobalConfig.KG_MIN_CONFIDENCE
        self.poll_interval, self.max_wait, self.display_name = poll_interval_secs, max_wait_secs, display_name
        self.model = f"models/{model}"
        _key = api_key or os.getenv("GEMINI_API_KEY")
        if not _key: raise RuntimeError("Missing GEMINI_API_KEY.")
        self.client = genai.Client(api_key=_key)

    def extract_batch(self, articles):
        if not articles: return []
        inline_requests = [{"key": str(i), "request": {
            "contents": [{"parts": [{"text": build_user_prompt(a.get("text",""), a.get("ticker","UNKNOWN"), a.get("date",""))}], "role": "user"}],
            "system_instruction": {"parts": [{"text": FINDKG_LITE_SYSTEM_PROMPT}]},
            "generation_config": _GEN_CONFIG_DICT}} for i, a in enumerate(articles)]
        print(f"  GeminiBatchAPIExtractor: submitting {len(inline_requests)} chunks")
        batch_job = self.client.batches.create(model=self.model, src=inline_requests, config={"display_name": self.display_name})
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
                    results[idx] = _filter_and_clamp(json.loads(resp.response.candidates[0].content.parts[0].text), self.min_relevance, self.min_confidence)
            except Exception: pass
        return results