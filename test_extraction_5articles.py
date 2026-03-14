#!/usr/bin/env python3
"""
test_extraction_5articles.py — Pipeline-Faithful Extraction Test
================================================================
Mirrors EXACTLY what KGGenNewsEmbedder._collect_day_triples_batch() does:

  1. Ticker normalization     → TICKER_ALIAS_MAP (AMAZON → AMZN, v.v.)
  2. Multi-ticker parsing     → "AAPL,GOOGL" → ["AAPL", "GOOGL"]
  3. SHA-1 corpus dedup       → mỗi article text chỉ extract 1 lần
  4. rescore_triples_for_ticker() → fan-out sang mỗi ticker
  5. Confidence + relevance filter sau rescore
  6. Dedup triples per (ticker, date)

Output:  Dict[ticker → List[RichTriple]]  — giống final graph input.

Modes:
  Sequential (default) : FinDKGLiteExtractor (1 article / call)
  Batch (--batch)      : GeminiBatchAPIExtractor (50% cost, async)

Usage:
  python test_extraction_5articles.py
  python test_extraction_5articles.py --n 10 --ticker AMZN
  python test_extraction_5articles.py --batch
  python test_extraction_5articles.py --dry-run
  python test_extraction_5articles.py --n 5 --save results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# TICKER ALIAS MAP  (GAP 1 FIX)
# Maps full company names / common variants → canonical ticker symbol.
# Add entries here as you encounter new ones in your parquet.
# ─────────────────────────────────────────────────────────────────────────────
TICKER_ALIAS_MAP: Dict[str, str] = {
    # Full names
    "AMAZON":           "AMZN",
    "AMAZON.COM":       "AMZN",
    "MICROSOFT":        "MSFT",
    "ALPHABET":         "GOOGL",
    "GOOGLE":           "GOOGL",
    "META PLATFORMS":   "META",
    "FACEBOOK":         "META",
    "APPLE":            "AAPL",
    "APPLE INC":        "AAPL",
    "NVIDIA":           "NVDA",
    "TESLA":            "TSLA",
    "NETFLIX":          "NFLX",
    "SALESFORCE":       "CRM",
    "SHOPIFY":          "SHOP",
    "PAYPAL":           "PYPL",
    "SQUARE":           "SQ",
    "BLOCK":            "SQ",
    "TWITTER":          "TWTR",
    "SNAPCHAT":         "SNAP",
    "SPOTIFY":          "SPOT",
    "DISNEY":           "DIS",
    "COMCAST":          "CMCSA",
    "VERIZON":          "VZ",
    "AT&T":             "T",
    "JPMORGAN":         "JPM",
    "JP MORGAN":        "JPM",
    "GOLDMAN SACHS":    "GS",
    "MORGAN STANLEY":   "MS",
    "BANK OF AMERICA":  "BAC",
    "WELLS FARGO":      "WFC",
    "CITIGROUP":        "C",
    "BLACKROCK":        "BLK",
    "VISA":             "V",
    "MASTERCARD":       "MA",
    "AMERICAN EXPRESS": "AXP",
    "JOHNSON & JOHNSON":"JNJ",
    "PFIZER":           "PFE",
    "MERCK":            "MRK",
    "ABBVIE":           "ABBV",
    "ELI LILLY":        "LLY",
    "EXXON":            "XOM",
    "EXXONMOBIL":       "XOM",
    "CHEVRON":          "CVX",
    "BOEING":           "BA",
    "LOCKHEED":         "LMT",
    "GENERAL ELECTRIC": "GE",
    "CATERPILLAR":      "CAT",
    "HONEYWELL":        "HON",
    "UPS":              "UPS",
    "FEDEX":            "FDX",
    "RIVIAN":           "RIVN",
    "LUCID":            "LCID",
    "TOYOTA":           "TM",
    "GENERAL MOTORS":   "GM",
    "FORD":             "F",
    "INTEL":            "INTC",
    "AMD":              "AMD",
    "BROADCOM":         "AVGO",
    "QUALCOMM":         "QCOM",
    "TAIWAN SEMICONDUCTOR": "TSM",
    "ORACLE":           "ORCL",
    "SAP":              "SAP",
    "IBM":              "IBM",
    "DELL":             "DELL",
    "HP":               "HPQ",
    "NIKE":             "NKE",
    "MCDONALD'S":       "MCD",
    "MCDONALDS":        "MCD",
    "STARBUCKS":        "SBUX",
}


def normalize_ticker(raw: str) -> str:
    """
    Normalize một ticker string thành canonical symbol.
    - Strip whitespace, upper-case
    - Lookup TICKER_ALIAS_MAP cho full-name variants
    """
    s = raw.strip().upper()
    return TICKER_ALIAS_MAP.get(s, s)


def parse_and_normalize_tickers(val: Any) -> List[str]:
    """
    Parse cột ticker (string hoặc list) → list of canonical tickers.
    "AAPL,GOOGL,AMAZON" → ["AAPL", "GOOGL", "AMZN"]
    ["APPLE", "MSFT"]   → ["AAPL", "MSFT"]
    """
    if isinstance(val, list):
        raw_list = [str(t) for t in val if str(t).strip()]
    elif isinstance(val, str):
        raw_list = [t for t in val.split(",") if t.strip()]
    else:
        return []
    return [normalize_ticker(t) for t in raw_list if normalize_ticker(t)]


# ─────────────────────────────────────────────────────────────────────────────
# IMPORT PIPELINE MODULES
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _import_pipeline() -> Dict[str, Any]:
    imports: Dict[str, Any] = {}
    errors: List[str] = []

    try:
        from data_pipeline.kg.extractor import FinDKGLiteExtractor
        imports["FinDKGLiteExtractor"] = FinDKGLiteExtractor
        print("  ✅ data_pipeline.kg.extractor")
    except ImportError as e:
        errors.append(f"  ❌ extractor: {e}")

    try:
        from data_pipeline.kg.extractor_batch import (
            AsyncConcurrentExtractor,
            GeminiBatchAPIExtractor,
            rescore_triples_for_ticker,
            build_user_prompt,
        )
        imports["AsyncConcurrentExtractor"]   = AsyncConcurrentExtractor
        imports["GeminiBatchAPIExtractor"]    = GeminiBatchAPIExtractor
        imports["rescore_triples_for_ticker"] = rescore_triples_for_ticker
        imports["build_user_prompt"]          = build_user_prompt
        print("  ✅ data_pipeline.kg.extractor_batch")
    except ImportError as e:
        errors.append(f"  ❌ extractor_batch: {e}")

    try:
        from data_pipeline.kg.prompts import (
            VALID_RELATIONS,
            VALID_ENTITY_TYPES,
        )
        imports["TICKER_SECTOR_MAP"]  = {}   # removed — no longer used
        imports["VALID_RELATIONS"]    = VALID_RELATIONS
        imports["VALID_ENTITY_TYPES"] = VALID_ENTITY_TYPES
        print("  ✅ data_pipeline.kg.prompts")
    except ImportError as e:
        errors.append(f"  ❌ prompts: {e}")

    if errors:
        print("\n  Import errors:")
        for e in errors:
            print(f"    {e}")
        print(f"\n  Run from project root:\n    cd {PROJECT_ROOT}\n    python test_extraction_5articles.py")
        if "FinDKGLiteExtractor" not in imports:
            raise ImportError("Cannot proceed without FinDKGLiteExtractor")

    return imports


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE-FAITHFUL BATCH COLLECTOR  (GAP 2 + 3 FIX)
# Mirrors KGGenNewsEmbedder._collect_day_triples_batch() exactly.
# ─────────────────────────────────────────────────────────────────────────────

def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def detect_primary_ticker(text: str, tickers: List[str]) -> str:
    """
    Pick the ticker mentioned most often in article text.
    Falls back to first ticker if none are explicitly mentioned.
    Ensures extract() uses the right TARGET_STOCK context.
    """
    if not tickers:
        return ""
    if len(tickers) == 1:
        return tickers[0]
    text_upper = text.upper()
    counts = {t: text_upper.count(t.upper()) for t in tickers}
    best_count = max(counts.values())
    if best_count == 0:
        return tickers[0]
    for t in tickers:       # preserve list order for ties
        if counts[t] == best_count:
            return t


def _cache_path(cache_dir: str, sha1: str) -> str:
    return os.path.join(cache_dir, f"{sha1}.json")


def _load_cache(cache_dir: str, sha1: str) -> Optional[List[Dict]]:
    """Return cached triples list, or None if not cached yet."""
    if not cache_dir:
        return None
    p = _cache_path(cache_dir, sha1)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f).get("triples", [])
    except Exception:
        return None


def _save_cache(cache_dir: str, sha1: str, triples: List[Dict]) -> None:
    """Persist triples to disk cache. Empty list [] is valid (confirmed no-event)."""
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    p = _cache_path(cache_dir, sha1)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"triples": triples, "_v": "v2"}, f, ensure_ascii=False)
    except Exception:
        pass


def collect_triples_pipeline_faithful(
    articles: List[Dict[str, Any]],
    # Each article: {text, tickers: List[str], date, primary_ticker}
    extractor,
    rescore_fn,
    ticker_sector_map: Dict[str, str],
    min_relevance: float = 0.30,
    min_confidence: float = 0.35,
    use_batch: bool = False,
    cache_dir: str = "",
    # Shared with KGGenNewsEmbedder cache: pass the same path to reuse production cache.
    # Default "" = no disk cache (memory-only, safe for one-off runs).
    # Recommended: set to "data/interim/kg_article_cache" (same as pipeline default).
) -> Tuple[Dict[str, List[Dict]], Dict[str, Any]]:
    """
    Process a list of articles through the full pipeline:

      1. Corpus-level dedup by SHA-1
      2. Disk cache lookup  — skips API call if already extracted
      3. Extract triples once per uncached unique text
      4. Write results to disk cache
      5. Fan-out + rescore_triples_for_ticker() for each target ticker
      6. Filter by confidence + relevance after rescore
      7. Dedup per (ticker, subject, relation, object)

    Returns:
        results   : Dict[ticker → List[RichTriple]]   — final graph input per ticker
        debug_info: Dict with extraction metadata for display
    """
    # ── Phase 1: Corpus-level dedup ──────────────────────────────────────────
    sha1_to_meta: Dict[str, Dict] = {}
    sha1_to_raw:  Dict[str, Optional[List[Dict]]] = {}

    for art in articles:
        text    = _norm(art.get("text", ""))
        tickers = art.get("tickers", [])
        primary = art.get("primary_ticker", tickers[0] if tickers else "UNKNOWN")
        date    = art.get("date", "")
        title   = art.get("title", "")

        if not text or not tickers:
            continue
        h = _sha1(text)
        if h in sha1_to_meta:
            sha1_to_meta[h]["tickers"] = list(
                set(sha1_to_meta[h]["tickers"]) | set(tickers)
            )
            continue

        sha1_to_meta[h] = {
            "text":           text,
            "primary_ticker": primary,
            "date":           date,
            "title":          title,
            "tickers":        tickers,
            "sector":         ticker_sector_map.get(primary, "Technology"),
        }
        sha1_to_raw[h] = None  # placeholder

    unique_count = len(sha1_to_meta)

    # ── Phase 2: Disk cache lookup ────────────────────────────────────────────
    cache_hits = 0
    if cache_dir:
        for h in sha1_to_meta:
            cached = _load_cache(cache_dir, h)
            if cached is not None:          # [] is a valid cached result
                sha1_to_raw[h] = cached
                cache_hits += 1

    uncached = [h for h, v in sha1_to_raw.items() if v is None]

    print(f"\n  📋 Unique articles: {unique_count} "
          f"(from {len(articles)} total rows)")
    if cache_dir:
        print(f"  💾 Cache hits: {cache_hits}/{unique_count} — "
              f"API calls needed: {len(uncached)}")
    else:
        print(f"  ⚠️  No cache_dir set — all {len(uncached)} articles will call API")
        print(f"     Pass cache_dir='data/interim/kg_article_cache' to reuse across runs")

    # ── Phase 3: Extract only uncached articles ───────────────────────────────
    timing: Dict[str, float] = {}
    extract_errors: List[str] = []

    if uncached:
        if use_batch:
            batch_input = [
                {
                    "text":   sha1_to_meta[h]["text"],
                    "ticker": sha1_to_meta[h]["primary_ticker"],
                    "date":   sha1_to_meta[h]["date"],
                    "sector": sha1_to_meta[h]["sector"],
                }
                for h in uncached
            ]
            t0 = time.time()
            try:
                batch_results = extractor.extract_batch(batch_input)
                elapsed_total = time.time() - t0
                per_art = elapsed_total / max(1, len(batch_input))
                for h, triples in zip(uncached, batch_results):
                    triples = triples or []
                    sha1_to_raw[h] = triples
                    timing[h] = per_art
                    _save_cache(cache_dir, h, triples)
            except Exception as e:
                extract_errors.append(str(e))
                for h in uncached:
                    sha1_to_raw[h] = []
        else:
            for h in uncached:
                meta = sha1_to_meta[h]
                t0 = time.time()
                try:
                    triples = extractor.extract(
                        text=meta["text"],
                        ticker=meta["primary_ticker"],
                        news_date=meta["date"],
                    )
                    triples = triples or []
                    sha1_to_raw[h] = triples
                    _save_cache(cache_dir, h, triples)
                except Exception as e:
                    extract_errors.append(f"{meta['primary_ticker']} {meta['date']}: {e}")
                    sha1_to_raw[h] = []
                timing[h] = time.time() - t0

    # ── Phase 3: Fan-out + rescore per ticker ─────────────────────────────────
    # ticker → { (subj, rel, obj) → triple }   (dedup key)
    per_ticker_seen:   Dict[str, set]       = defaultdict(set)
    per_ticker_triples: Dict[str, List[Dict]] = defaultdict(list)

    for h, raw_triples in sha1_to_raw.items():
        if not raw_triples:
            continue
        meta    = sha1_to_meta[h]
        primary = meta["primary_ticker"]

        for target_ticker in meta["tickers"]:
            # rescore_triples_for_ticker handles primary == target case (no-op)
            rescored = rescore_fn(
                raw_triples, primary, target_ticker, min_relevance
            )
            # Apply confidence filter after rescore
            rescored = [
                t for t in rescored
                if float(t.get("confidence", 0)) >= min_confidence
            ]
            # Dedup per ticker
            for t in rescored:
                key = (
                    t.get("subject", {}).get("name", ""),
                    t.get("relation", ""),
                    t.get("object",  {}).get("name", ""),
                )
                if key not in per_ticker_seen[target_ticker]:
                    per_ticker_seen[target_ticker].add(key)
                    per_ticker_triples[target_ticker].append(t)

    results = dict(per_ticker_triples)  # Dict[ticker → List[RichTriple]]

    debug_info = {
        "sha1_to_meta":   sha1_to_meta,
        "sha1_to_raw":    sha1_to_raw,
        "timing":         timing,
        "extract_errors": extract_errors,
        "unique_count":   unique_count,
    }
    return results, debug_info


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PARQUET = r"D:\ProjectNCKH\deep_finance\data\interim\concatenated_news_filtered.parquet"

COLUMN_CANDIDATES = {
    "text":    ["content", "text", "body", "article", "extracted_summary"],
    "date":    ["date", "datetime", "published_date", "publish_date", "timestamp"],
    "ticker":  ["symbols", "equity", "ticker", "symbol"],
    "title":   ["title", "headline"],
}


def detect_columns(df) -> Dict[str, Optional[str]]:
    cols_lower = {c.lower(): c for c in df.columns}
    result = {}
    for field, candidates in COLUMN_CANDIDATES.items():
        found = None
        for c in candidates:
            if c.lower() in cols_lower:
                found = cols_lower[c.lower()]
                break
        result[field] = found
    return result


def load_and_build_articles(
    parquet_path: str,
    n: int = 5,
    ticker_filter: Optional[str] = None,
    use_summary: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[str]]]:
    """
    Load parquet and build article dicts with multi-ticker support.

    Each returned article dict:
        text           : str
        tickers        : List[str]   canonical, e.g. ["AMZN", "RIVN"]
        primary_ticker : str         first ticker after normalization
        date           : str
        title          : str

    Returns:
        articles : List[article dict]
        cols     : detected column mapping (for debug display)
    """
    import pandas as pd

    print(f"\n📂 Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"   Rows: {len(df):,}  |  Columns: {list(df.columns)}")

    cols = detect_columns(df)
    print(f"\n📋 Column mapping:")
    for field, col in cols.items():
        print(f"   {field:<8}: {col or '❌ not found'}")

    if not cols["text"]:
        raise ValueError(f"No text column found. Available: {list(df.columns)}")

    # Choose text column
    text_col = cols["text"]
    if use_summary and "extracted_summary" in df.columns:
        text_col = "extracted_summary"
        print(f"\n   Using text column: extracted_summary (--use-summary)")
    else:
        print(f"\n   Using text column: {text_col}")

    df = df.dropna(subset=[text_col])
    df = df[df[text_col].astype(str).str.strip().str.len() > 50]

    # Parse tickers (multi-ticker aware)
    ticker_col = cols["ticker"]
    if not ticker_col:
        raise ValueError("No ticker column found.")

    df["_tickers_parsed"] = df[ticker_col].apply(parse_and_normalize_tickers)
    df = df[df["_tickers_parsed"].map(len) > 0]

    # Optional: filter to rows containing a specific ticker
    if ticker_filter:
        tf = normalize_ticker(ticker_filter)
        df = df[df["_tickers_parsed"].apply(lambda lst: tf in lst)]
        if len(df) == 0:
            print(f"\n⚠️  No articles for ticker '{tf}' after normalization — using all data")
            df = pd.read_parquet(parquet_path)
            df = df.dropna(subset=[text_col])
            df["_tickers_parsed"] = df[ticker_col].apply(parse_and_normalize_tickers)
            df = df[df["_tickers_parsed"].map(len) > 0]
        else:
            print(f"   Filtered to ticker={tf}: {len(df):,} rows")

    # Sample n rows (diverse tickers preferred)
    df = df.reset_index(drop=True)
    if len(df) > n * 3:
        # Try to sample across unique tickers
        unique_tickers = df["_tickers_parsed"].explode().unique()
        samples_idx = []
        per_t = max(1, n // max(1, len(unique_tickers)))
        for t in unique_tickers:
            mask = df["_tickers_parsed"].apply(lambda lst: t in lst)
            idx  = df[mask].head(per_t).index.tolist()
            samples_idx.extend(idx)
            if len(set(samples_idx)) >= n:
                break
        # Top up if needed
        remaining = [i for i in df.index if i not in set(samples_idx)]
        for i in remaining:
            if len(set(samples_idx)) >= n:
                break
            samples_idx.append(i)
        df = df.loc[list(dict.fromkeys(samples_idx))].head(n)
    else:
        df = df.head(n)

    # Build article dicts
    articles = []
    for _, row in df.iterrows():
        tickers = list(row["_tickers_parsed"])
        text_content = _norm(str(row.get(text_col, "")))
        articles.append({
            "text":           text_content,
            "tickers":        tickers,
            "primary_ticker": detect_primary_ticker(text_content, tickers),
            "date":           str(row[cols["date"]])[:10] if cols["date"] else "",
            "title":          str(row.get(cols["title"], ""))[:120] if cols["title"] else "",
        })

    print(f"\n✅ Loaded {len(articles)} articles")
    multi = sum(1 for a in articles if len(a["tickers"]) > 1)
    if multi:
        print(f"   Multi-ticker articles: {multi} "
              f"(fan-out to {sum(len(a['tickers']) for a in articles)} ticker-article pairs)")

    return articles, cols


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def impact_icon(score: float) -> str:
    if score >=  0.6: return "🟢🟢"
    if score >=  0.3: return "🟢"
    if score >= -0.3: return "⚪"
    if score >= -0.6: return "🔴"
    return "🔴🔴"


def bar(val: float, width: int = 10) -> str:
    filled = max(0, min(width, round(abs(val) * width)))
    return "█" * filled + "░" * (width - filled)


def print_article_header(art: Dict, idx: int, total: int):
    tickers_str = ", ".join(art["tickers"])
    multi_note  = f"  [multi-ticker → fan-out to: {tickers_str}]" if len(art["tickers"]) > 1 else ""
    print(f"\n{'═'*68}")
    print(f"  Article {idx}/{total}  |  Primary: {art['primary_ticker']}  |  Date: {art['date']}")
    if len(art["tickers"]) > 1:
        print(f"  All tickers: {tickers_str}")
    title = textwrap.fill(art["title"], 62, subsequent_indent="           ")
    print(f"  Title  : {title}")
    print(f"  Length : {len(art['text']):,} chars")
    print(f"{'═'*68}")


def print_triple(i: int, t: Dict, label: str = ""):
    subj   = t.get("subject", {})
    obj    = t.get("object",  {})
    rel    = t.get("relation", "?")
    conf   = float(t.get("confidence", 0))
    rel_s  = float(t.get("relevance_to_ticker", 0))
    impact = float(t.get("price_impact_score", 0))
    reason = t.get("reasoning", "")

    group_a = {"ANNOUNCES","RAISES","CUTS","INVESTS_IN","DIVESTS","APPOINTS"}
    group_b = {"POS_IMPACTS","NEG_IMPACTS","COMPETES_WITH","REGULATES","SUPPLIES_TO"}
    grp = "A" if rel in group_a else "B" if rel in group_b else "C"

    prefix = f"  [{label}] " if label else "  "
    print(f"\n{prefix}── Triple #{i+1}  [Group {grp}] {'─'*40}")
    print(f"  [{subj.get('type','?'):8}] {subj.get('name','?')}")
    print(f"      {rel}")
    print(f"  [{obj.get('type','?'):8}] {obj.get('name','?')}")
    print(f"  conf={conf:.2f} {bar(conf)}  rel={rel_s:.2f} {bar(rel_s)}  "
          f"impact={impact:+.2f} {impact_icon(impact)}")
    if reason:
        print(textwrap.fill(reason, 64,
                            initial_indent="  → ", subsequent_indent="    "))


def print_ticker_results(ticker: str, triples: List[Dict],
                         primary: str, rescored: bool):
    marker = " (rescored)" if rescored else " (primary)"
    print(f"\n  ┌── Ticker: {ticker}{marker}  —  {len(triples)} triples")
    if not triples:
        print(f"  │  ⚠️  All triples filtered out after rescore + threshold")
    for i, t in enumerate(triples):
        print_triple(i, t)
    print(f"  └{'─'*60}")


def print_summary(
    articles: List[Dict],
    results_by_ticker: Dict[str, List[Dict]],
    debug_info: Dict,
    use_batch: bool,
    elapsed_total: float,
):
    sha1_to_meta = debug_info["sha1_to_meta"]
    sha1_to_raw  = debug_info["sha1_to_raw"]
    timing       = debug_info["timing"]
    errors       = debug_info["extract_errors"]
    unique_count = debug_info["unique_count"]

    total_raw    = sum(len(v) for v in sha1_to_raw.values() if v)
    total_final  = sum(len(v) for v in results_by_ticker.values())
    tickers_out  = sorted(results_by_ticker.keys())

    print(f"\n{'━'*68}")
    print(f"  PIPELINE-FAITHFUL SUMMARY")
    print(f"{'━'*68}")
    print(f"  Mode          : {'Batch API (50% cost)' if use_batch else 'Sequential'}")
    print(f"  Input rows    : {len(articles)}")
    print(f"  Unique texts  : {unique_count}  (SHA-1 dedup)")
    print(f"  Raw triples   : {total_raw}  (before rescore/filter)")
    print(f"  Final triples : {total_final}  (after rescore + threshold)")
    print(f"  Output tickers: {len(tickers_out)}  →  {tickers_out}")
    print(f"  Total time    : {elapsed_total:.1f}s")
    if errors:
        print(f"\n  ⚠️  Extract errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"     {e}")

    # Per-ticker summary
    print(f"\n  {'Ticker':<8} {'Triples':>8}  {'Articles':>9}  Avg impact")
    print(f"  {'─'*44}")
    for tk in tickers_out:
        triples = results_by_ticker[tk]
        n_art   = sum(1 for a in articles if tk in a["tickers"])
        if triples:
            avg_imp = sum(float(t.get("price_impact_score", 0)) for t in triples) / len(triples)
            imp_str = f"{avg_imp:+.2f}  {impact_icon(avg_imp)}"
        else:
            imp_str = "  n/a"
        print(f"  {tk:<8} {len(triples):>8}  {n_art:>9}  {imp_str}")

    # Relation distribution (across all tickers)
    all_triples = [t for v in results_by_ticker.values() for t in v]
    if all_triples:
        rc = Counter(t.get("relation") for t in all_triples)
        print(f"\n  Relation distribution ({len(all_triples)} final triples):")
        group_a = {"ANNOUNCES","RAISES","CUTS","INVESTS_IN","DIVESTS","APPOINTS"}
        group_b = {"POS_IMPACTS","NEG_IMPACTS","COMPETES_WITH","REGULATES","SUPPLIES_TO"}
        for rel, cnt in sorted(rc.items(), key=lambda x: -x[1]):
            grp = "A" if rel in group_a else "B" if rel in group_b else "C"
            print(f"    [{grp}] {rel:<20} {'█'*cnt} {cnt}")

        n = len(all_triples)
        ga = sum(1 for t in all_triples if t.get("relation") in group_a)
        gb = sum(1 for t in all_triples if t.get("relation") in group_b)
        ab_pct = 100 * (ga + gb) / n
        quality = "✅ Good" if ab_pct >= 50 else "⚠️  Low"
        print(f"\n  Group A+B rate: {ga+gb}/{n} ({ab_pct:.0f}%)  {quality}")

    print(f"{'━'*68}\n")


# ─────────────────────────────────────────────────────────────────────────────
# ARGS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pipeline-faithful KG extraction test (mirrors KGGenNewsEmbedder)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          python test_extraction_5articles.py
          python test_extraction_5articles.py --n 10 --ticker AMZN
          python test_extraction_5articles.py --batch
          python test_extraction_5articles.py --dry-run
          python test_extraction_5articles.py --n 5 --save results.json
        """),
    )
    p.add_argument("--data",        default=DEFAULT_PARQUET,
                   help="Path to parquet file")
    p.add_argument("--n",           type=int, default=5,
                   help="Number of articles to test (default=5)")
    p.add_argument("--ticker",      default=None,
                   help="Filter articles containing this ticker (e.g. AMZN)")
    p.add_argument("--gemini-batch", action="store_true",
                   help=(
                       "Use Gemini Batch API (50%% cost, async job, 30s+ latency). "
                       "Default is AsyncConcurrentExtractor (standard cost, fast). "
                       "Use --gemini-batch only for large runs (>500 articles)."
                   ))
    p.add_argument("--max-concurrent", type=int, default=5,
                   help=(
                       "Max concurrent requests for AsyncConcurrentExtractor. "
                       "Free tier: 5 | Paid tier: 10-15. Default: 5."
                   ))
    p.add_argument("--dry-run",     action="store_true",
                   help="Preview data only, no API calls")
    p.add_argument("--save",        default=None,
                   help="Save results JSON to file (e.g. --save results.json)")
    p.add_argument("--use-summary", action="store_true",
                   help="Use 'extracted_summary' column instead of 'content'")
    p.add_argument("--min-relevance",  type=float, default=0.30)
    p.add_argument("--min-confidence", type=float, default=0.35)
    p.add_argument("--cache-dir",
                   default=os.path.join("data", "interim", "kg_article_cache"),
                   help=(
                       "Disk cache dir for extracted triples. "
                       "Same default as KGGenNewsEmbedder so test reuses production cache. "
                       "Pass '' to disable caching."
                   ))
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 68)
    print("  KG Extraction Test — Pipeline-Faithful Mode")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode    : {'Batch API' if args.batch else 'Sequential'}")
    print(f"  Mirrors : KGGenNewsEmbedder._collect_day_triples_batch()")
    print("=" * 68)

    # ── 1. Import ─────────────────────────────────────────────────────────────
    print("\n🔌 Importing pipeline modules ...")
    try:
        pipeline = _import_pipeline()
    except ImportError as e:
        print(f"\n❌ {e}")
        return

    FinDKGLiteExtractor     = pipeline["FinDKGLiteExtractor"]
    GeminiBatchAPIExtractor = pipeline.get("GeminiBatchAPIExtractor")
    rescore_fn              = pipeline.get("rescore_triples_for_ticker",
                                           lambda t, p, tg, mr: t)  # no-op fallback
    TICKER_SECTOR_MAP       = pipeline.get("TICKER_SECTOR_MAP", {})

    # ── 2. Load data ──────────────────────────────────────────────────────────
    try:
        articles, cols = load_and_build_articles(
            args.data, n=args.n,
            ticker_filter=args.ticker,
            use_summary=args.use_summary,
        )
    except FileNotFoundError:
        print(f"\n❌ File not found: {args.data}")
        print("   Use --data /path/to/file.parquet")
        return
    except Exception as e:
        print(f"\n❌ Load error: {e}")
        return

    # ── 3. Dry run ────────────────────────────────────────────────────────────
    if args.dry_run:
        print(f"\n{'─'*68}")
        print("  DRY-RUN — data preview (no API calls)")
        print(f"{'─'*68}")
        for i, art in enumerate(articles, 1):
            tickers_str = ", ".join(art["tickers"])
            print(f"\n  [{i}] primary={art['primary_ticker']}  date={art['date']}")
            print(f"       all tickers: {tickers_str}")
            print(f"       title  : {art['title'][:80]}")
            print(f"       length : {len(art['text']):,} chars")
            print(f"       preview: {art['text'][:160].replace(chr(10),' ')} ...")
        print(f"\n✅ Dry-run done. Remove --dry-run to run extraction.")
        return

    # ── 4. Check API key ──────────────────────────────────────────────────────
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\n❌ GEMINI_API_KEY not set.")
        print("   export GEMINI_API_KEY='your_key'   # Linux/Mac")
        print("   set GEMINI_API_KEY=your_key         # Windows")
        return

    # ── 5. Init extractor ─────────────────────────────────────────────────────
    print(f"\n🤖 Initializing extractor ...")
    use_batch = True   # always True — either Async or GeminiBatch
    AsyncConcurrentExtractor = pipeline.get("AsyncConcurrentExtractor")
    GeminiBatchAPIExtractor  = pipeline.get("GeminiBatchAPIExtractor")

    try:
        if args.gemini_batch:
            # Production mode: 50% cost, async job, 30s+ latency
            if not GeminiBatchAPIExtractor:
                raise RuntimeError("GeminiBatchAPIExtractor not imported")
            extractor = GeminiBatchAPIExtractor(
                api_key=api_key,
                min_relevance=args.min_relevance,
                min_confidence=args.min_confidence,
            )
            print("   ✅ GeminiBatchAPIExtractor ready  (50% cost, async job)")
        else:
            # Default: concurrent async, standard cost, fast results
            if not AsyncConcurrentExtractor:
                raise RuntimeError("AsyncConcurrentExtractor not imported")
            extractor = AsyncConcurrentExtractor(
                api_key=api_key,
                min_relevance=args.min_relevance,
                min_confidence=args.min_confidence,
                max_concurrent=args.max_concurrent,
            )
            print(f"   ✅ AsyncConcurrentExtractor ready  "
                  f"(max_concurrent={args.max_concurrent})")
    except Exception as e:
        print(f"❌ Extractor init failed: {e}")
        return

    # ── 6. Per-article display ────────────────────────────────────────────────
    for i, art in enumerate(articles, 1):
        print_article_header(art, i, len(articles))

    # ── 7. Run pipeline-faithful extraction ───────────────────────────────────
    print(f"\n{'─'*68}")
    print(f"  Running pipeline-faithful extraction ...")
    print(f"  • SHA-1 dedup → extract once per unique text")
    print(f"  • rescore_triples_for_ticker() for each target ticker")
    print(f"  • filter: relevance≥{args.min_relevance}, confidence≥{args.min_confidence}")
    print(f"{'─'*68}")

    cache_dir = args.cache_dir.strip() if args.cache_dir else ""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"\n  💾 Cache dir: {os.path.abspath(cache_dir)}")
    else:
        print("\n  ⚠️  Cache disabled (--cache-dir '')")

    t0 = time.time()
    results_by_ticker, debug_info = collect_triples_pipeline_faithful(
        articles=articles,
        extractor=extractor,
        rescore_fn=rescore_fn,
        ticker_sector_map=TICKER_SECTOR_MAP,
        min_relevance=args.min_relevance,
        min_confidence=args.min_confidence,
        use_batch=use_batch,
        cache_dir=cache_dir,
    )
    elapsed_total = time.time() - t0

    # ── 8. Display results per ticker ─────────────────────────────────────────
    sha1_to_meta = debug_info["sha1_to_meta"]
    sha1_to_raw  = debug_info["sha1_to_raw"]

    print(f"\n{'═'*68}")
    print(f"  RESULTS BY TICKER  (final graph input)")
    print(f"{'═'*68}")

    # Show raw extraction first (per unique article)
    print(f"\n  ── Raw extraction (before rescore) ──")
    for h, meta in sha1_to_meta.items():
        raw = sha1_to_raw.get(h) or []
        elapsed = debug_info["timing"].get(h, 0.0)
        title   = meta["title"][:60] + ".." if len(meta["title"]) > 62 else meta["title"]
        print(f"\n  [{meta['primary_ticker']}] {meta['date']} — {title}")
        print(f"  {len(raw)} raw triples extracted  ({elapsed:.1f}s)")
        for i, t in enumerate(raw):
            print_triple(i, t)

    # Show final per-ticker results (after rescore + filter)
    print(f"\n\n  ── Final results per ticker (after rescore + filter) ──")
    for ticker in sorted(results_by_ticker.keys()):
        triples = results_by_ticker[ticker]
        # Determine if any article used this as a non-primary ticker
        is_rescored = any(
            ticker != meta["primary_ticker"] and ticker in meta["tickers"]
            for meta in sha1_to_meta.values()
        )
        print_ticker_results(ticker, triples, ticker, rescored=is_rescored)

    # ── 9. Summary ────────────────────────────────────────────────────────────
    print_summary(articles, results_by_ticker, debug_info, use_batch, elapsed_total)

    # ── 10. Save ──────────────────────────────────────────────────────────────
    if args.save:
        # Serialize: Dict[ticker → List[RichTriple]]
        save_data = {
            "meta": {
                "timestamp":      datetime.now().isoformat(),
                "mode":           "batch" if use_batch else "sequential",
                "n_articles":     len(articles),
                "unique_texts":   debug_info["unique_count"],
                "min_relevance":  args.min_relevance,
                "min_confidence": args.min_confidence,
            },
            "results_by_ticker": results_by_ticker,
            "articles": [
                {k: v for k, v in a.items() if k != "text"}  # omit full text
                for a in articles
            ],
        }
        try:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
            print(f"💾 Saved: {args.save}")
        except Exception as e:
            print(f"⚠️  Save failed: {e}")


if __name__ == "__main__":
    main()