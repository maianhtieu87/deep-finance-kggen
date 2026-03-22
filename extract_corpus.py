# extract_corpus.py — V3.6
"""
Stage A: LLM Extraction.

V3.6 changes vs V3.4:
  - Cross-article dedup + caps applied on merged triple list after
    all articles for (ticker, date) are collected:
      all_triples = smart_dedup_triples(all_triples)   # Fix A, C
      all_triples = limit_signals_per_source(all_triples)  # Fix B
    Catches same-event duplicates that come from N different articles
    reporting the same stock drop or guidance cut on the same day.
  - _save_cache version bumped to v3.6.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from configs.config import GlobalConfig
from data_pipeline.kg.extractor_batch import (
    AsyncConcurrentExtractor, GeminiBatchAPIExtractor,
    detect_primary_ticker, build_combined_text, split_text_chunks,
    dedup_triples, apply_quality_filters, smart_dedup_triples, limit_signals_per_source,
    filter_triples_for_ticker, _sha1, _norm, _filter_and_clamp, _parse_tickers,
    tag_triples_source,
)

def _cache_path(cache_dir, sha1): return os.path.join(cache_dir, f"{sha1}.json")
def _load_cache(cache_dir, sha1):
    p = _cache_path(cache_dir, sha1)
    if not os.path.exists(p): return None
    try:
        with open(p, "r", encoding="utf-8") as f: return json.load(f).get("triples", [])
    except Exception: return None
def _save_cache(cache_dir, sha1, triples, meta=None):
    os.makedirs(cache_dir, exist_ok=True)
    payload = {"triples": triples, "_v": "v3.6"}
    if meta: payload["_meta"] = meta
    with open(_cache_path(cache_dir, sha1), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

def normalize_news_df(df):
    df = df.copy()
    col_map = {}
    if "headline" in df.columns and "title" not in df.columns: col_map["headline"] = "title"
    if "ticker" in df.columns and "equity" not in df.columns: col_map["ticker"] = "equity"
    if col_map: df = df.rename(columns=col_map)
    if "content" not in df.columns:
        for alt in ("body", "text"):
            if alt in df.columns: df = df.rename(columns={alt: "content"}); break
    if "content" not in df.columns: df["content"] = ""
    if "title"   not in df.columns: df["title"] = ""
    if "date" not in df.columns: raise ValueError(f"Missing date column. Has: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    ticker_col = next((c for c in ("symbols", "equity") if c in df.columns), None)
    if ticker_col is None: raise ValueError("No ticker column found.")
    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)
    df = df[df["_all_tickers"].map(len) > 0]
    df["primary_ticker"] = df.apply(
        lambda row: detect_primary_ticker(str(row.get("title","") or ""), str(row.get("content","") or ""), row["_all_tickers"]), axis=1)
    df = df.explode("_all_tickers").rename(columns={"_all_tickers": "equity"})
    df = df[df["equity"].notna() & (df["equity"] != "")].reset_index(drop=True)
    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)
    return df

def extract_and_cache_per_article(day_df, ticker, date_str, extractor, cache_dir, min_relevance, min_confidence):
    """
    Extract triples for (ticker, date).

    Per-article: apply_quality_filters() handles within-article dedup + caps.
    Cross-article merge: smart_dedup_triples() + limit_signals_per_source()
    applied on the merged list to catch cross-article duplicates
    (e.g. WMT -9.3% and WMT -9.2% from 2 different articles covering same drop).
    """
    sha1_to_meta, sha1_to_raw, cache_hits = {}, {}, 0
    for _, row in day_df.iterrows():
        title   = _norm(str(row.get("title","")   or ""))
        content = _norm(str(row.get("content","") or ""))
        full_text = build_combined_text([title] if title else [], [content] if content else [])
        if not full_text: continue
        h = _sha1(full_text)
        if h in sha1_to_raw:
            existing = set(sha1_to_meta[h]["all_tickers"])
            sha1_to_meta[h]["all_tickers"] = list(existing | set(row.get("_all_tickers") or [ticker]))
            continue
        sha1_to_meta[h] = {"full_text": full_text, "primary_ticker": str(row.get("primary_ticker") or ticker),
                           "date": date_str, "all_tickers": list(row.get("_all_tickers") or [ticker])}
        cached = _load_cache(cache_dir, h)
        if cached is not None: sha1_to_raw[h] = cached; cache_hits += 1
        else: sha1_to_raw[h] = None

    uncached = [h for h, v in sha1_to_raw.items() if v is None]
    if uncached:
        print(f"  [{ticker} {date_str}] cache_hits={cache_hits} to_extract={len(uncached)}")
        chunk_jobs = []
        for h in uncached:
            for chunk in split_text_chunks(sha1_to_meta[h]["full_text"]):
                chunk_jobs.append((h, {"text": chunk, "ticker": sha1_to_meta[h]["primary_ticker"], "date": sha1_to_meta[h]["date"]}))
        results = extractor.extract_batch([j[1] for j in chunk_jobs])
        sha1_chunks = defaultdict(list)
        for (sha1, _), triples in zip(chunk_jobs, results): sha1_chunks[sha1].extend(triples or [])
        for h in uncached:
            merged  = _filter_and_clamp(sha1_chunks[h], min_relevance, min_confidence)
            deduped = apply_quality_filters(merged)   # per-article filters
            deduped = tag_triples_source(deduped, h)  # traceability
            sha1_to_raw[h] = deduped
            _save_cache(cache_dir, h, deduped, meta=sha1_to_meta[h])
    else:
        if sha1_to_meta: print(f"  [{ticker} {date_str}] all {cache_hits} article(s) from cache")

    # ── Merge all articles for this (ticker, date) ────────────────────────
    all_triples = []
    for h, raw in sha1_to_raw.items():
        if not raw: continue
        meta = sha1_to_meta[h]
        filtered = filter_triples_for_ticker(raw, meta["primary_ticker"], ticker, min_relevance)
        filtered = [t for t in filtered if float(t.get("confidence",0)) >= min_confidence]
        all_triples.extend(filtered)

    # ── Cross-article dedup + caps (V3.6) ─────────────────────────────────
    # smart_dedup catches same-event duplicates across articles using
    # _normalize_object_for_dedup (Fix A: stock %, Fix C: guidance strings).
    # limit_signals_per_source re-applies caps on the merged pool
    # (Fix B: analyst COMP price target cap=1 per firm across all articles).
    all_triples = smart_dedup_triples(all_triples)
    all_triples = limit_signals_per_source(all_triples)
    return dedup_triples(all_triples)

def run_stage_a(news_df, cache_dir, use_gemini_batch=False, max_concurrent=None, min_relevance=None, min_confidence=None, ticker_filter=None, date_prefix=None):
    os.makedirs(cache_dir, exist_ok=True)
    _mr   = min_relevance  if min_relevance  is not None else GlobalConfig.KG_MIN_RELEVANCE
    _mc   = min_confidence if min_confidence is not None else GlobalConfig.KG_MIN_CONFIDENCE
    _conc = max_concurrent if max_concurrent is not None else GlobalConfig.KG_MAX_CONCURRENT
    df = normalize_news_df(news_df)
    if ticker_filter: df = df[df["equity"] == ticker_filter.upper()]
    if date_prefix:   df = df[df["date"].astype(str).str.startswith(date_prefix)]
    if len(df) == 0: print("No data after filter."); return {}
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise RuntimeError("GEMINI_API_KEY not set.")
    if use_gemini_batch:
        extractor = GeminiBatchAPIExtractor(api_key=api_key, min_relevance=_mr, min_confidence=_mc)
        print("Extractor: GeminiBatchAPIExtractor")
    else:
        extractor = AsyncConcurrentExtractor(api_key=api_key, min_relevance=_mr, min_confidence=_mc, max_concurrent=_conc)
        print(f"Extractor: AsyncConcurrentExtractor (max_concurrent={_conc})")
    print(f"Thresholds: min_relevance={_mr}  min_confidence={_mc}")
    tickers = sorted(df["equity"].unique())
    print(f"\nStage A: {len(tickers)} tickers x {df['date'].nunique()} dates\nCache: {cache_dir}\n")
    summary = {}
    for ticker in tickers:
        df_t = df[df["equity"] == ticker].copy(); summary[ticker] = {}
        for d in sorted(df_t["date"].unique()):
            date_str = str(d)
            triples  = extract_and_cache_per_article(df_t[df_t["date"]==d], ticker, date_str, extractor, cache_dir, _mr, _mc)
            summary[ticker][date_str] = len(triples)
    print("\nStage A complete.")
    return summary

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", action="store_true")
    p.add_argument("--max-concurrent", type=int,   default=None)
    p.add_argument("--min-relevance",  type=float, default=None)
    p.add_argument("--min-confidence", type=float, default=None)
    p.add_argument("--ticker", default=None)
    p.add_argument("--date",   default=None)
    p.add_argument("--news",   default=None)
    args = p.parse_args()
    news_path = args.news or os.path.join(GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet")
    if not os.path.exists(news_path): print(f"Not found: {news_path}"); sys.exit(1)
    df = pd.read_parquet(news_path)
    print(f"Loaded {len(df):,} rows")
    run_stage_a(news_df=df, cache_dir=GlobalConfig.kg_cache_dir(),
                use_gemini_batch=args.batch, max_concurrent=args.max_concurrent,
                min_relevance=args.min_relevance, min_confidence=args.min_confidence,
                ticker_filter=args.ticker, date_prefix=args.date)

if __name__ == "__main__": main()