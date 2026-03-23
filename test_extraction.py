#!/usr/bin/env python3
"""test_extraction.py — V5 Pipeline (ticker_aliases + no-chunk default)"""
from __future__ import annotations
import argparse, hashlib, json, os, random, re, sys, textwrap, time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from configs.config import GlobalConfig
from data_pipeline.kg.extractor_batch import (
    AsyncConcurrentExtractor, GeminiBatchAPIExtractor,
    build_combined_text, detect_primary_ticker, get_article_pieces,
    dedup_triples, apply_quality_filters, smart_dedup_triples, limit_signals_per_source,
    filter_triples_for_ticker, _sha1, _norm, _parse_tickers,
    _filter_and_clamp, TICKER_NAME_MAP, tag_triples_source,
)

DEFAULT_PARQUET = os.path.join(GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet")
DEFAULT_CACHE   = GlobalConfig.kg_cache_dir()

def _cache_path(cache_dir, sha1): return os.path.join(cache_dir, f"{sha1}.json")
def _load_cache(cache_dir, sha1):
    if not cache_dir: return None
    p = _cache_path(cache_dir, sha1)
    if not os.path.exists(p): return None
    try:
        with open(p, "r", encoding="utf-8") as f: return json.load(f).get("triples", [])
    except Exception: return None
def _save_cache(cache_dir, sha1, triples):
    if not cache_dir: return
    os.makedirs(cache_dir, exist_ok=True)
    with open(_cache_path(cache_dir, sha1), "w", encoding="utf-8") as f:
        json.dump({"triples": triples, "_v": "v5"}, f, ensure_ascii=False)

def load_and_normalize(parquet_path):
    df = pd.read_parquet(parquet_path)
    print(f"\n  Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"  Shape  : {df.shape}")
    DATE_CANDS = ["date","Date","DATE","published_at","publishedAt","publish_date","pub_date","created_at","createdAt","timestamp","time","news_date"]
    date_col = next((c for c in DATE_CANDS if c in df.columns), None)
    if date_col is None:
        date_col = next((c for c in df.columns if any(k in c.lower() for k in ("date","time","publish","creat"))), None)
    if date_col is None: print("  ERROR: Cannot find date column."); sys.exit(1)
    if date_col != "date": print(f"  Date column: '{date_col}' -> renamed to 'date'"); df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    for old, new in [("headline","title"),("body","content"),("text","content")]:
        if old in df.columns and new not in df.columns: df = df.rename(columns={old: new})
    if "title"   not in df.columns: df["title"]   = ""
    if "content" not in df.columns: df["content"] = ""
    TICK_CANDS = ["symbols","equity","ticker","symbol","stock","tickers"]
    ticker_col = next((c for c in TICK_CANDS if c in df.columns), None)
    if ticker_col is None: print("  ERROR: Cannot find ticker column."); sys.exit(1)
    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)
    df = df[df["_all_tickers"].map(len) > 0].reset_index(drop=True)
    df["primary_ticker"] = df.apply(
        lambda r: detect_primary_ticker(str(r.get("title","") or ""), str(r.get("content","") or ""), r["_all_tickers"]), axis=1)
    known = set(GlobalConfig.TICKERS)
    df = df[df["primary_ticker"].isin(known)].reset_index(drop=True)
    if len(df) == 0: print(f"  WARNING: No rows match known tickers {sorted(known)}."); sys.exit(1)
    print(f"  After normalize: {len(df):,} rows  |  {df['date'].nunique()} dates  |  {df['primary_ticker'].nunique()} tickers")
    return df

def sample_by_days(df, n, ticker_filter=None, date_start=None, date_end=None, seed=None):
    work = df.copy()
    if ticker_filter: work = work[work["primary_ticker"] == ticker_filter.upper()]
    if date_start:    work = work[work["date"].astype(str) >= date_start]
    if date_end:      work = work[work["date"].astype(str) <= date_end]
    if len(work) == 0: return []
    combos = [(str(t), str(d)) for (t, d) in work.groupby(["primary_ticker","date"]).groups]
    if seed is not None: random.seed(seed)
    selected = sorted(random.sample(combos, min(n, len(combos))), key=lambda x: x[1])
    return [(t, d, df[(df["primary_ticker"]==t) & (df["date"].astype(str)==d)].copy()) for t, d in selected]

def sample_by_tickers(df, tickers, seed=None):
    if seed is not None: random.seed(seed)
    result = []
    for ticker in tickers:
        sub = df[df["primary_ticker"] == ticker.upper()]
        if len(sub) == 0: print(f"  No articles for {ticker}"); continue
        date_counts = sub.groupby("date").size().sort_values(ascending=False)
        chosen = str(random.choice(date_counts.head(5).index.tolist()))
        result.append((ticker.upper(), chosen, sub[sub["date"].astype(str)==chosen].copy()))
    return result

def extract_one(ticker, date_str, day_df, extractor, cache_dir, min_relevance, min_confidence):
    """
    Extract triples for (ticker, date).

    V5: Uses get_article_pieces() which defaults to full-article (no chunking).
    Per-article: apply_quality_filters() handles within-article dedup + caps.
    Cross-article: smart_dedup_triples() (with ticker_aliases normalization)
    + limit_signals_per_source() (with PT/rating classification).
    """
    titles   = day_df["title"].fillna("").tolist()
    all_tickers_day = list({t for ts in day_df["_all_tickers"] for t in ts})
    sha1_to_meta, sha1_to_raw, cache_hits = {}, {}, 0
    for _, row in day_df.iterrows():
        title   = _norm(str(row.get("title","")   or ""))
        content = _norm(str(row.get("content","") or ""))
        combined = build_combined_text([title] if title else [], [content] if content else [])
        if not combined: continue
        h = _sha1(combined)
        if h in sha1_to_raw: continue
        sha1_to_meta[h] = {"full_text": combined, "primary": str(row.get("primary_ticker") or ticker),
                           "all_t": list(row.get("_all_tickers") or [ticker])}
        cached = _load_cache(cache_dir, h) if cache_dir else None
        if cached is not None: sha1_to_raw[h] = cached; cache_hits += 1
        else: sha1_to_raw[h] = None

    uncached = [h for h, v in sha1_to_raw.items() if v is None]
    n_pieces, elapsed = 0, 0.0
    if uncached:
        piece_jobs = []
        for h in uncached:
            pieces = get_article_pieces(sha1_to_meta[h]["full_text"])
            for piece in pieces:
                piece_jobs.append((h, {"text": piece, "ticker": sha1_to_meta[h]["primary"], "date": date_str}))
            n_pieces += len(pieces)

        if n_pieces == len(uncached):
            print(f"  Extracting: {len(uncached)} articles (full-text, no chunking)")
        else:
            print(f"  Extracting: {len(uncached)} articles -> {n_pieces} pieces (chunking enabled)")

        t0 = time.time()
        results = extractor.extract_batch([j[1] for j in piece_jobs])
        elapsed = time.time() - t0
        sha1_pieces = defaultdict(list)
        for (h, _), triples in zip(piece_jobs, results): sha1_pieces[h].extend(triples or [])
        for h in uncached:
            merged  = _filter_and_clamp(sha1_pieces[h], min_relevance, min_confidence)
            deduped = apply_quality_filters(merged)
            deduped = tag_triples_source(deduped, h)
            sha1_to_raw[h] = deduped
            _save_cache(cache_dir, h, deduped)
    else:
        print(f"  All {cache_hits} article(s) from cache")

    # ── Merge all articles for this (ticker, date) ────────────────────────
    all_raw = []
    for h, raw in sha1_to_raw.items():
        if not raw: continue
        meta = sha1_to_meta[h]
        filtered = filter_triples_for_ticker(raw, meta["primary"], ticker, min_relevance)
        filtered = [t for t in filtered if float(t.get("confidence",0)) >= min_confidence]
        all_raw.extend(filtered)

    # ── Cross-article dedup + caps ────────────────────────────────────────
    all_raw = smart_dedup_triples(all_raw)
    all_raw = limit_signals_per_source(all_raw)
    final   = dedup_triples(all_raw)

    cross = {}
    for target in all_tickers_day:
        if target == ticker: continue
        rs2 = []
        for h, raw in sha1_to_raw.items():
            if not raw: continue
            meta = sha1_to_meta[h]
            rs2.extend(filter_triples_for_ticker(raw, meta["primary"], target, min_relevance))
        cross[target] = dedup_triples([t for t in rs2 if float(t.get("confidence",0)) >= min_confidence])

    return {"ticker": ticker, "date": date_str, "n_articles": len(titles), "n_unique": len(sha1_to_meta),
            "cache_hits": cache_hits, "n_pieces": n_pieces if uncached else 0,
            "elapsed": elapsed, "triples": final, "cross": cross}

def _bar(v, w=10): return "█"*max(0,min(w,round(abs(v)*w))) + "░"*max(0,w-max(0,min(w,round(abs(v)*w))))
def _impact(v):
    if v >= .6: return "[++]"
    if v >= .3: return "[+] "
    if v >= -.3: return "[ ] "
    if v >= -.6: return "[-] "
    return "[--]"
def _grp(r):
    A = {"ANNOUNCES","RAISES","CUTS","INVESTS_IN","DIVESTS","APPOINTS"}
    B = {"POS_IMPACTS","NEG_IMPACTS","COMPETES_WITH","REGULATES","SUPPLIES_TO"}
    return "A" if r in A else "B" if r in B else "C"

def print_triple(i, t):
    s, o = t.get("subject",{}), t.get("object",{})
    rel = t.get("relation","?")
    cf, rv, imp = float(t.get("confidence",0)), float(t.get("relevance_to_ticker",0)), float(t.get("price_impact_score",0))
    rsn = t.get("reasoning","")
    src = t.get("_src", "?")
    print(f"\n  #{i+1} [{_grp(rel)}] [{src}]  {s.get('name','?')} —{rel}→ {o.get('name','?')}")
    print(f"       conf={cf:.2f} {_bar(cf)}  rel={rv:.2f} {_bar(rv)}  impact={imp:+.2f} {_impact(imp)}")
    if rsn: print(textwrap.fill(rsn, 64, initial_indent="       -> ", subsequent_indent="          "))

def print_result(r, show_cross=True):
    t, d = r["ticker"], r["date"]
    n_pieces = r.get("n_pieces", r.get("n_chunks", 0))
    print(f"\n{'='*66}\n  {t}  |  {d}")
    print(f"  {r['n_articles']} articles  ->  {r['n_unique']} unique  ->  {n_pieces} pieces  |  cache_hits={r['cache_hits']}  |  {r['elapsed']:.1f}s")
    print(f"  Primary triples after filter: {len(r['triples'])}")
    if not r["triples"]: print("  (no triples)")
    for i, tri in enumerate(r["triples"]): print_triple(i, tri)
    if show_cross:
        ce = {k:v for k,v in r["cross"].items() if v}
        if ce:
            print(f"\n  Cross-ticker (mention-only):")
            for ct, ct_triples in sorted(ce.items()):
                avg_rel = sum(float(x.get("relevance_to_ticker",0)) for x in ct_triples) / len(ct_triples)
                print(f"    {ct}: {len(ct_triples)} triples  avg_rel={avg_rel:.2f}")

def print_summary(results, total_elapsed):
    chunking = "chunking" if getattr(GlobalConfig, 'KG_ENABLE_CHUNKING', False) else "full-article"
    print(f"\n{chr(9473)*66}")
    print(f"  SUMMARY  —  V5 pipeline ({chunking} + ticker_aliases dedup)")
    print(f"{chr(9473)*66}")
    print(f"  Days tested : {len(results)}  Total time  : {total_elapsed:.1f}s\n")
    print(f"  {'Ticker':6}  {'Date':12}  {'Arts':>4}  {'Pieces':>6}  {'Triples':>7}  {'Avg impact':>11}")
    print(f"  {'-'*62}")
    for r in results:
        ts = r["triples"]
        n_pieces = r.get("n_pieces", r.get("n_chunks", 0))
        imp_s = f"{sum(float(t.get('price_impact_score',0)) for t in ts)/len(ts):+.2f} {_impact(sum(float(t.get('price_impact_score',0)) for t in ts)/len(ts))}" if ts else "  n/a"
        print(f"  {r['ticker']:6}  {r['date']:12}  {r['n_articles']:>4}  {n_pieces:>6}  {len(ts):>7}  {imp_s}")
    all_triples = [t for r in results for t in r["triples"]]
    if all_triples:
        from collections import Counter
        A = {"ANNOUNCES","RAISES","CUTS","INVESTS_IN","DIVESTS","APPOINTS"}
        B = {"POS_IMPACTS","NEG_IMPACTS","COMPETES_WITH","REGULATES","SUPPLIES_TO"}
        rc = Counter(t.get("relation") for t in all_triples)
        print(f"\n  Relations ({len(all_triples)} primary triples):")
        for rel, cnt in sorted(rc.items(), key=lambda x: -x[1]):
            print(f"    [{_grp(rel)}] {rel:<22} {'█'*min(cnt,20)} {cnt}")
        ga = sum(1 for t in all_triples if t.get("relation") in A)
        gb = sum(1 for t in all_triples if t.get("relation") in B)
        n  = len(all_triples)
        pct = 100*(ga+gb)/n if n else 0
        print(f"\n  Group A+B: {ga+gb}/{n} ({pct:.0f}%)  {'Good (>=50%)' if pct >= 50 else 'Low (<50%)'}")
    print(f"{chr(9473)*66}\n")

def parse_args():
    p = argparse.ArgumentParser(description="Test extraction V5")
    p.add_argument("--days",     type=int,   default=None)
    p.add_argument("--tickers",  nargs="+",  default=None)
    p.add_argument("--filter",   default=None)
    p.add_argument("--date-range", nargs=2, metavar=("FROM","TO"), default=None)
    p.add_argument("--seed",     type=int,   default=None)
    p.add_argument("--data",     default=DEFAULT_PARQUET)
    p.add_argument("--cache-dir",default=DEFAULT_CACHE)
    p.add_argument("--min-relevance",  type=float, default=GlobalConfig.KG_MIN_RELEVANCE)
    p.add_argument("--min-confidence", type=float, default=GlobalConfig.KG_MIN_CONFIDENCE)
    p.add_argument("--max-concurrent", type=int,   default=GlobalConfig.KG_MAX_CONCURRENT)
    p.add_argument("--gemini-batch",   action="store_true")
    p.add_argument("--dry-run",        action="store_true")
    p.add_argument("--no-cross",       action="store_true")
    p.add_argument("--save",   default=None)
    return p.parse_args()

def main():
    args = parse_args()
    if args.days is None and args.tickers is None:
        print("Need --days N or --tickers T1 T2 ..."); sys.exit(1)

    chunking = "chunking" if getattr(GlobalConfig, 'KG_ENABLE_CHUNKING', False) else "full-article"
    print("="*66)
    print(f"  Test Extraction — V5 Pipeline ({chunking})")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Thresholds: min_relevance={args.min_relevance}  min_confidence={args.min_confidence}")
    print("="*66)
    if not os.path.exists(args.data): print(f"File not found: {args.data}"); sys.exit(1)
    print(f"\nLoading: {args.data}")
    df = load_and_normalize(args.data)
    ds, de = (args.date_range or [None, None])
    samples = sample_by_tickers(df, args.tickers, args.seed) if args.tickers else sample_by_days(df, args.days, args.filter, ds, de, args.seed)
    if not samples: print("No data matching filter."); sys.exit(0)
    print(f"\nSampled {len(samples)} combination(s):")
    for ticker, date_str, day_df in samples:
        all_t = sorted({t for ts in day_df["_all_tickers"] for t in ts})
        print(f"  {ticker:6}  {date_str}  {len(day_df)} articles  all_tickers={all_t}")
    if args.dry_run: print("\nDry-run."); return
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: print("\nGEMINI_API_KEY not set."); sys.exit(1)
    if args.gemini_batch:
        extractor = GeminiBatchAPIExtractor(api_key=api_key, min_relevance=args.min_relevance, min_confidence=args.min_confidence)
        print("\nExtractor: GeminiBatchAPIExtractor")
    else:
        extractor = AsyncConcurrentExtractor(api_key=api_key, min_relevance=args.min_relevance, min_confidence=args.min_confidence, max_concurrent=args.max_concurrent)
        print(f"\nExtractor: AsyncConcurrentExtractor (max_concurrent={args.max_concurrent})")
    cache_dir = args.cache_dir.strip() if args.cache_dir else ""
    if cache_dir: os.makedirs(cache_dir, exist_ok=True); print(f"Cache    : {os.path.abspath(cache_dir)}")
    results, t0_total = [], time.time()
    for ticker, date_str, day_df in samples:
        print(f"\n{'-'*66}\n  {ticker}  {date_str}  ({len(day_df)} articles)")
        result = extract_one(ticker, date_str, day_df, extractor, cache_dir, args.min_relevance, args.min_confidence)
        print_result(result, show_cross=not args.no_cross)
        results.append(result)
    print_summary(results, time.time() - t0_total)
    if args.save:
        try:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump({"run_info": {"timestamp": datetime.now().isoformat(),
                           "min_relevance": args.min_relevance, "min_confidence": args.min_confidence},
                           "results": [{k:v for k,v in r.items() if k != "cross"} for r in results]},
                          f, ensure_ascii=False, indent=2, default=str)
            print(f"Saved: {args.save}")
        except Exception as e: print(f"Save failed: {e}")

if __name__ == "__main__": main()