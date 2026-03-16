#!/usr/bin/env python3
"""
test_extraction_5articles.py — Sample 5 Random Dates, Test Extraction Quality
==============================================================================
Chọn ngẫu nhiên 5 ngày từ parquet, lấy tất cả bài của ngày đó, chạy qua
pipeline extraction mới (chunk-based, title weight × 3, strict rescore tier2=0.75).

Mục đích:
  - Kiểm tra chất lượng prompt + rescore logic
  - Xác nhận chunk-based extraction không bị cắt ngang câu
  - Xác nhận _all_tickers được truyền đúng cho rescore
  - Xác nhận primary ticker detection (title weight × 3) hoạt động đúng

Usage:
    python test_extraction_5articles.py
    python test_extraction_5articles.py --n-days 5 --ticker TSLA
    python test_extraction_5articles.py --n-days 3 --date 2022-06
    python test_extraction_5articles.py --dry-run
    python test_extraction_5articles.py --max-concurrent 10
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import GlobalConfig
from data_pipeline.kg.extractor_batch import (
    AsyncConcurrentExtractor,
    GeminiBatchAPIExtractor,
    rescore_triples_for_ticker,
    build_combined_text,
    detect_primary_ticker,
    split_text_chunks,
    dedup_triples,
    _sha1,
    _norm,
    _parse_tickers,
    _filter_and_clamp,
    TICKER_NAME_MAP,
)

DEFAULT_PARQUET = os.path.join(
    GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet"
)


# ─────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(cache_dir: str, sha1: str) -> str:
    return os.path.join(cache_dir, f"{sha1}.json")


def _load_cache(cache_dir: str, sha1: str) -> Optional[List[Dict]]:
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
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    p = _cache_path(cache_dir, sha1)
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"triples": triples, "_v": "v3"}, f, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD AND NORMALIZE DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_news_df(parquet_path: str) -> pd.DataFrame:
    """Load và chuẩn hoá news DataFrame."""
    print(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"  {len(df):,} rows  |  columns: {list(df.columns)}")

    # Column renames
    if "headline" in df.columns and "title" not in df.columns:
        df = df.rename(columns={"headline": "title"})
    if "ticker" in df.columns and "equity" not in df.columns:
        df = df.rename(columns={"ticker": "equity"})
    if "content" not in df.columns:
        for alt in ("body", "text"):
            if alt in df.columns:
                df = df.rename(columns={alt: "content"})
                break

    if "content" not in df.columns:
        df["content"] = ""
    if "title" not in df.columns:
        df["title"] = ""

    # Date
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    # Ticker column
    ticker_col = next((c for c in ("symbols", "equity") if c in df.columns), None)
    if ticker_col is None:
        raise ValueError(f"No ticker column. Has: {list(df.columns)}")

    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)
    df = df[df["_all_tickers"].map(len) > 0].reset_index(drop=True)

    # Primary ticker (title weight × 3)
    df["primary_ticker"] = df.apply(
        lambda row: detect_primary_ticker(
            str(row.get("title",   "") or ""),
            str(row.get("content", "") or ""),
            row["_all_tickers"],
        ),
        axis=1,
    )

    return df


def sample_days(
    df: pd.DataFrame,
    n_days: int,
    ticker_filter: Optional[str] = None,
    date_prefix:   Optional[str] = None,
    seed: int = None,
) -> List[Tuple[str, str, pd.DataFrame]]:
    """
    Sample n_days ngẫu nhiên.
    Trả về list of (ticker, date_str, day_df).
    """
    work_df = df.copy()

    if ticker_filter:
        work_df = work_df[work_df["primary_ticker"] == ticker_filter.upper()]
        if len(work_df) == 0:
            # Fallback: tìm trong _all_tickers
            work_df = df[df["_all_tickers"].apply(lambda lst: ticker_filter.upper() in lst)]

    if date_prefix:
        work_df = work_df[work_df["date"].astype(str).str.startswith(date_prefix)]

    if len(work_df) == 0:
        print("No data matching filters.")
        return []

    # Group by (primary_ticker, date) để sample combinations
    combos = []
    for (ticker, date), group in work_df.groupby(["primary_ticker", "date"]):
        combos.append((str(ticker), str(date), group))

    if seed is not None:
        random.seed(seed)

    n_sample = min(n_days, len(combos))
    sampled  = random.sample(combos, n_sample)
    sampled.sort(key=lambda x: x[1])  # sort by date

    return sampled


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION PIPELINE (per day-ticker)
# ─────────────────────────────────────────────────────────────────────────────

def extract_day_ticker(
    ticker:     str,
    date_str:   str,
    day_df:     pd.DataFrame,
    extractor,
    cache_dir:  str,
    min_relevance:  float,
    min_confidence: float,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Extract cho 1 (ticker, date). Trả về (results, debug_info).

    results: {
        "ticker": str,
        "date": str,
        "n_articles": int,
        "n_chunks": int,
        "n_raw_triples": int,
        "triples_primary": List[Dict],     # triples cho primary ticker (trước rescore)
        "triples_by_ticker": Dict[str, List[Dict]],  # sau rescore cho tất cả tickers
    }
    """
    titles   = day_df["title"].fillna("").tolist()
    contents = day_df["content"].fillna("").tolist()
    all_t_sets = []
    for _, row in day_df.iterrows():
        all_t_sets.append(list(row.get("_all_tickers") or [ticker]))

    # Gộp tất cả tickers xuất hiện trong ngày này
    all_tickers_today = list({t for ts in all_t_sets for t in ts})

    combined = build_combined_text(titles, contents)
    combined = _norm(combined)

    if not combined:
        return {"ticker": ticker, "date": date_str, "n_articles": len(titles),
                "n_chunks": 0, "n_raw_triples": 0,
                "triples_primary": [], "triples_by_ticker": {}}, {}

    # Cache check (per article)
    sha1_to_meta: Dict[str, Dict] = {}
    sha1_to_raw:  Dict[str, Optional[List]] = {}
    cache_hits = 0

    for _, row in day_df.iterrows():
        title   = _norm(str(row.get("title",   "") or ""))
        content = _norm(str(row.get("content", "") or ""))
        full    = build_combined_text([title] if title else [],
                                       [content] if content else [])
        if not full:
            continue
        h = _sha1(full)
        if h in sha1_to_raw:
            continue

        primary = str(row.get("primary_ticker") or ticker)
        all_t   = list(row.get("_all_tickers") or [primary])

        sha1_to_meta[h] = {
            "full_text": full,
            "primary_ticker": primary,
            "all_tickers": all_t,
        }

        cached = _load_cache(cache_dir, h) if cache_dir else None
        if cached is not None:
            sha1_to_raw[h] = cached
            cache_hits += 1
        else:
            sha1_to_raw[h] = None

    # Extract uncached
    uncached = [h for h, v in sha1_to_raw.items() if v is None]

    if uncached:
        # Build chunk jobs
        chunk_jobs = []
        for h in uncached:
            meta   = sha1_to_meta[h]
            chunks = split_text_chunks(meta["full_text"])
            for chunk in chunks:
                chunk_jobs.append((h, {"text": chunk, "ticker": meta["primary_ticker"],
                                        "date": date_str}))

        articles_to_extract = [job[1] for job in chunk_jobs]
        print(f"  Extracting: {len(uncached)} articles → {len(articles_to_extract)} chunks")

        t0            = time.time()
        batch_results = extractor.extract_batch(articles_to_extract)
        elapsed       = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        # Group chunk results back
        sha1_chunks: Dict[str, List] = defaultdict(list)
        for (sha1, _), triples in zip(chunk_jobs, batch_results):
            sha1_chunks[sha1].extend(triples or [])

        for h in uncached:
            merged  = _filter_and_clamp(sha1_chunks[h], min_relevance, min_confidence)
            deduped = dedup_triples(merged)
            sha1_to_raw[h] = deduped
            _save_cache(cache_dir, h, deduped)
    else:
        print(f"  All {cache_hits} article(s) from cache")

    # Fan-out + rescore per ticker
    all_raw = []
    for h, raw in sha1_to_raw.items():
        all_raw.extend(raw or [])
    all_raw = dedup_triples(all_raw)

    # Per-ticker rescore
    triples_by_ticker: Dict[str, List] = {}
    for target in all_tickers_today:
        rescored = []
        for h, raw in sha1_to_raw.items():
            if not raw:
                continue
            meta = sha1_to_meta[h]
            rs = rescore_triples_for_ticker(
                raw,
                primary_ticker=meta["primary_ticker"],
                target_ticker=target,
                min_relevance=min_relevance,
                article_text=meta["full_text"],
                all_article_tickers=meta["all_tickers"],
            )
            rs = [t for t in rs if float(t.get("confidence", 0)) >= min_confidence]
            rescored.extend(rs)
        triples_by_ticker[target] = dedup_triples(rescored)

    n_chunks = sum(len(split_text_chunks(sha1_to_meta[h]["full_text"]))
                   for h in sha1_to_meta)

    return {
        "ticker":           ticker,
        "date":             date_str,
        "n_articles":       len(titles),
        "n_chunks":         n_chunks,
        "cache_hits":       cache_hits,
        "n_raw_triples":    len(all_raw),
        "triples_primary":  triples_by_ticker.get(ticker, []),
        "triples_by_ticker": triples_by_ticker,
    }, {"sha1_to_meta": sha1_to_meta, "sha1_to_raw": sha1_to_raw}


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def impact_icon(score: float) -> str:
    if score >=  0.6: return "[++]"
    if score >=  0.3: return "[+] "
    if score >= -0.3: return "[ ] "
    if score >= -0.6: return "[-] "
    return "[--]"


def bar(val: float, width: int = 10) -> str:
    filled = max(0, min(width, round(abs(val) * width)))
    return "█" * filled + "░" * (width - filled)


def print_triple(i: int, t: Dict):
    subj   = t.get("subject", {})
    obj    = t.get("object",  {})
    rel    = t.get("relation", "?")
    conf   = float(t.get("confidence", 0))
    rel_s  = float(t.get("relevance_to_ticker", 0))
    impact = float(t.get("price_impact_score", 0))
    reason = t.get("reasoning", "")

    GROUP_A = {"ANNOUNCES","RAISES","CUTS","INVESTS_IN","DIVESTS","APPOINTS"}
    GROUP_B = {"POS_IMPACTS","NEG_IMPACTS","COMPETES_WITH","REGULATES","SUPPLIES_TO"}
    grp = "A" if rel in GROUP_A else "B" if rel in GROUP_B else "C"

    print(f"\n  ── Triple #{i+1}  [Grp {grp}]")
    print(f"  [{subj.get('type','?'):8}] {subj.get('name','?')}")
    print(f"      {rel}")
    print(f"  [{obj.get('type','?'):8}] {obj.get('name','?')}")
    print(f"  conf={conf:.2f} {bar(conf)}  "
          f"rel={rel_s:.2f} {bar(rel_s)}  "
          f"impact={impact:+.2f} {impact_icon(impact)}")
    if reason:
        print(textwrap.fill(reason, 64,
                            initial_indent="  -> ", subsequent_indent="     "))


def print_day_results(result: Dict, show_cross_tickers: bool = True):
    ticker = result["ticker"]
    date   = result["date"]
    print(f"\n{'='*70}")
    print(f"  {ticker}  |  {date}  |  "
          f"{result['n_articles']} articles → {result['n_chunks']} chunks  |  "
          f"cache_hits={result['cache_hits']}")
    print(f"  Raw triples (before rescore): {result['n_raw_triples']}")

    # Primary ticker triples
    primary_triples = result["triples_primary"]
    print(f"\n  -- {ticker} (primary) — {len(primary_triples)} triples after rescore --")
    if not primary_triples:
        print("  (no triples)")
    for i, t in enumerate(primary_triples):
        print_triple(i, t)

    # Cross-ticker triples (brief)
    if show_cross_tickers:
        by_ticker = result["triples_by_ticker"]
        cross = {k: v for k, v in by_ticker.items()
                 if k != ticker and v}
        if cross:
            print(f"\n  -- Cross-ticker fan-out --")
            for t_name, triples in sorted(cross.items()):
                print(f"  {t_name}: {len(triples)} triples "
                      f"(avg rel={sum(float(t.get('relevance_to_ticker',0)) for t in triples)/len(triples):.2f})")


def print_summary(all_results: List[Dict], elapsed_total: float):
    print(f"\n{'━'*70}")
    print(f"  SUMMARY — V3 Pipeline (chunk-based, title weight×3, tier2=0.75)")
    print(f"{'━'*70}")
    print(f"  Total days tested : {len(all_results)}")
    print(f"  Total time        : {elapsed_total:.1f}s")
    print()

    total_chunks   = sum(r["n_chunks"] for r in all_results)
    total_raw      = sum(r["n_raw_triples"] for r in all_results)
    total_primary  = sum(len(r["triples_primary"]) for r in all_results)
    total_articles = sum(r["n_articles"] for r in all_results)
    cache_hits     = sum(r["cache_hits"] for r in all_results)

    print(f"  {'Day':24} {'Ticker':6} {'Arts':>4} {'Chnks':>5} "
          f"{'Raw':>4} {'Primary':>7} {'Avg imp':>8}")
    print(f"  {'-'*66}")
    for r in all_results:
        triples = r["triples_primary"]
        if triples:
            avg_imp = sum(float(t.get("price_impact_score", 0)) for t in triples) / len(triples)
            imp_str = f"{avg_imp:+.2f} {impact_icon(avg_imp)}"
        else:
            imp_str = "  n/a"
        print(f"  {r['date']:24} {r['ticker']:6} {r['n_articles']:>4} "
              f"{r['n_chunks']:>5} {r['n_raw_triples']:>4} "
              f"{len(r['triples_primary']):>7}  {imp_str}")

    print(f"  {'-'*66}")
    print(f"  {'TOTAL':24} {'':6} {total_articles:>4} {total_chunks:>5} "
          f"{total_raw:>4} {total_primary:>7}")
    print(f"\n  Cache hits: {cache_hits}/{total_articles} articles")

    # Relation distribution
    all_triples = [t for r in all_results for t in r["triples_primary"]]
    if all_triples:
        rc = Counter(t.get("relation") for t in all_triples)
        GROUP_A = {"ANNOUNCES","RAISES","CUTS","INVESTS_IN","DIVESTS","APPOINTS"}
        GROUP_B = {"POS_IMPACTS","NEG_IMPACTS","COMPETES_WITH","REGULATES","SUPPLIES_TO"}
        print(f"\n  Relation distribution ({len(all_triples)} primary triples):")
        for rel, cnt in sorted(rc.items(), key=lambda x: -x[1]):
            grp = "A" if rel in GROUP_A else "B" if rel in GROUP_B else "C"
            bar_str = "█" * min(cnt, 20)
            print(f"    [{grp}] {rel:<22} {bar_str} {cnt}")
        n  = len(all_triples)
        ga = sum(1 for t in all_triples if t.get("relation") in GROUP_A)
        gb = sum(1 for t in all_triples if t.get("relation") in GROUP_B)
        pct = 100 * (ga + gb) / n if n else 0
        qual = "Good (>=50%)" if pct >= 50 else "Low (<50%)"
        print(f"\n  Group A+B rate: {ga+gb}/{n} ({pct:.0f}%) — {qual}")

    print(f"{'━'*70}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Test extraction on random 5 dates (V3 pipeline)",
    )
    p.add_argument("--data",          default=DEFAULT_PARQUET)
    p.add_argument("--n-days",        type=int, default=5,
                   help="Number of random day-ticker combos to test")
    p.add_argument("--ticker",        default=None,
                   help="Filter to primary ticker")
    p.add_argument("--date",          default=None,
                   help="Filter to date prefix, e.g. '2022-06'")
    p.add_argument("--seed",          type=int, default=None,
                   help="Random seed for reproducible sampling")
    p.add_argument("--gemini-batch",  action="store_true")
    p.add_argument("--max-concurrent", type=int, default=5)
    p.add_argument("--min-relevance",  type=float, default=0.30)
    p.add_argument("--min-confidence", type=float, default=0.35)
    p.add_argument("--cache-dir",
                   default=os.path.join("data", "interim", "kg_article_cache"),
                   help="Disk cache dir (shared with production pipeline)")
    p.add_argument("--dry-run",       action="store_true",
                   help="Preview sampled days without calling API")
    p.add_argument("--no-cross-tickers", action="store_true",
                   help="Hide cross-ticker rescore results")
    p.add_argument("--save",          default=None,
                   help="Save results JSON (e.g. --save test_results.json)")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  KG Extraction Test — V3 Pipeline (5 random dates)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Chunk size: 3000 chars  |  Overlap: 200 chars")
    print(f"  Title weight: ×3  |  Tier-2 strict threshold: 0.75")
    print("=" * 70)

    # Load data
    if not os.path.exists(args.data):
        print(f"File not found: {args.data}")
        print("  Use --data /path/to/news.parquet")
        sys.exit(1)

    df = load_news_df(args.data)

    # Sample days
    sampled = sample_days(
        df,
        n_days=args.n_days,
        ticker_filter=args.ticker,
        date_prefix=args.date,
        seed=args.seed,
    )

    if not sampled:
        print("No days to process.")
        sys.exit(0)

    print(f"\nSampled {len(sampled)} day-ticker combination(s):")
    for ticker, date_str, day_df in sampled:
        n_articles = len(day_df)
        all_tickers = set(t for ts in day_df["_all_tickers"] for t in ts)
        print(f"  {date_str}  {ticker:6}  {n_articles} articles  "
              f"tickers: {sorted(all_tickers)}")

    if args.dry_run:
        print("\nDry-run complete. Remove --dry-run to run extraction.")
        return

    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\nGEMINI_API_KEY not set.")
        print("  export GEMINI_API_KEY='your_key'")
        sys.exit(1)

    # Init extractor
    if args.gemini_batch:
        extractor = GeminiBatchAPIExtractor(
            api_key=api_key,
            min_relevance=args.min_relevance,
            min_confidence=args.min_confidence,
        )
        print("\nExtractor: GeminiBatchAPIExtractor")
    else:
        extractor = AsyncConcurrentExtractor(
            api_key=api_key,
            min_relevance=args.min_relevance,
            min_confidence=args.min_confidence,
            max_concurrent=args.max_concurrent,
        )
        print(f"\nExtractor: AsyncConcurrentExtractor (max_concurrent={args.max_concurrent})")

    # Cache dir
    cache_dir = args.cache_dir.strip() if args.cache_dir else ""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Cache dir: {os.path.abspath(cache_dir)}")
    else:
        print("Cache disabled")

    # Extract each day
    all_results = []
    t0_total    = time.time()

    for ticker, date_str, day_df in sampled:
        print(f"\n{'─'*70}")
        print(f"Processing: {ticker}  {date_str}  ({len(day_df)} articles)")
        result, _ = extract_day_ticker(
            ticker=ticker,
            date_str=date_str,
            day_df=day_df,
            extractor=extractor,
            cache_dir=cache_dir,
            min_relevance=args.min_relevance,
            min_confidence=args.min_confidence,
        )
        print_day_results(result, show_cross_tickers=not args.no_cross_tickers)
        all_results.append(result)

    elapsed_total = time.time() - t0_total
    print_summary(all_results, elapsed_total)

    # Save
    if args.save:
        save_data = {
            "meta": {
                "timestamp":        datetime.now().isoformat(),
                "version":          "V3",
                "n_days":           len(all_results),
                "min_relevance":    args.min_relevance,
                "min_confidence":   args.min_confidence,
                "chunk_size":       3000,
                "chunk_overlap":    200,
                "title_weight":     3,
                "tier2_threshold":  0.75,
            },
            "results": [
                {k: v for k, v in r.items()
                 if k != "triples_by_ticker"}  # keep file size manageable
                for r in all_results
            ],
        }
        try:
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
            print(f"Saved: {args.save}")
        except Exception as e:
            print(f"Save failed: {e}")


if __name__ == "__main__":
    main()