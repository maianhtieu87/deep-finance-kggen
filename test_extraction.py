#!/usr/bin/env python3
"""
test_extraction.py — Kiểm tra extraction pipeline V4
=====================================================
Test nhanh bằng cách chọn ngẫu nhiên N ngày hoặc N mã cổ phiếu,
chạy qua toàn bộ luồng: chunk → extract → rescore → hiển thị kết quả.

Không cần chạy extract_corpus.py trước. File này tự xử lý cache.

Usage:
    # 3 ngày ngẫu nhiên (tất cả tickers)
    python test_extraction.py --days 3

    # 3 mã cụ thể, lấy 1 ngày mỗi mã
    python test_extraction.py --tickers TSLA AAPL MSFT

    # 5 ngày chỉ cho TSLA
    python test_extraction.py --days 5 --filter TSLA

    # Xem trước không gọi API
    python test_extraction.py --days 3 --dry-run

    # Dùng ngày cụ thể
    python test_extraction.py --date-range 2023-01-01 2023-03-31 --days 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import textwrap
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from configs.config import GlobalConfig
from data_pipeline.kg.extractor_batch import (
    AsyncConcurrentExtractor,
    GeminiBatchAPIExtractor,
    build_combined_text,
    detect_primary_ticker,
    split_text_chunks,
    dedup_triples,
    rescore_triples_for_ticker,
    _sha1,
    _norm,
    _parse_tickers,
    _filter_and_clamp,
    TICKER_NAME_MAP,
)

DEFAULT_PARQUET = os.path.join(
    GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet"
)
DEFAULT_CACHE = GlobalConfig.kg_cache_dir()


# ─────────────────────────────────────────────────────────────────────────────
# CACHE
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
    with open(_cache_path(cache_dir, sha1), "w", encoding="utf-8") as f:
        json.dump({"triples": triples, "_v": "v4"}, f, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_and_normalize(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)

    # ── Print columns để debug (hiện 1 lần) ──────────────────────────────────
    print(f"\n  Columns ({len(df.columns)}): {list(df.columns)}")
    print(f"  Shape  : {df.shape}")

    # ── Normalize: DATE column ────────────────────────────────────────────────
    DATE_CANDIDATES = [
        "date", "Date", "DATE",
        "published_at", "publishedAt", "publish_date", "pub_date",
        "created_at", "createdAt", "timestamp", "time",
        "news_date", "article_date", "trading_date",
    ]
    date_col = next((c for c in DATE_CANDIDATES if c in df.columns), None)
    if date_col is None:
        # Last resort: find any column whose name contains "date" or "time"
        date_col = next(
            (c for c in df.columns
             if any(k in c.lower() for k in ("date", "time", "publish", "creat"))),
            None,
        )
    if date_col is None:
        print(f"\n  ERROR: Cannot find a date column.")
        print(f"  Available columns: {list(df.columns)}")
        print(f"  Add --date-col COLNAME to specify manually, or rename the column to 'date'.")
        sys.exit(1)

    if date_col != "date":
        print(f"  Date column: '{date_col}' → renamed to 'date'")
        df = df.rename(columns={date_col: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    # ── Normalize: TITLE column ───────────────────────────────────────────────
    TITLE_CANDIDATES = ["title", "Title", "headline", "Headline", "subject",
                        "news_title", "article_title", "header"]
    title_col = next((c for c in TITLE_CANDIDATES if c in df.columns), None)
    if title_col and title_col != "title":
        df = df.rename(columns={title_col: "title"})
    if "title" not in df.columns:
        df["title"] = ""

    # ── Normalize: CONTENT column ─────────────────────────────────────────────
    CONTENT_CANDIDATES = ["content", "Content", "body", "Body", "text", "Text",
                          "article_body", "news_body", "full_text", "description",
                          "summary", "article_text"]
    content_col = next((c for c in CONTENT_CANDIDATES if c in df.columns), None)
    if content_col and content_col != "content":
        print(f"  Content column: '{content_col}' → renamed to 'content'")
        df = df.rename(columns={content_col: "content"})
    if "content" not in df.columns:
        df["content"] = ""

    # ── Normalize: TICKER column ──────────────────────────────────────────────
    TICKER_CANDIDATES = ["symbols", "equity", "ticker", "Ticker", "TICKER",
                         "symbol", "stock", "tickers", "equities",
                         "company_ticker", "stock_ticker"]
    ticker_col_name = next((c for c in TICKER_CANDIDATES if c in df.columns), None)
    if ticker_col_name is None:
        print(f"\n  ERROR: Cannot find a ticker column.")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    # Unify to "symbols" so _parse_tickers picks it up
    if ticker_col_name != "symbols":
        print(f"  Ticker column: '{ticker_col_name}' → used as-is")

    df["_all_tickers"] = df[ticker_col_name].apply(_parse_tickers)
    df = df[df["_all_tickers"].map(len) > 0].reset_index(drop=True)

    # ── Detect primary_ticker ─────────────────────────────────────────────────
    df["primary_ticker"] = df.apply(
        lambda r: detect_primary_ticker(
            str(r.get("title", "") or ""),
            str(r.get("content", "") or ""),
            r["_all_tickers"],
        ),
        axis=1,
    )

    # ── Filter to known tickers only ──────────────────────────────────────────
    known = set(GlobalConfig.TICKERS)
    df = df[df["primary_ticker"].isin(known)].reset_index(drop=True)
    if len(df) == 0:
        print(f"\n  WARNING: No rows match known tickers {sorted(known)}.")
        print(f"  Sample primary_ticker values from raw data:")
        raw_df = pd.read_parquet(parquet_path)
        raw_df["_all"] = raw_df[ticker_col_name].apply(_parse_tickers)
        sample_tickers = sorted({t for ts in raw_df["_all"].head(200) for t in ts})[:20]
        print(f"  {sample_tickers}")
        print("  Edit GlobalConfig.TICKERS in configs/config.py to match.")
        sys.exit(1)

    print(f"  After normalize: {len(df):,} rows  |  "
          f"{df['date'].nunique()} dates  |  "
          f"{df['primary_ticker'].nunique()} tickers")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLING
# ─────────────────────────────────────────────────────────────────────────────

def sample_by_days(
    df: pd.DataFrame,
    n: int,
    ticker_filter: Optional[str] = None,
    date_start: Optional[str] = None,
    date_end: Optional[str] = None,
    seed: Optional[int] = None,
) -> List[Tuple[str, str, pd.DataFrame]]:
    """Sample N (primary_ticker, date) combos randomly."""
    work = df.copy()
    if ticker_filter:
        work = work[work["primary_ticker"] == ticker_filter.upper()]
    if date_start:
        work = work[work["date"].astype(str) >= date_start]
    if date_end:
        work = work[work["date"].astype(str) <= date_end]
    if len(work) == 0:
        return []

    combos = [(str(t), str(d)) for (t, d) in work.groupby(["primary_ticker", "date"]).groups]
    if seed is not None:
        random.seed(seed)
    selected = random.sample(combos, min(n, len(combos)))
    selected.sort(key=lambda x: x[1])

    result = []
    for ticker, date_str in selected:
        mask = (df["primary_ticker"] == ticker) & (df["date"].astype(str) == date_str)
        result.append((ticker, date_str, df[mask].copy()))
    return result


def sample_by_tickers(
    df: pd.DataFrame,
    tickers: List[str],
    seed: Optional[int] = None,
) -> List[Tuple[str, str, pd.DataFrame]]:
    """For each ticker, pick 1 random date with most articles."""
    if seed is not None:
        random.seed(seed)
    result = []
    for ticker in tickers:
        mask = df["primary_ticker"] == ticker.upper()
        sub  = df[mask]
        if len(sub) == 0:
            print(f"  No articles for {ticker}")
            continue
        # Pick date with most articles for that ticker
        date_counts = sub.groupby("date").size().sort_values(ascending=False)
        top_dates   = date_counts.head(5).index.tolist()
        chosen_date = str(random.choice(top_dates))
        day_df      = sub[sub["date"].astype(str) == chosen_date].copy()
        result.append((ticker.upper(), chosen_date, day_df))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION (per day-ticker, chunk-based)
# ─────────────────────────────────────────────────────────────────────────────

def extract_one(
    ticker: str,
    date_str: str,
    day_df: pd.DataFrame,
    extractor,
    cache_dir: str,
    min_relevance: float,
    min_confidence: float,
) -> Dict[str, Any]:
    """
    Chạy extraction cho 1 (ticker, date).
    Trả về dict kết quả để hiển thị.
    """
    titles   = day_df["title"].fillna("").tolist()
    contents = day_df["content"].fillna("").tolist()

    # Gom tất cả tickers xuất hiện trong ngày
    all_tickers_day = list({t for ts in day_df["_all_tickers"] for t in ts})

    # Per-article cache
    sha1_to_meta: Dict[str, Dict] = {}
    sha1_to_raw:  Dict[str, Optional[List]] = {}
    cache_hits = 0

    for _, row in day_df.iterrows():
        title   = _norm(str(row.get("title",   "") or ""))
        content = _norm(str(row.get("content", "") or ""))
        combined = build_combined_text(
            [title]   if title   else [],
            [content] if content else [],
        )
        if not combined:
            continue
        h = _sha1(combined)
        if h in sha1_to_raw:
            continue

        primary = str(row.get("primary_ticker") or ticker)
        all_t   = list(row.get("_all_tickers") or [primary])

        sha1_to_meta[h] = {
            "full_text": combined,
            "primary":   primary,
            "all_t":     all_t,
        }
        cached = _load_cache(cache_dir, h) if cache_dir else None
        if cached is not None:
            sha1_to_raw[h] = cached
            cache_hits += 1
        else:
            sha1_to_raw[h] = None

    # Extract uncached
    uncached = [h for h, v in sha1_to_raw.items() if v is None]
    n_chunks = 0
    elapsed  = 0.0

    if uncached:
        chunk_jobs = []
        for h in uncached:
            chunks = split_text_chunks(sha1_to_meta[h]["full_text"])
            for chunk in chunks:
                chunk_jobs.append((h, {
                    "text":   chunk,
                    "ticker": sha1_to_meta[h]["primary"],
                    "date":   date_str,
                }))
            n_chunks += len(chunks)

        print(f"  Extracting: {len(uncached)} articles → {n_chunks} chunks")
        t0            = time.time()
        results       = extractor.extract_batch([j[1] for j in chunk_jobs])
        elapsed       = time.time() - t0

        # Group by sha1
        sha1_chunk_triples = defaultdict(list)
        for (h, _), triples in zip(chunk_jobs, results):
            sha1_chunk_triples[h].extend(triples or [])

        for h in uncached:
            merged  = _filter_and_clamp(sha1_chunk_triples[h], min_relevance, min_confidence)
            deduped = dedup_triples(merged)
            sha1_to_raw[h] = deduped
            _save_cache(cache_dir, h, deduped)
    else:
        print(f"  All {cache_hits} article(s) from cache")

    # Rescore cho target ticker
    all_raw: List[Dict] = []
    for h, raw in sha1_to_raw.items():
        if not raw:
            continue
        meta = sha1_to_meta[h]
        rs   = rescore_triples_for_ticker(
            raw,
            primary_ticker=meta["primary"],
            target_ticker=ticker,
            min_relevance=min_relevance,
            article_text=meta["full_text"],
            all_article_tickers=meta["all_t"],
        )
        rs = [t for t in rs if float(t.get("confidence", 0)) >= min_confidence]
        all_raw.extend(rs)

    final = dedup_triples(all_raw)

    # Also compute cross-ticker results
    cross: Dict[str, List] = {}
    for target in all_tickers_day:
        if target == ticker:
            continue
        rs2 = []
        for h, raw in sha1_to_raw.items():
            if not raw:
                continue
            meta = sha1_to_meta[h]
            rs2.extend(rescore_triples_for_ticker(
                raw,
                primary_ticker=meta["primary"],
                target_ticker=target,
                min_relevance=min_relevance,
                article_text=meta["full_text"],
                all_article_tickers=meta["all_t"],
            ))
        cross[target] = dedup_triples(
            [t for t in rs2 if float(t.get("confidence", 0)) >= min_confidence]
        )

    return {
        "ticker":      ticker,
        "date":        date_str,
        "n_articles":  len(titles),
        "n_unique":    len(sha1_to_meta),
        "cache_hits":  cache_hits,
        "n_chunks":    n_chunks if uncached else 0,
        "elapsed":     elapsed,
        "triples":     final,
        "cross":       cross,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

def _bar(v: float, w: int = 10) -> str:
    return "█" * max(0, min(w, round(abs(v) * w))) + "░" * max(0, w - max(0, min(w, round(abs(v) * w))))

def _impact(v: float) -> str:
    if v >= .6: return "[++]"
    if v >= .3: return "[+] "
    if v >= -.3: return "[ ] "
    if v >= -.6: return "[-] "
    return "[--]"

def _grp(rel: str) -> str:
    A = {"ANNOUNCES","RAISES","CUTS","INVESTS_IN","DIVESTS","APPOINTS"}
    B = {"POS_IMPACTS","NEG_IMPACTS","COMPETES_WITH","REGULATES","SUPPLIES_TO"}
    return "A" if rel in A else "B" if rel in B else "C"

def print_triple(i: int, t: Dict):
    s   = t.get("subject", {})
    o   = t.get("object",  {})
    rel = t.get("relation", "?")
    cf  = float(t.get("confidence", 0))
    rv  = float(t.get("relevance_to_ticker", 0))
    imp = float(t.get("price_impact_score", 0))
    rsn = t.get("reasoning", "")
    print(f"\n  #{i+1} [{_grp(rel)}]  {s.get('name','?')} —{rel}→ {o.get('name','?')}")
    print(f"       conf={cf:.2f} {_bar(cf)}  rel={rv:.2f} {_bar(rv)}  impact={imp:+.2f} {_impact(imp)}")
    if rsn:
        print(textwrap.fill(rsn, 64, initial_indent="       -> ", subsequent_indent="          "))

def print_result(r: Dict, show_cross: bool = True):
    t, d = r["ticker"], r["date"]
    arts, uniq, hits, chunks, elapsed = (
        r["n_articles"], r["n_unique"], r["cache_hits"], r["n_chunks"], r["elapsed"]
    )
    triples = r["triples"]

    print(f"\n{'═'*66}")
    print(f"  {t}  |  {d}")
    print(f"  {arts} articles  →  {uniq} unique  →  {chunks} chunks  "
          f"|  cache_hits={hits}  |  {elapsed:.1f}s")
    print(f"  Primary triples after rescore: {len(triples)}")

    if not triples:
        print("  (no triples — TYPE D articles or all filtered)")
    for i, tri in enumerate(triples):
        print_triple(i, tri)

    if show_cross:
        cross_nonempty = {k: v for k, v in r["cross"].items() if v}
        if cross_nonempty:
            print(f"\n  Cross-ticker:")
            for ct, ct_triples in sorted(cross_nonempty.items()):
                avg_rel = sum(float(x.get("relevance_to_ticker", 0)) for x in ct_triples) / len(ct_triples)
                print(f"    {ct}: {len(ct_triples)} triples  avg_rel={avg_rel:.2f}")

def print_summary(results: List[Dict], total_elapsed: float):
    from collections import Counter
    print(f"\n{'━'*66}")
    print(f"  SUMMARY  —  V4 pipeline (chunk-based, title×3, tier2=0.75)")
    print(f"{'━'*66}")
    print(f"  Days tested : {len(results)}")
    print(f"  Total time  : {total_elapsed:.1f}s\n")

    print(f"  {'Ticker':6}  {'Date':12}  {'Arts':>4}  {'Chunks':>6}  "
          f"{'Triples':>7}  {'Avg impact':>11}")
    print(f"  {'-'*62}")
    for r in results:
        ts = r["triples"]
        if ts:
            avg_imp = sum(float(t.get("price_impact_score", 0)) for t in ts) / len(ts)
            imp_s   = f"{avg_imp:+.2f} {_impact(avg_imp)}"
        else:
            imp_s = "  n/a"
        print(f"  {r['ticker']:6}  {r['date']:12}  {r['n_articles']:>4}  "
              f"{r['n_chunks']:>6}  {len(ts):>7}  {imp_s}")

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
        print(f"\n  Group A+B: {ga+gb}/{n} ({pct:.0f}%)  "
              f"{'Good (>=50%)' if pct >= 50 else 'Low (<50%)'}")
    print(f"{'━'*66}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Test extraction — V4 pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_extraction.py --days 3
  python test_extraction.py --tickers TSLA AAPL MSFT
  python test_extraction.py --days 5 --filter TSLA
  python test_extraction.py --days 3 --dry-run
  python test_extraction.py --days 3 --date-range 2023-01-01 2023-06-30
  python test_extraction.py --tickers TSLA AAPL --gemini-batch
""")
    p.add_argument("--days",       type=int, default=None,
                   help="Số ngày ngẫu nhiên cần test")
    p.add_argument("--tickers",    nargs="+", default=None,
                   help="Danh sách ticker (1 ngày/ticker)")
    p.add_argument("--filter",     default=None,
                   help="Chỉ lấy ngày của ticker này khi dùng --days")
    p.add_argument("--date-range", nargs=2, metavar=("FROM", "TO"), default=None,
                   help="Giới hạn phạm vi ngày, format YYYY-MM-DD")
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--data",       default=DEFAULT_PARQUET)
    p.add_argument("--cache-dir",  default=DEFAULT_CACHE)
    p.add_argument("--min-relevance",  type=float, default=0.30)
    p.add_argument("--min-confidence", type=float, default=0.35)
    p.add_argument("--max-concurrent", type=int,   default=5)
    p.add_argument("--gemini-batch",   action="store_true")
    p.add_argument("--dry-run",        action="store_true",
                   help="Chỉ xem trước danh sách ngày, không gọi API")
    p.add_argument("--no-cross",       action="store_true",
                   help="Không hiện cross-ticker results")
    p.add_argument("--save",       default=None,
                   help="Lưu kết quả JSON, ví dụ --save results.json")
    return p.parse_args()


def main():
    args = parse_args()

    # Validate
    if args.days is None and args.tickers is None:
        print("Cần chỉ định --days N hoặc --tickers TICKER1 TICKER2 ...")
        print("Ví dụ: python test_extraction.py --days 3")
        sys.exit(1)

    print("=" * 66)
    print("  Test Extraction — V4 Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 66)

    # Load data
    if not os.path.exists(args.data):
        print(f"File not found: {args.data}")
        sys.exit(1)

    print(f"\nLoading: {args.data}")
    df = load_and_normalize(args.data)
    print(f"  {len(df):,} rows  |  "
          f"{df['date'].nunique()} unique dates  |  "
          f"{df['primary_ticker'].nunique()} tickers")

    # Sample
    date_start = args.date_range[0] if args.date_range else None
    date_end   = args.date_range[1] if args.date_range else None

    if args.tickers:
        samples = sample_by_tickers(df, args.tickers, seed=args.seed)
    else:
        samples = sample_by_days(
            df, args.days,
            ticker_filter=args.filter,
            date_start=date_start,
            date_end=date_end,
            seed=args.seed,
        )

    if not samples:
        print("Không có dữ liệu phù hợp với filter.")
        sys.exit(0)

    print(f"\nSampled {len(samples)} combination(s):")
    for ticker, date_str, day_df in samples:
        all_t = sorted({t for ts in day_df["_all_tickers"] for t in ts})
        print(f"  {ticker:6}  {date_str}  "
              f"{len(day_df)} articles  all_tickers={all_t}")

    if args.dry_run:
        print("\nDry-run. Bỏ --dry-run để chạy thật.")
        return

    # Init extractor
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("\nGEMINI_API_KEY chưa được set.")
        print("  export GEMINI_API_KEY='your_key'")
        sys.exit(1)

    if args.gemini_batch:
        extractor = GeminiBatchAPIExtractor(
            api_key=api_key,
            min_relevance=args.min_relevance,
            min_confidence=args.min_confidence,
        )
        print(f"\nExtractor: GeminiBatchAPIExtractor")
    else:
        extractor = AsyncConcurrentExtractor(
            api_key=api_key,
            min_relevance=args.min_relevance,
            min_confidence=args.min_confidence,
            max_concurrent=args.max_concurrent,
        )
        print(f"\nExtractor: AsyncConcurrentExtractor (max_concurrent={args.max_concurrent})")

    cache_dir = args.cache_dir.strip() if args.cache_dir else ""
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Cache    : {os.path.abspath(cache_dir)}")

    # Run
    results     = []
    t0_total    = time.time()

    for ticker, date_str, day_df in samples:
        print(f"\n{'─'*66}")
        print(f"  {ticker}  {date_str}  ({len(day_df)} articles)")
        result = extract_one(
            ticker, date_str, day_df,
            extractor, cache_dir,
            args.min_relevance, args.min_confidence,
        )
        print_result(result, show_cross=not args.no_cross)
        results.append(result)

    print_summary(results, time.time() - t0_total)

    # Save
    if args.save:
        try:
            out = {
                "meta": {
                    "timestamp": datetime.now().isoformat(),
                    "pipeline":  "V4",
                    "min_relevance": args.min_relevance,
                    "min_confidence": args.min_confidence,
                },
                "results": [
                    {k: v for k, v in r.items() if k != "cross"}
                    for r in results
                ],
            }
            with open(args.save, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2, default=str)
            print(f"Saved: {args.save}")
        except Exception as e:
            print(f"Save failed: {e}")


if __name__ == "__main__":
    main()