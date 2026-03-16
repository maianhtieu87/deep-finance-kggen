# extract_corpus.py
"""
Stage A — LLM Extraction Only
==============================
Đọc news parquet → gộp headlines+content theo (ticker, date) → chunk →
gọi Gemini → lưu triples vào SHA-1 cache.

KHÔNG build graph, KHÔNG Voyage embedding, KHÔNG KMeans.
Kết quả cache có thể tái sử dụng vô hạn lần cho Stage B.

Usage:
    python extract_corpus.py
    python extract_corpus.py --batch          # dùng Gemini Batch API (50% cost)
    python extract_corpus.py --resume         # bỏ qua những ngày đã cache đủ
    python extract_corpus.py --ticker TSLA    # chỉ extract 1 ticker
    python extract_corpus.py --date 2022-01   # chỉ extract tháng cụ thể (prefix)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from configs.config import GlobalConfig

from data_pipeline.kg.extractor_batch import (
    AsyncConcurrentExtractor,
    GeminiBatchAPIExtractor,
    TICKER_NAME_MAP,
    detect_primary_ticker,
    build_combined_text,
    split_text_chunks,
    dedup_triples,
    rescore_triples_for_ticker,
    _sha1,
    _norm,
    _filter_and_clamp,
)


# ─────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(cache_dir: str, sha1: str) -> str:
    return os.path.join(cache_dir, f"{sha1}.json")


def _load_cache(cache_dir: str, sha1: str) -> Optional[List[Dict]]:
    p = _cache_path(cache_dir, sha1)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("triples", [])
    except Exception:
        return None


def _save_cache(cache_dir: str, sha1: str, triples: List[Dict],
                meta: Dict = None) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    p = _cache_path(cache_dir, sha1)
    payload = {"triples": triples, "_v": "v3"}
    if meta:
        payload["_meta"] = meta
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# DATAFRAME HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_tickers(val: Any) -> List[str]:
    if isinstance(val, list):
        return [t.strip().upper() for t in val if isinstance(t, str) and t.strip()]
    if isinstance(val, str):
        return [t.strip().upper() for t in val.split(",") if t.strip()]
    return []


def normalize_news_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hoá DataFrame:
      - rename columns
      - parse date
      - parse tickers → _all_tickers list (giữ list gốc trước khi explode)
      - detect primary_ticker (với title weight × 3)
      - explode 1 row per ticker → equity column
      - _all_tickers vẫn giữ full list để rescore dùng
    """
    df = df.copy()

    # Column renames
    col_map = {}
    if "headline" in df.columns and "title" not in df.columns:
        col_map["headline"] = "title"
    if "ticker" in df.columns and "equity" not in df.columns:
        col_map["ticker"] = "equity"
    if col_map:
        df = df.rename(columns=col_map)

    # Text columns
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
    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column. Has: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    # Ticker column detection
    ticker_col = next((c for c in ("symbols", "equity") if c in df.columns), None)
    if ticker_col is None:
        raise ValueError("No ticker column (symbols/equity) found.")

    # Parse all tickers — keep as list BEFORE explode
    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)
    df = df[df["_all_tickers"].map(len) > 0]

    # Detect primary ticker using title weight × 3
    df["primary_ticker"] = df.apply(
        lambda row: detect_primary_ticker(
            str(row.get("title", "") or ""),
            str(row.get("content", "") or ""),
            row["_all_tickers"],
        ),
        axis=1,
    )

    # Explode: 1 row per ticker, _all_tickers column stays intact
    df = df.explode("_all_tickers").rename(columns={"_all_tickers": "equity"})
    df = df[df["equity"].notna() & (df["equity"] != "")].reset_index(drop=True)

    # After explode, _all_tickers column is gone; restore from original data
    # We need to re-attach the original list — pandas explode keeps other columns
    # so we need to recreate _all_tickers from the unexploded ticker_col
    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# PER-DAY-TICKER EXTRACTION  (chunk-based, merge+dedup)
# ─────────────────────────────────────────────────────────────────────────────

def extract_day_ticker(
    day_df: pd.DataFrame,
    ticker: str,
    date_str: str,
    extractor,
    cache_dir: str,
    min_relevance: float,
    min_confidence: float,
) -> List[Dict]:
    """
    Gộp tất cả bài của (ticker, date):
      1. Gộp titles trước, content sau → combined_text
      2. Tính SHA-1 của combined_text → đây là cache key cho (ticker, date)
      3. Nếu cache hit → dùng luôn
      4. Nếu miss → chia chunks → extract từng chunk → merge+dedup
      5. Save cache

    Lưu ý: cache key là SHA-1 của combined_text cho (ticker, date), không phải per-article.
    Điều này đảm bảo cùng một ngày + ticker luôn dùng cùng cache entry.
    """
    # Gộp titles và contents
    titles   = day_df["title"].fillna("").tolist()
    contents = day_df["content"].fillna("").tolist()
    combined = build_combined_text(titles, contents)
    combined = _norm(combined)

    if not combined:
        return []

    cache_key = _sha1(combined)
    cached = _load_cache(cache_dir, cache_key)
    if cached is not None:
        return cached

    # Chia chunks
    chunks = split_text_chunks(combined)

    # Build chunk articles cho extractor
    chunk_articles = [
        {"text": chunk, "ticker": ticker, "date": date_str}
        for chunk in chunks
    ]

    # Extract
    if len(chunk_articles) > 0:
        print(f"  [{ticker} {date_str}] {len(titles)} articles → "
              f"{len(chunks)} chunk(s) → extracting...")
        batch_results = extractor.extract_batch(chunk_articles)
    else:
        batch_results = []

    # Merge và dedup tất cả triples từ các chunks
    all_triples: List[Dict] = []
    for chunk_triples in batch_results:
        all_triples.extend(chunk_triples or [])

    # Filter + dedup
    filtered = _filter_and_clamp(all_triples, min_relevance, min_confidence)
    deduped  = dedup_triples(filtered)

    # Save cache với metadata
    _save_cache(cache_dir, cache_key, deduped, meta={
        "ticker": ticker,
        "date":   date_str,
        "n_articles": len(titles),
        "n_chunks":   len(chunks),
    })

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# ALSO SAVE PER-ARTICLE CACHE  (để rebuild_graph_only có thể dùng)
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_cache_per_article(
    day_df: pd.DataFrame,
    ticker: str,
    date_str: str,
    extractor,
    cache_dir: str,
    min_relevance: float,
    min_confidence: float,
) -> List[Dict]:
    """
    Extract per-article (không gộp ngày), phù hợp cho việc build graph sau này.
    Cache key = SHA-1 của (title + content) của từng article.
    Fan-out + rescore được thực hiện sau khi có cache.

    Đây là entry point chính cho Stage A.
    Trả về list of triples đã được rescore cho ticker này.
    """
    sha1_to_meta: Dict[str, Dict] = {}
    sha1_to_raw:  Dict[str, Optional[List[Dict]]] = {}
    cache_hits = 0

    for _, row in day_df.iterrows():
        title   = _norm(str(row.get("title",   "") or ""))
        content = _norm(str(row.get("content", "") or ""))

        # Gộp title + content, title weight cao hơn bằng cách đặt ở đầu
        full_text = build_combined_text(
            [title] if title else [],
            [content] if content else [],
        )
        if not full_text:
            continue

        h = _sha1(full_text)
        if h in sha1_to_raw:
            # Bài trùng: bổ sung tickers
            existing_tickers = set(sha1_to_meta[h]["all_tickers"])
            new_tickers = set(row.get("_all_tickers") or [ticker])
            sha1_to_meta[h]["all_tickers"] = list(existing_tickers | new_tickers)
            continue

        primary = str(row.get("primary_ticker") or ticker)
        all_t   = list(row.get("_all_tickers") or [primary])

        sha1_to_meta[h] = {
            "full_text":     full_text,
            "primary_ticker": primary,
            "date":           date_str,
            "all_tickers":    all_t,
        }

        cached = _load_cache(cache_dir, h)
        if cached is not None:
            sha1_to_raw[h] = cached
            cache_hits += 1
        else:
            sha1_to_raw[h] = None  # cần extract

    # Extract uncached articles
    uncached_sha1s = [h for h, v in sha1_to_raw.items() if v is None]
    if uncached_sha1s:
        print(f"  [{ticker} {date_str}] cache_hits={cache_hits} "
              f"to_extract={len(uncached_sha1s)}")

        # Build chunk articles: mỗi article có thể tạo nhiều chunks
        # Mỗi chunk vẫn extract với primary_ticker của article gốc
        chunk_jobs: List[Tuple[str, int, Dict]] = []  # (sha1, chunk_idx, article_dict)
        for h in uncached_sha1s:
            meta   = sha1_to_meta[h]
            chunks = split_text_chunks(meta["full_text"])
            for i, chunk in enumerate(chunks):
                chunk_jobs.append((h, i, {
                    "text":   chunk,
                    "ticker": meta["primary_ticker"],
                    "date":   meta["date"],
                }))

        # Extract tất cả chunks concurrently
        articles_to_extract = [job[2] for job in chunk_jobs]
        batch_results = extractor.extract_batch(articles_to_extract)

        # Group chunk results back by sha1
        sha1_chunk_triples: Dict[str, List[Dict]] = defaultdict(list)
        for (sha1, chunk_idx, _), triples in zip(chunk_jobs, batch_results):
            sha1_chunk_triples[sha1].extend(triples or [])

        # Merge + dedup + cache per article
        for h in uncached_sha1s:
            merged  = _filter_and_clamp(sha1_chunk_triples[h], min_relevance, min_confidence)
            deduped = dedup_triples(merged)
            sha1_to_raw[h] = deduped
            _save_cache(cache_dir, h, deduped, meta={
                "primary_ticker": sha1_to_meta[h]["primary_ticker"],
                "date":           sha1_to_meta[h]["date"],
                "all_tickers":    sha1_to_meta[h]["all_tickers"],
            })
    else:
        if sha1_to_meta:
            print(f"  [{ticker} {date_str}] all {cache_hits} article(s) from cache")

    # Fan-out + rescore cho target ticker
    all_triples: List[Dict] = []
    for h, raw_triples in sha1_to_raw.items():
        if not raw_triples:
            continue
        meta    = sha1_to_meta[h]
        primary = meta["primary_ticker"]
        rescored = rescore_triples_for_ticker(
            raw_triples,
            primary_ticker=primary,
            target_ticker=ticker,
            min_relevance=min_relevance,
            article_text=meta["full_text"],
            all_article_tickers=meta["all_tickers"],
        )
        rescored = [t for t in rescored
                    if float(t.get("confidence", 0)) >= min_confidence]
        all_triples.extend(rescored)

    return dedup_triples(all_triples)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT — Stage A
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_a(
    news_df: pd.DataFrame,
    cache_dir: str,
    use_gemini_batch: bool = False,
    max_concurrent: int = 5,
    min_relevance: float = 0.30,
    min_confidence: float = 0.35,
    ticker_filter: Optional[str] = None,
    date_prefix: Optional[str]   = None,
) -> Dict[str, Dict[str, int]]:
    """
    Stage A: Extract triples từ news_df, lưu cache per-article.

    Args:
        news_df          : raw news DataFrame
        cache_dir        : thư mục cache SHA-1
        use_gemini_batch : True = GeminiBatch (50% cost), False = AsyncConcurrent
        max_concurrent   : số requests đồng thời (AsyncConcurrent)
        min_relevance    : ngưỡng relevance_to_ticker
        min_confidence   : ngưỡng confidence
        ticker_filter    : chỉ process 1 ticker (optional)
        date_prefix      : chỉ process ngày bắt đầu với prefix này, ví dụ "2022-01"

    Returns:
        summary: {ticker: {date: n_triples}}
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Chuẩn hoá DataFrame
    df = normalize_news_df(news_df)

    # Filter
    if ticker_filter:
        df = df[df["equity"] == ticker_filter.upper()]
        if len(df) == 0:
            print(f"No data for ticker {ticker_filter}")
            return {}
    if date_prefix:
        df = df[df["date"].astype(str).str.startswith(date_prefix)]
        if len(df) == 0:
            print(f"No data for date prefix {date_prefix}")
            return {}

    # Khởi tạo extractor
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")

    if use_gemini_batch:
        extractor = GeminiBatchAPIExtractor(
            api_key=api_key,
            min_relevance=min_relevance,
            min_confidence=min_confidence,
        )
        print("Extractor: GeminiBatchAPIExtractor (50% cost)")
    else:
        extractor = AsyncConcurrentExtractor(
            api_key=api_key,
            min_relevance=min_relevance,
            min_confidence=min_confidence,
            max_concurrent=max_concurrent,
        )
        print(f"Extractor: AsyncConcurrentExtractor (max_concurrent={max_concurrent})")

    tickers      = sorted(df["equity"].unique())
    total_dates  = df["date"].nunique()
    print(f"\nStage A: {len(tickers)} tickers × ~{total_dates} dates")
    print(f"Cache dir: {cache_dir}\n")

    summary: Dict[str, Dict[str, int]] = {}
    total_new    = 0
    total_cached = 0

    for ticker in tickers:
        df_t = df[df["equity"] == ticker].copy()
        summary[ticker] = {}

        for d in sorted(df_t["date"].unique()):
            date_str = str(d)
            day_df   = df_t[df_t["date"] == d]

            # Count pre-existing cache hits for this day
            pre_cached = sum(
                1 for _, row in day_df.iterrows()
                if _load_cache(cache_dir, _sha1(build_combined_text(
                    [_norm(str(row.get("title","") or ""))],
                    [_norm(str(row.get("content","") or ""))],
                ))) is not None
            )

            triples = extract_and_cache_per_article(
                day_df, ticker, date_str, extractor,
                cache_dir, min_relevance, min_confidence,
            )
            summary[ticker][date_str] = len(triples)
            total_new += max(0, len(day_df) - pre_cached)
            total_cached += pre_cached

    print(f"\nStage A complete.")
    print(f"  Cache hits: {total_cached}")
    print(f"  New API calls (articles): ~{total_new}")
    print(f"  Cache dir: {cache_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage A — LLM Extraction Only")
    parser.add_argument("--batch", action="store_true",
                        help="Use Gemini Batch API (50%% cost, >500 articles)")
    parser.add_argument("--max-concurrent", type=int, default=5)
    parser.add_argument("--min-relevance",  type=float, default=0.30)
    parser.add_argument("--min-confidence", type=float, default=0.35)
    parser.add_argument("--ticker", default=None, help="Process only this ticker")
    parser.add_argument("--date",   default=None, help="Process only dates starting with this prefix, e.g. 2022-01")
    parser.add_argument("--news",   default=None, help="Path to news parquet (default: GlobalConfig)")
    args = parser.parse_args()

    news_path = args.news or os.path.join(
        GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet"
    )
    if not os.path.exists(news_path):
        print(f"News file not found: {news_path}")
        sys.exit(1)

    print(f"Loading news: {news_path}")
    df = pd.read_parquet(news_path)
    print(f"Loaded {len(df):,} rows")

    cache_dir = GlobalConfig.kg_cache_dir()

    run_stage_a(
        news_df=df,
        cache_dir=cache_dir,
        use_gemini_batch=args.batch,
        max_concurrent=args.max_concurrent,
        min_relevance=args.min_relevance,
        min_confidence=args.min_confidence,
        ticker_filter=args.ticker,
        date_prefix=args.date,
    )


if __name__ == "__main__":
    main()