#!/usr/bin/env python3
# embed_news_batch.py
"""
Stage A.5 — News Embedding (BATCH COMPATIBLE VERSION)

V6 change vs V5:
  - THAY ĐỔI LOGIC ĐỌC CACHE: Quét trực tiếp thư mục cache và đọc `_meta` (date, primary_ticker) 
    thay vì băm SHA1 từng bài báo. Điều này giúp hỗ trợ đọc được cache của các cụm bài báo gộp 
    (Multi-article concat extraction - Phase 2).
  - Giữ nguyên luồng xử lý embedding bằng VoyageAI.
  - Vẫn xuất ra format: {"YYYY-MM-DD": {"TSLA": [1024D vector], "AAPL": [...]}}

Usage:
    python embed_news_batch.py
    python embed_news_batch.py --ticker TSLA
    python embed_news_batch.py --date 2023-07
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import GlobalConfig
from configs.ticker_aliases import TICKER_NAME_MAP


# ─────────────────────────────────────────────────────────────────────────────
# VOYAGE EMBEDDER (Giữ nguyên)
# ─────────────────────────────────────────────────────────────────────────────

class VoyageEmbedder:
    """Simple rate-limited Voyage embedder with disk cache."""

    def __init__(self, cache_dir: str):
        self.api_key = os.getenv("VOYAGE_API_KEY", GlobalConfig.VOYAGE_API_KEY)
        if not self.api_key or self.api_key.strip() in ("", "---"):
            raise RuntimeError("VOYAGE_API_KEY not set.")
        self.model        = getattr(GlobalConfig, "EMBED_MODEL", "voyage-3-large")
        self.max_texts    = getattr(GlobalConfig, "MAX_TEXTS_PER_REQ", 40)
        self.max_retries  = getattr(GlobalConfig, "MAX_RETRIES", 6)
        self.backoff_base = getattr(GlobalConfig, "BACKOFF_BASE", 30)
        payment_added = bool(getattr(GlobalConfig, "PAYMENT_ADDED", True))
        rl = GlobalConfig.VOYAGE_RATE_LIMITS[payment_added]
        self.base_sleep = float(rl.get("SLEEP", 1.0))
        self.rpm        = int(rl.get("RPM", 50))
        self.cache_dir  = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._req_times: List[float] = []

    def _sha1(self, s: str) -> str:
        return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

    def _cache_path(self, text: str) -> str:
        return os.path.join(self.cache_dir, f"{self._sha1(text)}.json")

    def _load(self, text: str) -> Optional[List[float]]:
        p = self._cache_path(text)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    emb = json.load(f).get("embedding")
                if isinstance(emb, list) and len(emb) > 0:
                    return emb
            except Exception:
                pass
        return None

    def _save(self, text: str, emb: List[float]):
        with open(self._cache_path(text), "w") as f:
            json.dump({"embedding": emb}, f)

    def _rpm_guard(self):
        now = time.time()
        self._req_times = [t for t in self._req_times if now - t < 60.0]
        if len(self._req_times) >= self.rpm:
            wait = max(0.0, 60.0 - (now - min(self._req_times))) + 0.5
            print(f"  Voyage RPM limit — sleep {wait:.1f}s")
            time.sleep(wait)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        out, miss_i, miss_t = [None] * len(texts), [], []
        for i, t in enumerate(texts):
            t = (t or "").strip()
            if not t: out[i] = [0.0] * 1024; continue
            cached = self._load(t)
            if cached is not None: out[i] = cached
            else: miss_i.append(i); miss_t.append(t)

        if not miss_t:
            return [o if o is not None else [0.0] * 1024 for o in out]

        url     = "https://api.voyageai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        def chunks(lst, n):
            for i in range(0, len(lst), n): yield lst[i: i + n]

        pos = 0
        for batch_texts in chunks(miss_t, self.max_texts):
            batch_idx = miss_i[pos: pos + len(batch_texts)]
            pos += len(batch_texts)
            payload = {"model": self.model, "input": batch_texts}
            for attempt in range(self.max_retries):
                try:
                    self._rpm_guard()
                    if self.base_sleep > 0: time.sleep(self.base_sleep)
                    r = requests.post(url, headers=headers, json=payload, timeout=(15, 120))
                    self._req_times.append(time.time())
                    if r.status_code == 429:
                        wait = self.backoff_base * (2 ** attempt) + 2.0
                        print(f"  429 rate limit — sleep {wait:.0f}s")
                        time.sleep(wait); continue
                    r.raise_for_status()
                    embs = r.json().get("data", [])
                    for bi, item in enumerate(embs):
                        emb = item.get("embedding", [])
                        idx = batch_idx[bi]
                        out[idx] = emb
                        self._save(batch_texts[bi], emb)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(f"  Voyage failed: {e}")
                        for idx in batch_idx: out[idx] = [0.0] * 1024
                    else:
                        time.sleep(self.backoff_base * (2 ** attempt))

        return [o if o is not None else [0.0] * 1024 for o in out]


# ─────────────────────────────────────────────────────────────────────────────
# UTILS & TICKER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _parse_tickers(val: Any) -> List[str]:
    if isinstance(val, list):
        return [t.strip().upper() for t in val if isinstance(t, str) and t.strip()]
    if isinstance(val, str):
        return [t.strip().upper() for t in val.split(",") if t.strip()]
    return []

TITLE_WEIGHT = 3

def detect_primary_ticker(title: str, content: str, tickers: List[str]) -> str:
    if not tickers: return ""
    if len(tickers) == 1: return tickers[0]
    t_up = (title   or "").upper()
    c_up = (content or "").upper()
    counts = {}
    for t in tickers:
        score = 0
        for name in TICKER_NAME_MAP.get(t, [t]):
            n_up = name.upper()
            score += t_up.count(n_up) * TITLE_WEIGHT + c_up.count(n_up)
        counts[t] = score
    best = max(counts.values())
    if best == 0: return tickers[0]
    for t in tickers:
        if counts[t] == best: return t

def _ticker_mentioned_in_triple(ticker: str, triple: Dict) -> bool:
    """True if ticker or any known name appears in subject or object."""
    tl = ticker.lower()
    sn = triple.get("subject", {}).get("name", "").lower()
    on = triple.get("object",  {}).get("name", "").lower()
    if tl in sn or tl in on: return True
    for name in TICKER_NAME_MAP.get(ticker.upper(), []):
        if name.lower() in sn or name.lower() in on: return True
    return False

def triples_to_text(triples: List[Dict], ticker: str) -> str:
    if not triples:
        return ""
    sorted_triples = sorted(
        triples,
        key=lambda t: abs(float(t.get("price_impact_score", 0))),
        reverse=True,
    )
    parts = [f"TARGET: {ticker}"]
    for t in sorted_triples:
        subj   = t.get("subject", {}).get("name", "")
        rel    = t.get("relation", "")
        obj    = t.get("object",  {}).get("name", "")
        reason = t.get("reasoning", "")
        if not subj or not obj:
            continue
        line = f"{subj} {rel} {obj}"
        if reason:
            line += f". {reason}"
        parts.append(line)
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# NEW LOGIC: PRELOAD CACHE & GROUP TRIPLES (BATCH COMPATIBLE)
# ─────────────────────────────────────────────────────────────────────────────

def preload_cache_directory(cache_dir: str) -> List[Dict]:
    """
    Quét trực tiếp thư mục cache. Lấy ra list các dict chứa thông tin:
    { "date": "2023-07-27", "primary_ticker": "TSLA", "triples": [...] }
    Hỗ trợ đọc hoàn hảo các file gộp nhiều articles.
    """
    entries = []
    if not os.path.exists(cache_dir):
        return entries
        
    files = [f for f in os.listdir(cache_dir) if f.endswith(".json") and not f.startswith("_")]
    
    for fname in files:
        path = os.path.join(cache_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            meta = data.get("_meta", {})
            date_val = meta.get("date")
            triples = data.get("triples", [])
            
            if not date_val or not triples:
                continue
                
            entries.append({
                "date": str(date_val),
                "primary_ticker": str(meta.get("primary_ticker", "")).upper(),
                "triples": triples
            })
        except Exception:
            pass
            
    return entries

def get_day_triples_from_cache(
    date_str: str,
    ticker: str,
    cache_entries: List[Dict],
    min_relevance: float,
    min_confidence: float,
) -> List[Dict]:
    """Trích xuất và deduplicate triples từ memory cache đã load cho (date, ticker)."""
    all_triples = []
    
    for entry in cache_entries:
        if entry["date"] != date_str:
            continue
            
        primary = entry["primary_ticker"]
        raw = entry["triples"]

        if primary == ticker.upper():
            filtered = [
                t for t in raw
                if float(t.get("confidence", 0)) >= min_confidence
                and float(t.get("relevance_to_ticker", 0)) >= min_relevance
            ]
        else:
            filtered = [
                t for t in raw
                if _ticker_mentioned_in_triple(ticker, t)
                and float(t.get("confidence", 0)) >= min_confidence
                and float(t.get("relevance_to_ticker", 0)) >= min_relevance
            ]

        all_triples.extend(filtered)

    # Khử trùng lặp (Deduplicate)
    seen = set()
    deduped = []
    for t in all_triples:
        key = (
            t.get("subject", {}).get("name", ""),
            t.get("relation", ""),
            t.get("object",  {}).get("name", ""),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Stage A.5
# ─────────────────────────────────────────────────────────────────────────────

def run_embed_news(
    news_df:        pd.DataFrame,
    cache_dir:      str,
    output_path:    str,
    voyage_cache:   str,
    min_relevance:  float = None,
    min_confidence: float = None,
    ticker_filter:  Optional[str] = None,
    date_prefix:    Optional[str] = None,
) -> str:
    if min_relevance  is None: min_relevance  = GlobalConfig.KG_MIN_RELEVANCE
    if min_confidence is None: min_confidence = GlobalConfig.KG_MIN_CONFIDENCE

    voyage = VoyageEmbedder(cache_dir=voyage_cache)

    # Lọc Tickers & Dates từ DataFrame để biết "universe" cần xử lý (Giữ nguyên từ code cũ)
    df = news_df.copy()
    if "headline" in df.columns and "title" not in df.columns:
        df = df.rename(columns={"headline": "title"})
    if "ticker" in df.columns and "equity" not in df.columns:
        df = df.rename(columns={"ticker": "equity"})

    if "date" not in df.columns:
        DATE_CANDS = ["created_at", "createdAt", "published_at", "publishedAt",
                      "publish_date", "pub_date", "Date", "DATE", "timestamp", "time", "news_date"]
        date_col = next((c for c in DATE_CANDS if c in df.columns), None)
        if date_col is None:
            date_col = next((c for c in df.columns if any(k in c.lower() for k in ("date", "time", "publish", "creat"))), None)
        if date_col is not None:
            df = df.rename(columns={date_col: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    ticker_col = next((c for c in ("symbols", "equity") if c in df.columns), None)
    if ticker_col is None:
        raise ValueError("No ticker column found.")

    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)
    df = df[df["_all_tickers"].map(len) > 0]
    df = df.explode("_all_tickers").rename(columns={"_all_tickers": "equity"})
    df = df[df["equity"].notna() & (df["equity"] != "")].reset_index(drop=True)

    if ticker_filter:
        df = df[df["equity"] == ticker_filter.upper()]
    if date_prefix:
        df = df[df["date"].astype(str).str.startswith(date_prefix)]

    tickers = sorted(df["equity"].unique())
    print(f"\nEmbed news V6 (Batch-Compatible): {len(tickers)} tickers")
    print(f"Cache dir : {cache_dir}")
    print(f"Output    : {output_path}\n")

    # BƯỚC MỚI: Load toàn bộ thư mục cache vào memory một lần duy nhất
    print("Scanning and preloading cache directory...")
    cache_entries = preload_cache_directory(cache_dir)
    print(f"  → Found {len(cache_entries)} cache entries with valid triples.")

    ticker_date_text: Dict[Tuple[str, str], str] = {}
    miss_count = 0

    for ticker in tickers:
        df_t  = df[df["equity"] == ticker].copy()
        dates = sorted(df_t["date"].unique())

        for d in dates:
            date_str = str(d)
            
            # Thay vì đọc file theo df row, giờ truy xuất trực tiếp từ cache_entries
            triples = get_day_triples_from_cache(
                date_str=date_str,
                ticker=ticker,
                cache_entries=cache_entries,
                min_relevance=min_relevance,
                min_confidence=min_confidence,
            )

            if not triples:
                miss_count += 1

            text = triples_to_text(triples, ticker)
            ticker_date_text[(ticker, date_str)] = text

    print(f"\nDays/Tickers with no cache entries : {miss_count}")
    print(f"Total (ticker, date) pairs       : {len(ticker_date_text)}")

    unique_texts = list(set(ticker_date_text.values()))
    non_empty    = [t for t in unique_texts if t]
    empty_vec    = [0.0] * 1024

    if non_empty:
        print(f"\nEmbedding {len(non_empty)} unique texts via Voyage...")
        embeddings_list = voyage.embed_texts(non_empty)
        text_to_emb     = {t: e for t, e in zip(non_empty, embeddings_list)}
    else:
        text_to_emb = {}

    output: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for (ticker, date_str), text in ticker_date_text.items():
        emb = text_to_emb.get(text, empty_vec)
        output[date_str][ticker] = emb

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dict(output), f, ensure_ascii=False)

    print(f"\nSaved news_embeddings.json: {len(output)} dates")
    print(f"Output path: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Stage A.5 — Embed news triples via Voyage (V6 Batch Compatible)")
    parser.add_argument("--news",       default=None,  help="Path to news parquet")
    parser.add_argument("--cache-dir",  default=None,  help="SHA-1 cache dir (default: GlobalConfig)")
    parser.add_argument("--output",     default=None,  help="Output JSON path (default: GlobalConfig)")
    parser.add_argument("--ticker",     default=None)
    parser.add_argument("--date",       default=None,  help="Date prefix filter, e.g. '2023-07'")
    parser.add_argument("--min-relevance",  type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    args = parser.parse_args()

    news_path = args.news or os.path.join(
        GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet"
    )
    if not os.path.exists(news_path):
        print(f"News file not found: {news_path}"); sys.exit(1)

    cache_dir = args.cache_dir or GlobalConfig.kg_cache_dir()
    if not os.path.exists(cache_dir):
        print(f"Cache dir not found: {cache_dir}")
        print("Run Stage A first."); sys.exit(1)

    output_path = args.output or os.path.join(
        GlobalConfig.INTERIM_PATH, "kg_embeddings", "news_embeddings.json"
    )
    voyage_cache = GlobalConfig.kg_voyage_cache_dir()

    df = pd.read_parquet(news_path)
    print(f"Loaded {len(df):,} rows from {news_path}")

    run_embed_news(
        news_df=df,
        cache_dir=cache_dir,
        output_path=output_path,
        voyage_cache=voyage_cache,
        min_relevance=args.min_relevance,
        min_confidence=args.min_confidence,
        ticker_filter=args.ticker,
        date_prefix=args.date,
    )

if __name__ == "__main__":
    main()