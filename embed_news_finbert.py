#!/usr/bin/env python3
# embed_news_finbert.py — FinBERT + KG Structured Features (13D hybrid)
"""
Stage A.5 Alternative — Thay thế Voyage 1024D bằng FinBERT 13D hybrid.

Output vector 13D per (date, ticker):
  [0:3]   FinBERT sentiment: [P_positive, P_negative, P_neutral]
           → từ article titles/summaries (ProsusAI/finbert)
  [3:10]  KG structured 7D:
           [0] net_impact        weighted mean(price_impact * confidence * relevance)
           [1] max_abs_impact    max(|price_impact_score|)
           [2] n_triples_norm    min(n_triples / 10, 1.0)
           [3] has_earnings      bool: earnings/revenue/eps triple tồn tại
           [4] has_regulatory    bool: REGULATES triple tồn tại
           [5] has_guidance      bool: guidance/outlook triple tồn tại
           [6] avg_confidence    mean confidence của tất cả triples
  [10:13] Flags 3D:
           [0] has_news          1.0 (luôn = 1 khi có articles; 0 → vector zeros → masked)
           [1] n_articles_norm   min(n_articles / 5, 1.0)
           [2] sentiment_sign    1=positive, 0.5=neutral, 0=negative (từ KG net_impact)

Output format giống hệt news_embeddings.json:
  {"YYYY-MM-DD": {"TICKER": [13 floats]}}

Days với 0 articles → KHÔNG có trong JSON → data_loader trả zeros → news_mask=True → excluded from attention.

Usage:
    # Cài dependencies trước:
    pip install transformers torch --break-system-packages

    python embed_news_finbert.py                         # tất cả tickers
    python embed_news_finbert.py --ticker TSLA           # 1 ticker, merge mode
    python embed_news_finbert.py --date 2024             # filter date prefix
    python embed_news_finbert.py --check-output          # inspect output file
    python embed_news_finbert.py --output /path/out.json # custom path
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import GlobalConfig

# Import utilities từ embed_news.py (tránh duplicate code)
from embed_news import (
    CacheStore,
    _dedup_triples,
    _load_existing_output,
    _parse_tickers,
    detect_primary_ticker,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FINBERT_OUTPUT_DIM = 13
FINBERT_MODEL_NAME = "ProsusAI/finbert"

_EARNINGS_KW  = frozenset((
    "earnings", "revenue", "eps", "ebit", "ebitda", "margin",
    "q1", "q2", "q3", "q4", "quarterly", "annual result", "net income",
    "operating income", "profit",
))
_GUIDANCE_KW  = frozenset((
    "guidance", "outlook", "forecast", "full-year", "full year",
    "raised guidance", "lowered guidance",
))

_DATE_CANDS = [
    "date", "Date", "DATE", "created_at", "createdAt",
    "published_at", "publishedAt", "publish_date", "pub_date",
    "timestamp", "time", "news_date",
]


# ─────────────────────────────────────────────────────────────────────────────
# FINBERT SCORER
# ─────────────────────────────────────────────────────────────────────────────

class FinBERTScorer:
    """
    Batched FinBERT inference.

    ProsusAI/finbert id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
    Output per text: [P_positive, P_negative, P_neutral]
    """

    def __init__(self, model_name: str = FINBERT_MODEL_NAME, batch_size: int = 32):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device     = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size

        print(f"Loading FinBERT [{model_name}] on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Verify label order — critical for correct interpretation
        id2label = self.model.config.id2label
        print(f"  Label order: {id2label}")
        # Expected: {0: 'positive', 1: 'negative', 2: 'neutral'}
        # If different, adjust _IDX_POS/_IDX_NEG/_IDX_NEU below
        self._idx_pos = next((k for k, v in id2label.items() if "pos" in v.lower()), 0)
        self._idx_neg = next((k for k, v in id2label.items() if "neg" in v.lower()), 1)
        self._idx_neu = next((k for k, v in id2label.items() if "neu" in v.lower()), 2)
        print(f"  Index mapping: pos={self._idx_pos} neg={self._idx_neg} neu={self._idx_neu}")

    def score_texts(self, texts: List[str]) -> np.ndarray:
        """
        Score a list of texts.
        Returns (N, 3) float32 array: each row = [P_positive, P_negative, P_neutral].
        """
        import torch
        import torch.nn.functional as F

        if not texts:
            return np.empty((0, 3), dtype=np.float32)

        all_probs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc   = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,   # titles/summaries fit easily; truncate if needed
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits          # (B, 3)
                probs  = F.softmax(logits, dim=-1).cpu().numpy()   # (B, 3)
            # Reorder to [P_pos, P_neg, P_neu] regardless of model's internal order
            reordered = np.stack([
                probs[:, self._idx_pos],
                probs[:, self._idx_neg],
                probs[:, self._idx_neu],
            ], axis=1)
            all_probs.append(reordered)

        return np.vstack(all_probs).astype(np.float32)   # (N, 3)

    def aggregate(self, texts: List[str]) -> np.ndarray:
        """
        Mean-pool FinBERT probabilities across multiple texts.
        Returns (3,) float32: [P_pos, P_neg, P_neu].
        Falls back to uniform [1/3, 1/3, 1/3] if no valid texts.
        """
        valid = [t for t in texts if t and t.strip()]
        if not valid:
            return np.array([1 / 3, 1 / 3, 1 / 3], dtype=np.float32)
        probs = self.score_texts(valid)
        return probs.mean(axis=0).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# KG FEATURE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_kg_7d(triples: List[dict]) -> np.ndarray:
    """
    7D structured features directly from KG triples.
    No text serialization — uses price_impact_score and metadata directly.

    Preserves the quantitative signal that was lost in Voyage serialization (F3 fix).
    """
    if not triples:
        return np.zeros(7, dtype=np.float32)

    weighted_impacts: List[float] = []
    abs_impacts:      List[float] = []
    confidences:      List[float] = []

    for t in triples:
        imp  = float(t.get("price_impact_score",  0.0))
        conf = float(t.get("confidence",           0.0))
        rel  = float(t.get("relevance_to_ticker",  0.0))
        weighted_impacts.append(imp * conf * rel)
        abs_impacts.append(abs(imp))
        confidences.append(conf)

    net_impact     = float(np.mean(weighted_impacts))
    max_abs_impact = float(np.max(abs_impacts))
    n_norm         = float(min(len(triples) / 10.0, 1.0))
    avg_conf       = float(np.mean(confidences))

    obj_names      = [t.get("object", {}).get("name", "").lower() for t in triples]
    has_earnings   = float(any(
        any(kw in name for kw in _EARNINGS_KW) for name in obj_names
    ))
    has_regulatory = float(any(t.get("relation", "") == "REGULATES" for t in triples))
    has_guidance   = float(any(
        any(kw in name for kw in _GUIDANCE_KW) for name in obj_names
    ))

    return np.array(
        [net_impact, max_abs_impact, n_norm,
         has_earnings, has_regulatory, has_guidance, avg_conf],
        dtype=np.float32,
    )


def compute_flags_3d(n_articles: int, net_impact: float) -> np.ndarray:
    """
    3D contextual flags.

    has_news is always 1.0 here because we only call this function when
    there ARE articles (no-article days return None → zero vector → mask=True).
    """
    n_norm = float(min(n_articles / 5.0, 1.0))

    # Map KG net_impact to sentiment sign: positive=1.0, neutral=0.5, negative=0.0
    if   net_impact >  0.05:
        sent = 1.0
    elif net_impact < -0.05:
        sent = 0.0
    else:
        sent = 0.5

    return np.array([1.0, n_norm, sent], dtype=np.float32)


def build_13d_vector(
    finbert_3d: np.ndarray,
    kg_7d:      np.ndarray,
    flags_3d:   np.ndarray,
) -> List[float]:
    """Concatenate 3+7+3=13D and return as Python list for JSON."""
    return np.concatenate([finbert_3d, kg_7d, flags_3d]).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# DATAFRAME NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize news DataFrame.
    Ensures columns: date (date), equity (str), title, summary, content, primary_ticker.
    """
    df = df.copy()

    col_map: Dict[str, str] = {}
    if "headline" in df.columns and "title"  not in df.columns:
        col_map["headline"] = "title"
    if "ticker"   in df.columns and "equity" not in df.columns:
        col_map["ticker"]   = "equity"
    if col_map:
        df = df.rename(columns=col_map)

    for alt in ("body", "text"):
        if "content" not in df.columns and alt in df.columns:
            df = df.rename(columns={alt: "content"})

    for col in ("title", "summary", "content"):
        if col not in df.columns:
            df[col] = ""

    # Auto-detect date column
    if "date" not in df.columns:
        dc = next((c for c in _DATE_CANDS if c in df.columns), None)
        if dc is None:
            dc = next(
                (c for c in df.columns
                 if any(k in c.lower() for k in ("date", "time", "publish", "creat"))),
                None,
            )
        if dc is None:
            raise ValueError(f"No date column. Available: {list(df.columns)}")
        df = df.rename(columns={dc: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    ticker_col = next((c for c in ("symbols", "equity") if c in df.columns), None)
    if ticker_col is None:
        raise ValueError("No ticker column found ('symbols' or 'equity').")

    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)
    df = df[df["_all_tickers"].map(len) > 0]

    df["primary_ticker"] = df.apply(
        lambda row: detect_primary_ticker(
            str(row.get("title",   "") or ""),
            str(row.get("content", "") or ""),
            row["_all_tickers"],
        ),
        axis=1,
    )

    df = df.explode("_all_tickers").rename(columns={"_all_tickers": "equity"})
    df = df[df["equity"].notna() & (df["equity"] != "")].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EMBED LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def run_embed_finbert(
    news_df:        pd.DataFrame,
    cache_dir:      str,
    output_path:    str,
    min_relevance:  Optional[float] = None,
    min_confidence: Optional[float] = None,
    ticker_filter:  Optional[str]   = None,
    date_prefix:    Optional[str]   = None,
    finbert_model:  str             = FINBERT_MODEL_NAME,
    batch_size:     int             = 32,
) -> str:

    if min_relevance  is None: min_relevance  = GlobalConfig.KG_MIN_RELEVANCE
    if min_confidence is None: min_confidence = GlobalConfig.KG_MIN_CONFIDENCE

    # ── Normalize DataFrame ────────────────────────────────────────────────
    df = _normalize_df(news_df)
    if ticker_filter:
        df = df[df["equity"] == ticker_filter.upper()]
    if date_prefix:
        df = df[df["date"].astype(str).str.startswith(date_prefix)]
    if len(df) == 0:
        print("No data after filters.")
        return output_path

    tickers = sorted(df["equity"].unique())
    print(f"\nFinBERT 13D embedding: {len(tickers)} tickers, "
          f"{df['date'].nunique()} unique dates")
    print(f"Output: {output_path}")
    if ticker_filter:
        print(f"Ticker filter: {ticker_filter.upper()} — will MERGE into existing output")
    print()

    # ── Load KG cache (same logic as embed_news.py) ────────────────────────
    cache_store = CacheStore(cache_dir)
    cache_store.load()

    # ── Load FinBERT ───────────────────────────────────────────────────────
    scorer = FinBERTScorer(model_name=finbert_model, batch_size=batch_size)
    print()

    # ── Merge-on-write: load existing output when filtering by ticker ──────
    existing: Dict[str, Dict] = {}
    if ticker_filter:
        existing = _load_existing_output(output_path)
        if existing:
            n_existing = len(set(t for d in existing.values() for t in d))
            print(f"Loaded existing output: {len(existing)} dates, ~{n_existing} tickers")

    # ── Compute 13D vectors ────────────────────────────────────────────────
    new_output: Dict[str, Dict[str, List[float]]] = defaultdict(dict)

    for ticker in tickers:
        df_t  = df[df["equity"] == ticker].copy()
        dates = sorted(df_t["date"].unique())
        n_with_news = 0

        for d in dates:
            date_str = str(d)
            day_df   = df_t[df_t["date"] == d]

            # Collect texts for FinBERT (title preferred; fallback to summary)
            texts: List[str] = []
            for _, row in day_df.iterrows():
                title   = str(row.get("title",   "") or "").strip()
                summary = str(row.get("summary", "") or "").strip()
                text    = title if title else summary
                if text:
                    texts.append(text)

            n_articles = len(texts)
            if n_articles == 0:
                # No article text → skip.
                # data_loader fills with zeros → news_mask[t,w]=True → excluded from attention.
                continue

            # KG triples for this (date, ticker) — per-ticker, not shared across tickers
            triples = cache_store.get_triples_meta(
                date_str, ticker, min_relevance, min_confidence
            )
            if not triples:
                # Fallback to SHA1-based lookup for old-format cache files
                triples = cache_store.get_triples_sha1(
                    day_df, ticker, min_relevance, min_confidence
                )
            triples = _dedup_triples(triples)

            # ── Feature extraction ─────────────────────────────────────────
            finbert_3d = scorer.aggregate(texts)                       # (3,)
            kg_7d      = compute_kg_7d(triples)                        # (7,)
            net_impact = float(kg_7d[0])
            flags_3d   = compute_flags_3d(n_articles, net_impact)      # (3,)

            new_output[date_str][ticker] = build_13d_vector(
                finbert_3d, kg_7d, flags_3d
            )
            n_with_news += 1

        print(f"  {ticker}: {n_with_news}/{len(dates)} dates have news  "
              f"({len(dates) - n_with_news} no-news days → zero/masked)")

    # ── Merge: existing other-ticker data + new ticker data ───────────────
    if ticker_filter and existing:
        for date_str, tickers_dict in existing.items():
            for t, vec in tickers_dict.items():
                if t not in new_output.get(date_str, {}):
                    new_output[date_str][t] = vec

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dict(new_output), f, ensure_ascii=False)

    total_pairs = sum(len(v) for v in new_output.values())
    print(f"\nSaved {output_path}")
    print(f"  Dates: {len(new_output)}, (date, ticker) pairs: {total_pairs}")
    if new_output:
        sample_date = next(iter(new_output))
        sample_t    = sorted(new_output[sample_date])[0]
        sample_vec  = new_output[sample_date][sample_t]
        print(f"  Sample [{sample_date}][{sample_t}]: dim={len(sample_vec)}")

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

_FEAT_LABELS = [
    "P_positive", "P_negative", "P_neutral",          # [0:3]  FinBERT
    "net_impact",  "max_abs_impact", "n_triples_norm", # [3:6]  KG
    "has_earnings", "has_regulatory", "has_guidance",  # [6:9]  KG
    "avg_confidence",                                   # [9]    KG
    "has_news", "n_articles_norm", "sentiment_sign",   # [10:13] Flags
]


def check_output(output_path: str):
    print(f"\nOutput: {output_path}")
    if not os.path.exists(output_path):
        print("  File not found.")
        return

    with open(output_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print("  Empty.")
        return

    dates = sorted(data.keys())
    all_tickers: Dict[str, int] = {}
    for d, td in data.items():
        for t in td:
            all_tickers[t] = all_tickers.get(t, 0) + 1

    print(f"  Dates: {len(dates)}  ({dates[0]} → {dates[-1]})")
    print(f"  Tickers:")
    sample_vec = None
    for t, cnt in sorted(all_tickers.items()):
        v = data.get(dates[len(dates)//2], {}).get(t)
        if v:
            sample_vec = (dates[len(dates)//2], t, v)
        dim = len(v) if v else "?"
        print(f"    {t}: {cnt} dates with news, dim={dim}")

    if sample_vec:
        d, t, vec = sample_vec
        print(f"\n  Feature breakdown [{d}][{t}]:")
        for lb, val in zip(_FEAT_LABELS, vec):
            bar = "█" * int(abs(val) * 20) if abs(val) <= 1.0 else "!overflow!"
            print(f"    {lb:20s}: {val:+.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Stage A.5 FinBERT — 13D hybrid news feature extraction"
    )
    ap.add_argument("--news",           default=None,
                    help="Path to news parquet (default: GlobalConfig.INTERIM_PATH/concatenated_news_filtered.parquet)")
    ap.add_argument("--cache-dir",      default=None,
                    help="KG article cache dir (default: GlobalConfig.kg_cache_dir())")
    ap.add_argument("--output",         default=None,
                    help="Output JSON path (default: GlobalConfig.finbert_emb_path())")
    ap.add_argument("--ticker",         default=None,
                    help="Filter to 1 ticker, e.g. TSLA (merge mode: preserves other tickers)")
    ap.add_argument("--date",           default=None,
                    help="Date prefix filter, e.g. '2024' or '2024-03'")
    ap.add_argument("--min-relevance",  type=float, default=None)
    ap.add_argument("--min-confidence", type=float, default=None)
    ap.add_argument("--batch-size",     type=int,   default=32,
                    help="FinBERT inference batch size (default: 32)")
    ap.add_argument("--finbert-model",  default=FINBERT_MODEL_NAME,
                    help="HuggingFace model name (default: ProsusAI/finbert)")
    ap.add_argument("--check-output",   action="store_true",
                    help="Print output stats and exit (no embedding)")
    args = ap.parse_args()

    output_path = args.output or GlobalConfig.finbert_emb_path()

    if args.check_output:
        check_output(output_path)
        return

    news_path = args.news or os.path.join(
        GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet"
    )
    if not os.path.exists(news_path):
        print(f"News file not found: {news_path}")
        sys.exit(1)

    cache_dir = args.cache_dir or GlobalConfig.kg_cache_dir()
    if not os.path.exists(cache_dir):
        print(f"KG cache dir not found: {cache_dir}")
        print("Run extract_corpus.py first.")
        sys.exit(1)

    df = pd.read_parquet(news_path)
    print(f"Loaded {len(df):,} rows from {news_path}")

    run_embed_finbert(
        news_df=df,
        cache_dir=cache_dir,
        output_path=output_path,
        min_relevance=args.min_relevance,
        min_confidence=args.min_confidence,
        ticker_filter=args.ticker,
        date_prefix=args.date,
        finbert_model=args.finbert_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()