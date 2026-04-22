#!/usr/bin/env python3
# extract_kg_features.py  — Bước 2
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import GlobalConfig
from configs.ticker_aliases import TICKER_ALIASES


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "net_impact",
    "max_abs_impact",
    "n_triples_norm",
    "has_earnings",
    "has_regulatory",
    "has_guidance",
    "avg_confidence",
]
FEATURE_DIM = len(FEATURE_NAMES)  # 7

# Relation types để detect event category
EARNINGS_RELATIONS = {"ANNOUNCES", "RAISES", "CUTS"}
EARNINGS_KEYWORDS  = {"eps", "revenue", "earnings", "profit", "margin", "q1", "q2", "q3", "q4",
                       "quarter", "annual", "guidance", "beat", "miss", "ebit", "ebitda",
                       "sales", "income", "loss"}
REGULATORY_RELATIONS = {"REGULATES"}
GUIDANCE_RELATIONS   = {"SIGNALS", "RAISES", "CUTS"}
GUIDANCE_KEYWORDS    = {"guidance", "outlook", "forecast", "target", "expect",
                         "project", "estimate", "forward", "next year", "fiscal"}


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _is_earnings_triple(triple: dict) -> bool:
    """Detect earnings/financial-results triple."""
    rel = triple.get("relation", "")
    if rel not in EARNINGS_RELATIONS:
        return False
    obj_name = triple.get("object", {}).get("name", "").lower()
    obj_type = triple.get("object", {}).get("type", "")
    if obj_type == "ECON_IND":
        return True
    return any(kw in obj_name for kw in EARNINGS_KEYWORDS)


def _is_regulatory_triple(triple: dict) -> bool:
    """Detect regulatory action triple."""
    return triple.get("relation", "") in REGULATORY_RELATIONS


def _is_guidance_triple(triple: dict) -> bool:
    """Detect forward-looking guidance triple."""
    rel      = triple.get("relation", "")
    obj_name = triple.get("object", {}).get("name", "").lower()
    if rel == "SIGNALS":
        return True
    if rel in GUIDANCE_RELATIONS:
        return any(kw in obj_name for kw in GUIDANCE_KEYWORDS)
    return False


def compute_features(
    triples:        List[dict],
    min_relevance:  float,
    min_confidence: float,
) -> List[float]:
    """
    Tính 7 scalar features từ danh sách triples của một (date, ticker).
    Trả về zero vector nếu không có triple hợp lệ.
    """
    # Filter theo threshold
    valid = [
        t for t in triples
        if float(t.get("relevance_to_ticker", 0)) >= min_relevance
        and float(t.get("confidence", 0))          >= min_confidence
    ]

    if not valid:
        return [0.0] * FEATURE_DIM

    # Feature 0: net_impact (tín hiệu hướng tổng hợp)
    net_impact = sum(
        float(t["price_impact_score"]) * float(t["confidence"]) * float(t["relevance_to_ticker"])
        for t in valid
    ) / len(valid)

    # Feature 1: max_abs_impact (event lớn nhất)
    max_abs_impact = max(abs(float(t["price_impact_score"])) for t in valid)

    # Feature 2: n_triples_norm (log-scaled density)
    n_triples_norm = math.log1p(len(valid)) / math.log1p(15)  # normalize vs 15 triples

    # Feature 3: has_earnings
    has_earnings = 1.0 if any(_is_earnings_triple(t) for t in valid) else 0.0

    # Feature 4: has_regulatory
    has_regulatory = 1.0 if any(_is_regulatory_triple(t) for t in valid) else 0.0

    # Feature 5: has_guidance
    has_guidance = 1.0 if any(_is_guidance_triple(t) for t in valid) else 0.0

    # Feature 6: avg_confidence (chất lượng trung bình)
    avg_confidence = sum(float(t["confidence"]) for t in valid) / len(valid)

    return [
        float(net_impact),
        float(max_abs_impact),
        float(n_triples_norm),
        float(has_earnings),
        float(has_regulatory),
        float(has_guidance),
        float(avg_confidence),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# CACHE READING — hỗ trợ cả new-format (_meta) và old-format (sha1-only)
# Tái sử dụng logic từ embed_news.py nhưng đơn giản hơn vì không cần SHA1
# ─────────────────────────────────────────────────────────────────────────────

def load_cache_by_meta(
    cache_dir:      str,
    min_relevance:  float,
    min_confidence: float,
    ticker_filter:  Optional[str] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Scan cache directory, đọc tất cả file có _meta.
    Trả về {date_str: {ticker: [7 floats]}}.
    """
    if not os.path.exists(cache_dir):
        print(f"Cache dir không tồn tại: {cache_dir}")
        return {}

    all_tickers = set(TICKER_ALIASES.keys())
    files = [f for f in os.listdir(cache_dir)
             if f.endswith(".json") and not f.startswith("_")]

    # Gom triples per (date, ticker)
    # date_ticker_triples[date][ticker] = [triple, triple, ...]
    date_ticker_triples: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))

    n_processed = 0
    n_skipped   = 0

    for fname in files:
        path = os.path.join(cache_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            n_skipped += 1
            continue

        triples = data.get("triples", [])
        meta    = data.get("_meta", {})
        date_val    = meta.get("date")
        primary_tk  = str(meta.get("primary_ticker", "")).upper()

        if not date_val or not triples:
            n_skipped += 1
            continue

        date_str = str(date_val)[:10]

        # Gom cho primary ticker
        if primary_tk in all_tickers:
            if ticker_filter is None or primary_tk == ticker_filter.upper():
                date_ticker_triples[date_str][primary_tk].extend(triples)

        # Cross-ticker: tìm ticker được mention trong triple
        # (tái sử dụng logic từ embed_news.py::_ticker_mentioned_in_triple)
        for ticker in all_tickers:
            if ticker == primary_tk:
                continue
            if ticker_filter and ticker != ticker_filter.upper():
                continue
            # Check nếu ticker được mention
            ticker_lower = ticker.lower()
            from configs.ticker_aliases import TICKER_ALIASES as _TA
            aliases = [a.lower() for a in _TA.get(ticker, [ticker])]
            for t in triples:
                sn = t.get("subject", {}).get("name", "").lower()
                on = t.get("object",  {}).get("name", "").lower()
                if any(al in sn or al in on for al in aliases):
                    date_ticker_triples[date_str][ticker].append(t)
                    break

        n_processed += 1

    print(f"  Cache: {n_processed} files với _meta, {n_skipped} skipped")

    # Tính features per (date, ticker)
    result: Dict[str, Dict[str, List[float]]] = {}
    n_pairs = 0

    for date_str, ticker_triples in date_ticker_triples.items():
        result[date_str] = {}
        for ticker, triples_list in ticker_triples.items():
            feats = compute_features(triples_list, min_relevance, min_confidence)
            # Chỉ lưu nếu có ít nhất 1 non-zero feature (tránh lãng phí bộ nhớ)
            if any(f != 0.0 for f in feats):
                result[date_str][ticker] = feats
                n_pairs += 1

    print(f"  Pairs có features: {n_pairs} (date, ticker)")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def check_features(output_path: str):
    """In thống kê về kg_features.json hiện tại."""
    print(f"\nOutput: {output_path}")
    if not os.path.exists(output_path):
        print("  File không tồn tại. Chạy extract_kg_features.py trước.")
        return

    with open(output_path, "r") as f:
        data = json.load(f)

    dates = sorted(data.keys())
    print(f"  Tổng số ngày: {len(dates)}")
    if dates:
        print(f"  Range: {dates[0]} → {dates[-1]}")

    ticker_counts: Dict[str, int] = {}
    feature_sums  = [0.0] * FEATURE_DIM
    n_pairs = 0

    for d, tk_dict in data.items():
        for tk, feats in tk_dict.items():
            ticker_counts[tk] = ticker_counts.get(tk, 0) + 1
            for i, v in enumerate(feats):
                feature_sums[i] += v
            n_pairs += 1

    print(f"  Tổng pairs: {n_pairs}")
    print(f"\n  Tickers:")
    for tk, cnt in sorted(ticker_counts.items()):
        coverage = cnt / max(len(dates), 1) * 100
        print(f"    {tk}: {cnt} ngày ({coverage:.1f}% coverage)")

    print(f"\n  Feature means (trung bình trên toàn bộ pairs):")
    for i, name in enumerate(FEATURE_NAMES):
        mean_val = feature_sums[i] / max(n_pairs, 1)
        print(f"    [{i}] {name:<20}: {mean_val:.4f}")

    # Phân tích sparsity (bao nhiêu pair có net_impact=0)
    zero_pairs = sum(
        1 for d, tk_dict in data.items()
        for tk, feats in tk_dict.items()
        if feats[0] == 0.0  # net_impact
    )
    print(f"\n  Pairs với net_impact=0: {zero_pairs}/{n_pairs} "
          f"({100*zero_pairs/max(n_pairs,1):.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bước 2: Extract 7D structured KG features từ triple cache"
    )
    parser.add_argument("--cache-dir",      default=None)
    parser.add_argument("--output",         default=None)
    parser.add_argument("--min-relevance",  type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=None)
    parser.add_argument("--ticker",         default=None, help="Filter 1 ticker, e.g. TSLA")
    parser.add_argument("--check",          action="store_true",
                        help="Chỉ kiểm tra output hiện tại, không extract")
    args = parser.parse_args()

    cache_dir = args.cache_dir or GlobalConfig.kg_cache_dir()
    output_path = args.output or os.path.join(
        GlobalConfig.INTERIM_PATH, "kg_embeddings", "kg_features.json"
    )

    if args.check:
        check_features(output_path)
        return

    min_rel  = args.min_relevance  or GlobalConfig.KG_MIN_RELEVANCE
    min_conf = args.min_confidence or GlobalConfig.KG_MIN_CONFIDENCE

    print(f"\n=== Extract KG Structured Features ===")
    print(f"Cache dir  : {cache_dir}")
    print(f"Output     : {output_path}")
    print(f"Thresholds : min_relevance={min_rel}, min_confidence={min_conf}")
    print(f"Ticker     : {args.ticker or 'all'}")
    print(f"Feature dim: {FEATURE_DIM} ({FEATURE_NAMES})")
    print()

    features = load_cache_by_meta(
        cache_dir=cache_dir,
        min_relevance=min_rel,
        min_confidence=min_conf,
        ticker_filter=args.ticker,
    )

    if not features:
        print("\nKhông có features nào được extract.")
        print("Kiểm tra cache_dir hoặc chạy extract_corpus.py trước.")
        return

    # Merge với file cũ nếu dùng ticker filter
    if args.ticker and os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing = json.load(f)
        for date_str, tk_dict in features.items():
            if date_str not in existing:
                existing[date_str] = {}
            existing[date_str].update(tk_dict)
        features = existing
        print(f"  Merged với existing file")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False)

    print(f"\nSaved: {output_path}")
    print(f"  Dates: {len(features)}")
    if features:
        sample_date = next(iter(features))
        sample_tickers = sorted(features[sample_date].keys())
        print(f"  Sample ({sample_date}): {sample_tickers}")
        print(f"\nFeature vector example ({sample_date}, {sample_tickers[0]}):")
        for i, (name, val) in enumerate(
            zip(FEATURE_NAMES, features[sample_date][sample_tickers[0]])
        ):
            print(f"  [{i}] {name:<20}: {val:.4f}")

    print(f"\nBước tiếp theo: chạy main_test.py để rebuild unified_dataset.pkl")
    print(f"  (DatasetBuilder đã tự động load kg_features.json nếu tồn tại)")


if __name__ == "__main__":
    main()