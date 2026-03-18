# build_graphs.py
"""
Stage B — Graph Building Only (No LLM, No KMeans)
==================================================
Đọc cache SHA-1 → rolling window → alias-based entity resolution →
Voyage node embeddings → tensorize → lưu .pt files → embedded_kg.json

KHÔNG gọi LLM. KHÔNG KMeans.
Entity resolution chỉ dùng canonical alias dict (deterministic, stable across windows).

Usage:
    python build_graphs.py
    python build_graphs.py --window 10       # thay đổi window size
    python build_graphs.py --ticker TSLA     # chỉ build 1 ticker
    python build_graphs.py --no-voyage       # bỏ qua Voyage (test nhanh)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from configs.config import GlobalConfig, TrainConfig

from data_pipeline.kg.extractor_batch import (
    rescore_triples_for_ticker,
    dedup_triples,
    build_combined_text,
    detect_primary_ticker,
    _sha1,
    _norm,
    _parse_tickers,
    TICKER_NAME_MAP,
)
from encoders.kg_graph_encoder import (
    KGGraphEncoderGATv2,
    build_node_info,
    build_node_features,
    build_rich_edge_data,
    NODE_FEATURE_DIM,
    EDGE_ATTR_DIM,
)


# ─────────────────────────────────────────────────────────────────────────────
# CANONICAL ALIAS DICTIONARY  (thay thế KMeans — deterministic)
# ─────────────────────────────────────────────────────────────────────────────

CANONICAL_ALIASES: Dict[str, str] = {
    # Các biến thể thường gặp → canonical name
    "apple inc":               "Apple",
    "apple inc.":              "Apple",
    "aapl":                    "Apple",
    "microsoft corporation":   "Microsoft",
    "microsoft corp":          "Microsoft",
    "msft":                    "Microsoft",
    "amazon.com":              "Amazon",
    "amazon.com inc":          "Amazon",
    "amazon.com, inc.":        "Amazon",
    "amzn":                    "Amazon",
    "alphabet inc":            "Alphabet",
    "alphabet inc.":           "Alphabet",
    "googl":                   "Alphabet",
    "goog":                    "Alphabet",
    "meta platforms":          "Meta",
    "meta platforms inc":      "Meta",
    "facebook":                "Meta",
    "tesla inc":               "Tesla",
    "tesla inc.":              "Tesla",
    "tsla":                    "Tesla",
    "jpmorgan chase":          "JPMorgan",
    "jpmorgan chase & co":     "JPMorgan",
    "jp morgan":               "JPMorgan",
    "j.p. morgan":             "JPMorgan",
    "jpm":                     "JPMorgan",
    "federal reserve":         "Federal Reserve",
    "fed":                     "Federal Reserve",
    "u.s. federal reserve":    "Federal Reserve",
    "the fed":                 "Federal Reserve",
    "u.s. fed":                "Federal Reserve",
    "boeing company":          "Boeing",
    "the boeing company":      "Boeing",
    "walmart inc":             "Walmart",
    "walmart inc.":            "Walmart",
    "netflix inc":             "Netflix",
    "netflix inc.":            "Netflix",
    "nvidia corporation":      "Nvidia",
    "nvidia corp":             "Nvidia",
    "nvda":                    "Nvidia",
    "intel corporation":       "Intel",
    "intel corp":              "Intel",
    "intc":                    "Intel",
    "advanced micro devices":  "AMD",
    "rivian automotive":       "Rivian",
    "catl":                    "CATL",
    "contemporary amperex":    "CATL",
    "sec":                     "SEC",
    "u.s. sec":                "SEC",
    "securities and exchange commission": "SEC",
    "ftc":                     "FTC",
    "federal trade commission":"FTC",
    "department of justice":   "DOJ",
    "u.s. doj":                "DOJ",
    "white house":             "White House",
    "u.s. government":         "U.S. Government",
}


def resolve_entity_name(name: str) -> str:
    """
    Chuẩn hoá entity name:
    1. Loại bỏ suffix pháp lý (Inc., Corp., Ltd., ...)
    2. Lookup CANONICAL_ALIASES (lowercase key)
    3. Trả về canonical hoặc tên gốc nếu không match
    """
    if not name or not isinstance(name, str):
        return name

    # Loại bỏ trailing legal suffixes
    cleaned = re.sub(
        r",?\s*(Inc\.?|Corp\.?|Corporation|Co\.?|Company|Ltd\.?|Limited|Group|"
        r"Holdings?|Partners?|LLC|LLP|PLC|N\.V\.|S\.A\.)\.?$",
        "", name, flags=re.IGNORECASE
    ).strip()

    key = cleaned.lower().strip()
    if key in CANONICAL_ALIASES:
        return CANONICAL_ALIASES[key]

    # Thử tên gốc trước khi clean
    key_orig = name.lower().strip()
    if key_orig in CANONICAL_ALIASES:
        return CANONICAL_ALIASES[key_orig]

    return cleaned if cleaned else name


import re  # needed for resolve_entity_name


def apply_entity_resolution(triples: List[Dict]) -> List[Dict]:
    """
    Áp dụng alias resolution cho tất cả entity names trong triples.
    Sau đó dedup theo (subject.name, relation, object.name).
    """
    resolved = []
    for t in triples:
        t2 = dict(t)
        t2["subject"] = dict(t.get("subject", {}))
        t2["object"]  = dict(t.get("object",  {}))
        t2["subject"]["name"] = resolve_entity_name(t2["subject"].get("name", ""))
        t2["object"]["name"]  = resolve_entity_name(t2["object"].get("name",  ""))
        # Bỏ self-loops sau resolution
        if t2["subject"]["name"] != t2["object"]["name"]:
            resolved.append(t2)
    return dedup_triples(resolved)


# ─────────────────────────────────────────────────────────────────────────────
# CACHE READING
# ─────────────────────────────────────────────────────────────────────────────

def _load_cache(cache_dir: str, sha1: str) -> Optional[List[Dict]]:
    p = os.path.join(cache_dir, f"{sha1}.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f).get("triples", [])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# VOYAGE EMBEDDER (inline — không cần VoyageEmbedder class từ news_processor)
# ─────────────────────────────────────────────────────────────────────────────

class _SimpleVoyageEmbedder:
    """Minimal Voyage embedder với disk cache cho Stage B."""

    def __init__(self, cache_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.cache_dir = cache_dir
        if enabled:
            import requests
            self._requests = requests
            self.api_key = os.getenv("VOYAGE_API_KEY", GlobalConfig.VOYAGE_API_KEY)
            if not self.api_key or self.api_key in ("", "---"):
                raise RuntimeError("VOYAGE_API_KEY not set.")
            os.makedirs(cache_dir, exist_ok=True)
        self.model = getattr(GlobalConfig, "EMBED_MODEL", "voyage-3-large")

    def _cache_path(self, text: str) -> str:
        return os.path.join(self.cache_dir, f"{_sha1(text)}.json")

    def _load(self, text: str) -> Optional[List[float]]:
        p = self._cache_path(text)
        if os.path.exists(p):
            try:
                with open(p) as f:
                    return json.load(f).get("embedding")
            except Exception:
                return None
        return None

    def _save(self, text: str, emb: List[float]):
        with open(self._cache_path(text), "w") as f:
            json.dump({"embedding": emb}, f)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not self.enabled:
            return [[0.0] * 1024 for _ in texts]

        out    = [None] * len(texts)
        miss_i = []
        miss_t = []
        for i, t in enumerate(texts):
            t = (t or "").strip()
            if not t:
                out[i] = [0.0] * 1024
                continue
            cached = self._load(t)
            if cached is not None:
                out[i] = cached
            else:
                miss_i.append(i)
                miss_t.append(t)

        if not miss_t:
            return [o if o is not None else [0.0]*1024 for o in out]

        # Batch call (up to 40 per request)
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        chunk_size = getattr(GlobalConfig, "MAX_TEXTS_PER_REQ", 40)

        for batch_start in range(0, len(miss_t), chunk_size):
            batch_texts = miss_t[batch_start: batch_start + chunk_size]
            batch_idx   = miss_i[batch_start: batch_start + chunk_size]
            payload     = {"model": self.model, "input": batch_texts}
            for attempt in range(6):
                try:
                    r = self._requests.post(
                        "https://api.voyageai.com/v1/embeddings",
                        headers=headers, json=payload, timeout=(15, 120)
                    )
                    if r.status_code == 429:
                        import time; time.sleep(30 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    embs = r.json().get("data", [])
                    for bi, item in enumerate(embs):
                        emb = item.get("embedding", [])
                        idx = batch_idx[bi]
                        out[idx] = emb
                        self._save(miss_t[batch_start + bi], emb)
                    break
                except Exception as e:
                    import time; time.sleep(30 * (attempt + 1))
                    if attempt == 5:
                        print(f"  Voyage embed failed: {e}")
                        for idx in batch_idx:
                            out[idx] = [0.0] * 1024
            import time; time.sleep(1.0)  # rate limit

        return [o if o is not None else [0.0]*1024 for o in out]


# ─────────────────────────────────────────────────────────────────────────────
# TENSORIZE (Stage B — saves node_x, edge_index, edge_attr — NO pre-encoded emb)
# ─────────────────────────────────────────────────────────────────────────────

def tensorize(
    ticker:       str,
    date_str:     str,
    rich_triples: List[Dict],
    voyage:       _SimpleVoyageEmbedder,
    tensors_dir:  str,
) -> str:
    """
    Tạo .pt file với node_x(1033D), edge_index, edge_attr(17D).
    KHÔNG pre-encode GATv2 — model.py sẽ encode lúc training.
    """
    out_dir = os.path.join(tensors_dir, ticker)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{date_str}.pt")

    if not rich_triples:
        torch.save({
            "node_x":     torch.zeros(0, NODE_FEATURE_DIM),
            "edge_index": torch.zeros(2, 0, dtype=torch.long),
            "edge_attr":  torch.zeros(0, EDGE_ATTR_DIM),
            "ticker_idx": 0,
            "nodes":      [],
            "node_info":  {},
        }, out_path)
        return out_path

    node_info             = build_node_info(rich_triples)
    nodes, node2id, x     = build_node_features(node_info, voyage, ticker)
    edge_index, edge_attr = build_rich_edge_data(rich_triples, node2id)

    torch.save({
        "node_x":     x,
        "edge_index": edge_index,
        "edge_attr":  edge_attr,
        "ticker_idx": node2id.get(ticker, 0),
        "nodes":      nodes,
        "node_info":  node_info,
    }, out_path)
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING WINDOW AGGREGATE
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_window(window_days: List[List[Dict]]) -> List[Dict]:
    """Gộp triples từ rolling window, dedup theo (subj, rel, obj)."""
    seen, result = set(), []
    for day in window_days:
        for t in day:
            key = (
                t.get("subject", {}).get("name", ""),
                t.get("relation", ""),
                t.get("object",  {}).get("name", ""),
            )
            if key not in seen:
                seen.add(key)
                result.append(t)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# STAGE B MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_stage_b(
    news_df:      pd.DataFrame,
    cache_dir:    str,
    interim_root: str,
    window_days:  int   = 20,
    min_relevance:  float = 0.30,
    min_confidence: float = 0.35,
    use_voyage:   bool  = True,
    ticker_filter: Optional[str] = None,
) -> str:
    """
    Stage B: Đọc cache → rolling window → alias resolution → tensorize.

    Returns:
        Path to embedded_kg.json index file.
    """
    from data_pipeline.processors.news_processor import normalize_news_df as _norm_df

    # Normalize
    df = _norm_df(news_df)
    if "content" not in df.columns and "text" in df.columns:
        df = df.rename(columns={"text": "content"})
    df["content"] = df.get("content", pd.Series(dtype=str)).fillna("")
    df["title"]   = df.get("title",   pd.Series(dtype=str)).fillna("")

    if ticker_filter:
        df = df[df["equity"] == ticker_filter.upper()]

    # Directories
    kg_dir      = os.path.join(interim_root, "kg")
    tensors_dir = os.path.join(kg_dir, "tensors")
    emb_dir     = os.path.join(interim_root, "kg_embeddings")
    for d in [tensors_dir, emb_dir]:
        os.makedirs(d, exist_ok=True)

    # Voyage embedder
    voyage_cache = GlobalConfig.kg_voyage_cache_dir()
    voyage = _SimpleVoyageEmbedder(cache_dir=voyage_cache, enabled=use_voyage)
    if not use_voyage:
        print("  Voyage disabled — using zero node features (test mode)")

    tickers = sorted(df["equity"].unique())
    print(f"\nStage B: {len(tickers)} tickers, window={window_days}")

    results_json: Dict[str, List[Dict]] = {}
    miss_total = total_articles = 0

    for ticker in tickers:
        df_t = df[df["equity"] == ticker].copy()
        window_triples: List[List[Dict]] = []

        for d in sorted(df_t["date"].unique()):
            date_str = str(d)
            day_df   = df_t[df_t["date"] == d]
            day_triples: List[Dict] = []
            sha1_seen: set = set()

            for _, row in day_df.iterrows():
                total_articles += 1
                title   = _norm(str(row.get("title",   "") or ""))
                content = _norm(str(row.get("content", "") or ""))
                full_text = build_combined_text(
                    [title]   if title   else [],
                    [content] if content else [],
                )
                if not full_text:
                    continue
                h = _sha1(full_text)
                if h in sha1_seen:
                    continue
                sha1_seen.add(h)

                cached = _load_cache(cache_dir, h)
                if cached is None:
                    miss_total += 1
                    continue

                primary  = str(row.get("primary_ticker") or ticker)
                all_t    = list(row.get("_all_tickers") or [primary])

                rescored = rescore_triples_for_ticker(
                    cached,
                    primary_ticker=primary,
                    target_ticker=ticker,
                    min_relevance=min_relevance,
                    article_text=full_text,
                    all_article_tickers=all_t,
                )
                rescored = [t for t in rescored
                            if float(t.get("confidence", 0)) >= min_confidence]
                day_triples.extend(rescored)

            day_triples = dedup_triples(day_triples)

            # Rolling window
            window_triples.append(day_triples)
            if len(window_triples) > window_days:
                window_triples.pop(0)

            window_graph = aggregate_window(window_triples)

            # Entity resolution (alias dict, deterministic)
            stable = apply_entity_resolution(window_graph)

            # Tensorize — NO pre-encoded GATv2 embedding
            kg_path = tensorize(ticker, date_str, stable, voyage, tensors_dir)

            results_json.setdefault(date_str, []).append({
                "date":           date_str,
                "equity":         ticker,
                "kg_tensor_path": kg_path,
            })

    print(f"\nStage B complete. Cache-hit: {total_articles - miss_total}/{total_articles}")
    if miss_total > 0:
        print(f"  {miss_total} articles had no cache entry → empty graph for those days")
        print(f"  Run Stage A first to fill cache: python extract_corpus.py")

    # Save index
    out_path = os.path.join(emb_dir, "embedded_kg.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False)
    print(f"  KG index: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Stage B — Graph Building Only")
    parser.add_argument("--window",     type=int,   default=20)
    parser.add_argument("--ticker",     default=None)
    parser.add_argument("--no-voyage",  action="store_true")
    parser.add_argument("--min-relevance",  type=float, default=GlobalConfig.KG_MIN_RELEVANCE)
    parser.add_argument("--min-confidence", type=float, default=GlobalConfig.KG_MIN_CONFIDENCE)
    parser.add_argument("--news",       default=None)
    args = parser.parse_args()

    news_path = args.news or os.path.join(
        GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet"
    )
    if not os.path.exists(news_path):
        print(f"News file not found: {news_path}")
        sys.exit(1)

    cache_dir = GlobalConfig.kg_cache_dir()
    if not os.path.exists(cache_dir) or not os.listdir(cache_dir):
        print(f"Cache dir empty: {cache_dir}")
        print("Run Stage A first: python extract_corpus.py")
        sys.exit(1)

    df = pd.read_parquet(news_path)
    print(f"Loaded {len(df):,} rows from {news_path}")

    run_stage_b(
        news_df=df,
        cache_dir=cache_dir,
        interim_root=GlobalConfig.INTERIM_PATH,
        window_days=args.window,
        min_relevance=args.min_relevance,
        min_confidence=args.min_confidence,
        use_voyage=not args.no_voyage,
        ticker_filter=args.ticker,
    )


if __name__ == "__main__":
    main()