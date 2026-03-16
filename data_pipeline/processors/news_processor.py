# data_pipeline/processors/news_processor.py
"""
V3: Tách biệt extraction (Stage A) và graph building (Stage B).

Các thay đổi chính:
  - normalize_news_df() exported ở module level (dùng bởi cả Stage A và B)
  - KGGenNewsEmbedder: bỏ KMeans, bỏ pre-encode GATv2, giữ compat cho main_test.py
  - process_and_save() → gọi extract_corpus.run_stage_a() + build_graphs.run_stage_b()
  - rebuild_graph_only() → gọi build_graphs.run_stage_b()
  - NewsProcessor.align_to_trading_days() giữ nguyên

Fixes từ prompts.py v2 (applied vào V3):
  - KGGenNewsEmbedder: min_relevance default 0.30 → 0.50
  - KGGenNewsEmbedder: min_confidence default 0.35 → 0.65
"""

import os
import re
import json
import time
import random
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import requests

from configs.config import GlobalConfig

from data_pipeline.kg.extractor import (
    FinDKGLiteExtractor,
    upgrade_legacy_triple,
)
from data_pipeline.kg.extractor_batch import (
    AsyncConcurrentExtractor,
    GeminiBatchAPIExtractor,
    rescore_triples_for_ticker,
    build_combined_text,
    detect_primary_ticker,
    dedup_triples,
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

RichTriple = Dict[str, Any]


# ─────────────────────────────────────────────────────────────────────────────
# normalize_news_df — module-level (exported, used by Stage A and B)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_news_df(df: pd.DataFrame, symbols_col: str = "symbols") -> pd.DataFrame:
    """
    Chuẩn hoá DataFrame đầu vào.

    - Rename headline→title, ticker→equity nếu cần
    - Parse date
    - Parse _all_tickers list (giữ list gốc, trước explode)
    - detect_primary_ticker dùng title weight × 3
    - Explode 1 row per ticker, _all_tickers column vẫn giữ full list

    Fix lỗi 2: sau explode, _all_tickers được recreate từ cột symbols gốc
    để rescore_triples_for_ticker nhận đúng all_article_tickers.
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

    # Text fallbacks
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

    # Detect ticker column
    ticker_col = None
    for col in (symbols_col, "equity", "ticker"):
        if col in df.columns:
            ticker_col = col
            break
    if ticker_col is None:
        raise ValueError(f"No ticker column. Expected: {symbols_col}, equity, ticker.")

    # Parse tickers → _all_tickers (list)
    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)
    df = df[df["_all_tickers"].map(len) > 0]

    # Primary ticker detection (title weight × 3)
    df["primary_ticker"] = df.apply(
        lambda row: detect_primary_ticker(
            str(row.get("title",   "") or ""),
            str(row.get("content", "") or ""),
            row["_all_tickers"],
        ),
        axis=1,
    )

    # Explode 1 row per ticker
    df = df.explode("_all_tickers").rename(columns={"_all_tickers": "equity"})
    df = df[df["equity"].notna() & (df["equity"] != "")].reset_index(drop=True)

    # Fix lỗi 2: recreate _all_tickers from original symbols column
    # After explode, ticker_col still holds original "AAPL,GOOGL,TSLA" string
    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# NewsProcessor — align helper (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class NewsProcessor:
    def __init__(self):
        pass

    def align_to_trading_days(self, news_input, trading_days):
        if isinstance(news_input, pd.DataFrame):
            df = news_input.copy()
        elif isinstance(news_input, str):
            df = (pd.read_parquet(news_input) if news_input.endswith(".parquet")
                  else pd.read_csv(news_input))
        else:
            raise TypeError(f"Expected DataFrame or path, got {type(news_input)}")

        if "equity" not in df.columns and "ticker" in df.columns:
            df = df.rename(columns={"ticker": "equity"})
        if "title" not in df.columns and "headline" in df.columns:
            df = df.rename(columns={"headline": "title"})
        if "content" not in df.columns:
            for alt in ("body", "text"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "content"})
                    break

        if "date" not in df.columns:
            raise ValueError(f"Missing 'date' column. Has: {list(df.columns)}")
        df["date"] = pd.to_datetime(df["date"]).dt.date

        if trading_days is not None:
            td = set(pd.to_datetime(trading_days).date)
            df = df[df["date"].isin(td)]

        cols = ["date", "equity"]
        for c in ("content", "title"):
            if c in df.columns:
                cols.append(c)
        return df[cols].copy()


# ─────────────────────────────────────────────────────────────────────────────
# VoyageEmbedder (kept for backward compat; Stage B uses _SimpleVoyageEmbedder)
# ─────────────────────────────────────────────────────────────────────────────

class VoyageEmbedder:
    """Rate-limited Voyage AI embedder with disk cache."""

    def __init__(self, cache_dir: str):
        self.api_key = os.getenv("VOYAGE_API_KEY", GlobalConfig.VOYAGE_API_KEY)
        if not self.api_key or self.api_key == "---":
            raise RuntimeError("Missing VOYAGE_API_KEY.")

        self.model        = getattr(GlobalConfig, "EMBED_MODEL", "voyage-3-large")
        self.max_texts    = getattr(GlobalConfig, "MAX_TEXTS_PER_REQ", 40)
        self.max_retries  = getattr(GlobalConfig, "MAX_RETRIES", 6)
        self.backoff_base = getattr(GlobalConfig, "BACKOFF_BASE", 30)

        payment_added = bool(getattr(GlobalConfig, "PAYMENT_ADDED", True))
        rl = GlobalConfig.VOYAGE_RATE_LIMITS[payment_added]
        self.rpm        = int(rl["RPM"])
        self.base_sleep = float(rl["SLEEP"])
        self.cache_dir  = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self._req_times: List[float] = []

    def _cache_path(self, text: str) -> str:
        return os.path.join(self.cache_dir, f"{_sha1(text)}.json")

    def _load_cached(self, text: str) -> Optional[List[float]]:
        p = self._cache_path(text)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                emb = obj.get("embedding")
                if isinstance(emb, list) and emb:
                    return emb
            except Exception:
                pass
        return None

    def _save_cache(self, text: str, emb: List[float]) -> None:
        with open(self._cache_path(text), "w", encoding="utf-8") as f:
            json.dump({"embedding": emb}, f)

    def _rpm_sleep_if_needed(self):
        now = time.time()
        self._req_times = [t for t in self._req_times if now - t < 60.0]
        if len(self._req_times) >= self.rpm:
            wait = max(0.0, 60.0 - (now - min(self._req_times))) + 0.5
            print(f"Voyage RPM limit. Sleep {wait:.1f}s")
            time.sleep(wait)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        texts = [(_norm(t)[:6000] if t else "") for t in texts]
        out: List[Optional[List[float]]] = [None] * len(texts)
        missing_idx, missing_texts = [], []

        for i, t in enumerate(texts):
            if not t.strip():
                out[i] = []
                continue
            cached = self._load_cached(t)
            if cached is not None:
                out[i] = cached
            else:
                missing_idx.append(i)
                missing_texts.append(t)

        if not missing_texts:
            return [o if o is not None else [] for o in out]

        url     = "https://api.voyageai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type":  "application/json"}

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i: i + n]

        pos = 0
        for batch in chunks(missing_texts, self.max_texts):
            batch_idx = missing_idx[pos: pos + len(batch)]
            pos += len(batch)
            payload = {"model": self.model, "input": batch}
            for attempt in range(self.max_retries):
                try:
                    self._rpm_sleep_if_needed()
                    if self.base_sleep > 0:
                        time.sleep(self.base_sleep)
                    r = requests.post(url, headers=headers, json=payload,
                                      timeout=(15, 120))
                    self._req_times.append(time.time())
                    if r.status_code == 429:
                        wait = self.backoff_base * (2 ** attempt) + random.uniform(0, 3)
                        time.sleep(wait)
                        continue
                    r.raise_for_status()
                    embs = r.json().get("data", [])
                    for bi, one in enumerate(embs):
                        emb = one.get("embedding")
                        if not isinstance(emb, list):
                            raise RuntimeError("Invalid Voyage embedding format")
                        idx = batch_idx[bi]
                        out[idx] = emb
                        self._save_cache(texts[idx], emb)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(self.backoff_base * (2 ** attempt) + random.uniform(0, 3))

        return [o if o is not None else [] for o in out]


# ─────────────────────────────────────────────────────────────────────────────
# KGGenNewsEmbedder — backward compat wrapper
# ─────────────────────────────────────────────────────────────────────────────

class KGGenNewsEmbedder:
    """
    Backward-compatible wrapper.

    Internally calls:
      process_and_save()    → Stage A (extract) + Stage B (build)
      rebuild_graph_only()  → Stage B only

    Threshold defaults aligned với prompts.py:
      min_relevance  = 0.50  (was 0.30)
      min_confidence = 0.65  (was 0.35)
    """

    def __init__(
        self,
        interim_root: str = None,
        window_days:  int  = 20,
        min_relevance:  float = 0.50,   # ← was 0.30; aligned với prompt
        min_confidence: float = 0.65,   # ← was 0.35; aligned với prompt
        max_triples_cap_per_day: Optional[int] = None,
        debug_print_samples: bool = False,
        enable_cache:        bool = True,
        cache_dirname:       str  = "kg_article_cache",
        allow_llm_when_missing: bool = False,
        use_voyage_resolution:  bool = True,
        use_voyage_node_features: bool = True,
        voyage_cache_dirname: str = "kg_voyage_emb_cache",
        use_graph_encoder_embedding: bool = False,  # NOW IGNORED — model.py handles encoding
        graph_out_dim:    int   = 128,
        graph_hidden_dim: int   = 128,
        graph_num_layers: int   = 2,
        graph_num_heads:  int   = 4,
        graph_dropout:    float = 0.1,
        symbols_col:        str = "symbols",
        batch_display_name: str = "findkg-lite-v3",
        max_concurrent: int = 5,
        top_triples_per_article: int = 0,
        graph_use_gat: bool = True,
        # New V3 params
        use_gemini_batch: bool = False,
    ):
        if interim_root is None:
            interim_root = os.path.join("data", "interim")
        self.interim_root         = interim_root
        self.window_days          = window_days
        self.min_relevance        = min_relevance
        self.min_confidence       = min_confidence
        self.allow_llm            = allow_llm_when_missing
        self.use_voyage           = use_voyage_node_features
        self.symbols_col          = symbols_col
        self.max_concurrent       = max_concurrent
        self.use_gemini_batch     = use_gemini_batch
        self.cache_dir = os.path.join(interim_root, cache_dirname)
        if enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        if use_graph_encoder_embedding:
            print("  Note: use_graph_encoder_embedding=True is ignored in V3. "
                  "GATv2 encoding happens in model.py during training.")

    def process_and_save(self, news_df: pd.DataFrame,
                          use_gemini_batch: bool = None) -> str:
        """Stage A + Stage B."""
        from extract_corpus import run_stage_a
        from build_graphs   import run_stage_b

        if use_gemini_batch is None:
            use_gemini_batch = self.use_gemini_batch

        if self.allow_llm:
            print("Stage A: LLM extraction...")
            run_stage_a(
                news_df=news_df,
                cache_dir=self.cache_dir,
                use_gemini_batch=use_gemini_batch,
                max_concurrent=self.max_concurrent,
                min_relevance=self.min_relevance,
                min_confidence=self.min_confidence,
            )

        print("Stage B: Graph building...")
        return run_stage_b(
            news_df=news_df,
            cache_dir=self.cache_dir,
            interim_root=self.interim_root,
            window_days=self.window_days,
            min_relevance=self.min_relevance,
            min_confidence=self.min_confidence,
            use_voyage=self.use_voyage,
        )

    def rebuild_graph_only(self) -> str:
        """Stage B only — NO LLM."""
        from build_graphs import run_stage_b

        news_path = os.path.join(self.interim_root, "concatenated_news_filtered.parquet")
        if not os.path.exists(news_path):
            raise FileNotFoundError(f"News parquet not found: {news_path}")
        df = pd.read_parquet(news_path)

        return run_stage_b(
            news_df=df,
            cache_dir=self.cache_dir,
            interim_root=self.interim_root,
            window_days=self.window_days,
            min_relevance=self.min_relevance,
            min_confidence=self.min_confidence,
            use_voyage=self.use_voyage,
        )