# data_pipeline/processors/news_processor.py
"""
KGGenNewsEmbedder V2 — Batch API + Multi-Ticker Support

THAY ĐỔI SO VỚI PHIÊN BẢN CŨ:
1. Multi-ticker: cột symbols="AAPL,GOOGL,TSLA" → explode → 1 ticker/row
2. Batch extraction: GeminiBatchAPIExtractor (50% cost) thay vì sequential
3. Corpus-level dedup: 1 article text = 1 LLM call, dù nó xuất hiện nhiều tickers
4. Fan-out: broadcast kết quả extraction sang tất cả tickers liên quan
5. SDK: extractor.py đã migrate sang google-genai

KHÔNG THAY ĐỔI:
- NewsProcessor.align_to_trading_days()
- VoyageEmbedder
- KMeans entity resolution
- rebuild_graph_only() — graph-only rebuild không cần LLM
- Tất cả path/directory structure
- API surface: process_and_save(), rebuild_graph_only()
"""

import os
import re
import json
import time
import random
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import requests
from sklearn.cluster import KMeans

from configs.config import GlobalConfig

from data_pipeline.kg.extractor import (
    FinDKGLiteExtractor,
    upgrade_legacy_triple,
    upgrade_legacy_cache_file,
)
from data_pipeline.kg.extractor_batch import (
    GeminiBatchAPIExtractor,
    rescore_triples_for_ticker,
    build_user_prompt,
)
from data_pipeline.kg.prompts import TICKER_SECTOR_MAP
from encoders.kg_graph_encoder import (
    KGGraphEncoderGATv2,
    build_node_info,
    build_node_features,
    build_rich_edge_data,
    NODE_FEATURE_DIM,
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ─────────────────────────────────────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────────────────────────────────────

RichTriple = Dict[str, Any]


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _parse_tickers(val: Any) -> List[str]:
    """
    Parse ticker value thành list.
    Hỗ trợ: "AAPL,GOOGL,TSLA" | ["AAPL","GOOGL"] | "AAPL"
    """
    if isinstance(val, list):
        return [t.strip().upper() for t in val if isinstance(t, str) and t.strip()]
    if isinstance(val, str):
        return [t.strip().upper() for t in val.split(",") if t.strip()]
    return []


# ─────────────────────────────────────────────────────────────────────────────
# NewsProcessor (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class NewsProcessor:
    """Unchanged — align news to trading days."""

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
# VoyageEmbedder (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class VoyageEmbedder:
    """Unchanged from original — rate limit + retry + cache."""

    def __init__(self, cache_dir: str):
        self.api_key = os.getenv("VOYAGE_API_KEY", GlobalConfig.VOYAGE_API_KEY)
        if not self.api_key or self.api_key == "---":
            raise RuntimeError("Missing VOYAGE_API_KEY.")

        self.model       = getattr(GlobalConfig, "EMBED_MODEL", "voyage-3-large")
        self.max_texts   = getattr(GlobalConfig, "MAX_TEXTS_PER_REQ", 40)
        self.max_retries = getattr(GlobalConfig, "MAX_RETRIES", 6)
        self.backoff_base = getattr(GlobalConfig, "BACKOFF_BASE", 30)

        payment_added = bool(getattr(GlobalConfig, "PAYMENT_ADDED", True))
        rl = GlobalConfig.VOYAGE_RATE_LIMITS[True if payment_added else False]
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
                emb = obj.get("embedding", None)
                if isinstance(emb, list) and len(emb) > 0:
                    return emb
            except Exception:
                return None
        return None

    def _save_cache(self, text: str, emb: List[float]) -> None:
        p = self._cache_path(text)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"embedding": emb}, f)

    def _rpm_sleep_if_needed(self):
        now = time.time()
        self._req_times = [t for t in self._req_times if now - t < 60.0]
        if len(self._req_times) >= self.rpm:
            oldest = min(self._req_times)
            wait   = max(0.0, 60.0 - (now - oldest)) + 0.5
            print(f"⏳ Voyage RPM limit. Sleep {wait:.1f}s …")
            time.sleep(wait)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        texts  = [(_normalize_space(t)[:6000] if t else "") for t in texts]
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
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
                    r = requests.post(url, headers=headers, json=payload, timeout=(15, 120))
                    self._req_times.append(time.time())
                    if r.status_code == 429:
                        wait = self.backoff_base * (2 ** attempt) + random.uniform(0, 3)
                        time.sleep(wait)
                        continue
                    r.raise_for_status()
                    embs = r.json().get("data", [])
                    for bi, one in enumerate(embs):
                        emb = one.get("embedding", None)
                        if not isinstance(emb, list):
                            raise RuntimeError("Invalid Voyage embedding format")
                        idx = batch_idx[bi]
                        out[idx] = emb
                        self._save_cache(texts[idx], emb)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    wait = self.backoff_base * (2 ** attempt) + random.uniform(0, 3)
                    time.sleep(wait)

        return [o if o is not None else [] for o in out]


# ─────────────────────────────────────────────────────────────────────────────
# KGGenNewsEmbedder V2
# ─────────────────────────────────────────────────────────────────────────────

class KGGenNewsEmbedder:
    """
    V2: FinDKG-Lite schema + GATv2 + Batch API + Multi-Ticker.

    Entry points:
        process_and_save(news_df)  — full build với LLM extraction
        rebuild_graph_only()       — graph rebuild từ existing cache, no LLM
    """

    def __init__(
        self,
        interim_root: str = None,
        window_days:  int  = 20,
        kmeans_k:     int  = 128,
        min_relevance:      float = 0.30,
        min_confidence:     float = 0.35,
        max_triples_cap_per_day: Optional[int] = None,
        debug_print_samples: bool = False,
        enable_cache:        bool = True,
        cache_dirname:       str  = "kg_article_cache",
        allow_llm_when_missing: bool = False,
        use_voyage_resolution:  bool = True,
        use_voyage_node_features: bool = True,
        voyage_cache_dirname: str = "kg_voyage_emb_cache",
        # GNN params
        use_graph_encoder_embedding: bool = True,
        graph_out_dim:    int   = 128,
        graph_hidden_dim: int   = 128,
        graph_num_layers: int   = 2,
        graph_num_heads:  int   = 4,
        graph_dropout:    float = 0.1,
        # Multi-ticker + batch params (NEW)
        symbols_col: str = "symbols",
        # Tên cột chứa ticker(s). Hỗ trợ "AAPL,GOOGL" hoặc ["AAPL","GOOGL"].
        # Fallback tự động sang "equity" hoặc "ticker" nếu không tìm thấy.
        batch_display_name: str = "findkg-lite-v2",
        # Legacy compat
        top_triples_per_article: int = 0,
        graph_use_gat: bool = True,
    ):
        if interim_root is None:
            interim_root = os.path.join("data", "interim")

        self.interim_root           = interim_root
        self.window_days            = window_days
        self.kmeans_k               = kmeans_k
        self.min_relevance          = min_relevance
        self.min_confidence         = min_confidence
        self.max_triples_cap_per_day = max_triples_cap_per_day
        self.debug_print_samples    = debug_print_samples
        self.enable_cache           = enable_cache
        self.allow_llm_when_missing = allow_llm_when_missing
        self.use_voyage_resolution  = use_voyage_resolution
        self.use_voyage_node_features = use_voyage_node_features
        self.graph_out_dim          = graph_out_dim
        self.symbols_col            = symbols_col
        self.batch_display_name     = batch_display_name

        # Directories
        self.base_dir     = os.path.join(interim_root, "kg")
        self.dir_triples  = os.path.join(self.base_dir, "extracted_triples")
        self.dir_raw      = os.path.join(self.base_dir, "window_graph_raw")
        self.dir_stable   = os.path.join(self.base_dir, "window_graph_stable")
        self.dir_tensors  = os.path.join(self.base_dir, "tensors")
        self.emb_dir      = os.path.join(interim_root, "kg_embeddings")
        for d in [self.dir_triples, self.dir_raw, self.dir_stable,
                  self.dir_tensors, self.emb_dir]:
            os.makedirs(d, exist_ok=True)

        self.cache_dir = os.path.join(interim_root, cache_dirname)
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.voyage_cache_dir = os.path.join(interim_root, voyage_cache_dirname)
        self.voyage = VoyageEmbedder(cache_dir=self.voyage_cache_dir)

        # GATv2 encoder
        self.graph_encoder = None
        if use_graph_encoder_embedding:
            self.graph_encoder = KGGraphEncoderGATv2(
                node_dim=NODE_FEATURE_DIM,
                hidden_dim=graph_hidden_dim,
                output_dim=graph_out_dim,
                num_heads=graph_num_heads,
                num_layers=graph_num_layers,
                dropout=graph_dropout,
            )
            self.graph_encoder.eval()

        # LLM extractors (lazy init)
        self._llm_extractor   = None
        self._batch_extractor = None

    # ── Lazy extractors ───────────────────────────────────────────────────────

    def _get_extractor(self) -> FinDKGLiteExtractor:
        """Sequential extractor (fallback / test)."""
        if self._llm_extractor is None:
            self._llm_extractor = FinDKGLiteExtractor(
                api_key=os.getenv("GEMINI_API_KEY"),
                min_relevance=self.min_relevance,
                min_confidence=self.min_confidence,
            )
        return self._llm_extractor

    def _get_batch_extractor(self) -> GeminiBatchAPIExtractor:
        """Batch extractor (50% cost, default for process_and_save)."""
        if self._batch_extractor is None:
            self._batch_extractor = GeminiBatchAPIExtractor(
                api_key=os.getenv("GEMINI_API_KEY"),
                min_relevance=self.min_relevance,
                min_confidence=self.min_confidence,
                display_name=self.batch_display_name,
            )
        return self._batch_extractor

    # ── Cache helpers ─────────────────────────────────────────────────────────

    def _article_cache_path(self, text: str) -> str:
        return os.path.join(self.cache_dir, f"{_sha1(text)}.json")

    def _load_article_cache(self, text: str) -> Optional[List[RichTriple]]:
        if not self.enable_cache:
            return None
        p = self._article_cache_path(text)
        if not os.path.exists(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
            raw = obj.get("triples", [])
            if not raw:
                return []
            if isinstance(raw[0], (list, tuple)):
                raw = [upgrade_legacy_triple(t) for t in raw if t]
                raw = [t for t in raw if t is not None]
                obj["triples"] = raw
                obj["_format_version"] = "v2"
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False)
            return [t for t in raw if isinstance(t, dict)]
        except Exception:
            return None

    def _save_article_cache(self, text: str, triples: List[RichTriple]) -> None:
        if not self.enable_cache:
            return
        p = self._article_cache_path(text)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {"text_sha1": _sha1(text), "triples": triples, "_format_version": "v2"},
                f, ensure_ascii=False,
            )

    # ── DataFrame normalization ───────────────────────────────────────────────

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Chuẩn hóa DataFrame: detect cột ticker, explode multi-ticker rows.

        Input:  1 row có symbols="AAPL,GOOGL,TSLA"
        Output: 3 rows, mỗi row có equity="AAPL"|"GOOGL"|"TSLA"

        Cũng thêm cột 'primary_ticker' = ticker đầu tiên trong symbols list
        (dùng cho extraction — chỉ extract 1 lần cho primary ticker).
        """
        df = df.copy()

        # 1. Detect ticker column
        ticker_col = None
        for col in (self.symbols_col, "equity", "ticker"):
            if col in df.columns:
                ticker_col = col
                break
        if ticker_col is None:
            raise ValueError(
                f"No ticker column found. Expected one of: "
                f"{self.symbols_col}, equity, ticker. "
                f"Actual columns: {list(df.columns)}"
            )

        # 2. Parse thành list, lấy primary ticker (element đầu tiên)
        df["_ticker_list"]   = df[ticker_col].apply(_parse_tickers)
        df["primary_ticker"] = df["_ticker_list"].apply(
            lambda lst: lst[0] if lst else None
        )

        # 3. Explode: 1 row per ticker
        df = df.explode("_ticker_list")
        df = df.rename(columns={"_ticker_list": "equity"})
        df = df[df["equity"].notna() & (df["equity"] != "")]
        df = df.reset_index(drop=True)

        return df

    # ── Day-level batch extraction ────────────────────────────────────────────

    def _collect_day_triples_batch(
        self,
        day_df: pd.DataFrame,
        text_col: str,
        ticker: str,
        date_str: str,
    ) -> List[RichTriple]:
        """
        Collect triples cho 1 (ticker, date):
        1. Load từ cache nếu có
        2. Gom các bài CHƯA cache → batch LLM call
        3. Save cache cho bài mới
        4. Fan-out: rescore nếu primary_ticker của bài != ticker hiện tại
        5. Dedup + cap

        Key invariant: mỗi article text chỉ gọi LLM 1 lần dù xuất hiện
        nhiều lần trong day_df (cho các tickers khác nhau).
        """
        # sha1 → (text, primary_ticker) để track unique articles
        sha1_to_meta: Dict[str, Dict] = {}
        # sha1 → triples (None = chưa xử lý)
        sha1_to_raw: Dict[str, Optional[List[RichTriple]]] = {}

        # ── Pass 1: cache lookup ──────────────────────────────────────────
        for _, row in day_df.iterrows():
            text           = _normalize_space(str(row.get(text_col, "") or ""))
            primary_ticker = str(row.get("primary_ticker") or ticker)
            if not text:
                continue
            h = _sha1(text)
            if h in sha1_to_raw:
                continue  # đã xử lý text này trong vòng lặp này

            sha1_to_meta[h] = {
                "text":           text,
                "primary_ticker": primary_ticker,
                "date":           date_str,
                "sector":         TICKER_SECTOR_MAP.get(primary_ticker, "Technology"),
            }

            cached = self._load_article_cache(text)
            if cached is not None:
                sha1_to_raw[h] = cached  # có thể [] (confirmed no-event)
            elif self.allow_llm_when_missing:
                sha1_to_raw[h] = None   # placeholder → sẽ gọi LLM

        # ── Pass 2: batch LLM cho bài chưa cache ─────────────────────────
        uncached_sha1s = [h for h, v in sha1_to_raw.items() if v is None]
        if uncached_sha1s:
            articles_to_extract = [
                {
                    "text":   sha1_to_meta[h]["text"],
                    "ticker": sha1_to_meta[h]["primary_ticker"],
                    "date":   sha1_to_meta[h]["date"],
                    "sector": sha1_to_meta[h].get("sector"),
                }
                for h in uncached_sha1s
            ]
            print(
                f"   🔄 [{ticker} {date_str}] "
                f"Batch extracting {len(articles_to_extract)} uncached articles ..."
            )
            batch_results = self._get_batch_extractor().extract_batch(articles_to_extract)

            for h, triples in zip(uncached_sha1s, batch_results):
                triples = triples or []
                sha1_to_raw[h] = triples
                self._save_article_cache(sha1_to_meta[h]["text"], triples)

        # ── Pass 3: fan-out — rescore cho target ticker ───────────────────
        all_triples: List[RichTriple] = []
        for h, raw_triples in sha1_to_raw.items():
            if not raw_triples:
                continue
            primary = sha1_to_meta[h]["primary_ticker"]
            rescored = rescore_triples_for_ticker(
                raw_triples, primary, ticker, self.min_relevance
            )
            # Also apply confidence threshold after rescore
            rescored = [
                t for t in rescored
                if float(t.get("confidence", 0)) >= self.min_confidence
            ]
            all_triples.extend(rescored)

        # ── Pass 4: dedup ─────────────────────────────────────────────────
        seen    = set()
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

        if self.max_triples_cap_per_day and len(deduped) > self.max_triples_cap_per_day:
            deduped = sorted(
                deduped,
                key=lambda t: float(t.get("relevance_to_ticker", 0)),
                reverse=True,
            )[: self.max_triples_cap_per_day]

        return deduped

    # ── Keep sequential version for backward compat / test ───────────────────

    def _collect_day_triples(
        self, day_df: pd.DataFrame, text_col: str, ticker: str, date_str: str
    ) -> List[RichTriple]:
        """
        Sequential version (1 article per call).
        Dùng khi allow_llm_when_missing=True nhưng KHÔNG muốn batch.
        Trong V2, process_and_save() dùng _collect_day_triples_batch() thay thế.
        """
        all_triples: List[RichTriple] = []
        for _, r in day_df.iterrows():
            text = _normalize_space(str(r.get(text_col, "") or ""))
            if not text:
                continue
            cached = self._load_article_cache(text)
            if cached is not None:
                triples = [
                    t for t in cached
                    if float(t.get("relevance_to_ticker", 0)) >= self.min_relevance
                    and float(t.get("confidence", 0)) >= self.min_confidence
                ]
            elif self.allow_llm_when_missing:
                primary = str(r.get("primary_ticker") or ticker)
                triples = self._get_extractor().extract(text, primary, date_str)
                self._save_article_cache(text, triples)
            else:
                triples = []

            # Rescore nếu cần
            primary = str(r.get("primary_ticker") or ticker)
            triples = rescore_triples_for_ticker(triples, primary, ticker, self.min_relevance)
            all_triples.extend(triples)

        seen    = set()
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

        if self.max_triples_cap_per_day and len(deduped) > self.max_triples_cap_per_day:
            deduped = sorted(
                deduped,
                key=lambda t: float(t.get("relevance_to_ticker", 0)),
                reverse=True,
            )[: self.max_triples_cap_per_day]

        return deduped

    @staticmethod
    def aggregate_window(per_day_triples: List[List[RichTriple]]) -> List[RichTriple]:
        """Rolling window aggregation with dedup."""
        seen   = set()
        result = []
        for day in per_day_triples:
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

    # ── Entity resolution (unchanged) ────────────────────────────────────────

    def resolve_triples(self, triples: List[RichTriple]) -> List[RichTriple]:
        if not triples or not self.use_voyage_resolution:
            return triples

        ents = list({
            name
            for t in triples
            for name in [
                t.get("subject", {}).get("name", ""),
                t.get("object",  {}).get("name", ""),
            ]
            if name
        })

        if len(ents) <= 1:
            return triples

        emb_list = self.voyage.embed_texts(ents)
        emb = torch.tensor(emb_list, dtype=torch.float32).numpy()
        k   = min(self.kmeans_k, len(ents))
        labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(emb)

        canon: Dict[int, str] = {}
        for e, cid in zip(ents, labels):
            canon.setdefault(int(cid), e)
        mapping = {e: canon[int(cid)] for e, cid in zip(ents, labels)}

        out  = []
        seen = set()
        for t in triples:
            t2 = dict(t)
            t2["subject"] = dict(t["subject"])
            t2["object"]  = dict(t["object"])
            t2["subject"]["name"] = mapping.get(t["subject"].get("name", ""), t["subject"].get("name", ""))
            t2["object"]["name"]  = mapping.get(t["object"].get("name",  ""), t["object"].get("name",  ""))
            key = (t2["subject"]["name"], t2["relation"], t2["object"]["name"])
            if key not in seen:
                seen.add(key)
                out.append(t2)
        return out

    # ── Tensorize + GATv2 encode (unchanged) ─────────────────────────────────

    def tensorize_and_embed(
        self, ticker: str, date_str: str, rich_triples: List[RichTriple]
    ) -> Tuple[str, List[float]]:
        out_dir  = os.path.join(self.dir_tensors, ticker)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{date_str}.pt")
        zero_emb = [0.0] * self.graph_out_dim

        if not rich_triples:
            torch.save(
                {
                    "node_x":    torch.zeros(0, NODE_FEATURE_DIM),
                    "edge_index": torch.zeros(2, 0, dtype=torch.long),
                    "edge_attr":  torch.zeros(0, 17),
                    "ticker_idx": 0,
                    "nodes":      [],
                    "node_info":  {},
                    "graph_emb":  torch.tensor(zero_emb),
                },
                out_path,
            )
            return out_path, zero_emb

        node_info              = build_node_info(rich_triples)
        nodes, node2id, x      = build_node_features(node_info, self.voyage, ticker)
        edge_index, edge_attr  = build_rich_edge_data(rich_triples, node2id)

        if self.debug_print_samples:
            print(
                f"   Graph {ticker} {date_str}: "
                f"{len(nodes)} nodes, {edge_index.shape[1]} edges"
            )

        if (
            self.graph_encoder is not None
            and edge_index.shape[1] > 0
            and x.shape[0] > 0
        ):
            with torch.no_grad():
                g = self.graph_encoder(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    batch=None,
                )
            graph_emb = g.squeeze(0).tolist()
        else:
            graph_emb = zero_emb

        torch.save(
            {
                "node_x":    x,
                "edge_index": edge_index,
                "edge_attr":  edge_attr,
                "ticker_idx": node2id.get(ticker, 0),
                "nodes":      nodes,
                "node_info":  node_info,
                "graph_emb":  torch.tensor(graph_emb),
            },
            out_path,
        )
        return out_path, graph_emb

    # ── ENTRY POINT A: Full build ─────────────────────────────────────────────

    def process_and_save(self, news_df: pd.DataFrame) -> str:
        """
        Full pipeline: normalize → batch extract → resolve → tensorize → save.

        Hỗ trợ multi-ticker: bài báo "AAPL,GOOGL,TSLA" được xử lý 1 lần,
        kết quả fan-out sang cả 3 tickers.
        """
        # ── Normalize ────────────────────────────────────────────────────
        df = self._normalize_dataframe(news_df)

        # Standard column normalization
        if "content" not in df.columns and "text" in df.columns:
            df = df.rename(columns={"text": "content"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        text_col   = "content" if "content" in df.columns else "title"
        df[text_col] = df[text_col].fillna("").astype(str)

        results_json: Dict[str, List[Dict]] = {}

        tickers = sorted(df["equity"].unique())
        print(f"🚀 Processing {len(tickers)} tickers across "
              f"{df['date'].nunique()} dates ...")

        for ticker in tickers:
            df_t        = df[df["equity"] == ticker].copy()
            dates_sorted = sorted(df_t["date"].unique())
            window_triples: List[List[RichTriple]] = []

            for d in dates_sorted:
                date_str = str(d)
                day_df   = df_t[df_t["date"] == d]

                # Batch extraction (with multi-ticker dedup + fan-out)
                day_triples = self._collect_day_triples_batch(
                    day_df, text_col, ticker, date_str
                )
                self._save_day_json(ticker, date_str, "extracted_triples", day_triples)

                window_triples.append(day_triples)
                if len(window_triples) > self.window_days:
                    window_triples.pop(0)

                raw_graph = self.aggregate_window(window_triples)
                self._save_day_json(ticker, date_str, "window_graph_raw", raw_graph)

                stable = self.resolve_triples(raw_graph)
                self._save_day_json(ticker, date_str, "window_graph_stable", stable)

                kg_path, graph_emb = self.tensorize_and_embed(ticker, date_str, stable)

                results_json.setdefault(date_str, []).append({
                    "date":          date_str,
                    "equity":        ticker,
                    "kg_tensor_path": kg_path,
                    "embedding":      graph_emb,
                })

        return self._save_index(results_json)

    # ── ENTRY POINT B: Graph-only rebuild ────────────────────────────────────

    def rebuild_graph_only(self) -> str:
        """Graph rebuild từ existing cache — KHÔNG gọi LLM."""
        news_path = os.path.join(self.interim_root, "concatenated_news_filtered.parquet")
        if not os.path.exists(news_path):
            raise FileNotFoundError(f"News parquet not found: {news_path}")

        df = pd.read_parquet(news_path)
        df = self._normalize_dataframe(df)

        if "content" not in df.columns and "text" in df.columns:
            df = df.rename(columns={"text": "content"})
        df["date"] = pd.to_datetime(df["date"]).dt.date
        text_col   = "content" if "content" in df.columns else "title"
        df[text_col] = df[text_col].fillna("").astype(str)

        print("🚀 REBUILD GRAPH-ONLY (GATv2, NO LLM)")
        results_json: Dict[str, List[Dict]] = {}
        miss_total = total = 0

        for ticker in sorted(df["equity"].unique()):
            df_t         = df[df["equity"] == ticker].copy()
            window_triples: List[List[RichTriple]] = []

            for d in sorted(df_t["date"].unique()):
                date_str = str(d)
                day_df   = df_t[df_t["date"] == d]
                day_triples: List[RichTriple] = []
                sha1_seen: set = set()

                for _, r in day_df.iterrows():
                    total += 1
                    text    = _normalize_space(str(r.get(text_col, "") or ""))
                    primary = str(r.get("primary_ticker") or ticker)
                    if not text:
                        continue
                    h = _sha1(text)
                    if h in sha1_seen:
                        continue  # skip duplicate texts in day
                    sha1_seen.add(h)

                    cached = self._load_article_cache(text)
                    if cached is None:
                        miss_total += 1
                        continue
                    rescored = rescore_triples_for_ticker(
                        cached, primary, ticker, self.min_relevance
                    )
                    day_triples.extend(rescored)

                # Dedup
                seen    = set()
                deduped = []
                for t in day_triples:
                    key = (
                        t.get("subject", {}).get("name", ""),
                        t.get("relation", ""),
                        t.get("object",  {}).get("name", ""),
                    )
                    if key not in seen:
                        seen.add(key)
                        deduped.append(t)
                day_triples = deduped

                self._save_day_json(ticker, date_str, "extracted_triples", day_triples)

                window_triples.append(day_triples)
                if len(window_triples) > self.window_days:
                    window_triples.pop(0)

                raw_graph = self.aggregate_window(window_triples)
                self._save_day_json(ticker, date_str, "window_graph_raw", raw_graph)

                stable = self.resolve_triples(raw_graph)
                self._save_day_json(ticker, date_str, "window_graph_stable", stable)

                kg_path, graph_emb = self.tensorize_and_embed(ticker, date_str, stable)
                results_json.setdefault(date_str, []).append({
                    "date":           date_str,
                    "equity":         ticker,
                    "kg_tensor_path": kg_path,
                    "embedding":      graph_emb,
                })

        print(f"✅ Rebuild done. Cache-hit: {total-miss_total}/{total}")
        return self._save_index(results_json)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _save_day_json(self, ticker: str, date_str: str, subdir: str, data: Any):
        d = os.path.join(self.base_dir, subdir, ticker)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{date_str}.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"date": date_str, "ticker": ticker, "triples": data},
                f, ensure_ascii=False,
            )

    def _save_index(self, results_json: Dict) -> str:
        out_path = os.path.join(self.emb_dir, "embedded_kg.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False)
        print(f"✅ KG index saved: {out_path}")
        return out_path