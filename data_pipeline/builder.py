# data_pipeline/builder.py
"""
V5.7 — DatasetBuilder

V5.7 vs V5.6:
  Phase 2 quality support: loads news_quality_finbert.json (4D per ticker/day)
  generated automatically by embed_news.py --finbert (V8.1+).

  Requires config.py to have:
    GlobalConfig.QUALITY_DIM = 4
    GlobalConfig.finbert_quality_path()
    GlobalConfig.news_quality_path()

  Backward-compatible: if config.py not yet updated or quality file absent,
  news_quality entries are simply empty dicts → data_loader fills with zeros.

BUG FIXED vs uploaded version:
  Quality assignment block was placed BEFORE the main date loop (lines 97-104
  in uploaded file), causing NameError on date_obj / date_str. Moved to inside
  the loop as section 2.4b.
"""

import os
import json
import pickle
import pandas as pd
from configs.config import GlobalConfig as Config, TrainConfig
from src.data_loader import NEWS_EMB_DIM


class DatasetBuilder:
    def create_synchronized_data(
        self,
        price_macro_dict,
        news_df,
        filing_path,
        embedding_path,
    ):
        """
        Sync price + macro + news_embedding + news_quality + filings.

        V5.7: Adds section 2.4b — news_quality per ticker per day.
          Source: news_embeddings_finbert_quality.json (4D vectors)
          Resolved via Config.news_quality_path() → None for Voyage (skipped).
        """
        # ── 0) Filings (optional) ─────────────────────────────────────────────
        if filing_path and os.path.exists(filing_path):
            filing_df = pd.read_parquet(filing_path)
            filing_df["filedAt"] = (
                pd.to_datetime(filing_df["filedAt"], errors="coerce").dt.normalize()
            )
            filing_df = filing_df.dropna(subset=["filedAt"])
        else:
            filing_df = pd.DataFrame(
                columns=["filedAt", "ticker", "formType", "content_summary"]
            )

        # ── 0.1) News dtype fix ───────────────────────────────────────────────
        if news_df is None:
            news_df = pd.DataFrame(columns=["date", "equity"])
        else:
            news_df = news_df.copy()

        if "date" in news_df.columns:
            news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
            news_df = news_df.dropna(subset=["date"])
            news_df["date"] = news_df["date"].dt.normalize()

        for c in ["title", "content", "summary", "source", "url"]:
            if c not in news_df.columns:
                news_df[c] = None

        # ── 1) Load news embeddings JSON ──────────────────────────────────────
        embedding_data: dict = {}
        n_dim_mismatch = 0
        if not embedding_path:
            embedding_path = Config.news_emb_path()
        _embedder_label = {"finbert": "FinBERT per-triple CLS", "voyage": "Voyage-finance-2"}
        _label = _embedder_label.get(TrainConfig.news_embedder, TrainConfig.news_embedder)
        if embedding_path and os.path.exists(embedding_path):
            print(f"Loading news embeddings from {embedding_path}...")
            with open(embedding_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for k, v in raw.items():
                embedding_data[str(k)[:10]] = v
            print(f"  Loaded {len(embedding_data)} dates")
            print(f"  Embedder: {_label} | Expected dim: {NEWS_EMB_DIM}D")
        else:
            print(f"  news embeddings not found at {embedding_path}")
            print(
                f"  Run: python embed_news.py "
                f"{'--finbert' if TrainConfig.news_embedder == 'finbert' else ''}"
            )

        # ── 1b) Load news quality JSON (V5.7) — BEFORE loop ──────────────────
        # quality_data is loaded once here, then read per-date inside the loop.
        # Using getattr/hasattr for backward-compat if config.py not yet updated.
        quality_data: dict = {}
        _quality_dim: int  = getattr(Config, "QUALITY_DIM", 4)

        try:
            quality_path = Config.news_quality_path()
        except AttributeError:
            quality_path = None
            print(
                "  [INFO] Config.news_quality_path() not found — "
                "add QUALITY_DIM + news_quality_path() to config.py for Phase 2 quality gate."
            )

        if quality_path and os.path.exists(quality_path):
            print(f"Loading news quality stats from {quality_path}...")
            with open(quality_path, "r", encoding="utf-8") as f:
                raw_q = json.load(f)
            for k, v in raw_q.items():
                quality_data[str(k)[:10]] = v
            print(f"  Loaded quality stats for {len(quality_data)} dates ({_quality_dim}D)")
        elif quality_path:
            print(
                f"  Quality stats not found: {quality_path}\n"
                f"  Re-run: python embed_news.py --finbert  "
                f"(V8.1+ generates quality automatically alongside embeddings)"
            )
        # If quality_path is None (e.g. Voyage embedder): silently skip.

        # ── Shared lookup structures ──────────────────────────────────────────
        synchronized_data: dict = {}
        mapping = Config.TICKER_MAPPING

        # ── 2) Iterate trading dates ──────────────────────────────────────────
        for date_obj, data in price_macro_dict.items():
            date_dt  = pd.to_datetime(date_obj).normalize()
            date_str = str(date_obj)[:10]

            synchronized_data[date_obj] = {}

            # ── 2.1 Price ─────────────────────────────────────────────────────
            synchronized_data[date_obj]["price"] = {}
            for t, v in data.items():
                if t != "macro" and t in mapping:
                    synchronized_data[date_obj]["price"][mapping[t]] = v

            # ── 2.2 Macro ─────────────────────────────────────────────────────
            synchronized_data[date_obj]["macro"] = data.get("macro", {})

            # ── 2.3 News objects (reference only) ─────────────────────────────
            synchronized_data[date_obj]["news"] = {}
            if len(news_df) > 0:
                date_news = news_df[news_df["date"] == date_dt]
                for ticker in date_news["equity"].unique():
                    if ticker in mapping:
                        clean_ticker = mapping[ticker]
                        news_records = date_news[date_news["equity"] == ticker][
                            ["title", "content", "summary", "source", "url"]
                        ].to_dict(orient="records")
                        synchronized_data[date_obj]["news"].setdefault(clean_ticker, [])
                        synchronized_data[date_obj]["news"][clean_ticker].extend(news_records)

            # ── 2.4 News embeddings ────────────────────────────────────────────
            synchronized_data[date_obj]["news_embedding"] = {}
            if date_str in embedding_data:
                day_embs = embedding_data[date_str]
                for raw_ticker, emb in day_embs.items():
                    if raw_ticker in mapping:
                        clean_ticker = mapping[raw_ticker]
                        if isinstance(emb, list) and len(emb) == NEWS_EMB_DIM:
                            synchronized_data[date_obj]["news_embedding"][clean_ticker] = emb
                        elif isinstance(emb, list) and len(emb) > 0:
                            n_dim_mismatch += 1
                            synchronized_data[date_obj]["news_embedding"][clean_ticker] = []
                        else:
                            synchronized_data[date_obj]["news_embedding"][clean_ticker] = []

            # ── 2.4b News quality (V5.7) ───────────────────────────────────────
            # Populated only when quality JSON exists (finbert embedder).
            # Empty dict → data_loader._load_rows() returns None → s_q stays zeros.
            synchronized_data[date_obj]["news_quality"] = {}
            if date_str in quality_data:
                for raw_ticker, q in quality_data[date_str].items():
                    if raw_ticker in mapping:
                        clean_ticker = mapping[raw_ticker]
                        if isinstance(q, list) and len(q) == _quality_dim:
                            synchronized_data[date_obj]["news_quality"][clean_ticker] = q

            # ── 2.5 Filings ───────────────────────────────────────────────────
            date_filings = filing_df[filing_df["filedAt"] == date_dt]
            synchronized_data[date_obj]["filing_q"] = {}
            synchronized_data[date_obj]["filing_k"] = {}

            if len(date_filings) > 0:
                for ticker in date_filings["ticker"].unique():
                    if ticker in mapping:
                        clean_ticker = mapping[ticker]
                        tf    = date_filings[date_filings["ticker"] == ticker]
                        q_txt = tf[tf["formType"] == "10-Q"]["content_summary"].tolist()
                        k_txt = tf[tf["formType"] == "10-K"]["content_summary"].tolist()
                        if q_txt:
                            synchronized_data[date_obj]["filing_q"][clean_ticker] = \
                                " ".join(q_txt)
                        if k_txt:
                            synchronized_data[date_obj]["filing_k"][clean_ticker] = \
                                " ".join(k_txt)

        # ── Post-loop summary ─────────────────────────────────────────────────
        if n_dim_mismatch > 0:
            print(
                f"  [WARN] {n_dim_mismatch} embeddings rejected: wrong dimension "
                f"(expected {NEWS_EMB_DIM}D). "
                f"Rebuild with: python embed_news.py "
                f"{'--finbert' if TrainConfig.news_embedder == 'finbert' else '(no --finbert for Voyage)'}"
            )

        return synchronized_data

    def save(self, data, filename="unified_dataset.pkl"):
        os.makedirs(Config.PROCESSED_PATH, exist_ok=True)
        path = os.path.join(Config.PROCESSED_PATH, filename)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Dataset saved: {path}")