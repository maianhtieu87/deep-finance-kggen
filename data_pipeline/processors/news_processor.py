# data_pipeline/processors/news_processor.py
"""
V4 — Slim version.

Removed (no longer in pipeline):
  - KGGenNewsEmbedder (called build_graphs.py / build_kg.py — both deleted)
  - VoyageEmbedder inline class (embed_news.py has its own standalone version)
  - from encoders.kg_graph_encoder import ... (kg_graph_encoder.py deleted)
  - from data_pipeline.kg.extractor import ... (only needed by KGGenNewsEmbedder)

Kept:
  - normalize_news_df()              utility exported for external use
  - NewsProcessor.align_to_trading_days()   called by main_test.py
"""

import pandas as pd

from data_pipeline.kg.extractor_batch import (
    detect_primary_ticker,
    _parse_tickers,
    _norm,
    TICKER_NAME_MAP,
)


# ─────────────────────────────────────────────────────────────────────────────
# normalize_news_df — module-level export
# Used by: (external callers / tests)
# Not used by: extract_corpus.py (has own inline version),
#              embed_news.py (has own inline version)
# ─────────────────────────────────────────────────────────────────────────────

def normalize_news_df(df: pd.DataFrame, symbols_col: str = "symbols") -> pd.DataFrame:
    """
    Standardise raw news DataFrame:
      - Rename headline→title, ticker→equity
      - Parse date
      - Parse _all_tickers list
      - detect_primary_ticker (title weight ×3)
      - Explode one row per ticker; restore _all_tickers after explode
    """
    df = df.copy()

    col_map = {}
    if "headline" in df.columns and "title" not in df.columns:
        col_map["headline"] = "title"
    if "ticker" in df.columns and "equity" not in df.columns:
        col_map["ticker"] = "equity"
    if col_map:
        df = df.rename(columns=col_map)

    if "content" not in df.columns:
        for alt in ("body", "text"):
            if alt in df.columns:
                df = df.rename(columns={alt: "content"})
                break
    if "content" not in df.columns:
        df["content"] = ""
    if "title" not in df.columns:
        df["title"] = ""

    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column. Has: {list(df.columns)}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    ticker_col = None
    for col in (symbols_col, "equity", "ticker"):
        if col in df.columns:
            ticker_col = col
            break
    if ticker_col is None:
        raise ValueError(f"No ticker column. Expected: {symbols_col}, equity, ticker.")

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
    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# NewsProcessor — align helper
# Used by: main_test.py Phase B
# ─────────────────────────────────────────────────────────────────────────────

class NewsProcessor:
    """Thin wrapper. Only align_to_trading_days() is used in the V4 pipeline."""

    def align_to_trading_days(self, news_input, trading_days):
        if isinstance(news_input, pd.DataFrame):
            df = news_input.copy()
        elif isinstance(news_input, str):
            df = (
                pd.read_parquet(news_input)
                if news_input.endswith(".parquet")
                else pd.read_csv(news_input)
            )
        else:
            raise TypeError(f"Expected DataFrame or path, got {type(news_input)}")

        # Column normalisation
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