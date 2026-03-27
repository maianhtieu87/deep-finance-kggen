# data_pipeline/processors/news_processor.py
"""
V5.2 — Slim version.

V5.2 change: Added auto-detection of date column (created_at, publishedAt, etc.)
             to avoid crash when column is not literally named "date".

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
# DATE COLUMN AUTO-DETECTION (shared helper)
# ─────────────────────────────────────────────────────────────────────────────

_DATE_CANDIDATES = [
    "date", "Date", "DATE",
    "created_at", "createdAt",
    "published_at", "publishedAt",
    "publish_date", "pub_date",
    "timestamp", "time", "news_date",
]

def _ensure_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Auto-detect and rename date column to 'date' if needed."""
    if "date" in df.columns:
        return df
    date_col = next((c for c in _DATE_CANDIDATES if c in df.columns), None)
    if date_col is None:
        date_col = next(
            (c for c in df.columns if any(k in c.lower() for k in ("date", "time", "publish", "creat"))),
            None,
        )
    if date_col is None:
        raise ValueError(f"Cannot find date column. Has: {list(df.columns)}")
    return df.rename(columns={date_col: "date"})


# ─────────────────────────────────────────────────────────────────────────────
# normalize_news_df — module-level export
# ─────────────────────────────────────────────────────────────────────────────

def normalize_news_df(df: pd.DataFrame, symbols_col: str = "symbols") -> pd.DataFrame:
    """
    Standardise raw news DataFrame:
      - Auto-detect and rename date column
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

    df = _ensure_date_column(df)
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
    """Thin wrapper. Only align_to_trading_days() is used in the V4+ pipeline."""

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
        #if "equity" not in df.columns and "ticker" in df.columns:
            #df = df.rename(columns={"ticker": "equity"})
            
        if "equity" not in df.columns:
            if "symbols" in df.columns:
                df = df.rename(columns={"symbols": "equity"})
            elif "ticker" in df.columns:
                df = df.rename(columns={"ticker": "equity"})    
        if "title" not in df.columns and "headline" in df.columns:
            df = df.rename(columns={"headline": "title"})
        if "content" not in df.columns:
            for alt in ("body", "text"):
                if alt in df.columns:
                    df = df.rename(columns={alt: "content"})
                    break

        df = _ensure_date_column(df)
        df["date"] = pd.to_datetime(df["date"]).dt.date

        if trading_days is not None:
            td = set(pd.to_datetime(trading_days).date)
            df = df[df["date"].isin(td)]

        cols = ["date", "equity"]
        for c in ("content", "title"):
            if c in df.columns:
                cols.append(c)
        return df[cols].copy()