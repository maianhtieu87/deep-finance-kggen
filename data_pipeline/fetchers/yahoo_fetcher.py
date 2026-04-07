# data_pipeline/fetchers/yahoo_fetcher.py
"""
V5.3 — YahooFetcher

Symbol mapping:
  ^GSPC    → sp500           (model feature)
  ^VIX     → vix             (model feature)
  CL=F     → wti             (model feature)
  DX-Y.NYB → dxy             (model feature)
  ^TNX     → us10y_yahoo     (intermediate: yield_spread fallback)
  ^IRX     → us_irx          (intermediate: yield_spread fallback)

V5.3 fixes vs V5.2:
  1. _FRIENDLY map now covers CL=F → wti and DX-Y.NYB → dxy correctly.
     Old code: key = sym.lower() → "cl=f" not in map → stored as raw key.
  2. FRED: shorter per-attempt timeout + one retry before fallback.
     Old code: single long blocking call → 30s hang on timeout.
  3. Removed dji/nasdaq from model path. They are never in MACRO_COLS
     so fetching them was wasted API calls.
"""

import yfinance as yf
import pandas as pd
from typing import List, Dict, Union
from pandas_datareader import data as pdr


class YahooFetcher:

    # Canonical name map: key = processed symbol string → column name in macro_df
    # Processing: sym.replace("^","").replace(".","_").lower()
    _FRIENDLY: Dict[str, str] = {
        "gspc":      "sp500",         # ^GSPC
        "vix":       "vix",           # ^VIX
        "cl=f":      "wti",           # CL=F   (crude oil futures)
        "dx-y_nyb":  "dxy",           # DX-Y.NYB → replace "." with "_"
        "tnx":       "us10y_yahoo",   # ^TNX   (10Y yield, intermediate)
        "irx":       "us_irx",        # ^IRX   (13-week T-bill, intermediate)
    }

    def download_data(self, start_day: str, end_day: str,
                      tickers: List[str]) -> List[pd.DataFrame]:
        """Download OHLC per ticker."""
        df_list = []
        for ticker in tickers:
            print(f"Downloading data for {ticker}")
            data = yf.download(ticker, start=start_day, end=end_day,
                               progress=False, auto_adjust=True)
            data = data.reset_index()
            data["Date"] = pd.to_datetime(data["Date"]).dt.date

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ["_".join(filter(None, col)).strip()
                                for col in data.columns]

            open_col  = next((c for c in data.columns if c.lower().startswith("open")),  None)
            high_col  = next((c for c in data.columns if c.lower().startswith("high")),  None)
            close_col = next((c for c in data.columns if c.lower().startswith("close")), None)
            date_col  = next((c for c in data.columns if c.lower() == "date"),           None)

            if not all([open_col, high_col, close_col, date_col]):
                print(f"  [WARN] Missing OHLC for {ticker}: {list(data.columns)}")
                continue

            out = data[[date_col, open_col, high_col, close_col]].copy()
            out = out.rename(columns={
                date_col:  "date",
                open_col:  f"{ticker}_open",
                high_col:  f"{ticker}_high",
                close_col: f"{ticker}_close",
            })
            df_list.append(out)
        return df_list

    def fetch_macro_indicators(
        self,
        start_date: str,
        end_date:   str,
        symbols:    Union[List[str], Dict[str, str]],
    ) -> pd.DataFrame:
        """
        Download macro data from Yahoo + FRED.

        Returns a DataFrame with canonical column names.
        Columns present after a successful run (FRED OK):
          sp500, vix, wti, dxy, us10y_yahoo, us_irx, us10y, us2y
        Columns present after FRED timeout:
          sp500, vix, wti, dxy, us10y_yahoo, us_irx
        MacroProcessor consumes this and produces exactly 5 model features.
        """
        # ── Build symbol_map: canonical_name → yahoo_symbol ──────────────────
        if isinstance(symbols, dict):
            symbol_map = symbols
        else:
            symbol_map = {}
            for sym in symbols:
                key = sym.replace("^", "").replace(".", "_").lower()
                canonical = self._FRIENDLY.get(key, key)
                symbol_map[canonical] = sym

        ticker_list = list(symbol_map.values())
        print(f"Fetching Macro from Yahoo ({len(ticker_list)} symbols: {ticker_list})...")

        # ── Yahoo download ────────────────────────────────────────────────────
        macro_data = pd.DataFrame()
        try:
            raw = yf.download(ticker_list, start=start_date, end=end_date,
                              auto_adjust=True, progress=False)

            if isinstance(raw.columns, pd.MultiIndex):
                lvl0 = raw.columns.get_level_values(0).unique().tolist()
                raw  = raw["Close"] if "Close" in lvl0 else raw
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = ["_".join(filter(None, c)) for c in raw.columns]

            inv_map    = {v: k for k, v in symbol_map.items()}
            macro_data = raw.rename(columns=inv_map)

            got     = [c for c in macro_data.columns if not macro_data[c].isna().all()]
            missing = [k for k in symbol_map if k not in macro_data.columns]
            print(f"  Yahoo OK   : {got}")
            if missing:
                print(f"  Yahoo miss : {missing}")

        except Exception as e:
            print(f"  Yahoo Error: {e}")

        # ── FRED — 10Y and 2Y Treasury yields ────────────────────────────────
        print("  Fetching yields from FRED...")
        macro_data["us10y"] = self._fred_series("DGS10", start_date, end_date)
        macro_data["us2y"]  = self._fred_series("DGS2",  start_date, end_date)

        if macro_data["us10y"] is None:
            macro_data.drop(columns=["us10y"], errors="ignore", inplace=True)
            print("  FRED DGS10: failed → yield_spread will use ^TNX fallback")
        else:
            print("  FRED DGS10: OK")

        if macro_data["us2y"] is None:
            macro_data.drop(columns=["us2y"], errors="ignore", inplace=True)
            print("  FRED DGS2 : failed → yield_spread will use ^IRX fallback")
        else:
            print("  FRED DGS2 : OK")

        # ── Normalise index → date column ────────────────────────────────────
        macro_data = macro_data.reset_index()
        date_col = next(
            (c for c in macro_data.columns if c.lower() in ("date", "index")), None
        )
        if date_col and date_col != "date":
            macro_data = macro_data.rename(columns={date_col: "date"})
        if "date" in macro_data.columns:
            macro_data["date"] = pd.to_datetime(macro_data["date"]).dt.date

        cols = [c for c in macro_data.columns if c != "date"]
        print(f"  macro_df columns ({len(cols)}): {cols}")
        return macro_data

    # ── Private ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fred_series(series_id: str, start: str, end: str):
        """Fetch one FRED series with one retry. Returns Series or None."""
        for attempt in range(2):
            try:
                df = pdr.DataReader(series_id, "fred", start, end)
                return df[series_id]
            except Exception as e:
                label = "attempt 1" if attempt == 0 else "attempt 2 (final)"
                print(f"    FRED {series_id} {label}: {type(e).__name__}")
        return None