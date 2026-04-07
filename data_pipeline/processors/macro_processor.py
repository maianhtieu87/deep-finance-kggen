# data_pipeline/processors/macro_processor.py
"""
V5.3 — MacroProcessor

Design: 5 model features exactly.
  vix, sp500, dxy, wti, yield_spread_10y_2y

Intermediate columns (^TNX → us10y_yahoo, ^IRX → us_irx) are used
internally to compute yield_spread_10y_2y when FRED is unavailable.
They are NOT stored in price_dict and do NOT reach the model.

Fixes vs V5.2:
  1. MACRO_COLS = 5 items only (previous code had 6 with sp500_return,
     plus the 3 missing columns: yield_spread, dxy, wti → silent zeros)
  2. bfill() before ffill() to handle leading NaN at series start
     (ffill alone cannot fill NaN that have no prior value)
  3. yield_spread_10y_2y computed with explicit fallback chain:
       FRED (DGS10-DGS2) → Yahoo (^TNX-^IRX) → constant 0 with warning
  4. Transparent NaN report on merge so issues are visible
"""

import pandas as pd
import numpy as np


class MacroProcessor:

    # ── The 5 columns that reach the model ───────────────────────────────────
    MACRO_COLS = [
        "vix",
        "sp500",
        "dxy",
        "wti",
        "yield_spread_10y_2y",
    ]

    # ── Intermediate column names (fetch targets, not model features) ─────────
    _COL_10Y = "us10y_yahoo"   # mapped from ^TNX in yahoo_fetcher
    _COL_2Y  = "us_irx"        # mapped from ^IRX in yahoo_fetcher

    def process_and_enrich(self, price_dict: dict, macro_df: pd.DataFrame) -> dict:
        """
        Enrich price_dict with lagged macro indicators.
        Only the 5 columns in MACRO_COLS are written into price_dict.

        Pipeline:
          1. Standardise DatetimeIndex
          2. Compute yield_spread_10y_2y (from FRED or fallback)
          3. bfill → ffill → shift(1)
          4. Merge MACRO_COLS into price_dict
          5. Report NaN count
        """
        # ── 1. Index ──────────────────────────────────────────────────────────
        if "date" in macro_df.columns:
            macro_df = macro_df.set_index(
                pd.to_datetime(macro_df["date"])
            ).drop(columns=["date"]).sort_index()
        elif not isinstance(macro_df.index, pd.DatetimeIndex):
            macro_df.index = pd.to_datetime(macro_df.index)
        macro_df = macro_df.copy().sort_index()

        # ── 2. Derived: yield_spread_10y_2y ──────────────────────────────────
        self._compute_yield_spread(macro_df)

        # ── 3. Gap-fill then lag ──────────────────────────────────────────────
        # bfill: fills leading NaN at series start (nothing before → fill forward
        #        from the first available value going backward)
        # ffill: fills remaining mid-series gaps
        # shift(1): lag by 1 trading day → no lookahead leakage
        macro_df = macro_df.bfill().ffill().shift(1)

        # ── 4. Determine available columns ───────────────────────────────────
        available = [c for c in self.MACRO_COLS if c in macro_df.columns]
        missing   = [c for c in self.MACRO_COLS if c not in macro_df.columns]
        if missing:
            print(f"  [WARN] macro columns unavailable (will be 0): {missing}")
        print(f"  Macro model features ({len(available)}/5): {available}")

        # ── 5. Merge into price_dict ──────────────────────────────────────────
        all_dates  = sorted(price_dict.keys())
        nan_count  = 0
        ok_count   = 0

        for date in all_dates:
            ts  = pd.to_datetime(date)
            idx = macro_df.index[macro_df.index <= ts]

            if "macro" not in price_dict[date]:
                price_dict[date]["macro"] = {}

            if len(idx) == 0:
                # Before macro data starts → zeros (safe default)
                for col in available:
                    price_dict[date]["macro"][col] = 0.0
                continue

            row = macro_df.loc[idx.max()]
            for col in available:
                val = row.get(col, np.nan) if hasattr(row, "get") else row[col]
                if pd.isna(val):
                    price_dict[date]["macro"][col] = 0.0
                    nan_count += 1
                else:
                    price_dict[date]["macro"][col] = float(val)
                    ok_count += 1

            # Columns not available at all → set 0
            for col in missing:
                price_dict[date]["macro"][col] = 0.0

        total  = ok_count + nan_count
        pct    = 100 * nan_count / max(total, 1)
        status = "[WARN]" if pct > 5 else "[OK] "
        print(f"  {status} Macro merge: {ok_count} filled, "
              f"{nan_count} NaN→0 ({pct:.1f}%)")
        return price_dict

    # ── Private ───────────────────────────────────────────────────────────────

    def _compute_yield_spread(self, macro_df: pd.DataFrame) -> None:
        """
        Compute yield_spread_10y_2y in-place.

        Priority chain:
          1. FRED data: us10y - us2y          (most accurate, daily data)
          2. Yahoo fallback: us10y_yahoo - us_irx   (if FRED timed out)
          3. Cannot compute → leave absent (will become 0 with warning)
        """
        ten_y = self._best_series(macro_df, ["us10y", self._COL_10Y])
        two_y = self._best_series(macro_df, ["us2y",  self._COL_2Y])

        if ten_y is not None and two_y is not None:
            macro_df["yield_spread_10y_2y"] = (
                ten_y.interpolate(method="time").ffill().bfill()
                - two_y.interpolate(method="time").ffill().bfill()
            )
            src_10 = "FRED" if "us10y" in macro_df.columns else "Yahoo ^TNX"
            src_2  = "FRED" if "us2y"  in macro_df.columns else "Yahoo ^IRX"
            print(f"  yield_spread_10y_2y: {src_10} − {src_2}")
        else:
            missing = []
            if ten_y is None: missing.append("10Y yield")
            if two_y is None: missing.append("2Y yield")
            print(f"  [WARN] yield_spread_10y_2y: cannot compute ({', '.join(missing)} missing)")

    @staticmethod
    def _best_series(df: pd.DataFrame, candidates: list):
        """Return first non-all-NaN series from candidates list."""
        for col in candidates:
            if col in df.columns and not df[col].isna().all():
                return df[col]
        return None