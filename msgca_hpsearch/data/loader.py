"""
DataPrepare — standalone data loader for MSGCA hyperparameter search.

Loads the unified_dataset_test.pkl produced by the deep-finance-kggen pipeline
and splits it into train/val/test splits for nested cross-validation.

Key differences from src/data_loader.py:
  - Config (window_size, news_dim, etc.) is passed as constructor args, not
    imported from TrainConfig. This makes the module fully self-contained.
  - Tickers and their IDs are configurable.
  - load_and_split() helper builds all splits needed for 2-phase MSGCA training.
"""

import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Default tickers — must match what was in the dataset when it was built
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["TSLA", "AAPL", "AMZN", "MSFT", "GOOGL", "META", "BA", "JPM", "WMT"]

VOL_WINDOW   = 20
VOL_FALLBACK = 0.02


def _extract_float(d, *keys):
    for k in keys:
        v = d.get(k) if isinstance(d, dict) else None
        if v is None:
            continue
        if hasattr(v, "iloc"):
            v = v.iloc[0] if len(v) > 0 else None
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return None


class DataPrepare:
    """
    Loads the unified pkl dataset and provides per-ticker train/val/test splits.

    Parameters
    ----------
    dataset_path : str
        Path to unified_dataset_test.pkl
    window_size : int
        Rolling window length (default 20)
    news_dim : int
        News embedding dimension (768 for FinBERT, 1024 for Voyage)
    quality_dim : int
        Quality stats dimension (4 for pipeline quality)
    tickers : list[str]
        Tickers to use — must be a subset of those in the dataset
    ticker_to_id : dict
        Mapping ticker → integer ID for the embedding layer
    price_mode : str
        "vol_adjusted" | "pct_first" | "absolute"
    label_mode : str
        "rolling" | "fixed" | "volatility"
    include_ticker_id : bool
        Whether to include ticker_id in output dicts
    """

    def __init__(
        self,
        dataset_path:      str,
        window_size:       int        = 20,
        news_dim:          int        = 768,
        quality_dim:       int        = 4,
        tickers:           List[str]  = None,
        ticker_to_id:      Dict[str, int] = None,
        price_mode:        str        = "vol_adjusted",
        label_mode:        str        = "rolling",
        include_ticker_id: bool       = True,
    ):
        with open(dataset_path, "rb") as f:
            self.raw_data = pickle.load(f)

        self.window_size       = window_size
        self.news_dim          = news_dim
        self.quality_dim       = quality_dim
        self.price_mode        = price_mode
        self.label_mode        = label_mode
        self.include_ticker_id = include_ticker_id
        self._cache_rows: Dict[str, list] = {}

        self.tickers      = tickers or DEFAULT_TICKERS
        self.ticker_to_id = ticker_to_id or {t: i for i, t in enumerate(DEFAULT_TICKERS)}

        print(f"Dataset loaded : {len(self.raw_data)} trading days")
        print(f"Window size    : {window_size}")
        print(f"News dim       : {news_dim}D")
        print(f"Quality dim    : {quality_dim}D")
        print(f"Price mode     : {price_mode}")
        print(f"Label mode     : {label_mode}")
        print(f"Tickers        : {self.tickers}")

    # ── Row loading ────────────────────────────────────────────────────────

    def _load_rows(self, ticker: str) -> list:
        if ticker in self._cache_rows:
            return self._cache_rows[ticker]

        dates = sorted(self.raw_data.keys())
        rows  = []
        for date_key in dates:
            day = self.raw_data[date_key]
            p   = day.get("price", {}).get(ticker)
            if p is None:
                continue
            s_o = _extract_float(p, "Open", "open")
            s_h = _extract_float(p, "High", "high")
            s_c = _extract_float(p, "Close", "close")
            if None in (s_o, s_h, s_c):
                continue
            rows.append({
                "date":         date_key,
                "s_o":          s_o,
                "s_h":          s_h,
                "s_c":          s_c,
                "macro":        day.get("macro", {}),
                "news_emb":     day.get("news_embedding", {}).get(ticker, None),
                "news_quality": day.get("news_quality", {}).get(ticker, None),
            })

        self._cache_rows[ticker] = rows
        return rows

    def get_max_T(self, ticker: str) -> int:
        rows = self._load_rows(ticker)
        if len(rows) < self.window_size + 1:
            return 0
        return len(rows) - self.window_size

    # ── Data preparation ───────────────────────────────────────────────────

    def prepare_data(
        self,
        ticker: str,
        train_end: int = None,
        val_end:   int = None,
        test_end:  int = None,
        train_ratio: float = 0.70,
        valid_ratio: float = 0.15,
    ) -> Tuple[dict, dict, dict]:
        rows = self._load_rows(ticker)
        if len(rows) < self.window_size + 1:
            return {}, {}, {}

        T_max = len(rows) - self.window_size

        if train_end is None or val_end is None or test_end is None:
            train_end = int(T_max * train_ratio)
            val_end   = int(T_max * (train_ratio + valid_ratio))
            test_end  = T_max

        T = test_end

        close_s = pd.Series([r["s_c"] for r in rows])
        ret_s   = close_s.pct_change().fillna(0)
        labels  = self._build_labels(ret_s, T, train_end)

        close_arr = np.array([r["s_c"] for r in rows], dtype=np.float64)
        open_arr  = np.array([r["s_o"] for r in rows], dtype=np.float64)
        high_arr  = np.array([r["s_h"] for r in rows], dtype=np.float64)
        s_o_all, s_h_all, s_c_all = self._build_price_features(
            close_arr, open_arr, high_arr, T, train_end
        )

        macro_keys = sorted(rows[0]["macro"].keys())
        macro_dim  = len(macro_keys)
        s_m_all    = np.zeros((T, self.window_size, macro_dim), dtype=np.float32)
        for t in range(T):
            for w, row in enumerate(rows[t : t + self.window_size]):
                s_m_all[t, w, :] = [row["macro"].get(k, 0.0) for k in macro_keys]
        s_m_all = self._znorm(s_m_all, s_m_all[:train_end])

        s_n_all, news_mask_all = self._build_news(rows, T, train_end)
        s_q_all = self._build_news_quality(rows, T, train_end)

        # NaN cleanup
        for arr in [s_o_all, s_h_all, s_c_all, s_m_all, s_n_all, s_q_all]:
            if np.isnan(arr).any():
                arr[:] = np.nan_to_num(arr, nan=0.0)

        s_o_t  = torch.tensor(s_o_all,       dtype=torch.float32)
        s_h_t  = torch.tensor(s_h_all,       dtype=torch.float32)
        s_c_t  = torch.tensor(s_c_all,       dtype=torch.float32)
        s_m_t  = torch.tensor(s_m_all,       dtype=torch.float32)
        s_n_t  = torch.tensor(s_n_all,       dtype=torch.float32)
        s_q_t  = torch.tensor(s_q_all,       dtype=torch.float32)
        mask_t = torch.tensor(news_mask_all,  dtype=torch.bool)
        lbl_t  = torch.tensor(labels,         dtype=torch.long)
        tid_t  = torch.full(
            (T,), self.ticker_to_id.get(ticker, 0), dtype=torch.long
        )

        def _s(x, a, b): return x[a:b]

        def _make(a, b):
            if a >= b:
                return {}
            d = {
                "s_o":          _s(s_o_t,  a, b),
                "s_h":          _s(s_h_t,  a, b),
                "s_c":          _s(s_c_t,  a, b),
                "s_m":          _s(s_m_t,  a, b),
                "s_n":          _s(s_n_t,  a, b),
                "news_mask":    _s(mask_t,  a, b),
                "label":        _s(lbl_t,   a, b),
                "news_quality": _s(s_q_t,   a, b),
            }
            if self.include_ticker_id:
                d["ticker_id"] = _s(tid_t, a, b)
            return d

        return _make(0, train_end), _make(train_end, val_end), _make(val_end, test_end)

    # ── Feature builders ───────────────────────────────────────────────────

    def _build_price_features(self, close_arr, open_arr, high_arr, T, train_end):
        if self.price_mode == "vol_adjusted":
            return self._vol_adjusted(close_arr, open_arr, high_arr, T, train_end)
        elif self.price_mode == "pct_first":
            return self._pct_first(close_arr, open_arr, high_arr, T, train_end)
        else:
            return self._absolute(close_arr, open_arr, high_arr, T, train_end)

    def _vol_adjusted(self, close_arr, open_arr, high_arr, T, train_end):
        n = len(close_arr)
        close_lr = np.zeros(n, dtype=np.float64)
        open_lr  = np.zeros(n, dtype=np.float64)
        high_lr  = np.zeros(n, dtype=np.float64)
        for i in range(1, n):
            if close_arr[i - 1] > 0: close_lr[i] = np.log(close_arr[i] / close_arr[i - 1])
            if open_arr[i - 1]  > 0: open_lr[i]  = np.log(open_arr[i]  / open_arr[i - 1])
            if high_arr[i - 1]  > 0: high_lr[i]  = np.log(high_arr[i]  / high_arr[i - 1])

        realized_vol = np.full(n, VOL_FALLBACK, dtype=np.float64)
        for i in range(VOL_WINDOW, n):
            v = close_lr[i - VOL_WINDOW : i].std()
            realized_vol[i] = max(v, 1e-6)
        realized_vol[:VOL_WINDOW] = realized_vol[VOL_WINDOW]

        close_va = close_lr / realized_vol
        open_va  = open_lr  / realized_vol
        high_va  = high_lr  / realized_vol

        s_o = np.zeros((T, self.window_size, 1), dtype=np.float32)
        s_h = np.zeros((T, self.window_size, 1), dtype=np.float32)
        s_c = np.zeros((T, self.window_size, 1), dtype=np.float32)
        for t in range(T):
            s_o[t, :, 0] = open_va[t  : t + self.window_size]
            s_h[t, :, 0] = high_va[t  : t + self.window_size]
            s_c[t, :, 0] = close_va[t : t + self.window_size]

        s_o = self._znorm(s_o, s_o[:train_end])
        s_h = self._znorm(s_h, s_h[:train_end])
        s_c = self._znorm(s_c, s_c[:train_end])
        return s_o, s_h, s_c

    def _pct_first(self, close_arr, open_arr, high_arr, T, train_end):
        s_o = np.zeros((T, self.window_size, 1), dtype=np.float32)
        s_h = np.zeros((T, self.window_size, 1), dtype=np.float32)
        s_c = np.zeros((T, self.window_size, 1), dtype=np.float32)
        for t in range(T):
            base = close_arr[t]
            if base > 0:
                s_o[t, :, 0] = open_arr[t  : t + self.window_size] / base - 1.0
                s_h[t, :, 0] = high_arr[t  : t + self.window_size] / base - 1.0
                s_c[t, :, 0] = close_arr[t : t + self.window_size] / base - 1.0
        s_o = self._znorm(s_o, s_o[:train_end])
        s_h = self._znorm(s_h, s_h[:train_end])
        s_c = self._znorm(s_c, s_c[:train_end])
        return s_o, s_h, s_c

    def _absolute(self, close_arr, open_arr, high_arr, T, train_end):
        s_o = np.zeros((T, self.window_size, 1), dtype=np.float32)
        s_h = np.zeros((T, self.window_size, 1), dtype=np.float32)
        s_c = np.zeros((T, self.window_size, 1), dtype=np.float32)
        for t in range(T):
            s_o[t, :, 0] = open_arr[t  : t + self.window_size]
            s_h[t, :, 0] = high_arr[t  : t + self.window_size]
            s_c[t, :, 0] = close_arr[t : t + self.window_size]
        s_o = self._znorm(s_o, s_o[:train_end])
        s_h = self._znorm(s_h, s_h[:train_end])
        s_c = self._znorm(s_c, s_c[:train_end])
        return s_o, s_h, s_c

    def _build_news(self, rows, T, train_end):
        s_n  = np.zeros((T, self.window_size, self.news_dim), dtype=np.float32)
        mask = np.ones ((T, self.window_size), dtype=bool)
        for t in range(T):
            for w, row in enumerate(rows[t : t + self.window_size]):
                emb = row["news_emb"]
                if emb is None or len(emb) != self.news_dim:
                    continue
                arr = np.array(emb, dtype=np.float32)
                if np.allclose(arr, 0):
                    continue
                s_n[t, w, :] = arr
                mask[t, w]   = False
        s_n = self._znorm(s_n, s_n[:train_end])
        s_n[mask] = 0.0
        return s_n, mask

    def _build_news_quality(self, rows: list, T: int, train_end: int) -> np.ndarray:
        q_dim = self.quality_dim
        s_q   = np.zeros((T, self.window_size, q_dim), dtype=np.float32)
        for t in range(T):
            for w, row in enumerate(rows[t : t + self.window_size]):
                q = row.get("news_quality")
                if q is not None and len(q) == q_dim:
                    s_q[t, w, :] = q
        s_q = self._znorm(s_q, s_q[:train_end])
        return s_q

    # ── Label builders ─────────────────────────────────────────────────────

    def _build_labels(self, ret_s: pd.Series, T: int, train_end: int) -> np.ndarray:
        if self.label_mode == "rolling":
            return self._labels_rolling(ret_s, T, train_end)
        elif self.label_mode == "fixed":
            return self._labels_fixed(ret_s, T, train_end)
        elif self.label_mode == "volatility":
            return self._labels_volatility(ret_s, T, train_end)
        raise ValueError(f"Unknown label_mode: {self.label_mode!r}")

    def _labels_rolling(self, ret_s, T, train_end):
        ws  = self.window_size
        rl  = ret_s.rolling(20).quantile(0.33).shift(1)
        rh  = ret_s.rolling(20).quantile(0.66).shift(1)
        tr  = [ret_s.iloc[t + ws] for t in range(train_end)]
        fb_lo = np.percentile(tr, 33)
        fb_hi = np.percentile(tr, 66)
        labels = np.ones(T, dtype=int)
        for t in range(T):
            ret = ret_s.iloc[t + ws]
            lo  = rl.iloc[t + ws]
            hi  = rh.iloc[t + ws]
            lo, hi = (fb_lo, fb_hi) if pd.isna(lo) else (lo, hi)
            if ret < lo:   labels[t] = 0
            elif ret > hi: labels[t] = 2
        return labels

    def _labels_fixed(self, ret_s, T, train_end):
        ws  = self.window_size
        tr  = [ret_s.iloc[t + ws] for t in range(train_end)]
        q33 = np.percentile(tr, 33)
        q66 = np.percentile(tr, 66)
        labels = np.ones(T, dtype=int)
        for t in range(T):
            r = ret_s.iloc[t + ws]
            if r < q33:   labels[t] = 0
            elif r > q66: labels[t] = 2
        return labels

    def _labels_volatility(self, ret_s, T, train_end):
        ws  = self.window_size
        tr  = np.array([ret_s.iloc[t + ws] for t in range(train_end)])
        thr = 0.5 * tr.std()
        labels = np.ones(T, dtype=int)
        for t in range(T):
            r = ret_s.iloc[t + ws]
            if r < -thr:  labels[t] = 0
            elif r > thr: labels[t] = 2
        return labels

    @staticmethod
    def _znorm(arr: np.ndarray, ref: np.ndarray) -> np.ndarray:
        mu  = ref.mean()
        std = ref.std() + 1e-8
        return (arr - mu) / std


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function: load data and build all splits for MSGCA training
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split(
    pkl_path:    str,
    tickers:     List[str]  = None,
    ticker_to_id: Dict[str, int] = None,
    news_dim:    int  = 768,
    quality_dim: int  = 4,
    window_size: int  = 20,
    price_mode:  str  = "vol_adjusted",
    label_mode:  str  = "rolling",
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
) -> dict:
    """
    Load pkl and return all data splits needed for MSGCA 2-phase training.

    Split strategy:
      global_T_max = min T_max over all tickers (ensures identical test sets)
      inner_T      = int(global_T_max * (train_ratio + valid_ratio))
      hval_split   = int(inner_T * 0.80)

      train_hval: [0 : hval_split]          used for HP search phase 1
      val_hval  : [hval_split : inner_T]    used for HP search validation
      train_full: [0 : inner_T]             used for final eval phase 2
      test      : [inner_T : global_T_max]  held-out outer test

    Returns dict with keys:
      global_T_max, inner_T, hval_split,
      train_hval, val_hval, train_full, test,
      macro_dim, news_dim
    """
    tickers      = tickers or DEFAULT_TICKERS
    ticker_to_id = ticker_to_id or {t: i for i, t in enumerate(DEFAULT_TICKERS)}

    dp = DataPrepare(
        dataset_path=pkl_path,
        window_size=window_size,
        news_dim=news_dim,
        quality_dim=quality_dim,
        tickers=tickers,
        ticker_to_id=ticker_to_id,
        price_mode=price_mode,
        label_mode=label_mode,
        include_ticker_id=True,
    )

    valid_T      = [dp.get_max_T(t) for t in tickers if dp.get_max_T(t) > 0]
    if not valid_T:
        raise RuntimeError("No valid tickers found in dataset.")
    global_T_max = min(valid_T)
    inner_T      = int(global_T_max * (train_ratio + valid_ratio))
    hval_split   = int(inner_T * 0.80)

    print(f"\n{'='*60}")
    print(f"Data split summary")
    print(f"{'='*60}")
    print(f"  global_T_max = {global_T_max}")
    print(f"  inner_T      = {inner_T}  (train+val boundary)")
    print(f"  hval_split   = {hval_split}  (HP search train/val boundary)")
    print(f"  Test range   : [{inner_T} : {global_T_max}]")

    def _add_ind(s):
        if not s or not len(s.get("label", [])):
            return s
        s = dict(s)
        s["indicators"] = torch.cat([s["s_o"], s["s_h"], s["s_c"]], dim=-1)
        return s

    def _merge(dicts, shuffle=False):
        if not dicts:
            return {}
        m: dict = {}
        for key in dicts[0]:
            parts = [d[key] for d in dicts if key in d and isinstance(d[key], torch.Tensor)]
            if parts:
                m[key] = torch.cat(parts, dim=0)
        if shuffle and "label" in m:
            idx = torch.randperm(len(m["label"]))
            for k in m:
                m[k] = m[k][idx]
        return m

    macro_dim = news_dim_detected = None
    trhv, vahv, trfl, tel = [], [], [], []

    for ticker in tickers:
        if dp.get_max_T(ticker) == 0:
            continue
        trh, vah, _ = dp.prepare_data(ticker, train_end=hval_split,  val_end=inner_T,      test_end=inner_T)
        trf, _,  te = dp.prepare_data(ticker, train_end=inner_T,     val_end=inner_T,      test_end=global_T_max)
        trh = _add_ind(trh); vah = _add_ind(vah)
        trf = _add_ind(trf); te  = _add_ind(te)

        if macro_dim is None and trh and len(trh.get("label", [])) > 0:
            macro_dim        = trh["s_m"].shape[-1]
            news_dim_detected = trh["s_n"].shape[-1]

        if trh and len(trh.get("label", [])): trhv.append(trh)
        if vah and len(vah.get("label", [])): vahv.append(vah)
        if trf and len(trf.get("label", [])): trfl.append(trf)
        if te  and len(te.get("label",  [])): tel.append(te)

        print(f"  {ticker}: hval_tr={len(trh.get('label',[]))} "
              f"hval_va={len(vah.get('label',[]))} "
              f"te={len(te.get('label',[]))}")

    train_hval = _merge(trhv, shuffle=True)
    val_hval   = _merge(vahv)
    train_full = _merge(trfl, shuffle=True)
    test       = _merge(tel)

    print(f"\n  Merged sizes:")
    print(f"    train_hval={len(train_hval.get('label',[]))}")
    print(f"    val_hval  ={len(val_hval.get('label',[]))}")
    print(f"    train_full={len(train_full.get('label',[]))}")
    print(f"    test      ={len(test.get('label',[]))}")
    print(f"    macro_dim ={macro_dim}  news_dim={news_dim_detected}")

    return {
        "global_T_max": global_T_max,
        "inner_T":      inner_T,
        "hval_split":   hval_split,
        "train_hval":   train_hval,
        "val_hval":     val_hval,
        "train_full":   train_full,
        "test":         test,
        "macro_dim":    macro_dim,
        "news_dim":     news_dim_detected,
    }
