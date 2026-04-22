# src/data_loader.py
"""
Data Loader — V5.6

V5.6 change: NEWS_EMB_DIM now derived from GlobalConfig.news_emb_dim()
  which reads TrainConfig.news_embedder ("finbert" → 768D, "voyage" → 1024D).
  No more hardcoded dimension — switch embedder in config.py only.
"""

import pickle
import numpy as np
import pandas as pd
import torch

from configs.config import TrainConfig, GlobalConfig

# Derived from TrainConfig.news_embedder — change config.py only
NEWS_EMB_DIM = GlobalConfig.news_emb_dim()

VOL_WINDOW   = 20
VOL_FALLBACK = 0.02

ALL_TICKERS  = ["TSLA", "AAPL", "AMZN", "MSFT", "GOOGL",
                "META", "BA",   "JPM",  "WMT"]
TICKER_TO_ID = {t: i for i, t in enumerate(ALL_TICKERS)}
N_TICKERS    = len(ALL_TICKERS)


def _extract_float(d, *keys):
    for k in keys:
        v = d.get(k) if isinstance(d, dict) else None
        if v is None: continue
        if hasattr(v, "iloc"):
            v = v.iloc[0] if len(v) > 0 else None
        try:
            return float(v)
        except (TypeError, ValueError):
            pass
    return None


class data_prepare:

    _EMBEDDER_LABEL = {
        "finbert": "FinBERT per-triple CLS",
        "voyage":  "Voyage-finance-2",
    }

    def __init__(
        self,
        dataset_path:      str,
        price_mode:        str  = "vol_adjusted",
        label_mode:        str  = "rolling",
        include_ticker_id: bool = True,
    ):
        with open(dataset_path, "rb") as f:
            self.raw_data = pickle.load(f)
        self.window_size       = TrainConfig.window_size
        self.news_dim          = NEWS_EMB_DIM
        self.price_mode        = price_mode
        self.label_mode        = label_mode
        self.include_ticker_id = include_ticker_id
        self._cache_rows       = {}

        embedder       = TrainConfig.news_embedder
        embedder_label = self._EMBEDDER_LABEL.get(embedder, embedder)

        print(f"Loaded dataset : {len(self.raw_data)} trading days")
        print(f"Price mode     : {self.price_mode}")
        print(f"Label mode     : {self.label_mode}")
        print(f"News embed dim : {self.news_dim}D ({embedder_label})")
        print(f"Ticker ID      : {'enabled' if include_ticker_id else 'disabled'}")

    def _load_rows(self, target_ticker: str):
        if target_ticker in self._cache_rows:
            return self._cache_rows[target_ticker]

        dates = sorted(self.raw_data.keys())
        rows  = []
        for date_key in dates:
            day = self.raw_data[date_key]
            p   = day.get("price", {}).get(target_ticker)
            if p is None: continue

            s_o = _extract_float(p, "Open",  "open")
            s_h = _extract_float(p, "High",  "high")
            s_c = _extract_float(p, "Close", "close")
            if None in (s_o, s_h, s_c): continue

            rows.append({
                "date":     date_key,
                "s_o":      s_o,
                "s_h":      s_h,
                "s_c":      s_c,
                "macro":    day.get("macro", {}),
                "news_emb": day.get("news_embedding", {}).get(target_ticker, None),
            })

        self._cache_rows[target_ticker] = rows
        return rows

    def get_max_T(self, target_ticker: str) -> int:
        rows = self._load_rows(target_ticker)
        if len(rows) < self.window_size + 1: return 0
        return len(rows) - self.window_size

    def prepare_data(self, target_ticker: str, train_end=None, val_end=None, test_end=None):
        rows = self._load_rows(target_ticker)
        if len(rows) < self.window_size + 1:
            return {}, {}, {}

        T_max = len(rows) - self.window_size

        if train_end is None or val_end is None or test_end is None:
            train_ratio = getattr(TrainConfig, "train_ratio", 0.70)
            valid_ratio = getattr(TrainConfig, "valid_ratio", 0.15)
            train_end   = int(T_max * train_ratio)
            val_end     = int(T_max * (train_ratio + valid_ratio))
            test_end    = T_max

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
            for w, row in enumerate(rows[t: t + self.window_size]):
                s_m_all[t, w, :] = [row["macro"].get(k, 0.0) for k in macro_keys]
        s_m_all = self._znorm(s_m_all, s_m_all[:train_end])

        s_n_all, news_mask_all = self._build_news(rows, T, train_end)

        for name, arr in [("s_o", s_o_all), ("s_h", s_h_all), ("s_c", s_c_all),
                          ("s_m", s_m_all), ("s_n", s_n_all)]:
            n_nan = int(np.isnan(arr).sum())
            if n_nan > 0: arr[:] = np.nan_to_num(arr, nan=0.0)

        s_o_t  = torch.tensor(s_o_all,      dtype=torch.float32)
        s_h_t  = torch.tensor(s_h_all,      dtype=torch.float32)
        s_c_t  = torch.tensor(s_c_all,      dtype=torch.float32)
        s_m_t  = torch.tensor(s_m_all,      dtype=torch.float32)
        s_n_t  = torch.tensor(s_n_all,      dtype=torch.float32)
        mask_t = torch.tensor(news_mask_all, dtype=torch.bool)
        lbl_t  = torch.tensor(labels,        dtype=torch.long)

        tid_t  = torch.full(
            (T,), TICKER_TO_ID.get(target_ticker, 0), dtype=torch.long
        )

        def _s(x, a, b): return x[a:b]

        def _make(a, b):
            if a >= b: return {}
            d = {
                "s_o":       _s(s_o_t,  a, b),
                "s_h":       _s(s_h_t,  a, b),
                "s_c":       _s(s_c_t,  a, b),
                "s_m":       _s(s_m_t,  a, b),
                "s_n":       _s(s_n_t,  a, b),
                "news_mask": _s(mask_t, a, b),
                "label":     _s(lbl_t,  a, b),
            }
            if self.include_ticker_id:
                d["ticker_id"] = _s(tid_t, a, b)
            return d

        tr = _make(0,         train_end)
        va = _make(train_end, val_end)
        te = _make(val_end,   test_end)
        return tr, va, te

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
            if close_arr[i-1] > 0: close_lr[i] = np.log(close_arr[i] / close_arr[i-1])
            if open_arr[i-1] > 0:  open_lr[i]  = np.log(open_arr[i]  / open_arr[i-1])
            if high_arr[i-1] > 0:  high_lr[i]  = np.log(high_arr[i]  / high_arr[i-1])

        realized_vol = np.full(n, VOL_FALLBACK, dtype=np.float64)
        for i in range(VOL_WINDOW, n):
            v = close_lr[i - VOL_WINDOW: i].std()
            realized_vol[i] = max(v, 1e-6)
        realized_vol[:VOL_WINDOW] = realized_vol[VOL_WINDOW]

        close_va = close_lr / realized_vol
        open_va  = open_lr  / realized_vol
        high_va  = high_lr  / realized_vol

        s_o = np.zeros((T, self.window_size, 1), dtype=np.float32)
        s_h = np.zeros((T, self.window_size, 1), dtype=np.float32)
        s_c = np.zeros((T, self.window_size, 1), dtype=np.float32)
        for t in range(T):
            s_o[t, :, 0] = open_va[t:  t + self.window_size]
            s_h[t, :, 0] = high_va[t:  t + self.window_size]
            s_c[t, :, 0] = close_va[t: t + self.window_size]

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
                s_o[t, :, 0] = open_arr[t:  t + self.window_size]  / base - 1.0
                s_h[t, :, 0] = high_arr[t:  t + self.window_size]  / base - 1.0
                s_c[t, :, 0] = close_arr[t: t + self.window_size]  / base - 1.0
        s_o = self._znorm(s_o, s_o[:train_end])
        s_h = self._znorm(s_h, s_h[:train_end])
        s_c = self._znorm(s_c, s_c[:train_end])
        return s_o, s_h, s_c

    def _absolute(self, close_arr, open_arr, high_arr, T, train_end):
        s_o = np.zeros((T, self.window_size, 1), dtype=np.float32)
        s_h = np.zeros((T, self.window_size, 1), dtype=np.float32)
        s_c = np.zeros((T, self.window_size, 1), dtype=np.float32)
        for t in range(T):
            s_o[t, :, 0] = open_arr[t:  t + self.window_size]
            s_h[t, :, 0] = high_arr[t:  t + self.window_size]
            s_c[t, :, 0] = close_arr[t: t + self.window_size]
        s_o = self._znorm(s_o, s_o[:train_end])
        s_h = self._znorm(s_h, s_h[:train_end])
        s_c = self._znorm(s_c, s_c[:train_end])
        return s_o, s_h, s_c

    def _build_news(self, rows, T, train_end):
        s_n  = np.zeros((T, self.window_size, self.news_dim), dtype=np.float32)
        mask = np.ones ((T, self.window_size),                dtype=bool)

        for t in range(T):
            for w, row in enumerate(rows[t: t + self.window_size]):
                emb = row["news_emb"]
                if emb is None or len(emb) != self.news_dim: continue
                arr = np.array(emb, dtype=np.float32)
                if np.allclose(arr, 0): continue
                s_n[t, w, :]  = arr
                mask[t, w]    = False

        s_n = self._znorm(s_n, s_n[:train_end])
        s_n[mask] = 0.0
        return s_n, mask

    def _build_labels(self, ret_s: pd.Series, T: int, train_end: int) -> np.ndarray:
        if self.label_mode == "rolling":
            return self._labels_rolling(ret_s, T, train_end)
        elif self.label_mode == "fixed":
            return self._labels_fixed(ret_s, T, train_end)
        elif self.label_mode == "volatility":
            return self._labels_volatility(ret_s, T, train_end)
        raise ValueError(f"Unknown label_mode: {self.label_mode!r}")

    def _labels_rolling(self, ret_s, T, train_end):
        ws    = self.window_size
        rl    = ret_s.rolling(20).quantile(0.33).shift(1)
        rh    = ret_s.rolling(20).quantile(0.66).shift(1)
        tr    = [ret_s.iloc[t + ws] for t in range(train_end)]
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
        ws    = self.window_size
        tr    = [ret_s.iloc[t + ws] for t in range(train_end)]
        q33   = np.percentile(tr, 33)
        q66   = np.percentile(tr, 66)
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