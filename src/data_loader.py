# src/data_loader.py
"""
V4 — Data Loader (Voyage embedding, no GATv2)

Thay đổi so với V3:
  - Đọc "news_embedding" key thay vì "kg_tensor" key
  - s_n là tensor (T, 1024) thay vì list of PyG Data objects
  - _create_zero_news() trả về zeros(1024) thay vì empty graph
  - Không cần PyG Data, không cần _load_graph_from_path

Luồng:
  unified_dataset.pkl
    → date → "news_embedding" → {ticker: [1024D vector]}
    → s_n_all: (T, window_size, 1024)
    → model.py: NewsEncoder(1024 → 128) → MSGCA

Fusion order (model.py):
  price (v_i) → Stage1 MSGCA với news (v_n) → Stage2 MSGCA với macro (v_m) → predict
"""

import os
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import torch

from configs.config import TrainConfig, GlobalConfig


# News embedding dimension: Voyage-3-large output = 1024
NEWS_EMB_DIM = 1024


class data_prepare:

    def __init__(self, dataset_path: str):
        with open(dataset_path, "rb") as f:
            self.raw_data = pickle.load(f)
        self.window_size = TrainConfig.window_size
        # news_embed_dim phải khớp với Voyage dim = 1024
        # model.py: NewsEncoder(1024, dim) — project xuống dim=128 khi train
        self.news_dim = NEWS_EMB_DIM
        print(f"Loaded dataset: {len(self.raw_data)} trading days")
        print(f"News embedding dim: {self.news_dim} (Voyage-3-large)")

    def prepare_data(self, target_ticker: str):
        dates = sorted(self.raw_data.keys())
        rows  = []

        for date_key in dates:
            day_data   = self.raw_data[date_key]
            price_data = day_data.get("price", {}).get(target_ticker)
            if price_data is None:
                continue

            def _extract(d, *keys):
                for k in keys:
                    v = d.get(k) if isinstance(d, dict) else None
                    if v is not None:
                        if hasattr(v, "iloc"):
                            v = v.iloc[0] if len(v) > 0 else None
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            pass
                return None

            s_o = _extract(price_data, "Open",  "open")
            s_h = _extract(price_data, "High",  "high")
            s_c = _extract(price_data, "Close", "close")
            if s_o is None or s_h is None or s_c is None:
                continue

            macro_data = day_data.get("macro", {})

            # Read news_embedding — 1024D Voyage vector
            news_emb_dict = day_data.get("news_embedding", {})
            news_emb = news_emb_dict.get(target_ticker, None)

            rows.append({
                "date":     date_key,
                "s_o":      s_o,
                "s_h":      s_h,
                "s_c":      s_c,
                "macro":    macro_data,
                "news_emb": news_emb,  # list of 1024 floats or None
            })

        if len(rows) < self.window_size + 1:
            print(f"  {target_ticker}: Not enough data ({len(rows)} days)")
            return {}, {}, {}

        print(f"  {target_ticker}: {len(rows)} days with valid price data")

        # Count coverage
        n_with_news = sum(1 for r in rows if r["news_emb"] and len(r["news_emb"]) > 0)
        print(f"  News embedding coverage: {n_with_news}/{len(rows)} days")

        # Rolling Quantile Labeling (no look-ahead)
        close_prices   = pd.Series([r["s_c"] for r in rows])
        returns_series = close_prices.pct_change().fillna(0)
        roll_w    = 20
        roll_low  = returns_series.rolling(roll_w).quantile(0.33).shift(1)
        roll_high = returns_series.rolling(roll_w).quantile(0.66).shift(1)

        T = len(rows) - self.window_size

        s_o_all = np.zeros((T, self.window_size, 1))
        s_h_all = np.zeros((T, self.window_size, 1))
        s_c_all = np.zeros((T, self.window_size, 1))

        first_macro = rows[0]["macro"]
        macro_keys  = sorted(first_macro.keys())
        macro_dim   = len(macro_keys)
        s_m_all     = np.zeros((T, self.window_size, macro_dim))

        # s_n_all: (T, window_size, news_dim)
        s_n_all     = np.zeros((T, self.window_size, self.news_dim))

        labels = np.zeros(T, dtype=int)

        for t in range(T):
            for w, row in enumerate(rows[t: t + self.window_size]):
                s_o_all[t, w, 0] = row["s_o"]
                s_h_all[t, w, 0] = row["s_h"]
                s_c_all[t, w, 0] = row["s_c"]
                s_m_all[t, w, :]  = [row["macro"].get(k, 0) for k in macro_keys]

                # Fill news embedding for this window position
                emb = row["news_emb"]
                if emb and len(emb) == self.news_dim:
                    s_n_all[t, w, :] = emb
                # else: stays zeros (no news for this day)

            # Label for day t+window_size
            idx     = t + self.window_size
            ret     = returns_series.iloc[idx]
            r_low   = roll_low.iloc[idx]
            r_high  = roll_high.iloc[idx]

            if pd.isna(r_low) or pd.isna(r_high):
                labels[t] = 1
            elif abs(ret) < 0.001:
                labels[t] = 1
            elif ret < r_low:
                labels[t] = 0
            elif ret > r_high:
                labels[t] = 2
            else:
                labels[t] = 1

        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Label distribution: {dict(zip(unique, counts))}")

        # Fix lỗi Nan
        s_o_all = np.nan_to_num(s_o_all, nan=0.0)
        s_h_all = np.nan_to_num(s_h_all, nan=0.0)
        s_c_all = np.nan_to_num(s_c_all, nan=0.0)
        s_m_all = np.nan_to_num(s_m_all, nan=0.0)
        s_n_all = np.nan_to_num(s_n_all, nan=0.0)

        # Leakage-free normalization (train stats only)
        train_ratio = getattr(TrainConfig, "train_ratio", 0.7)
        valid_ratio = getattr(TrainConfig, "valid_ratio", 0.15)
        train_end   = int(T * train_ratio)
        valid_end   = int(T * (train_ratio + valid_ratio))

        def _norm_arr(arr, ref):
            mu  = ref.mean()
            std = ref.std() + 1e-8
            return (arr - mu) / std

        s_o_all = _norm_arr(s_o_all, s_o_all[:train_end])
        s_h_all = _norm_arr(s_h_all, s_h_all[:train_end])
        s_c_all = _norm_arr(s_c_all, s_c_all[:train_end])

        mu_m  = s_m_all[:train_end].mean(axis=(0, 1), keepdims=True)
        std_m = s_m_all[:train_end].std( axis=(0, 1), keepdims=True) + 1e-8
        s_m_all = (s_m_all - mu_m) / std_m

        # Normalize news embedding per-dimension using train stats
        mu_n  = s_n_all[:train_end].mean(axis=(0, 1), keepdims=True)
        std_n = s_n_all[:train_end].std( axis=(0, 1), keepdims=True) + 1e-8
        s_n_all = (s_n_all - mu_n) / std_n

        s_o_t = torch.tensor(s_o_all, dtype=torch.float32)
        s_h_t = torch.tensor(s_h_all, dtype=torch.float32)
        s_c_t = torch.tensor(s_c_all, dtype=torch.float32)
        s_m_t = torch.tensor(s_m_all, dtype=torch.float32)
        s_n_t = torch.tensor(s_n_all, dtype=torch.float32)
        lbl_t = torch.tensor(labels,  dtype=torch.long)

        def _split(tensor, a, b):
            return tensor[a:b]

        train_data = {
            "s_o": _split(s_o_t, 0, train_end),
            "s_h": _split(s_h_t, 0, train_end),
            "s_c": _split(s_c_t, 0, train_end),
            "s_m": _split(s_m_t, 0, train_end),
            "s_n": _split(s_n_t, 0, train_end),       # (T_train, window, 1024)
            "label": _split(lbl_t, 0, train_end),
        }
        valid_data = {
            "s_o": _split(s_o_t, train_end, valid_end),
            "s_h": _split(s_h_t, train_end, valid_end),
            "s_c": _split(s_c_t, train_end, valid_end),
            "s_m": _split(s_m_t, train_end, valid_end),
            "s_n": _split(s_n_t, train_end, valid_end),
            "label": _split(lbl_t, train_end, valid_end),
        }
        test_data = {
            "s_o": _split(s_o_t, valid_end, T),
            "s_h": _split(s_h_t, valid_end, T),
            "s_c": _split(s_c_t, valid_end, T),
            "s_m": _split(s_m_t, valid_end, T),
            "s_n": _split(s_n_t, valid_end, T),
            "label": _split(lbl_t, valid_end, T),
        }

        print(f"  {target_ticker}: Train={len(train_data['label'])} "
              f"Valid={len(valid_data['label'])} Test={len(test_data['label'])}")
        return train_data, valid_data, test_data