# src/data_loader.py
"""
Fixed Data Loader:
- _load_graph_from_path: reads 'node_x' key (matches KGGenNewsEmbedder .pt format)
  and also handles 'edge_attr' for GATv2 edge features
- Rolling Quantile Labeling (no look-ahead)
- Leakage-free normalization (train stats only)
"""

import torch
import pickle
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from configs.config import TrainConfig, GlobalConfig
import os


class data_prepare:

    def __init__(self, dataset_path: str):
        with open(dataset_path, "rb") as f:
            self.raw_data = pickle.load(f)
        self.window_size = TrainConfig.window_size
        self.news_dim    = TrainConfig.news_embed_dim
        print(f"📦 Loaded dataset with {len(self.raw_data)} trading days")

    def _load_graph_from_path(self, kg_tensor_path):
        """
        Load PyG Data from a .pt file saved by KGGenNewsEmbedder.

        KGGenNewsEmbedder saves:
            {"node_x": ..., "edge_index": ..., "edge_attr": ...,
             "ticker_idx": ..., "nodes": ..., "node_info": ..., "graph_emb": ...}

        CRITICAL FIX: key is 'node_x', NOT 'x' or 'node_features'.
        Also loads 'edge_attr' for GATv2 relation-aware attention.
        """
        if not kg_tensor_path or not isinstance(kg_tensor_path, str):
            return None
        if not os.path.exists(kg_tensor_path):
            return None

        try:
            d = torch.load(kg_tensor_path, map_location="cpu")

            if isinstance(d, Data):
                # Already a PyG Data object (old format)
                return d

            if isinstance(d, dict):
                # New format from KGGenNewsEmbedder
                # Key is 'node_x', not 'x'
                x = d.get("node_x")
                if x is None:
                    # Fallback for any legacy format
                    x = d.get("x") or d.get("node_features")
                if x is None:
                    return None

                if x.dim() == 1:
                    x = x.unsqueeze(0)

                edge_index = d.get("edge_index",
                                   torch.zeros(2, 0, dtype=torch.long))
                edge_attr  = d.get("edge_attr", None)   # 17D for GATv2

                data = Data(x=x, edge_index=edge_index)
                if edge_attr is not None and edge_attr.shape[0] > 0:
                    data.edge_attr = edge_attr
                return data

            # Bare tensor (very old format)
            x = d
            if x.dim() == 1:
                x = x.unsqueeze(0)
            return Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long))

        except Exception as e:
            return None

    def _create_empty_graph(self):
        return Data(
            x=torch.zeros(1, self.news_dim),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
        )

    def prepare_data(self, target_ticker: str):
        dates = sorted(self.raw_data.keys())

        rows = []
        for date_key in dates:
            day_data   = self.raw_data[date_key]
            price_data = day_data.get("price", {}).get(target_ticker)
            if price_data is None:
                continue

            # Extract OHLC — handle dict or Series
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

            macro_data     = day_data.get("macro", {})
            kg_tensor_path = day_data.get("kg_tensor", {}).get(target_ticker)

            rows.append({
                "date":   date_key,
                "s_o":    s_o,
                "s_h":    s_h,
                "s_c":    s_c,
                "macro":  macro_data,
                "kg_path": kg_tensor_path,
            })

        if len(rows) < self.window_size + 1:
            print(f"⚠️ {target_ticker}: Not enough data ({len(rows)} days)")
            return {}, {}, {}

        print(f"📊 {target_ticker}: {len(rows)} days with valid price data")

        # ── Rolling Quantile Labeling (no look-ahead) ─────────────────────────
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

        graph_paths_all = []
        labels = np.zeros(T, dtype=int)

        for t in range(T):
            for w, row in enumerate(rows[t: t + self.window_size]):
                s_o_all[t, w, 0] = row["s_o"]
                s_h_all[t, w, 0] = row["s_h"]
                s_c_all[t, w, 0] = row["s_c"]
                s_m_all[t, w, :]  = [row["macro"].get(k, 0) for k in macro_keys]

            graph_paths_all.append(rows[t + self.window_size]["kg_path"])

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
        print(f" ⚖️ Label Distribution: {dict(zip(unique, counts))}")

        # ── Leakage-free normalization ────────────────────────────────────────
        train_ratio = getattr(TrainConfig, "train_ratio", 0.7)
        valid_ratio = getattr(TrainConfig, "valid_ratio", 0.15)
        train_end   = int(T * train_ratio)
        valid_end   = int(T * (train_ratio + valid_ratio))

        def _norm(arr, ref):
            mu  = ref.mean()
            std = ref.std() + 1e-8
            return (arr - mu) / std

        s_o_all = _norm(s_o_all, s_o_all[:train_end])
        s_h_all = _norm(s_h_all, s_h_all[:train_end])
        s_c_all = _norm(s_c_all, s_c_all[:train_end])

        mu_m  = s_m_all[:train_end].mean(axis=(0, 1), keepdims=True)
        std_m = s_m_all[:train_end].std( axis=(0, 1), keepdims=True) + 1e-8
        s_m_all = (s_m_all - mu_m) / std_m

        s_o_t = torch.tensor(s_o_all, dtype=torch.float32)
        s_h_t = torch.tensor(s_h_all, dtype=torch.float32)
        s_c_t = torch.tensor(s_c_all, dtype=torch.float32)
        s_m_t = torch.tensor(s_m_all, dtype=torch.float32)
        lbl_t = torch.tensor(labels,   dtype=torch.long)

        # Load graphs
        print(f"📊 Loading {T} graphs for {target_ticker}...")
        graph_list = []
        for path in graph_paths_all:
            g = self._load_graph_from_path(path)
            if g is None:
                g = self._create_empty_graph()
            graph_list.append(g)

        def _split(tensor, a, b):
            return tensor[a:b]

        train_data = {
            "s_o": _split(s_o_t, 0, train_end),
            "s_h": _split(s_h_t, 0, train_end),
            "s_c": _split(s_c_t, 0, train_end),
            "s_m": _split(s_m_t, 0, train_end),
            "s_n_graphs": graph_list[:train_end],
            "label": _split(lbl_t, 0, train_end),
        }
        valid_data = {
            "s_o": _split(s_o_t, train_end, valid_end),
            "s_h": _split(s_h_t, train_end, valid_end),
            "s_c": _split(s_c_t, train_end, valid_end),
            "s_m": _split(s_m_t, train_end, valid_end),
            "s_n_graphs": graph_list[train_end:valid_end],
            "label": _split(lbl_t, train_end, valid_end),
        }
        test_data = {
            "s_o": _split(s_o_t, valid_end, T),
            "s_h": _split(s_h_t, valid_end, T),
            "s_c": _split(s_c_t, valid_end, T),
            "s_m": _split(s_m_t, valid_end, T),
            "s_n_graphs": graph_list[valid_end:],
            "label": _split(lbl_t, valid_end, T),
        }

        print(f"✅ {target_ticker}: Train={len(train_data['label'])} "
              f"Valid={len(valid_data['label'])} Test={len(test_data['label'])}")
        return train_data, valid_data, test_data