# baselines/data_adapter.py
"""
BaselineDataPrepare — Kế thừa data_prepare gốc, bổ sung:
  - indicators   : (T, W, 3)   — s_o, s_h, s_c ghép lại
  - s_news_per_day: (T, W, 128) — graph/news embedding từng ngày trong cửa sổ
  - s_graph_emb  : (T, 128)    — graph embedding của ngày target (từ file .pt)
  - s_m          : (T, W, macro_dim) — giữ nguyên như cũ
  - label        : (T,)        — rolling-quantile label (giống data_loader.py)

KHÔNG thay đổi bất kỳ file gốc nào.
"""

import os
import torch
import numpy as np
import pandas as pd

from configs.config import TrainConfig
from src.data_loader import data_prepare

# Chiều embedding của GNN output (graph_out_dim=128 trong configs)
BASELINE_NEWS_DIM = 128


class BaselineDataPrepare(data_prepare):
    """
    Mở rộng data_prepare để trích xuất flat embeddings cho baseline models.

    Sử dụng:
        adapter = BaselineDataPrepare(pkl_path)
        train, valid, test = adapter.prepare_baseline_data("TSLA")
    """

    NEWS_DIM = BASELINE_NEWS_DIM

    # ------------------------------------------------------------------
    # Helper: lấy news_embedding vector từ một ngày trong raw_data
    # ------------------------------------------------------------------
    def _get_news_emb_for_day(self, day_data: dict, target_ticker: str) -> np.ndarray:
        """
        Trả về vector 128-dim từ day_data["news_embedding"][ticker].
        Nếu thiếu → zero vector.
        """
        emb = day_data.get("news_embedding", {}).get(target_ticker)
        if emb is None or not isinstance(emb, list) or len(emb) == 0:
            return np.zeros(self.NEWS_DIM, dtype=np.float32)

        arr = np.array(emb, dtype=np.float32)
        # Padding / truncation nếu không khớp chiều
        if len(arr) < self.NEWS_DIM:
            arr = np.pad(arr, (0, self.NEWS_DIM - len(arr)))
        elif len(arr) > self.NEWS_DIM:
            arr = arr[: self.NEWS_DIM]
        return arr

    # ------------------------------------------------------------------
    # Helper: load graph_emb từ file .pt được lưu bởi tensorize_and_embed
    # ------------------------------------------------------------------
    def _get_graph_emb_from_pt(self, kg_path) -> np.ndarray:
        """
        Load pre-computed graph embedding (128-dim) từ file .pt.
        File được tạo bởi KGGenNewsEmbedder.tensorize_and_embed().
        """
        if not kg_path or not isinstance(kg_path, str) or not os.path.exists(kg_path):
            return np.zeros(self.NEWS_DIM, dtype=np.float32)
        try:
            data = torch.load(kg_path, map_location="cpu", weights_only=False)
            if isinstance(data, dict) and "graph_emb" in data:
                emb = data["graph_emb"]
                arr = emb.numpy().astype(np.float32) if isinstance(emb, torch.Tensor) else np.array(emb, dtype=np.float32)
                if len(arr) < self.NEWS_DIM:
                    arr = np.pad(arr, (0, self.NEWS_DIM - len(arr)))
                return arr[: self.NEWS_DIM]
        except Exception:
            pass
        return np.zeros(self.NEWS_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------
    def prepare_baseline_data(self, target_ticker: str):
        """
        Chuẩn bị dữ liệu cho baseline models.

        Returns
        -------
        train_dict, valid_dict, test_dict : dict
            Mỗi dict chứa:
              - 'indicators'     : Tensor (N, W, 3)   — giá ghép
              - 's_o','s_h','s_c': Tensor (N, W, 1)   — giá riêng lẻ
              - 's_m'            : Tensor (N, W, M)   — macro
              - 's_news_per_day' : Tensor (N, W, 128) — news emb mỗi ngày trong window
              - 's_graph_emb'    : Tensor (N, 128)    — graph emb ngày target
              - 'label'          : Tensor (N,)        — 0=DOWN,1=FLAT,2=UP
        """
        dates = sorted(self.raw_data.keys())

        # ── 1. Thu thập rows ──────────────────────────────────────────
        rows = []
        for date_key in dates:
            day_data = self.raw_data[date_key]
            price_data = day_data.get("price", {}).get(target_ticker)
            if price_data is None:
                continue

            if isinstance(price_data, dict):
                s_o = price_data.get("Open") or price_data.get("open")
                s_h = price_data.get("High") or price_data.get("high")
                s_c = price_data.get("Close") or price_data.get("close")
            else:
                continue

            if s_o is None or s_h is None or s_c is None:
                continue
            try:
                s_o, s_h, s_c = float(s_o), float(s_h), float(s_c)
            except (ValueError, TypeError):
                continue

            macro_data = day_data.get("macro", {})
            kg_path    = day_data.get("kg_tensor", {}).get(target_ticker)
            # news_embedding là GNN graph embedding (128-dim) lưu trong builder.py
            news_emb   = self._get_news_emb_for_day(day_data, target_ticker)

            rows.append({
                "date": date_key,
                "s_o": s_o, "s_h": s_h, "s_c": s_c,
                "macro": macro_data,
                "kg_path": kg_path,
                "news_emb": news_emb,
            })

        if len(rows) < self.window_size + 1:
            print(f"⚠️  {target_ticker}: Không đủ dữ liệu ({len(rows)} ngày)")
            return {}, {}, {}

        print(f"📊 {target_ticker}: {len(rows)} ngày hợp lệ")

        # ── 2. Rolling Quantile Labels (giống data_loader.py) ────────
        close_prices   = pd.Series([r["s_c"] for r in rows])
        returns_series = close_prices.pct_change().fillna(0)
        roll_low  = returns_series.rolling(20).quantile(0.33).shift(1)
        roll_high = returns_series.rolling(20).quantile(0.66).shift(1)

        # ── 3. Khởi tạo arrays ───────────────────────────────────────
        T          = len(rows) - self.window_size
        macro_keys = sorted(rows[0]["macro"].keys())
        macro_dim  = len(macro_keys)

        s_o_all        = np.zeros((T, self.window_size, 1),             dtype=np.float32)
        s_h_all        = np.zeros((T, self.window_size, 1),             dtype=np.float32)
        s_c_all        = np.zeros((T, self.window_size, 1),             dtype=np.float32)
        s_m_all        = np.zeros((T, self.window_size, macro_dim),     dtype=np.float32)
        s_news_per_day = np.zeros((T, self.window_size, self.NEWS_DIM), dtype=np.float32)
        s_graph_emb    = np.zeros((T, self.NEWS_DIM),                   dtype=np.float32)
        labels         = np.zeros(T, dtype=np.int64)

        # ── 4. Xây dựng windows ──────────────────────────────────────
        for t in range(T):
            window_rows = rows[t : t + self.window_size]
            target_row  = rows[t + self.window_size]

            for w, row in enumerate(window_rows):
                s_o_all[t, w, 0] = row["s_o"]
                s_h_all[t, w, 0] = row["s_h"]
                s_c_all[t, w, 0] = row["s_c"]
                s_m_all[t, w]    = [row["macro"].get(k, 0.0) for k in macro_keys]
                s_news_per_day[t, w] = row["news_emb"]

            # Graph embedding của ngày target (từ file .pt)
            s_graph_emb[t] = self._get_graph_emb_from_pt(target_row["kg_path"])

            # Label
            idx      = t + self.window_size
            cur_ret  = returns_series.iloc[idx]
            cur_low  = roll_low.iloc[idx]
            cur_high = roll_high.iloc[idx]

            if pd.isna(cur_low) or pd.isna(cur_high):
                labels[t] = 1
            elif abs(cur_ret) < 0.001:
                labels[t] = 1
            elif cur_ret < cur_low:
                labels[t] = 0   # DOWN
            elif cur_ret > cur_high:
                labels[t] = 2   # UP
            else:
                labels[t] = 1   # FLAT

        label_dist = dict(zip(*np.unique(labels, return_counts=True)))
        print(f"   ⚖️  Label: {label_dist}")

        # ── 5. Normalization (chỉ trên Train để tránh leakage) ───────
        train_ratio = getattr(TrainConfig, "train_ratio", 0.7)
        valid_ratio = getattr(TrainConfig, "valid_ratio", 0.15)
        train_end   = int(T * train_ratio)
        valid_end   = int(T * (train_ratio + valid_ratio))

        def _znorm_inplace(arr, train_end):
            mu  = arr[:train_end].mean()
            std = arr[:train_end].std() + 1e-8
            arr -= mu
            arr /= std

        _znorm_inplace(s_o_all, train_end)
        _znorm_inplace(s_h_all, train_end)
        _znorm_inplace(s_c_all, train_end)

        mu_m  = s_m_all[:train_end].mean(axis=(0, 1), keepdims=True)
        std_m = s_m_all[:train_end].std(axis=(0, 1),  keepdims=True) + 1e-8
        s_m_all = (s_m_all - mu_m) / std_m

        # indicators = cat(s_o, s_h, s_c) → (T, W, 3)
        indicators = np.concatenate([s_o_all, s_h_all, s_c_all], axis=-1)

        # ── 6. Chuyển sang tensor và split ───────────────────────────
        def _to_dict(sl: slice) -> dict:
            return {
                "indicators":      torch.tensor(indicators[sl],      dtype=torch.float32),
                "s_o":             torch.tensor(s_o_all[sl],         dtype=torch.float32),
                "s_h":             torch.tensor(s_h_all[sl],         dtype=torch.float32),
                "s_c":             torch.tensor(s_c_all[sl],         dtype=torch.float32),
                "s_m":             torch.tensor(s_m_all[sl],         dtype=torch.float32),
                "s_news_per_day":  torch.tensor(s_news_per_day[sl],  dtype=torch.float32),
                "s_graph_emb":     torch.tensor(s_graph_emb[sl],     dtype=torch.float32),
                "label":           torch.tensor(labels[sl],          dtype=torch.long),
            }

        train_data = _to_dict(slice(None, train_end))
        valid_data = _to_dict(slice(train_end, valid_end))
        test_data  = _to_dict(slice(valid_end, None))

        print(f"✅ {target_ticker}: Train={train_end} | Valid={valid_end-train_end} | Test={T-valid_end}")
        return train_data, valid_data, test_data