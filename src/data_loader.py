# src/data_loader.py
import pandas as pd
import numpy as np
import torch
from configs.config import TrainConfig  # Đồng bộ với Config

class data_prepare:
    def __init__(self, data_path) -> None:
        self.data_path = data_path

    # ======================================================
    # LABEL: RETURN Calculation
    # r_t = close_t / close_{t-1} - 1
    # ======================================================
    def create_return(self, price_df):
        df = price_df.copy()
        df["return"] = df["close"] / df["close"].shift(1) - 1
        df.dropna(inplace=True)
        return df[["return"]]

    def make_window(self, data, window_size):
        """
        data: numpy array (T, D)
        return: (N, window_size, D)
        """
        X = []
        for i in range(len(data) - window_size + 1):
            X.append(data[i:i + window_size])
        return np.array(X)

    def prepare_data(
        self,
        stock_name,
        window_size=TrainConfig.window_size, 
        future_days=1,
        train_ratio=getattr(TrainConfig, 'train_ratio', 0.70),
        valid_ratio=getattr(TrainConfig, 'valid_ratio', 0.15),
        flat_ratio=30 # Giữ lại tham số cũ dù không dùng trong logic mới để tránh lỗi gọi hàm
    ):
        # ==========================
        # 1. LOAD DATA
        # ==========================
        try:
            Data = pd.read_pickle(self.data_path)
        except Exception as e:
            print(f"❌ Error loading pickle file: {e}")
            return {}, {}, {}

        rows = {}
        
        for d, content in Data.items():
            if "price" not in content or stock_name not in content["price"]:
                continue

            price = content["price"][stock_name]
            macro = content["macro"]

            # Lấy embedding tin tức (Giữ nguyên logic của bạn)
            news_section = content.get("news_embedding", {})
            raw_vec = news_section.get(stock_name)

            if raw_vec is None:
                news_vec = np.zeros(TrainConfig.news_embed_dim, dtype=np.float32)
            else:
                try:
                    news_vec = np.array(raw_vec, dtype=np.float32)
                except Exception:
                    news_vec = np.zeros(TrainConfig.news_embed_dim, dtype=np.float32)

            news_dict = {f"news_{i}": v for i, v in enumerate(news_vec)}

            rows[d] = {
                **price,
                **macro,
                **news_dict
            }
        
        if not rows:
            print(f"❌ No data found for stock {stock_name}")
            return {}, {}, {}

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.sort_index(inplace=True)

        # ==========================
        # 2. PRE-PROCESS FEATURES
        # ==========================
        price_df = df[["open", "high", "close"]].astype(float)

        macro_df = df[
            ["vix", "yield_spread_10y_2y",
             "sp500", "sp500_return", "dxy", "wti"]
        ].astype(float)

        news_cols = [c for c in df.columns if c.startswith("news_")]
        news_df = df[news_cols]
        news_df = news_df.apply(pd.to_numeric, errors="coerce")
        news_df = news_df.fillna(0.0)

        # Tạo Return DataFrame (để tính nhãn)
        return_df = self.create_return(price_df)

        # Align Step 1
        price_df = price_df.loc[return_df.index]
        macro_df = macro_df.loc[return_df.index]
        news_df  = news_df.loc[return_df.index]

        # Log-Return cho Input Price (Chuẩn hóa input giá)
        price_df = np.log(price_df / price_df.shift(1))
        price_df.dropna(inplace=True)

        # Align Step 2 (Final Alignment)
        macro_df  = macro_df.loc[price_df.index]
        news_df   = news_df.loc[price_df.index]
        return_df = return_df.loc[price_df.index]

        # Macro Clean
        macro_df = macro_df.replace([np.inf, -np.inf], np.nan)
        macro_df = macro_df.ffill().bfill()

        # ==========================
        # 3. WINDOWING
        # ==========================
        price_np  = price_df.values            
        macro_np  = macro_df.values            
        news_np   = news_df.values             
        return_np = return_df.values           

        price_win = self.make_window(price_np, window_size)
        macro_win = self.make_window(macro_np, window_size)
        news_win  = self.make_window(news_np, window_size)

        # Cắt input để khớp với label (bỏ đoạn đuôi future_days)
        price_win = price_win[:-future_days]
        macro_win = macro_win[:-future_days]
        news_win  = news_win[:-future_days]

        # ==============================================================================
        # [STRATEGY UPDATE] ROLLING QUANTILE LABELING (Dynamic & No Look-Ahead)
        # ==============================================================================
        
        # 1. Prepare Full Series
        full_returns_series = pd.Series(return_np.flatten())
        rolling_window = 20
        
        # 2. Tính Quantile động (33% và 66%) trên cửa sổ quá khứ
        # [CRITICAL]: .shift(1) để loại bỏ Look-Ahead Bias. 
        # Giá trị ngưỡng tại ngày t được tính từ [t-20 ... t-1], KHÔNG bao gồm t.
        roll_low  = full_returns_series.rolling(window=rolling_window).quantile(0.33).shift(1)
        roll_high = full_returns_series.rolling(window=rolling_window).quantile(0.66).shift(1)
        
        # 3. Vectorized Labeling
        # Mặc định là FLAT (1)
        labels_temp = np.full(len(full_returns_series), 1, dtype=int)
        
        # Điều kiện:
        is_down = full_returns_series < roll_low
        is_up   = full_returns_series > roll_high
        
        # [NOISE FILTER]: Nếu biến động tuyệt đối < 0.1% (0.001), ép về FLAT
        # Tránh việc ép model học nhiễu trong thị trường đi ngang biên độ cực hẹp
        is_noise = full_returns_series.abs() < 0.001
        
        # Gán nhãn (Thứ tự quan trọng: Noise filter ghi đè tất cả)
        labels_temp[is_down] = 0
        labels_temp[is_up]   = 2
        labels_temp[is_noise] = 1 # Force Flat
        
        # Xử lý NaN đầu chuỗi (do rolling window) -> Mặc định Flat
        labels_temp[np.isnan(roll_low)] = 1
        
        # 4. Slicing Label để khớp với Window Input
        start_idx = window_size - 1 + future_days
        if start_idx < len(labels_temp):
            label_all = labels_temp[start_idx:]
        else:
            label_all = np.array([])

        # [LOGGING] Kiểm tra phân phối
        unique, counts = np.unique(label_all, return_counts=True)
        dist = dict(zip(unique, counts))
        total_lbl = sum(counts)
        
        print(f" ⚖️  Label Distribution (Rolling Quantile 33/66): {dist}")
        if total_lbl > 0:
            p0 = dist.get(0,0)/total_lbl
            p1 = dist.get(1,0)/total_lbl
            p2 = dist.get(2,0)/total_lbl
            print(f"      Down: {p0:.2%}, Flat: {p1:.2%}, Up: {p2:.2%}")

        # ==========================
        # 4. SPLIT DATASETS & NORMALIZATION (ANTI-LEAKAGE)
        # ==========================
        total_len = len(price_win)
        idx_train = int(total_len * train_ratio)
        idx_valid = int(total_len * (train_ratio + valid_ratio))

        # [STRATEGY UPDATE]: Normalization (Fit on Train, Apply on All)
        # Tính Mean/Std CHỈ trên tập Train
        macro_mean = macro_win[:idx_train].mean(axis=(0, 1), keepdims=True)
        macro_std  = macro_win[:idx_train].std(axis=(0, 1), keepdims=True) + 1e-6
        
        news_mean = news_win[:idx_train].mean(axis=(0, 1), keepdims=True)
        news_std  = news_win[:idx_train].std(axis=(0, 1), keepdims=True) + 1e-6

        # Transform toàn bộ
        macro_win = (macro_win - macro_mean) / macro_std
        news_win  = (news_win - news_mean) / news_std

        def create_dataset(start, end):
            if start >= end: return {}
            return {
                "s_o": torch.tensor(price_win[start:end, :, 0:1], dtype=torch.float32),
                "s_h": torch.tensor(price_win[start:end, :, 1:2], dtype=torch.float32),
                "s_c": torch.tensor(price_win[start:end, :, 2:3], dtype=torch.float32),
                "s_m": torch.tensor(macro_win[start:end], dtype=torch.float32),
                "s_n": torch.tensor(news_win[start:end], dtype=torch.float32),
                "label": torch.tensor(label_all[start:end], dtype=torch.long),
            }

        train_data = create_dataset(0, idx_train)
        valid_data = create_dataset(idx_train, idx_valid)
        test_data  = create_dataset(idx_valid, total_len)

        print(f"Stats: Train={len(train_data.get('label', []))}, Valid={len(valid_data.get('label', []))}, Test={len(test_data.get('label', []))}")
        
        return train_data, valid_data, test_data