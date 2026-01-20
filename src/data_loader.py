# src/data_loader.py
import pandas as pd
import numpy as np
import torch
from configs.config import TrainConfig  # Đồng bộ với Config mới

class data_prepare:
    def __init__(self, data_path) -> None:
        self.data_path = data_path

    # ======================================================
    # LABEL: GIỮ NGUYÊN CÔNG THỨC RETURN
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
        # Lấy window_size từ Config để đảm bảo khớp với Model Input
        window_size=TrainConfig.window_size, 
        future_days=1,
        # Tỷ lệ chia tập dữ liệu (cần thêm valid_ratio vào Config hoặc truyền tay)
        train_ratio=getattr(TrainConfig, 'train_ratio', 0.70),
        valid_ratio=getattr(TrainConfig, 'valid_ratio', 0.15),
        flat_ratio=30
    ):
        # ==========================
        # LOAD DATA
        # ==========================
        Data = pd.read_pickle(self.data_path)
        rows = {}
        
        # In ra cấu trúc keys của 1 ngày đầu tiên để debug (nếu cần)
        # first_date = next(iter(Data))
        # print(f"DEBUG: Keys in Data[{first_date}]: {list(Data[first_date].keys())}")

        for d, content in Data.items():
            # Kiểm tra xem ngày này có dữ liệu giá của cổ phiếu không
            if "price" not in content or stock_name not in content["price"]:
                continue

            price = content["price"][stock_name]
            macro = content["macro"]

            # [CRITICAL FIX]: LẤY DỮ LIỆU TỪ KEY 'news_embedding'
            # Thay vì content["news"] (chứa text), ta lấy content["news_embedding"]
            news_section = content.get("news_embedding", {})
            
            # Lấy vector của stock hiện tại, nếu không có thì trả về None
            raw_vec = news_section.get(stock_name)

            if raw_vec is None:
                # Không có tin hoặc không có embedding -> Zero Vector
                news_vec = np.zeros(TrainConfig.news_embed_dim, dtype=np.float32)
            else:
                # Có dữ liệu -> Đảm bảo là Numpy Array Float
                # Theo builder.py: rec['embedding'] là list of floats -> OK
                try:
                    news_vec = np.array(raw_vec, dtype=np.float32)
                except Exception as e:
                    print(f"⚠️ Error converting embedding on {d}: {e}")
                    news_vec = np.zeros(TrainConfig.news_embed_dim, dtype=np.float32)

            # Map thành dict để pandas dễ xử lý column name
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
        # SPLIT MODALITIES
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

        # ==========================
        # RETURN (PAST RETURN)
        # ==========================
        return_df = self.create_return(price_df)

        # ALIGN STEP 1: theo return
        price_df = price_df.loc[return_df.index]
        macro_df = macro_df.loc[return_df.index]
        news_df  = news_df.loc[return_df.index]

        # ==========================
        # PRICE INPUT: LOG-RETURN
        # ==========================
        price_df = np.log(price_df / price_df.shift(1))
        price_df.dropna(inplace=True)

        # ALIGN STEP 2 (QUAN TRỌNG NHẤT)
        macro_df  = macro_df.loc[price_df.index]
        news_df   = news_df.loc[price_df.index]
        return_df = return_df.loc[price_df.index]

        # ==========================
        # MACRO CLEAN
        # ==========================
        macro_df = macro_df.replace([np.inf, -np.inf], np.nan)
        macro_df = macro_df.ffill().bfill()

        # SAFETY CHECK
        assert len(price_df) == len(macro_df) == len(news_df) == len(return_df), \
            "❌ Modality length mismatch before windowing"

        # ==========================
        # NUMPY & WINDOWING
        # ==========================
        price_np  = price_df.values           # (T, 3)
        macro_np  = macro_df.values           # (T, Dm)
        news_np   = news_df.values            # (T, 1024)
        return_np = return_df.values          # (T, 1)

        price_win = self.make_window(price_np, window_size)
        macro_win = self.make_window(macro_np, window_size)
        news_win  = self.make_window(news_np, window_size)

        # LABEL = future return (NO LEAK)
        label_raw = return_np[window_size - 1 + future_days:]

        price_win = price_win[:-future_days]
        macro_win = macro_win[:-future_days]
        news_win  = news_win[:-future_days]

        assert len(price_win) == len(macro_win) == len(news_win) == len(label_raw), \
            "❌ Window length mismatch"

        # ==========================
        # [NEW] SPLIT 3 TẬP (TRAIN - VALID - TEST)
        # ==========================
        total_len = len(price_win)
        idx_train = int(total_len * train_ratio)
        idx_valid = int(total_len * (train_ratio + valid_ratio))
        
        # Tập Train: 0 -> idx_train
        # Tập Valid: idx_train -> idx_valid
        # Tập Test : idx_valid -> Hết

        # ==============================================================================
        # [UPDATED] LABELING STRATEGY: ROLLING Z-SCORE (ADAPTIVE THRESHOLD)
        # ==============================================================================
        # Thay vì dùng ngưỡng cố định từ quá khứ, ta dùng ngưỡng động dựa trên độ biến động
        # của 20 ngày gần nhất. Giải quyết triệt để vấn đề "Mode Collapse" khi thị trường Sideway.
        # Công thức: Z_t = (R_t - Mean_20) / Std_20
        # ==============================================================================
        
        # 1. Chuyển đổi return_np thành Pandas Series để dùng hàm rolling
        # return_np shape (T, 1) -> flatten thành (T,)
        full_returns_series = pd.Series(return_np.flatten())
        
        # 2. Tính Rolling Stats (Cửa sổ 20 ngày - tương đương 1 tháng giao dịch)
        # shift(1) để đảm bảo không nhìn thấy tương lai (dùng volatility của quá khứ để xét hiện tại)
        # Tuy nhiên, ở đây ta đang xét nhãn cho chính ngày đó, nên ta chuẩn hóa độ lớn của Return
        # dựa trên độ biến động của cửa sổ bao quanh nó (hoặc liền trước nó).
        # Cách chuẩn nhất: Z-score của chính return đó so với độ lệch chuẩn của 20 ngày gần nhất.
        rolling_window = 20
        
        # Mean và Std của 20 ngày gần nhất
        roll_mean = full_returns_series.rolling(window=rolling_window).mean()
        roll_std  = full_returns_series.rolling(window=rolling_window).std()
        
        # 3. Tính Z-Score
        # Thêm 1e-6 để tránh chia cho 0
        z_scores = (full_returns_series - roll_mean) / (roll_std + 1e-6)
        
        # 4. Define Threshold (0.5 Sigma)
        z_threshold = 0.5 
        
        def map_z_label(z):
            if np.isnan(z): return 1 # Handle NaN ở đầu chuỗi -> FLAT
            if z < -z_threshold: return 0  # DOWN (Biến động tiêu cực lớn hơn 0.5 std)
            elif z > z_threshold: return 2 # UP   (Biến động tích cực lớn hơn 0.5 std)
            else: return 1                 # FLAT (Biến động nhỏ trong vùng 0.5 std)

        # 5. Áp dụng Mapping
        # Lưu ý: z_scores có độ dài bằng return_np (T)
        # label_raw được cắt từ index: window_size - 1 + future_days
        # Nên ta cũng cắt z_scores y hệt như vậy để khớp index
        start_idx = window_size - 1 + future_days
        
        # Đảm bảo start_idx không vượt quá độ dài
        if start_idx < len(z_scores):
            z_scores_sliced = z_scores.values[start_idx:]
            label_all = np.array([map_z_label(z) for z in z_scores_sliced])
        else:
            # Fallback nếu dữ liệu quá ngắn
            label_all = np.array([])

        # [DEBUG] In ra phân phối nhãn để kiểm tra độ cân bằng
        unique, counts = np.unique(label_all, return_counts=True)
        dist = dict(zip(unique, counts))
        total_lbl = sum(counts)
        print(f"   ⚖️  Label Distribution (Rolling Z): {dist}")
        if total_lbl > 0:
            print(f"      Down: {dist.get(0,0)/total_lbl:.2%}, Flat: {dist.get(1,0)/total_lbl:.2%}, Up: {dist.get(2,0)/total_lbl:.2%}")

        # ==========================
        # NORMALIZATION (FIT ON TRAIN ONLY - CHỐNG LEAKAGE)
        # ==========================
        # Tính Mean/Std chỉ trên tập Train
        macro_mean = macro_win[:idx_train].mean(axis=(0, 1), keepdims=True)
        macro_std  = macro_win[:idx_train].std(axis=(0, 1), keepdims=True) + 1e-6
        
        news_mean = news_win[:idx_train].mean(axis=(0, 1), keepdims=True)
        news_std  = news_win[:idx_train].std(axis=(0, 1), keepdims=True) + 1e-6

        # Áp dụng chuẩn hóa cho TOÀN BỘ dữ liệu (Train, Valid, Test)
        macro_win = (macro_win - macro_mean) / macro_std
        news_win  = (news_win - news_mean) / news_std

        # ==========================
        # HELPER FUNCTION: CREATE DICT
        # ==========================
        def create_dataset(start, end):
            if start >= end: # Handle edge cases
                return {}
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