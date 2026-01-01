import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class data_prepare:
    def __init__(self, data_path) -> None:
        self.data_path = data_path

        # ===== Scalers (fit on train only) =====
        self.price_scaler = StandardScaler()
        self.macro_scaler = StandardScaler()

    def create_label(self, price_df, window_size=3, threshold=0.01):
        """
        Output label: 0 = DOWN, 1 = FLAT, 2 = UP
        """
        df = price_df.copy()

        df["future_return"] = (
            df["close"].shift(-window_size) / df["close"] - 1
        )

        conditions = [
            df["future_return"] < -threshold,   # DOWN
            (df["future_return"] >= -threshold) &
            (df["future_return"] <= threshold), # FLAT
            df["future_return"] > threshold     # UP
        ]

        choices = [0, 1, 2]
        df["label"] = np.select(conditions, choices)

        df.dropna(inplace=True)

        return df[["label"]]

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
        window_size=20,
        future_days=1,
        train_ratio=0.8
    ):
        Data = pd.read_pickle(self.data_path)

        df = pd.DataFrame({
            d: {
                **content["price"][stock_name],
                **content["macro"],
            }
            for d, content in Data.items()
        }).T

        price_df = df[["open", "high", "close"]]
        macro_df = df[
            ["vix", "yield_spread_10y_2y",
            "sp500", "sp500_return", "dxy", "wti"]
        ]

        # ===== LABEL =====
        label_df = self.create_label(price_df, window_size=future_days)

        price_df = price_df.loc[label_df.index]
        macro_df = macro_df.loc[label_df.index]

        # ==================================================
        # 🔥 NORMALIZATION
        # ==================================================

        # ----- Price: log-return -----
        price_df = np.log(price_df / price_df.shift(1))
        price_df = price_df.dropna()

        # align again
        macro_df = macro_df.loc[price_df.index]
        label_df = label_df.loc[price_df.index]

        # ----- Macro: z-score (fit later) -----
        macro_df = macro_df.replace([np.inf, -np.inf], np.nan)
        macro_df = macro_df.fillna(method="ffill").fillna(method="bfill")

        # ===== numpy =====
        price_np = price_df.values
        macro_np = macro_df.values
        label_np = label_df.values

        # ===== sliding window =====
        price_win = self.make_window(price_np, window_size)
        macro_win = self.make_window(macro_np, window_size)
        label_win = label_np[window_size - 1:]

        # ===== split =====
        split_idx = int(len(price_win) * train_ratio)

        # ----- fit macro scaler on train -----
        macro_mean = macro_win[:split_idx].mean(axis=(0, 1), keepdims=True)
        macro_std = macro_win[:split_idx].std(axis=(0, 1), keepdims=True) + 1e-6

        macro_win = (macro_win - macro_mean) / macro_std

        train_data = {
            "s_o": torch.tensor(price_win[:split_idx, :, 0:1], dtype=torch.float32),
            "s_h": torch.tensor(price_win[:split_idx, :, 1:2], dtype=torch.float32),
            "s_c": torch.tensor(price_win[:split_idx, :, 2:3], dtype=torch.float32),
            "s_m": torch.tensor(macro_win[:split_idx], dtype=torch.float32),
            "label": label_win[:split_idx].tolist()
        }

        test_data = {
            "s_o": torch.tensor(price_win[split_idx:, :, 0:1], dtype=torch.float32),
            "s_h": torch.tensor(price_win[split_idx:, :, 1:2], dtype=torch.float32),
            "s_c": torch.tensor(price_win[split_idx:, :, 2:3], dtype=torch.float32),
            "s_m": torch.tensor(macro_win[split_idx:], dtype=torch.float32),
            "label": label_win[split_idx:].tolist()
        }

        return train_data, test_data
