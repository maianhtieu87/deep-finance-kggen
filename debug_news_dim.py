# debug_news_dim.py

import os
import torch
from src.data_loader import data_prepare
from configs.config import GlobalConfig, TrainConfig

if __name__ == "__main__":
    pkl_path = os.path.join(
        GlobalConfig.PROCESSED_PATH,
        "unified_dataset_test.pkl"
    )
    print("Using PKL:", pkl_path)

    dp = data_prepare(pkl_path)

    # Lấy tạm 1 ticker bất kỳ có data, ví dụ TSLA
    train_data, valid_data, test_data = dp.prepare_data("TSLA")

    # In shape các tensor
    for name, tensor in [
        ("s_o", train_data["s_o"]),
        ("s_h", train_data["s_h"]),
        ("s_c", train_data["s_c"]),
        ("s_m", train_data["s_m"]),
        ("s_n", train_data["s_n"]),
    ]:
        print(f"{name} shape:", tensor.shape)

    # Quan trọng nhất: last dim của s_n
    news_dim = train_data["s_n"].shape[-1]
    print("\n👉 news_feature_dim (từ KG/GNN) =", news_dim)
