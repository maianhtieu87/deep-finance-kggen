# =========================================================
# FILE: debug_pipeline.py
# Mục tiêu: Phân tích chi tiết performance per-ticker
# trên unified_dataset_test.pkl (sau khi build bằng main_test.py)
# =========================================================

import os
import sys
import numpy as np
import torch
from collections import Counter
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    matthews_corrcoef,
)

# Import project modules
from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig, GlobalConfig

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("output", "best_model.pt")

# ✅ Đường dẫn .pkl lấy từ GlobalConfig.PROCESSED_PATH
DATA_PATH = os.path.join(
    GlobalConfig.PROCESSED_PATH,
    "unified_dataset_test.pkl",
)


def print_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"🔎 {title}")
    print(f"{'=' * 60}")


def load_data_per_ticker(tickers):
    """
    Load dữ liệu Test riêng biệt cho từng mã để phân tích behavior.
    Sử dụng logic Rolling Z-Score mới nhất từ data_loader.data_prepare.
    """
    if not os.path.exists(DATA_PATH):
        print(f"❌ DATA_PATH not found: {DATA_PATH}")
        print("   → Hãy chạy main_test.py để tạo unified_dataset_test.pkl trước.")
        return {}

    dp = data_prepare(DATA_PATH)
    ticker_datasets = {}

    print(f"📥 Loading TEST data for: {tickers}")
    for t in tickers:
        try:
            # prepare_data trả về: train, valid, test
            _, _, test_data = dp.prepare_data(
                stock_name=t,
                window_size=TrainConfig.window_size,
                # Các tham số khác sẽ lấy default từ Config nếu có
            )

            if test_data and len(test_data.get("label", [])) > 0:
                ticker_datasets[t] = test_data
                print(f"   ✅ {t}: {len(test_data['label'])} samples")
            else:
                print(f"   ⚠️ {t}: No data or empty test set")
        except Exception as e:
            print(f"   ❌ {t}: Error {e}")

    return ticker_datasets


def run_prediction(model: StockMovementModel, data_dict: dict):
    """
    Chạy forward pass để lấy logits + preds + probs.
    Ở đây ta tái sử dụng đúng pipeline trong model:
      - multimodal_encoder
      - fusion_news / fusion_macro
      - movement_predictor
    """
    model.eval()
    with torch.no_grad():
        s_o = data_dict["s_o"].to(DEVICE)
        s_h = data_dict["s_h"].to(DEVICE)
        s_c = data_dict["s_c"].to(DEVICE)
        s_m = data_dict["s_m"].to(DEVICE)
        s_n = data_dict["s_n"].to(DEVICE)

        # 1. Encoder
        v_m, v_i, v_n = model.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)

        # 2. Fusion (khớp với logic trong StockMovementModel)
        fused_news = model.fusion_news(primary=v_i, aux=v_n)
        fused_macro = model.fusion_macro(primary=v_i, aux=v_m)
        v_fused_total = (fused_news + fused_macro) / 2.0

        # 3. Predictor
        logits = model.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

    return preds.cpu().numpy(), data_dict["label"].numpy(), probs.cpu().numpy()


def analyze_performance():
    # 0. Check data & model tồn tại
    print_header("0. CHECK FILES")

    if not os.path.exists(DATA_PATH):
        print(f"❌ Cannot find dataset at {DATA_PATH}")
        print("   → Hãy chạy main_test.py để tạo unified_dataset_test.pkl trước.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Cannot find model at {MODEL_PATH}")
        print("   → Hãy train model bằng main.py trước.")
        return

    # 1. Load Model
    print_header("1. LOADING MODEL")

    # Lấy macro_dim thực tế từ dữ liệu để init model đúng shape
    dp = data_prepare(DATA_PATH)
    dummy_train, _, _ = dp.prepare_data("TSLA")
    if dummy_train and "s_m" in dummy_train:
        macro_dim = dummy_train["s_m"].shape[-1]
    else:
        macro_dim = 6  # fallback nếu không lấy được (ít khi dùng)
    print(f"🔧 Model Config: Dim={TrainConfig.dim}, Heads={TrainConfig.num_head}, Macro_dim={macro_dim}")

    model = StockMovementModel(
        price_dim=1,
        macro_dim=macro_dim,
        news_dim=TrainConfig.news_embed_dim,   # phải khớp với lúc train
        dim=TrainConfig.dim,
        input_dim=TrainConfig.window_size,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        device=DEVICE,
        dropout=0.0,          # eval không cần dropout
        class_weights=None,   # eval không tính loss
        use_focal_loss=False, # eval không dùng focal
    ).to(DEVICE)

    try:
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        print("✅ Weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading weights from {MODEL_PATH}: {e}")
        print("💡 Hint: Kiểm tra Dim/Heads/news_dim/macro_dim/num_classes trong Config có khớp với lúc train không?")
        return

    # 2. Load Data
    print_header("2. LOADING DATA")
    target_tickers = ["TSLA", "AMZN", "MSFT", "NFLX"]
    datasets = load_data_per_ticker(target_tickers)

    if not datasets:
        print("❌ No datasets loaded.")
        return

    # 3. Deep Dive Analysis per ticker
    print_header("3. DEEP DIVE ANALYSIS PER-TICKER")

    all_preds = []
    all_labels = []

    print(
        f"{'TICKER':<10} | {'SAMPLES':<8} | {'ACTUAL (0/1/2)':<20} | "
        f"{'PRED (0/1/2)':<20} | {'ACC':<8} | {'MCC':<8}"
    )
    print("-" * 100)

    for ticker, data in datasets.items():
        preds, labels, probs = run_prediction(model, data)

        all_preds.extend(preds)
        all_labels.extend(labels)

        acc = accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)

        act_counts = Counter(labels)
        pred_counts = Counter(preds)

        act_dist = f"{act_counts.get(0,0)}/{act_counts.get(1,0)}/{act_counts.get(2,0)}"
        pred_dist = f"{pred_counts.get(0,0)}/{pred_counts.get(1,0)}/{pred_counts.get(2,0)}"

        print(
            f"{ticker:<10} | {len(labels):<8} | "
            f"{act_dist:<20} | {pred_dist:<20} | {acc:.4f} | {mcc:.4f}"
        )

    # 4. Global Analysis
    print_header("4. GLOBAL SUMMARY (ALL TICKERS COMBINED)")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    unique_act, counts_act = np.unique(all_labels, return_counts=True)
    unique_pred, counts_pred = np.unique(all_preds, return_counts=True)

    print("📉 ACTUAL Labels Distribution (Ground Truth):")
    print(f"   {dict(zip(unique_act, counts_act))}")

    print("\n🔮 PREDICTED Labels Distribution:")
    print(f"   {dict(zip(unique_pred, counts_pred))}")

    # Check mode collapse
    if len(unique_pred) == 1:
        print("\n⚠️  CRITICAL WARNING: MODE COLLAPSE DETECTED!")
        print(f"   Mô hình chỉ dự đoán duy nhất lớp {unique_pred[0]} cho toàn bộ dữ liệu.")
        print("   → Đây là lý do MCC ~ 0.0 hoặc rất thấp.")

    print("\n📊 Confusion Matrix (labels: 0=DOWN, 1=FLAT, 2=UP):")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print(f"      Pred 0  Pred 1  Pred 2")
    print(f"Act 0   {cm[0][0]:<7} {cm[0][1]:<7} {cm[0][2]:<7}")
    print(f"Act 1   {cm[1][0]:<7} {cm[1][1]:<7} {cm[1][2]:<7}")
    print(f"Act 2   {cm[2][0]:<7} {cm[2][1]:<7} {cm[2][2]:<7}")

    print("\n📋 Classification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=["DOWN", "FLAT", "UP"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    analyze_performance()