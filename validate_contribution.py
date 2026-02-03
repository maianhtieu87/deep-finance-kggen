import torch
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef, classification_report

# Import modules từ dự án
from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig, GlobalConfig
from main import merge_datasets, set_seed  # KHÔNG cần StockDataset, compute_class_weights

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("output", "best_model.pt")  # Đường dẫn model tốt nhất

# ✅ Dùng đúng path trong project hiện tại
DATA_PATH = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
TARGET_TICKERS = ["TSLA", "AMZN", "MSFT", "NFLX"]


def run_ablation_test():
    print("=" * 60)
    print("🧪 MODULE CONTRIBUTION ANALYSIS (ABLATION STUDY)")
    print("=" * 60)

    # 1. LOAD DATA (TEST SET)
    print("\n📥 Loading TEST Data...")
    print(f"   Using DATA_PATH = {DATA_PATH}")
    dp = data_prepare(DATA_PATH)
    list_test = []

    # Load 1 sample để lấy dimension config
    sample_dim_check = None

    for ticker in TARGET_TICKERS:
        try:
            _, _, te = dp.prepare_data(ticker)
            if te and len(te.get("label", [])) > 0:
                list_test.append(te)
                if sample_dim_check is None:
                    sample_dim_check = te
                print(f"   ✅ {ticker}: {len(te['label'])} test samples")
            else:
                print(f"   ⚠️ {ticker}: no test data.")
        except Exception as e:
            print(f"   ❌ {ticker}: error when loading test data: {e}")

    if not list_test:
        print("❌ Error: No test data found.")
        return

    final_test = merge_datasets(list_test, shuffle=False)
    print(f"\n✅ Total Test Samples (merged): {len(final_test['label'])}")

    # Lấy Dimension thực tế từ dữ liệu
    s_m_dim = sample_dim_check["s_m"].shape[-1]

    # 2. LOAD TRAINED MODEL
    print(f"\n🤖 Loading Model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found! Please train a model first.")
        return

    # Dummy weights chỉ để init; eval không dùng loss
    dummy_weights = torch.tensor([1.0, 1.0, 1.0])

    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m_dim,
        news_dim=TrainConfig.news_embed_dim,
        dim=TrainConfig.dim,
        input_dim=TrainConfig.window_size,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        dropout=0.0,  # Tắt dropout khi test
        class_weights=dummy_weights,
        use_focal_loss=True,
        device=DEVICE,
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("🔧 Loss Strategy: FOCAL LOSS (γ=2.0) + ALPHA BALANCING ✅")
    print("   ► Alpha weights: [1. 1. 1.]")
    print("✅ Model Loaded Successfully.")

    # 3. DEFINE ABLATION EXPERIMENTS
    experiments = [
        {"name": "BASELINE (Full Info)", "mask_price": False, "mask_macro": False, "mask_news": False},
        {"name": "NO MACRO (Price+News)", "mask_price": False, "mask_macro": True, "mask_news": False},
        {"name": "NO NEWS  (Price+Macro)", "mask_price": False, "mask_macro": False, "mask_news": True},
        {"name": "PRICE ONLY", "mask_price": True, "mask_macro": True, "mask_news": True},
        {
            "name": "NEWS ONLY (KG branch)",
            "mask_price": True,
            "mask_macro": True,
            "mask_news": False,
            "is_news_only": True,
        },
    ]

    results = []

    print("\n🚀 Running Inference Tests...")

    news_only_y_true = None
    news_only_y_pred = None

    for exp in experiments:
        exp_name = exp["name"]
        print(f"   Running: {exp_name}...", end="")

        is_news_only = exp.get("is_news_only", False)

        acc, mcc, y_true, y_pred = evaluate_with_masking(
            model,
            final_test,
            mask_price=exp["mask_price"],
            mask_macro=exp["mask_macro"],
            mask_news=exp["mask_news"],
            return_preds=is_news_only,
        )

        results.append(
            {
                "Scenario": exp_name,
                "ACC": acc,
                "MCC": mcc,
                "Diff MCC": 0.0,
            }
        )
        print(f" Done. (MCC: {mcc:.4f})")

        if is_news_only and y_true is not None and y_pred is not None:
            news_only_y_true = y_true
            news_only_y_pred = y_pred

    print("\n" + "=" * 60)
    print("📊 CONTRIBUTION REPORT")
    print("=" * 60)

    baseline_mcc = results[0]["MCC"]

    df_res = pd.DataFrame(results)
    df_res["Diff MCC"] = df_res["MCC"] - baseline_mcc
    df_res["Impact"] = df_res["Diff MCC"].apply(
        lambda x: "🔻 HURT" if x < -0.01 else ("✅ HELP" if x > 0.01 else "⚪ NEUTRAL")
    )

    print(
        df_res.to_string(
            index=False,
            formatters={
                "ACC": "{:.4f}".format,
                "MCC": "{:.4f}".format,
                "Diff MCC": "{:+.4f}".format,
            },
        )
    )

    print("\n📝 INTERPRETATION:")
    print("   - Nếu 'Diff MCC' ÂM LỚN (vd: -0.05): Module đó QUAN TRỌNG (bỏ đi làm model ngu đi).")
    print("   - Nếu 'Diff MCC' GẦN 0 (vd: -0.00): Module đó ÍT ĐƯỢC DÙNG (model gần như ignore).")
    print("   - Nếu 'Diff MCC' DƯƠNG (vd: +0.02): Module đó CÓ THỂ GÂY NHIỄU (bỏ đi model lại chạy tốt hơn).")

    # 6. PHÂN TÍCH RIÊNG CASE NEWS ONLY
    if news_only_y_true is not None and news_only_y_pred is not None:
        print("\n" + "=" * 60)
        print("🔍 DEEP DIVE: NEWS ONLY (KG branch)")
        print("=" * 60)

        unique_act, counts_act = np.unique(news_only_y_true, return_counts=True)
        unique_pred, counts_pred = np.unique(news_only_y_pred, return_counts=True)

        print("📉 ACTUAL label distribution (0=DOWN,1=FLAT,2=UP):")
        print(f"   {dict(zip(unique_act, counts_act))}")

        print("\n🔮 PREDICTED label distribution (NEWS ONLY):")
        print(f"   {dict(zip(unique_pred, counts_pred))}")

        print("\n📊 Confusion Matrix (NEWS ONLY):")
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(news_only_y_true, news_only_y_pred, labels=[0, 1, 2])
        print(f"      Pred 0  Pred 1  Pred 2")
        print(f"Act 0   {cm[0][0]:<7} {cm[0][1]:<7} {cm[0][2]:<7}")
        print(f"Act 1   {cm[1][0]:<7} {cm[1][1]:<7} {cm[1][2]:<7}")
        print(f"Act 2   {cm[2][0]:<7} {cm[2][1]:<7} {cm[2][2]:<7}")

        print("\n📋 Classification Report (NEWS ONLY):")
        print(
            classification_report(
                news_only_y_true,
                news_only_y_pred,
                target_names=["DOWN", "FLAT", "UP"],
                zero_division=0,
            )
        )


def evaluate_with_masking(
    model,
    data_dict,
    mask_price: bool = False,
    mask_macro: bool = False,
    mask_news: bool = False,
    return_preds: bool = False,
):
    """
    Evaluate với khả năng 'tắt' (mask) các nguồn dữ liệu bằng cách đưa về 0.
    """

    # Price
    s_o = data_dict["s_o"].to(DEVICE)
    s_h = data_dict["s_h"].to(DEVICE)
    s_c = data_dict["s_c"].to(DEVICE)

    if mask_price:
        s_o = torch.zeros_like(s_o).to(DEVICE)
        s_h = torch.zeros_like(s_h).to(DEVICE)
        s_c = torch.zeros_like(s_c).to(DEVICE)

    # Macro
    if mask_macro:
        s_m = torch.zeros_like(data_dict["s_m"]).to(DEVICE)
    else:
        s_m = data_dict["s_m"].to(DEVICE)

    # News / KG
    if mask_news:
        s_n = torch.zeros_like(data_dict["s_n"]).to(DEVICE)
    else:
        s_n = data_dict["s_n"].to(DEVICE)

    label = data_dict["label"].to(DEVICE)

    with torch.no_grad():
        v_m, v_i, v_n = model.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)

        fused_news = model.fusion_news(primary=v_i, aux=v_n)
        fused_macro = model.fusion_macro(primary=v_i, aux=v_m)

        v_fused_total = (fused_news + fused_macro) / 2.0

        logits = model.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    y_true = label.cpu().numpy()
    y_pred = preds.cpu().numpy()

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    if return_preds:
        return acc, mcc, y_true, y_pred
    else:
        return acc, mcc, None, None


if __name__ == "__main__":
    set_seed(42)
    run_ablation_test()
