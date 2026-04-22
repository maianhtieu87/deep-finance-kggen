import os
import glob
import torch
import numpy as np

# Adjust path to find modules
import sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig, GlobalConfig
from baselines.run_ablation import evaluate, build_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    pkl_path = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    if not os.path.exists(pkl_path):
        print(f"Data not found: {pkl_path}. Please run main_test.py first to build data.")
        return
    
    print("Loading test data...")
    dp = data_prepare(pkl_path, include_ticker_id=True)
    valid_T = [dp.get_max_T(t) for t in GlobalConfig.TICKERS if dp.get_max_T(t) > 0]
    
    if not valid_T:
        print("No valid tickers found in data.")
        return
        
    global_T_max = min(valid_T)
    inner_T = int(global_T_max * 0.85)

    te_list = []
    macro_dim = None
    news_dim = None
    
    for t in GlobalConfig.TICKERS:
        if dp.get_max_T(t) == 0: continue
        _, _, te = dp.prepare_data(t, train_end=inner_T, val_end=inner_T, test_end=global_T_max)
        
        if macro_dim is None and te and len(te.get("label", [])) > 0:
            macro_dim = te["s_m"].shape[-1]
            news_dim = te["s_n"].shape[-1]
            
        if te and len(te.get("label", [])) > 0:
            te_list.append(te)

    # Merge testing data
    test_data = {}
    if te_list:
        for key in te_list[0].keys():
            parts = [d[key] for d in te_list if key in d]
            if parts and isinstance(parts[0], torch.Tensor):
                test_data[key] = torch.cat(parts, dim=0)

    if not test_data:
        print("Test data is empty!")
        return

    print(f"Test set size: {len(test_data['label'])} samples")
    print(f"Feature dims -> Macro: {macro_dim}, News: {news_dim}")

    # Find the most recently trained model
    pattern = os.path.join(GlobalConfig.BASE_DIR, "output", "best_model_*_standard.pt")
    matches = glob.glob(pattern)
    if not matches:
        print(f"No trained model found matching {pattern}")
        print("Please run main_test.py or main.py first to train a model.")
        return
        
    pt_path = max(matches, key=os.path.getmtime)
    print(f"\nLoading model weights from: {os.path.basename(pt_path)}")

    model = build_model(macro_dim, news_dim, use_focal=False, cw=None)
    state = torch.load(pt_path, map_location=DEVICE, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()

    # 1. Evaluate WITH News (Baseline)
    print(f"\n--- EVALUATING MODEL WITH NEWS ---")
    acc_with, mcc_with = evaluate(model, test_data, zero_news=False)
    print(f"Accuracy : {acc_with:.4f}")
    print(f"MCC      : {mcc_with:.4f}")

    # 2. Evaluate WITHOUT News (Ablation - zeroing out news features)
    print(f"\n--- EVALUATING MODEL WITHOUT NEWS (ZEROED OUT) ---")
    acc_without, mcc_without = evaluate(model, test_data, zero_news=True)
    print(f"Accuracy : {acc_without:.4f}")
    print(f"MCC      : {mcc_without:.4f}")

    # Conclusion
    print("\n" + "="*50)
    print("CONCLUSION (Impact of News module during inference):")
    print("="*50)
    mcc_diff = mcc_without - mcc_with
    if mcc_diff > 0.005:
        print(f"Removing news IMPROVED MCC by {mcc_diff:.4f}.")
        print("Interpretation: The news module is adding noise or the model hasn't learned to use it effectively.")
    elif mcc_diff < -0.005:
        print(f"Removing news DECREASED MCC by {abs(mcc_diff):.4f}.")
        print("Interpretation: The news module provides valuable signal that helps the model's performance.")
    else:
        print(f"Removing news had NEGLIGIBLE EFFECT on MCC (Diff: {mcc_diff:+.4f}).")
        print("Interpretation: The model is mostly relying on price/macro features and ignoring the news.")

if __name__ == "__main__":
    main()
