"""
analyze_performance.py — Deep Finance V5.3

Phân tích chi tiết model predictions: per-ticker metrics,
confusion matrix, mode collapse detection, confidence distribution.

Usage:
    python analyze_performance.py                        # tất cả 9 tickers
    python analyze_performance.py --ticker TSLA          # 1 ticker
    python analyze_performance.py --ticker TSLA AAPL WMT
    python analyze_performance.py --model output/best_model_tsla.pt --ticker TSLA
"""

import argparse
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

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig, GlobalConfig

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — chỉnh ở đây nếu cần override
# ─────────────────────────────────────────────────────────────────────────────

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sep(char="─", width=75):
    print(char * width)

def _header(title: str):
    _sep("═")
    print(f"  {title}")
    _sep("═")


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(tickers: list[str], data_path: str) -> dict:
    """
    Load test split cho từng ticker.
    data_prepare.prepare_data(ticker) → (train, valid, test)
    """
    dp = data_prepare(data_path)
    datasets = {}

    print(f"\nLoading test data: {tickers}")
    for t in tickers:
        try:
            _, _, test_data = dp.prepare_data(t)
            if test_data and len(test_data.get("label", [])) > 0:
                datasets[t] = test_data
                n = len(test_data["label"])
                news_nonzero = int(
                    (test_data["s_n"].abs().sum(dim=-1).sum(dim=-1) > 0).sum()
                )
                print(f"  {t}: {n} samples  |  news coverage: {news_nonzero}/{n} windows")
            else:
                print(f"  {t}: no test data")
        except Exception as e:
            print(f"  {t}: error — {e}")

    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str, macro_dim: int) -> StockMovementModel:
    """
    Khởi tạo và load weights StockMovementModel.
    dropout=0.0 và class_weights=None cho eval mode.
    """
    model = StockMovementModel(
        price_dim=1,
        macro_dim=macro_dim,
        news_dim=TrainConfig.news_embed_dim,     # 1024
        dim=TrainConfig.dim,                     # 256
        input_dim=TrainConfig.window_size,       # 20
        output_dim=TrainConfig.output_dim,       # 3
        num_head=TrainConfig.num_head,           # 4
        device=DEVICE,
        dropout=0.0,                             # eval mode — không cần dropout
        class_weights=None,
        use_focal_loss=False,                    # eval mode — không tính loss
    ).to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"  Weights loaded: {model_path}")
    print(f"  dim={TrainConfig.dim}, heads={TrainConfig.num_head}, "
          f"macro={macro_dim}, news={TrainConfig.news_embed_dim}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: StockMovementModel, data: dict):
    """
    Forward pass theo đúng kiến trúc V5.3:
      Price → News (Stage 1 MSGCA) → Macro (Stage 2 MSGCA) → Predictor

    model.multimodal_encoder  → (v_m, v_i, v_n)
    model.fusion_stage1(primary=v_i, aux=v_n)  → H1
    model.fusion_stage2(primary=H1,  aux=v_m)  → H_final
    model.movement_predictor(fused_seq=H_final, orig_seq=v_i) → logits
    """
    s_o = data["s_o"].to(DEVICE)
    s_h = data["s_h"].to(DEVICE)
    s_c = data["s_c"].to(DEVICE)
    s_m = data["s_m"].to(DEVICE)
    s_n = data["s_n"].to(DEVICE)

    # 1. Encode
    v_m, v_i, v_n = model.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)
    if v_n is None:
        v_n = torch.zeros_like(v_i)

    # 2. Sequential 2-stage MSGCA fusion
    H1      = model.fusion_stage1(primary=v_i, aux=v_n)  # Price + News
    H_final = model.fusion_stage2(primary=H1,  aux=v_m)  # Fused + Macro

    # 3. Predict
    logits = model.movement_predictor(fused_seq=H_final, orig_seq=v_i)
    logits = torch.clamp(logits, -15, 15)

    probs  = torch.softmax(logits, dim=1)
    preds  = torch.argmax(logits, dim=1)

    return (
        preds.cpu().numpy(),
        data["label"].numpy(),
        probs.cpu().numpy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["DOWN", "FLAT", "UP"]

def _dist_str(arr) -> str:
    c = Counter(arr)
    return f"{c.get(0,0):>4}/{c.get(1,0):>4}/{c.get(2,0):>4}"

def _pct_str(arr) -> str:
    c = Counter(arr)
    n = len(arr)
    if n == 0:
        return "—"
    return (f"D:{c.get(0,0)/n*100:.0f}% "
            f"F:{c.get(1,0)/n*100:.0f}% "
            f"U:{c.get(2,0)/n*100:.0f}%")

def analyze(tickers: list[str], model_path: str, data_path: str):
    # ── Sanity checks ─────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        print("Run main_test.py first.")
        sys.exit(1)

    # ── Load data first to get macro_dim ──────────────────────────────────
    _header("DATA LOADING")
    datasets = load_test_data(tickers, data_path)

    if not datasets:
        print("No datasets loaded. Exiting.")
        return

    # Infer macro_dim from first available ticker
    macro_dim = next(iter(datasets.values()))["s_m"].shape[-1]

    # ── Load model ────────────────────────────────────────────────────────
    _header("MODEL LOADING")
    model = load_model(model_path, macro_dim)

    # ── Per-ticker analysis ───────────────────────────────────────────────
    _header("PER-TICKER ANALYSIS")
    fmt = "{:<8} | {:>7} | {:>16} | {:>16} | {:>7} | {:>7}"
    print(fmt.format("TICKER", "SAMPLES",
                     "ACTUAL D/F/U", "PRED D/F/U", "ACC", "MCC"))
    _sep()

    all_preds, all_labels, all_probs = [], [], []
    ticker_results = {}

    for ticker, data in datasets.items():
        preds, labels, probs = run_inference(model, data)

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.append(probs)

        acc = accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        ticker_results[ticker] = dict(preds=preds, labels=labels, probs=probs,
                                      acc=acc, mcc=mcc)

        print(fmt.format(
            ticker,
            len(labels),
            _dist_str(labels),
            _dist_str(preds),
            f"{acc:.4f}",
            f"{mcc:.4f}",
        ))

    # ── Global summary ────────────────────────────────────────────────────
    _header("GLOBAL SUMMARY")
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.vstack(all_probs)

    print(f"Total samples : {len(all_labels)}")
    print(f"Overall ACC   : {accuracy_score(all_labels, all_preds):.4f}")
    print(f"Overall MCC   : {matthews_corrcoef(all_labels, all_preds):.4f}")

    print(f"\nActual distribution  : {_pct_str(all_labels)}")
    print(f"Predicted distribution: {_pct_str(all_preds)}")

    # Mode collapse check
    unique_preds = np.unique(all_preds)
    if len(unique_preds) == 1:
        print(f"\n[!] MODE COLLAPSE: model predicts only class "
              f"{unique_preds[0]} ({CLASS_NAMES[unique_preds[0]]}) for all samples.")
    elif len(unique_preds) == 2:
        missing = [c for c in [0, 1, 2] if c not in unique_preds]
        print(f"\n[!] PARTIAL COLLAPSE: class {missing} ({[CLASS_NAMES[m] for m in missing]}) "
              f"never predicted.")

    # ── Confusion matrix ──────────────────────────────────────────────────
    _header("CONFUSION MATRIX")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print(f"{'':12}  {'Pred DOWN':>10}  {'Pred FLAT':>10}  {'Pred UP':>10}")
    for i, name in enumerate(CLASS_NAMES):
        row = cm[i]
        # Per-row recall
        total = row.sum()
        recall = row[i] / total if total > 0 else 0.0
        print(f"Act {name:<8}  {row[0]:>10}  {row[1]:>10}  {row[2]:>10}"
              f"   recall={recall:.2f}")

    # ── Classification report ─────────────────────────────────────────────
    _header("CLASSIFICATION REPORT")
    print(classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    # ── Confidence analysis ───────────────────────────────────────────────
    _header("CONFIDENCE ANALYSIS")
    max_conf = all_probs.max(axis=1)

    print(f"Mean confidence (max-prob):  {max_conf.mean():.4f}")
    print(f"Median confidence:           {np.median(max_conf):.4f}")
    print(f"Std:                         {max_conf.std():.4f}")

    thresholds = [0.40, 0.45, 0.50, 0.55, 0.60]
    print(f"\n{'Threshold':>12}  {'Kept':>8}  {'Kept%':>8}  {'ACC on kept':>12}  {'MCC on kept':>12}")
    _sep("-", 60)
    for thr in thresholds:
        mask = max_conf >= thr
        n_kept = mask.sum()
        if n_kept == 0:
            continue
        acc_t = accuracy_score(all_labels[mask], all_preds[mask])
        mcc_t = matthews_corrcoef(all_labels[mask], all_preds[mask])
        print(f"{thr:>12.2f}  {n_kept:>8}  {n_kept/len(all_labels)*100:>7.1f}%"
              f"  {acc_t:>12.4f}  {mcc_t:>12.4f}")

    # ── Per-ticker detailed report (optional verbose) ─────────────────────
    if len(datasets) > 1:
        _header("PER-TICKER CLASSIFICATION REPORTS")
        for ticker, r in ticker_results.items():
            print(f"\n--- {ticker} ---")
            print(classification_report(
                r["labels"], r["preds"],
                target_names=CLASS_NAMES,
                zero_division=0,
            ))

    _sep("═")
    print("  Analysis complete.")
    _sep("═")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze Deep Finance model performance"
    )
    parser.add_argument(
        "--ticker", nargs="+", default=None,
        help="Tickers to analyze (default: all 9 from GlobalConfig)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to model .pt file (default: output/best_model.pt)",
    )
    parser.add_argument(
        "--data", default=DATA_PATH,
        help=f"Path to unified_dataset.pkl (default: {DATA_PATH})",
    )
    args = parser.parse_args()

    tickers = [t.upper() for t in args.ticker] if args.ticker else GlobalConfig.TICKERS

    # Auto-detect model path
    if args.model:
        model_path = args.model
    else:
        # Try single-ticker model first
        if len(tickers) == 1:
            single = os.path.join("output", f"best_model_{tickers[0].lower()}.pt")
            if os.path.exists(single):
                model_path = single
                print(f"Auto-detected single-ticker model: {single}")
            else:
                model_path = os.path.join("output", "best_model.pt")
        else:
            model_path = os.path.join("output", "best_model.pt")

    analyze(tickers=tickers, model_path=model_path, data_path=args.data)


if __name__ == "__main__":
    main()