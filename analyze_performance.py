# analyze_performance.py
"""
FIXED VERSION — Sử dụng model.forward() thay vì thủ công gọi submodules.

BUG ĐÃ SỬA:
  analyze_performance.py cũ chạy sai forward pass:
    - Sequential fusion thay vì parallel + gated merge
    - Bỏ qua modality_gate và pre_predict_proj (3*dim → 2*dim)
    - Truyền sai input vào movement_predictor
  → Predictor nhận input hoàn toàn khác với lúc training
  → Logits gần uniform (confidence ≈ 0.342 ≈ 1/3)
  → argmax luôn trả về DOWN (class 0) → ảo giác "98% collapse"

FIX: Gọi model.forward(mode="eval") để dùng đúng forward pass như lúc train.

Usage:
    python analyze_performance_fixed.py --model output/best_model_label=rolling_price=vol_adjusted_tid_seed42_wf.pt
    python analyze_performance_fixed.py --ticker TSLA AAPL
    python analyze_performance_fixed.py --no-ticker-id
"""

import argparse
import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
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
from src.data_loader import data_prepare, N_TICKERS
from configs.config import TrainConfig, GlobalConfig

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
CLASS_NAMES = ["DOWN", "FLAT", "UP"]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _sep(char="─", width=75):
    print(char * width)

def _header(title: str):
    _sep("═")
    print(f"  {title}")
    _sep("═")

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


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class StockDataset(Dataset):
    _BASE_KEYS = ["s_o", "s_h", "s_c", "s_m", "s_n", "news_mask", "label"]

    def __init__(self, d: dict):
        self.d    = d
        self.keys = list(self._BASE_KEYS)
        if "ticker_id" in d:
            self.keys.append("ticker_id")

    def __len__(self) -> int:
        return len(self.d["label"])

    def __getitem__(self, i: int) -> dict:
        return {k: self.d[k][i] for k in self.keys}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_test_data(tickers: list, data_path: str, include_ticker_id: bool) -> dict:
    dp = data_prepare(data_path, include_ticker_id=include_ticker_id)
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

def load_model(model_path: str, macro_dim: int, news_dim: int) -> StockMovementModel:
    model = StockMovementModel(
        price_dim=1,
        macro_dim=macro_dim,
        news_dim=news_dim,
        dim=TrainConfig.dim,
        input_dim=TrainConfig.window_size,
        output_dim=3,
        num_head=TrainConfig.num_head,
        device=DEVICE,
        n_tickers=N_TICKERS,
        dropout=0.0,
        class_weights=None,
        use_focal_loss=False,
    ).to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    
    # strict=False: bỏ qua loss_fn.weight (class weights từ training,
    # không cần thiết cho inference)
    missing, unexpected = model.load_state_dict(state, strict=False)
    
    expected_skip = {"loss_fn.weight"}
    real_unexpected = [k for k in unexpected if k not in expected_skip]
    real_missing    = [k for k in missing    if k not in expected_skip]
    
    if real_unexpected:
        print(f"  [WARN] Unexpected keys: {real_unexpected}")
    if real_missing:
        print(f"  [WARN] Missing keys: {real_missing}")
    if not real_unexpected and not real_missing:
        print(f"  Weights loaded OK (loss_fn.weight skipped — inference only)")
    
    model.eval()
    print(f"  Model: {model_path}")
    print(f"  dim={TrainConfig.dim}, heads={TrainConfig.num_head}, "
          f"macro={macro_dim}, news={news_dim}, n_tickers={N_TICKERS}")
    return model

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE — SỬ DỤNG model.forward() ĐÚNG CÁCH
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(model: StockMovementModel, data_dict: dict, include_ticker_id: bool):
    """
    FIX: Gọi model.forward(mode="eval") để dùng ĐÚNG forward pass như lúc training.
    
    Model forward path (src/model.py):
      1. Encode: v_m, v_i, v_n = multimodal_encoder(...)
      2. Parallel fusion: H_news = fusion_stage1(v_i, v_n)
                          H_macro = fusion_stage2(v_i, v_m)
      3. Gated merge: H_fused = w*H_news + (1-w)*H_macro
      4. Concat + proj: combined = cat([H_fused, v_i, v_t_seq]) → pre_predict_proj
      5. Predict: movement_predictor(H_final[:,:,:dim], H_final[:,:,dim:])
    
    Tất cả bước trên phải đi qua model.forward() để tránh bug.
    """
    ds  = StockDataset(data_dict)
    ldr = DataLoader(ds, batch_size=64, shuffle=False)

    all_preds, all_labels, all_probs = [], [], []

    for batch in ldr:
        labels = batch["label"].numpy()

        # ── Gọi model.forward() — dùng mode="eval" để lấy logits ──────────
        # model.forward với mode không phải "train" hay "test" → trả về logits
        logits = model(
            batch["s_o"].to(DEVICE),
            batch["s_h"].to(DEVICE),
            batch["s_c"].to(DEVICE),
            batch["s_m"].to(DEVICE),
            batch["s_n"].to(DEVICE),
            label=batch["label"].to(DEVICE),
            mode="eval",          # mode != "train" và != "test" → return logits
            ticker_id=batch.get("ticker_id", None),
            news_mask=batch.get("news_mask", None),
        )

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels)

    return (
        np.array(all_preds),
        np.array(all_labels),
        np.array(all_probs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze(tickers: list, model_path: str, data_path: str, include_ticker_id: bool):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)
    if not os.path.exists(data_path):
        print(f"Dataset not found: {data_path}")
        sys.exit(1)

    _header("DATA LOADING")
    datasets = load_test_data(tickers, data_path, include_ticker_id)
    if not datasets:
        print("No datasets loaded.")
        return

    first_data = next(iter(datasets.values()))
    macro_dim  = first_data["s_m"].shape[-1]
    news_dim   = first_data["s_n"].shape[-1]

    _header("MODEL LOADING")
    model = load_model(model_path, macro_dim, news_dim)

    _header("PER-TICKER ANALYSIS")
    fmt = "{:<8} | {:>7} | {:>16} | {:>16} | {:>7} | {:>7}"
    print(fmt.format("TICKER", "SAMPLES", "ACTUAL D/F/U", "PRED D/F/U", "ACC", "MCC"))
    _sep()

    all_preds, all_labels, all_probs = [], [], []
    ticker_results = {}

    for ticker, data in datasets.items():
        preds, labels, probs = run_inference(model, data, include_ticker_id)

        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.append(probs)

        acc = accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        ticker_results[ticker] = dict(preds=preds, labels=labels, probs=probs, acc=acc, mcc=mcc)

        print(fmt.format(
            ticker,
            len(labels),
            _dist_str(labels),
            _dist_str(preds),
            f"{acc:.4f}",
            f"{mcc:.4f}",
        ))

    _header("GLOBAL SUMMARY")
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs  = np.vstack(all_probs)

    overall_acc = accuracy_score(all_labels, all_preds)
    overall_mcc = matthews_corrcoef(all_labels, all_preds)

    print(f"Total samples : {len(all_labels)}")
    print(f"Overall ACC   : {overall_acc:.4f}")
    print(f"Overall MCC   : {overall_mcc:.4f}")
    print(f"\nActual distribution   : {_pct_str(all_labels)}")
    print(f"Predicted distribution: {_pct_str(all_preds)}")

    unique_preds = np.unique(all_preds)
    if len(unique_preds) == 1:
        print(f"\n[!] MODE COLLAPSE: model predicts only class "
              f"{unique_preds[0]} ({CLASS_NAMES[unique_preds[0]]}) for all samples.")
    elif len(unique_preds) == 2:
        missing = [c for c in [0, 1, 2] if c not in unique_preds]
        print(f"\n[!] PARTIAL COLLAPSE: class {missing} "
              f"({[CLASS_NAMES[m] for m in missing]}) never predicted.")
    else:
        print(f"\n[OK] Model predicts all 3 classes.")

    _header("CONFUSION MATRIX")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print(f"{'':12}  {'Pred DOWN':>10}  {'Pred FLAT':>10}  {'Pred UP':>10}")
    for i, name in enumerate(CLASS_NAMES):
        row = cm[i]
        total = row.sum()
        recall = row[i] / total if total > 0 else 0.0
        print(f"Act {name:<8}  {row[0]:>10}  {row[1]:>10}  {row[2]:>10}   recall={recall:.2f}")

    _header("CLASSIFICATION REPORT")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, zero_division=0))

    _header("CONFIDENCE ANALYSIS")
    max_conf = all_probs.max(axis=1)
    print(f"Mean confidence (max-prob):  {max_conf.mean():.4f}")
    print(f"Median confidence:           {np.median(max_conf):.4f}")
    print(f"Std:                         {max_conf.std():.4f}")
    print(f"\nPer-class avg probability:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {all_probs[:, i].mean():.4f}")

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

    if len(datasets) > 1:
        _header("PER-TICKER CLASSIFICATION REPORTS")
        for ticker, r in ticker_results.items():
            print(f"\n--- {ticker} ---")
            print(classification_report(r["labels"], r["preds"], target_names=CLASS_NAMES, zero_division=0))

    _sep("═")
    print("  Analysis complete.")
    _sep("═")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Analyze Deep Finance model (FIXED forward pass)")
    parser.add_argument("--ticker",       nargs="+", default=None)
    parser.add_argument("--model",        default=None)
    parser.add_argument("--data",         default=DATA_PATH)
    parser.add_argument("--no-ticker-id", action="store_true")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.ticker] if args.ticker else GlobalConfig.TICKERS
    include_tid = not args.no_ticker_id

    if args.model:
        model_path = args.model
    else:
        output_dir = "output"
        pt_files = [
            f for f in os.listdir(output_dir)
            if f.startswith("best_model") and f.endswith(".pt")
        ] if os.path.isdir(output_dir) else []

        if not pt_files:
            print(f"Không tìm thấy file .pt nào trong '{output_dir}/'")
            print("Chạy main.py trước để train model, hoặc dùng --model <path>")
            sys.exit(1)

        # Lấy file mới nhất theo thời gian chỉnh sửa
        pt_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
        model_path = os.path.join(output_dir, pt_files[0])
        print(f"Auto-detected model: {model_path}")
        if len(pt_files) > 1:
            print(f"  (Có {len(pt_files)} model, dùng mới nhất. Dùng --model để chọn cụ thể)")

    analyze(tickers=tickers, model_path=model_path, data_path=args.data,
            include_ticker_id=include_tid)


if __name__ == "__main__":
    main()