# export_predictions.py
"""
Export real model predictions for the interactive demo.

Usage:
    python export_predictions.py
    python export_predictions.py --pkl data/processed/unified_dataset_test.pkl
    python export_predictions.py --tickers TSLA AAPL

Output:
    demo_predictions.json  (drop-in replacement for generateData() in the widget)
"""

import os, json, argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import StockMovementModel
from src.data_loader import data_prepare, N_TICKERS, NEWS_EMB_DIM
from configs.config import TrainConfig, GlobalConfig
import glob

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABEL_MAP = {0: "DOWN", 1: "FLAT", 2: "UP"}


class StockDataset(Dataset):
    _KEYS = ["s_o","s_h","s_c","s_m","s_n","news_mask","label","news_quality","ticker_id"]
    def __init__(self, d):
        self.d = d
        self.keys = [k for k in self._KEYS if k in d]
    def __len__(self): return len(self.d["label"])
    def __getitem__(self, i): return {k: self.d[k][i] for k in self.keys}


def find_model(seed=42, output_dir="output"):
    pattern = os.path.join(output_dir, f"best_model_*_seed{seed}_fixed.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def load_model(pt_path, macro_dim, news_dim):
    quality_dim = getattr(GlobalConfig, "QUALITY_DIM", 4)
    model = StockMovementModel(
        price_dim=1, macro_dim=macro_dim, news_dim=news_dim,
        dim=TrainConfig.dim, input_dim=TrainConfig.window_size,
        output_dim=3, num_head=TrainConfig.num_head,
        dropout=0.0, class_weights=None, use_focal_loss=False,
        device=DEVICE, n_tickers=N_TICKERS,
        quality_dim=quality_dim,
    ).to(DEVICE)
    state = torch.load(pt_path, map_location=DEVICE, weights_only=True)
    missing, _ = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if "loss_fn" not in k]
    if real_missing:
        print(f"  [WARN] Missing keys: {real_missing}")
    model.eval()
    return model


def get_predictions(model, data_dict):
    """Run inference, return (preds, labels) arrays."""
    if not data_dict or len(data_dict.get("label", [])) == 0:
        return [], []
    ds = StockDataset(data_dict)
    ldr = DataLoader(ds, batch_size=128, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in ldr:
            q = batch.get("news_quality")
            _, _, preds = model(
                batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                batch["s_c"].to(DEVICE), batch["s_m"].to(DEVICE),
                batch["s_n"].to(DEVICE), batch["label"].to(DEVICE),
                mode="test", return_preds=True,
                ticker_id=batch.get("ticker_id"),
                news_mask=batch.get("news_mask"),
                news_quality=q.to(DEVICE) if q is not None else None,
            )
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch["label"].cpu().numpy().tolist())
    return all_preds, all_labels


def extract_close_prices(raw_data, ticker, test_start_idx, window_size=20):
    """
    Extract closing prices for the test period.
    test_start_idx: the global row index where test data starts.
    Returns list of floats (one per test sample).
    """
    dates = sorted(raw_data.keys())
    prices = []
    for date_key in dates:
        day = raw_data[date_key]
        p = day.get("price", {}).get(ticker)
        if p is None:
            continue
        for k in ("Close", "close"):
            v = p.get(k)
            if v is not None:
                try:
                    prices.append(float(v))
                    break
                except (TypeError, ValueError):
                    pass

    # Test samples start at test_start_idx (each sample = row t in the price series)
    # The "current" price for sample t is prices[t], "next" is prices[t + window_size]
    result = []
    for t in range(test_start_idx, len(prices) - window_size):
        result.append({
            "price": round(prices[t + window_size - 1], 2),       # last day of window
            "next_price": round(prices[t + window_size], 2),       # prediction target day
        })
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default=None)
    ap.add_argument("--model", default=None, help="Path to .pt (auto-detected if omitted)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--output", default="demo_predictions.json")
    args = ap.parse_args()

    pkl = args.pkl or os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    if not os.path.exists(pkl):
        print(f"PKL not found: {pkl}")
        return

    tickers = [t.upper() for t in args.tickers] if args.tickers else GlobalConfig.TICKERS
    print(f"Exporting predictions for: {tickers}")
    print(f"Device: {DEVICE}")

    # ── Find model ────────────────────────────────────────────────────────────
    pt_path = args.model or find_model(args.seed)
    if not pt_path or not os.path.exists(pt_path):
        print(f"\nNo model found. Run: python main.py --seed {args.seed}")
        print("Then re-run this script.")
        return
    print(f"Loading model: {os.path.basename(pt_path)}")

    # ── Load data ─────────────────────────────────────────────────────────────
    dp = data_prepare(pkl, include_ticker_id=True)
    valid_T = [dp.get_max_T(t) for t in tickers if dp.get_max_T(t) > 0]
    if not valid_T:
        print("No valid ticker data.")
        return

    global_T_max = min(valid_T)
    train_ratio  = getattr(TrainConfig, "train_ratio", 0.70)
    valid_ratio  = getattr(TrainConfig, "valid_ratio", 0.15)
    val_end      = int(global_T_max * (train_ratio + valid_ratio))   # = test start

    print(f"T_max={global_T_max}  test_start={val_end}  test_n={global_T_max - val_end}")

    # Determine macro/news dim from first valid ticker
    first_ticker = next(t for t in tickers if dp.get_max_T(t) > 0)
    _, _, te_sample = dp.prepare_data(
        first_ticker, train_end=int(global_T_max*train_ratio),
        val_end=val_end, test_end=global_T_max
    )
    macro_dim = te_sample["s_m"].shape[-1]
    news_dim  = te_sample["s_n"].shape[-1]
    print(f"macro_dim={macro_dim}  news_dim={news_dim}")

    # ── Load model ────────────────────────────────────────────────────────────
    model = load_model(pt_path, macro_dim, news_dim)

    # ── Per-ticker export ─────────────────────────────────────────────────────
    output = {}
    all_preds_global, all_labels_global = [], []

    for ticker in tickers:
        if dp.get_max_T(ticker) == 0:
            print(f"  {ticker}: no data, skip")
            continue

        _, _, test_data = dp.prepare_data(
            ticker,
            train_end=int(global_T_max * train_ratio),
            val_end=val_end,
            test_end=global_T_max,
        )
        if not test_data or len(test_data.get("label", [])) == 0:
            print(f"  {ticker}: empty test set, skip")
            continue

        preds, labels = get_predictions(model, test_data)
        acc = accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        n   = len(preds)

        # Get raw closing prices from the PKL
        price_data = extract_close_prices(dp.raw_data, ticker, val_end)
        # Align to number of predictions
        price_data = price_data[:n]
        while len(price_data) < n:
            price_data.append({"price": 0.0, "next_price": 0.0})

        days = []
        for i in range(n):
            days.append({
                "day":       i + 1,
                "predicted": LABEL_MAP[preds[i]],
                "actual":    LABEL_MAP[labels[i]],
                "correct":   preds[i] == labels[i],
                "price":     price_data[i]["price"],
                "next_price":price_data[i]["next_price"],
            })

        output[ticker] = {
            "acc":   round(acc, 4),
            "mcc":   round(mcc, 4),
            "n":     n,
            "days":  days,
        }
        all_preds_global.extend(preds)
        all_labels_global.extend(labels)

        correct = sum(1 for p, l in zip(preds, labels) if p == l)
        print(f"  {ticker}: n={n}  ACC={acc:.4f}  MCC={mcc:.4f}  correct={correct}/{n}")

    if all_preds_global:
        g_acc = accuracy_score(all_labels_global, all_preds_global)
        g_mcc = matthews_corrcoef(all_labels_global, all_preds_global)
        output["_meta"] = {
            "model": os.path.basename(pt_path),
            "tickers": tickers,
            "global_acc": round(g_acc, 4),
            "global_mcc": round(g_mcc, 4),
            "test_start": val_end,
            "global_T_max": global_T_max,
        }
        print(f"\nGlobal  ACC={g_acc:.4f}  MCC={g_mcc:.4f}  n={len(all_preds_global)}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved -> {args.output}")
    print("Paste this file's content into the demo widget (see instructions below).")


if __name__ == "__main__":
    main()