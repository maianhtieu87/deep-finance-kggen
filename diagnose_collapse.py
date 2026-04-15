#!/usr/bin/env python3
# diagnose_collapse.py
"""
Script chẩn đoán nhanh: phân biệt collapse thật vs collapse do bug inference.

Chạy script này TRƯỚC để xác nhận vấn đề thực sự:
  python diagnose_collapse.py --model output/best_model_label=rolling_price=vol_adjusted_tid_seed42_wf.pt

Nếu kết quả A và B khác nhau nhiều → Bug #1 (analyze_performance.py sai forward pass)
Nếu cả hai đều collapse → Cần fix model (Bugs #2-5)
"""

import os
import sys
import numpy as np
import torch
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import StockMovementModel
from src.data_loader import data_prepare, N_TICKERS
from configs.config import TrainConfig, GlobalConfig
from torch.utils.data import DataLoader, Dataset
import argparse


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MinimalDataset(Dataset):
    def __init__(self, d):
        self.d    = d
        self.keys = [k for k in ["s_o","s_h","s_c","s_m","s_n","news_mask","label","ticker_id"] if k in d]
    def __len__(self): return len(self.d["label"])
    def __getitem__(self, i): return {k: self.d[k][i] for k in self.keys}


def load_one_ticker_test(pkl_path, ticker="TSLA"):
    dp = data_prepare(pkl_path, include_ticker_id=True)
    _, _, test = dp.prepare_data(ticker)
    return test


def load_model(model_path, macro_dim, news_dim):
    m = StockMovementModel(
        price_dim=1, macro_dim=macro_dim, news_dim=news_dim,
        dim=TrainConfig.dim, input_dim=TrainConfig.window_size,
        output_dim=3, num_head=TrainConfig.num_head, device=DEVICE,
        n_tickers=N_TICKERS, dropout=0.0, class_weights=None, use_focal_loss=False,
    ).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def infer_correct(model, data):
    """Path A: Dùng model.forward() đúng."""
    ds  = MinimalDataset(data)
    ldr = DataLoader(ds, batch_size=64, shuffle=False)
    all_preds, all_probs = [], []
    for batch in ldr:
        logits = model(
            batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
            batch["s_c"].to(DEVICE), batch["s_m"].to(DEVICE),
            batch["s_n"].to(DEVICE),
            label=batch["label"].to(DEVICE),
            mode="eval",
            ticker_id=batch.get("ticker_id"),
            news_mask=batch.get("news_mask"),
        )
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_probs)


@torch.no_grad()
def infer_wrong(model, data):
    """Path B: Dùng forward pass giống analyze_performance.py cũ (SAI)."""
    ds  = MinimalDataset(data)
    ldr = DataLoader(ds, batch_size=64, shuffle=False)
    all_preds, all_probs = [], []
    for batch in ldr:
        s_o = batch["s_o"].to(DEVICE)
        s_h = batch["s_h"].to(DEVICE)
        s_c = batch["s_c"].to(DEVICE)
        s_m = batch["s_m"].to(DEVICE)
        s_n = batch["s_n"].to(DEVICE)

        # Sai: sequential + bỏ qua gate + bỏ qua pre_predict_proj
        v_m, v_i, v_n = model.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)
        if v_n is None:
            v_n = torch.zeros_like(v_i)
        H1      = model.fusion_stage1(primary=v_i, aux=v_n)
        H_final = model.fusion_stage2(primary=H1, aux=v_m)
        logits  = model.movement_predictor(fused_seq=H_final, orig_seq=v_i)
        logits  = torch.clamp(logits, -15, 15)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return np.array(all_preds), np.array(all_probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ticker", default="TSLA")
    ap.add_argument("--pkl",   default=os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl"))
    args = ap.parse_args()

    print(f"\n{'='*60}")
    print(f"COLLAPSE DIAGNOSIS: {args.ticker}")
    print(f"{'='*60}")

    data = load_one_ticker_test(args.pkl, args.ticker)
    if not data or len(data.get("label", [])) == 0:
        print(f"No test data for {args.ticker}")
        return

    macro_dim = data["s_m"].shape[-1]
    news_dim  = data["s_n"].shape[-1]
    labels    = data["label"].numpy()
    model     = load_model(args.model, macro_dim, news_dim)

    print(f"\nTrue label dist: {dict(Counter(labels))}")

    # Path A: Correct
    preds_a, probs_a = infer_correct(model, data)
    conf_a = probs_a.max(axis=1).mean()
    print(f"\n[A] CORRECT forward pass (model.forward):")
    print(f"    Pred dist: {dict(Counter(preds_a.tolist()))}")
    print(f"    Mean max-prob: {conf_a:.4f}  (random=0.333, perfect=1.0)")
    classes_predicted_a = len(set(preds_a.tolist()))
    print(f"    Classes predicted: {classes_predicted_a}/3")

    # Path B: Wrong (like old analyze_performance.py)
    preds_b, probs_b = infer_wrong(model, data)
    conf_b = probs_b.max(axis=1).mean()
    print(f"\n[B] WRONG forward pass (old analyze_performance.py):")
    print(f"    Pred dist: {dict(Counter(preds_b.tolist()))}")
    print(f"    Mean max-prob: {conf_b:.4f}  (random=0.333, perfect=1.0)")
    classes_predicted_b = len(set(preds_b.tolist()))
    print(f"    Classes predicted: {classes_predicted_b}/3")

    print(f"\n{'='*60}")
    print("DIAGNOSIS:")
    if conf_b < 0.36 and conf_a > 0.40:
        print("  ✅ BUG CONFIRMED: analyze_performance.py sai forward pass.")
        print("  → Model thực sự KHÔNG collapse (A OK), nhưng B cho ảo giác collapse.")
        print("  → FIX: dùng analyze_performance_fixed.py")
    elif conf_a < 0.36 and conf_b < 0.36:
        print("  ⚠️  REAL COLLAPSE: Cả 2 paths đều collapse.")
        print("  → Cần retrain với main_fixed.py (focal loss + stronger weights)")
    elif classes_predicted_a < 3:
        print(f"  ⚠️  REAL PARTIAL COLLAPSE: Model chỉ predict {classes_predicted_a}/3 classes.")
        print("  → Cần retrain với main_fixed.py")
    else:
        print("  ✅ Model hoạt động bình thường.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()