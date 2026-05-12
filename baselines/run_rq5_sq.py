# -*- coding: utf-8 -*-
# baselines/run_rq5.py  — V2 (OFAT methodology, khớp paper)
"""
RQ5: Kích thước không gian ẩn và tốc độ học ảnh hưởng như thế nào đến
     hiệu quả dự báo của mô hình đề xuất?

═══════════════════════════════════════════════════════════════════
METHODOLOGY — One-Factor-At-a-Time (OFAT):

  Thay đổi MỘT biến, giữ CÁC BIẾN KHÁC cố định ở giá trị best.
  "Best" lấy từ kết quả grid search trong run_experiments.py:
    best_dim     = 64      (từ best_hparams.json["MSGCA_FV"])
    best_lr      = 1e-4    (từ best_hparams.json["MSGCA_FV"])
    best_dropout = 0.2     (từ best_hparams.json["MSGCA_FV"])

  Sweep A — dim sensitivity (lr=1e-4 cố định):
    d ∈ [16, 32, 64, 128]
    Câu hỏi: model underfit ở dim nhỏ, overfit ở dim lớn?

  Sweep B — lr sensitivity (dim=64 cố định):
    lr ∈ [1e-3, 5e-4, 1e-4, 5e-5]
    Câu hỏi: lr nào hội tụ tốt nhất trong ngân sách epoch hợp lý?

  Tổng: 4 + 4 = 8 combos × n_seeds runs
        (so với grid 4×4=16 — tiết kiệm 50% compute)

  Kết quả paper gốc (MSGCA, 4 datasets):
    dim: tăng từ 16→64 cải thiện, giảm ở 128 (overfit)
    lr : 1e-4 tốt nhất; 5e-4 và 1e-3 bỏ qua optimal; 5e-5 too slow

  Protocol (khớp run_experiments.py MSGCA_FV):
    Model  : MSGCA_FV (CE loss, không class weights — fair comparison)
    Phase 1: train=[0:hval_split], val=[hval_split:inner_T]
             LinearLR warmup(15ep) → CosineAnnealingLR + early stopping (patience=30)
             → tìm best_epoch
    Phase 2: train=[0:inner_T] for best_epoch → eval test=[inner_T:T_max]
    Dropout: 0.2 (best_hparams.json)
    ModDrop: TrainConfig.news_modality_dropout (30%)

  NOTE — patience=30, KHÔNG dùng TrainConfig.early_stop_patience:
    TrainConfig.early_stop_patience có thể = 9999 (disabled cho main.py).
    RQ5 cần early stopping để khớp run_experiments.py (avg_ep≈43).
    Nếu không có early stopping, seed=42 sẽ chạy hết 200ep → overtrain.

═══════════════════════════════════════════════════════════════════

Usage:
  python baselines/run_rq5.py
  python baselines/run_rq5.py --n-seeds 3
  python baselines/run_rq5.py --dims 16 32 64 128 --lrs 1e-3 5e-4 1e-4 5e-5
  python baselines/run_rq5.py --sweep dim      # only Sweep A
  python baselines/run_rq5.py --sweep lr       # only Sweep B
  python baselines/run_rq5.py --skip-existing  # resume from checkpoint

Outputs:
  baselines/results/rq5_table.txt
  baselines/results/rq5_raw.json
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import TrainConfig, GlobalConfig
from src.data_loader import data_prepare, N_TICKERS, NEWS_EMB_DIM
from src.model import StockMovementModel

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")

SEEDS = [42, 123, 256, 512, 1024]

# ── OFAT fixed values — lấy từ best_hparams.json["MSGCA_FV"] ─────────────────
BEST_DIM     = 64     # giữ cố định khi sweep lr
BEST_LR      = 1e-4   # giữ cố định khi sweep dim
BEST_DROPOUT = 0.2    # giữ cố định trong cả hai sweeps

# ── Sweep grids ────────────────────────────────────────────────────────────────
DEFAULT_DIMS = [32, 64, 128, 256]             # Sweep A
DEFAULT_LRS  = [1e-3, 5e-4, 1e-4, 5e-5]     # Sweep B

# ── patience=30, KHÔNG kế thừa TrainConfig.early_stop_patience ────────────────
# TrainConfig.early_stop_patience có thể = 9999 → overtrain → MCC drop
# 30 khớp run_experiments.py: avg_ep≈43 khi dim=64, lr=1e-4
RQ5_PATIENCE = 30
_MOD_DROPOUT = TrainConfig.news_modality_dropout  # 0.30


# =============================================================================
# DATA
# =============================================================================

class StockDataset(Dataset):
    _KEYS = ["s_o", "s_h", "s_c", "s_m", "s_n",
             "news_mask", "label", "ticker_id", "news_quality"]

    def __init__(self, d: dict):
        self.d    = d
        self.keys = [k for k in self._KEYS if k in d]

    def __len__(self):
        return len(self.d["label"])

    def __getitem__(self, i):
        return {k: self.d[k][i] for k in self.keys}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def merge(dicts: list, shuffle: bool = False) -> dict:
    valid = [d for d in dicts if d and len(d.get("label", [])) > 0]
    if not valid:
        return {}
    m: dict = {}
    for key in valid[0]:
        parts = [d[key] for d in valid if key in d and isinstance(d[key], torch.Tensor)]
        if parts:
            m[key] = torch.cat(parts, dim=0)
    if shuffle and "label" in m:
        idx = torch.randperm(len(m["label"]))
        for k in m:
            m[k] = m[k][idx]
    return m


def load_splits(pkl_path: str, tickers: list) -> dict:
    """Split logic IDENTICAL với run_experiments.py."""
    dp = data_prepare(pkl_path, include_ticker_id=True)
    valid_T      = [dp.get_max_T(t) for t in tickers if dp.get_max_T(t) > 0]
    global_T_max = min(valid_T)
    inner_T      = int(global_T_max * 0.85)
    hval_split   = int(inner_T * 0.80)

    print(f"  global_T_max={global_T_max}  inner_T={inner_T}  hval_split={hval_split}")
    print(f"  test=[{inner_T}:{global_T_max}] ({global_T_max - inner_T} steps) "
          f"← identical to main.py ✓")

    tr_hv_list, va_hv_list, tr_fl_list, te_list = [], [], [], []
    macro_dim = news_dim = None

    for t in tickers:
        if dp.get_max_T(t) == 0:
            continue
        tr_hv, va_hv, _ = dp.prepare_data(
            t, train_end=hval_split, val_end=inner_T, test_end=inner_T
        )
        tr_fl, _, te = dp.prepare_data(
            t, train_end=inner_T, val_end=inner_T, test_end=global_T_max
        )
        if macro_dim is None and tr_hv and len(tr_hv.get("label", [])) > 0:
            macro_dim = tr_hv["s_m"].shape[-1]
            news_dim  = tr_hv["s_n"].shape[-1]

        if tr_hv and len(tr_hv.get("label", [])): tr_hv_list.append(tr_hv)
        if va_hv and len(va_hv.get("label", [])): va_hv_list.append(va_hv)
        if tr_fl and len(tr_fl.get("label", [])): tr_fl_list.append(tr_fl)
        if te    and len(te.get("label",    [])): te_list.append(te)

    print(f"  macro_dim={macro_dim}  news_dim={news_dim}  (NEWS_EMB_DIM={NEWS_EMB_DIM})")
    return {
        "inner_T":      inner_T,
        "global_T_max": global_T_max,
        "hval_split":   hval_split,
        "train_hval":   merge(tr_hv_list, shuffle=True),
        "val_fixed":    merge(va_hv_list, shuffle=False),
        "train_full":   merge(tr_fl_list, shuffle=True),
        "test":         merge(te_list,    shuffle=False),
        "macro_dim":    macro_dim,
        "news_dim":     news_dim,
    }


# =============================================================================
# MODEL
# =============================================================================

def _safe_num_head(dim: int) -> int:
    nh = TrainConfig.num_head
    if dim % nh == 0:
        return nh
    for c in range(nh, 0, -1):
        if dim % c == 0:
            return c
    return 1


def _make_adamw(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    no_kw = ["bias", "LayerNorm.weight", "layernorm.weight",
             "norm.weight", "attn_norm.weight", "out_norm.weight"]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if any(k in name for k in no_kw) else decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay,    "weight_decay": getattr(TrainConfig, "weight_decay", 1e-4)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def _build_msgca_fv(macro_dim, news_dim, dim, lr, dropout=BEST_DROPOUT):
    model = StockMovementModel(
        price_dim=1, macro_dim=macro_dim, news_dim=news_dim,
        dim=dim, input_dim=TrainConfig.window_size,
        output_dim=3, num_head=_safe_num_head(dim),
        dropout=dropout,
        class_weights=None, use_focal_loss=False, focal_gamma=2.0,
        device=DEVICE, n_tickers=N_TICKERS,
        quality_dim=getattr(GlobalConfig, "QUALITY_DIM", 4),
    ).to(DEVICE)
    return model, _make_adamw(model, lr)


# =============================================================================
# TRAIN / EVAL
# =============================================================================

def evaluate(model, data: dict) -> Tuple[float, float]:
    if not data or len(data.get("label", [])) == 0:
        return 0.0, 0.0
    model.eval()
    ldr = DataLoader(StockDataset(data),
                     batch_size=256 if DEVICE.type == "cuda" else 64,
                     shuffle=False, num_workers=0,
                     pin_memory=(DEVICE.type == "cuda"))
    preds_all, labels_all = [], []
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
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(batch["label"].numpy())
    if len(set(labels_all)) < 2:
        return float(accuracy_score(labels_all, preds_all)), 0.0
    return (float(accuracy_score(labels_all, preds_all)),
            float(matthews_corrcoef(labels_all, preds_all)))


def _train_epoch(model, loader, opt) -> float:
    model.train()
    total = 0.0
    for batch in loader:
        opt.zero_grad(set_to_none=True)
        s_n     = batch["s_n"].to(DEVICE)
        mask_in = batch.get("news_mask")
        q_in    = batch.get("news_quality")
        if _MOD_DROPOUT > 0.0 and torch.rand(1).item() < _MOD_DROPOUT:
            s_n = torch.zeros_like(s_n)
            if mask_in is not None:
                mask_in = torch.ones_like(mask_in, dtype=torch.bool)
            q_in = None
        loss = model(
            batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
            batch["s_c"].to(DEVICE), batch["s_m"].to(DEVICE),
            s_n, batch["label"].to(DEVICE), mode="train",
            ticker_id=batch.get("ticker_id"),
            news_mask=mask_in.to(DEVICE) if mask_in is not None else None,
            news_quality=q_in.to(DEVICE) if q_in is not None else None,
        )
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
    return total


def run_one_seed(seed, splits, dim, lr, max_epochs=200, patience=RQ5_PATIENCE):
    """2-phase training — khớp run_experiments.py _run_msgca_one_seed()."""
    warmup_epochs = 15
    set_seed(seed)
    macro_dim, news_dim = splits["macro_dim"], splits["news_dim"]

    # Phase 1: find best_epoch
    model, opt = _build_msgca_fv(macro_dim, news_dim, dim, lr)
    ldr = DataLoader(StockDataset(splits["train_hval"]),
                     batch_size=getattr(TrainConfig, "batch_size", 32),
                     shuffle=True, drop_last=False, num_workers=0,
                     pin_memory=(DEVICE.type == "cuda"))
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(max_epochs - warmup_epochs, 1), eta_min=1e-6)

    best_mcc, best_epoch, no_improve = -2.0, 1, 0
    min_active = max(warmup_epochs, 40)

    for epoch in range(max_epochs):
        _train_epoch(model, ldr, opt)
        (warmup if epoch < warmup_epochs else cosine).step()
        if epoch >= min_active:
            _, mcc = evaluate(model, splits["val_fixed"])
            if mcc > best_mcc:
                best_mcc = mcc; best_epoch = epoch + 1; no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    # Phase 2: retrain full
    set_seed(seed)
    model2, opt2 = _build_msgca_fv(macro_dim, news_dim, dim, lr)
    ldr2 = DataLoader(StockDataset(splits["train_full"]),
                      batch_size=getattr(TrainConfig, "batch_size", 32),
                      shuffle=True, drop_last=False, num_workers=0,
                      pin_memory=(DEVICE.type == "cuda"))
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=max(best_epoch, 1), eta_min=1e-6)
    for _ in range(best_epoch):
        _train_epoch(model2, ldr2, opt2)
        sched2.step()

    acc, mcc = evaluate(model2, splits["test"])
    return acc, mcc, best_epoch


def run_sweep(sweep_vals, fixed_dim, fixed_lr, sweep_dim,
              splits, n_seeds, max_epochs, patience, verbose=True):
    results = {}
    for val in sweep_vals:
        dim = val if sweep_dim else fixed_dim
        lr  = fixed_lr if sweep_dim else val
        key = f"dim{dim}" if sweep_dim else f"lr{_lr_label(lr)}"
        print(f"\n    [{key}]  dim={dim}  lr={_lr_label(lr)}  "
              f"num_head={_safe_num_head(dim)}")
        acc_list, mcc_list, ep_list = [], [], []
        for seed in SEEDS[:n_seeds]:
            t0 = time.time()
            acc, mcc, ep = run_one_seed(seed, splits, dim, lr, max_epochs, patience)
            acc_list.append(acc); mcc_list.append(mcc); ep_list.append(ep)
            if verbose:
                print(f"      seed={seed:5d}  ACC={acc:.4f}  MCC={mcc:.4f}  "
                      f"ep={ep:3d}  ({time.time()-t0:.0f}s)")
        r = {
            "dim":      dim, "lr": lr, "num_head": _safe_num_head(dim),
            "acc_mean": float(np.mean(acc_list)), "acc_std": float(np.std(acc_list)),
            "mcc_mean": float(np.mean(mcc_list)), "mcc_std": float(np.std(mcc_list)),
            "ep_mean":  float(np.mean(ep_list)),
            "acc_list": [float(x) for x in acc_list],
            "mcc_list": [float(x) for x in mcc_list],
            "n_seeds":  len(acc_list), "sweep": "dim" if sweep_dim else "lr",
            "is_best":  (dim == BEST_DIM and abs(lr - BEST_LR) < 1e-10),
        }
        results[key] = r
        print(f"    → ACC={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}  "
              f"MCC={r['mcc_mean']:.4f}+/-{r['mcc_std']:.4f}  ep≈{r['ep_mean']:.0f}")
    return results


# =============================================================================
# REFERENCE
# =============================================================================

def load_reference(results_dir: str) -> dict:
    raw_path = os.path.join(results_dir, "raw_results.json")
    if not os.path.exists(raw_path):
        print("  [Ref] raw_results.json not found — run run_experiments.py first.")
        return {}
    try:
        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)
        if "MSGCA_FV" in raw:
            r = raw["MSGCA_FV"]
            print(f"  [Ref] MSGCA_FV: MCC={r['mcc_mean']:.4f}+/-{r['mcc_std']:.4f}  "
                  f"(dim={BEST_DIM}, lr={_lr_label(BEST_LR)}, dropout={BEST_DROPOUT})")
            return {"MSGCA_FV": r}
    except Exception as e:
        print(f"  [Ref] Error: {e}")
    return {}


# =============================================================================
# FORMATTERS
# =============================================================================

def _lr_label(lr: float) -> str:
    s = f"{lr:.0e}"; exp = int(s.split("e")[1])
    return f"{s.split('e')[0]}e{exp}"


def format_sweep_table(sweep_results, sweep_type, ref_mcc):
    sep = "─" * 90
    if sweep_type == "dim":
        header = (f"  Sweep A — Hidden Dimension Size  "
                  f"(lr={_lr_label(BEST_LR)} fixed, dropout={BEST_DROPOUT})")
        sort_key = lambda k: sweep_results[k]["dim"]
    else:
        header = (f"  Sweep B — Learning Rate  "
                  f"(dim={BEST_DIM} fixed, dropout={BEST_DROPOUT})")
        sort_key = lambda k: -sweep_results[k]["lr"]

    lines = [sep, header, sep,
             f"  {'val':<10} {'num_head':>9} {'ACC':>16} {'MCC':>16}  "
             f"{'dMCC(ref)':>10}  {'ep_mean':>8}",
             "  " + "─" * 65]

    sorted_keys  = sorted(sweep_results.keys(), key=sort_key)
    best_mcc_val = max(sweep_results[k]["mcc_mean"] for k in sorted_keys)

    for k in sorted_keys:
        r   = sweep_results[k]
        val = _lr_label(r["lr"]) if sweep_type == "lr" else str(r["dim"])
        acc = f"{r['acc_mean']:.4f}+/-{r['acc_std']:.4f}"
        mcc = f"{r['mcc_mean']:.4f}+/-{r['mcc_std']:.4f}"
        dmcc = (f"{r['mcc_mean']-ref_mcc:+.4f}" if ref_mcc is not None else "N/A")
        star     = " ✓" if abs(r["mcc_mean"] - best_mcc_val) < 1e-6 else "  "
        ref_mark = " ←ref" if r.get("is_best") else ""
        lines.append(f"  {val:<10} {r['num_head']:>9} {acc:>16} {mcc:>16}  "
                     f"{dmcc:>10}  {r['ep_mean']:>8.1f}{star}{ref_mark}")
    lines.append(sep)
    return "\n".join(lines)


def format_interpretation(dim_results, lr_results, ref_mcc):
    lines = ["\n" + "=" * 90, "  INTERPRETATION (RQ5)", "=" * 90]

    # Sweep A
    lines.append("\n  A) Hidden Dimension Effect:")
    sorted_d = sorted(dim_results, key=lambda k: dim_results[k]["dim"])
    dim_mccs = [(dim_results[k]["dim"], dim_results[k]["mcc_mean"]) for k in sorted_d]
    mccs     = [x[1] for x in dim_mccs]
    peak_idx = mccs.index(max(mccs))
    best_dim = dim_mccs[peak_idx][0]
    lines.append(f"     Best dim: {best_dim}  (MCC={max(mccs):.4f})")
    if 0 < peak_idx < len(mccs) - 1:
        lines.append(f"     Pattern : increases {dim_mccs[0][0]}→{best_dim}, "
                     f"then decreases → underfitting (small) / overfitting (large)")
    if len(dim_mccs) >= 2:
        gains = "  ".join(f"{dim_mccs[i][0]}→{dim_mccs[i+1][0]}: "
                          f"{dim_mccs[i+1][1]-dim_mccs[i][1]:+.4f}"
                          for i in range(len(dim_mccs)-1))
        lines.append(f"     ΔMCCs  : {gains}")

    # Sweep B
    lines.append("\n  B) Learning Rate Effect:")
    sorted_l = sorted(lr_results, key=lambda k: -lr_results[k]["lr"])
    for k in sorted_l:
        r  = lr_results[k]
        lr = r["lr"]
        note = ("too fast — skips optimal" if lr >= 5e-4 else
                "too slow — low efficiency" if lr <= 5e-5 else "balanced")
        lines.append(f"     lr={_lr_label(lr)}: MCC={r['mcc_mean']:.4f}  "
                     f"avg_ep≈{r['ep_mean']:.0f}  [{note}]")

    # Conclusion
    lines.append("\n  CONCLUSION:")
    all_r  = list(dim_results.values()) + list(lr_results.values())
    best_r = max(all_r, key=lambda r: r["mcc_mean"])
    if ref_mcc is not None:
        delta = best_r["mcc_mean"] - ref_mcc
        if abs(delta) < 0.005:
            lines.append(f"  dim={BEST_DIM}, lr={_lr_label(BEST_LR)} confirmed near-optimal "
                         f"(best Δ={delta:+.4f}).")
        elif delta > 0:
            lines.append(f"  Better: dim={best_r['dim']}, lr={_lr_label(best_r['lr'])}  "
                         f"(Δ={delta:+.4f} vs reference). Consider updating TrainConfig.")
        else:
            lines.append(f"  Reference settings (dim={BEST_DIM}, "
                         f"lr={_lr_label(BEST_LR)}) are optimal.")
    lines.append("=" * 90)
    return "\n".join(lines)


def format_latex(dim_results, lr_results):
    lines = ["\n  [LaTeX]",
             "  % Sweep A",
             "  dim & num\\_head & ACC & MCC \\\\"]
    for k in sorted(dim_results, key=lambda k: dim_results[k]["dim"]):
        r = dim_results[k]
        lines.append(f"  {r['dim']} & {r['num_head']} "
                     f"& {r['acc_mean']:.4f}$\\pm${r['acc_std']:.4f} "
                     f"& {r['mcc_mean']:.4f}$\\pm${r['mcc_std']:.4f} \\\\")
    lines += ["  % Sweep B", "  lr & ACC & MCC & avg\\_ep \\\\"]
    for k in sorted(lr_results, key=lambda k: -lr_results[k]["lr"]):
        r = lr_results[k]
        lines.append(f"  ${_lr_label(r['lr'])}$ "
                     f"& {r['acc_mean']:.4f}$\\pm${r['acc_std']:.4f} "
                     f"& {r['mcc_mean']:.4f}$\\pm${r['mcc_std']:.4f} "
                     f"& {r['ep_mean']:.0f} \\\\")
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="RQ5: Hyperparameter sensitivity — OFAT, MSGCA_FV")
    ap.add_argument("--pkl",           default=None)
    ap.add_argument("--dims",          nargs="+", type=int,   default=DEFAULT_DIMS,
                    help="Dim values for Sweep A (default: 16 32 64 128)")
    ap.add_argument("--lrs",           nargs="+", type=float, default=DEFAULT_LRS,
                    help="LR values for Sweep B (default: 1e-3 5e-4 1e-4 5e-5)")
    ap.add_argument("--n-seeds",       type=int,  default=5)
    ap.add_argument("--epochs",        type=int,  default=200)
    ap.add_argument("--patience",      type=int,  default=RQ5_PATIENCE,
                    help=f"Early stopping patience (default={RQ5_PATIENCE}). "
                         f"TrainConfig.early_stop_patience={TrainConfig.early_stop_patience}"
                         f" is INTENTIONALLY IGNORED.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Resume from rq5_raw.json checkpoint")
    ap.add_argument("--sweep",         choices=["both", "dim", "lr"], default="both",
                    help="Which sweep to run: 'dim' (Sweep A), 'lr' (Sweep B), 'both'")
    ap.add_argument("--best-dim",      type=int,   default=BEST_DIM,
                    help=f"Fixed dim when sweeping lr (default={BEST_DIM}, "
                         f"from best_hparams.json[MSGCA_FV])")
    ap.add_argument("--best-lr",       type=float, default=BEST_LR,
                    help=f"Fixed lr when sweeping dim (default={BEST_LR:.0e}, "
                         f"from best_hparams.json[MSGCA_FV])")
    ap.add_argument("--verbose",       action="store_true", default=True)
    args = ap.parse_args()
    # Override module-level constants if user specified via CLI
    best_dim = args.best_dim
    best_lr  = args.best_lr

    pkl_path = args.pkl or os.path.join(
        GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    if not os.path.exists(pkl_path):
        print(f"Dataset not found: {pkl_path}"); sys.exit(1)

    tickers  = GlobalConfig.TICKERS
    run_dim  = args.sweep in ("both", "dim")
    run_lr   = args.sweep in ("both", "lr")
    n_combos = (len(args.dims) if run_dim else 0) + (len(args.lrs) if run_lr else 0)

    print(f"\n{'='*70}")
    print(f"  RQ5 — Hyperparameter Sensitivity  (OFAT, MSGCA_FV)")
    print(f"  Device   : {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print(f"  Method   : One-Factor-At-a-Time (OFAT)")
    print(f"  Fixed    : best_dim={best_dim}  best_lr={_lr_label(best_lr)}  "
          f"dropout={BEST_DROPOUT}")
    if run_dim:
        print(f"  Sweep A  : dim={args.dims}  (lr={_lr_label(best_lr)} fixed)")
    if run_lr:
        print(f"  Sweep B  : lr={[_lr_label(l) for l in args.lrs]}  "
              f"(dim={best_dim} fixed)")
    print(f"  Combos   : {n_combos} × n_seeds={args.n_seeds} = "
          f"{n_combos * args.n_seeds} runs")
    print(f"  Epochs   : max={args.epochs}  warmup=15  patience={args.patience}")
    print(f"  WARNING  : patience={args.patience} (run_experiments protocol), "
          f"TrainConfig.early_stop_patience={TrainConfig.early_stop_patience} IGNORED")
    print(f"{'='*70}")

    print("\nLoading reference...")
    refs    = load_reference(RESULTS_DIR)
    ref_mcc = refs.get("MSGCA_FV", {}).get("mcc_mean")

    print("\nLoading data splits...")
    splits = load_splits(pkl_path, tickers)

    raw_path = os.path.join(RESULTS_DIR, "rq5_raw.json")
    all_raw: dict = {"dim_sweep": {}, "lr_sweep": {}}
    if args.skip_existing and os.path.exists(raw_path):
        try:
            with open(raw_path, encoding="utf-8") as f:
                all_raw = json.load(f)
            print(f"  Checkpoint: {len(all_raw.get('dim_sweep',{}))} dim, "
                  f"{len(all_raw.get('lr_sweep',{}))} lr combos loaded")
        except Exception as e:
            print(f"  [WARN] Checkpoint load failed: {e}")

    total_t0 = time.time()

    if run_dim:
        print(f"\n{'─'*70}")
        print(f"  Sweep A — dim  (lr={_lr_label(best_lr)} fixed)")
        print(f"{'─'*70}")
        to_run = ([d for d in args.dims if f"dim{d}" not in all_raw["dim_sweep"]]
                  if args.skip_existing else args.dims)
        if args.skip_existing and len(to_run) < len(args.dims):
            print(f"  [SKIP] already done: {[d for d in args.dims if d not in to_run]}")
        new = run_sweep(to_run, best_dim, best_lr, sweep_dim=True,
                        splits=splits, n_seeds=args.n_seeds,
                        max_epochs=args.epochs, patience=args.patience,
                        verbose=args.verbose)
        all_raw["dim_sweep"].update(new)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(all_raw, f, indent=2)

    if run_lr:
        print(f"\n{'─'*70}")
        print(f"  Sweep B — lr  (dim={best_dim} fixed)")
        print(f"{'─'*70}")
        to_run = ([l for l in args.lrs if f"lr{_lr_label(l)}" not in all_raw["lr_sweep"]]
                  if args.skip_existing else args.lrs)
        if args.skip_existing and len(to_run) < len(args.lrs):
            print(f"  [SKIP] already done: {[_lr_label(l) for l in args.lrs if l not in to_run]}")
        new = run_sweep(to_run, best_dim, best_lr, sweep_dim=False,
                        splits=splits, n_seeds=args.n_seeds,
                        max_epochs=args.epochs, patience=args.patience,
                        verbose=args.verbose)
        all_raw["lr_sweep"].update(new)
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(all_raw, f, indent=2)

    print(f"\nTotal: {(time.time()-total_t0)/60:.1f} min\n")

    dim_results = all_raw.get("dim_sweep", {})
    lr_results  = all_raw.get("lr_sweep",  {})

    lines = ["=" * 90,
             "  RQ5 — Effect of Hidden Dim and Learning Rate  (MSGCA_FV, CE loss)",
             f"  Protocol : OFAT — one variable at a time, others fixed at best",
             f"  Best ref : dim={BEST_DIM}, lr={_lr_label(BEST_LR)}, "
             f"dropout={BEST_DROPOUT}  (from run_experiments.py)",
             "=" * 90]
    if dim_results:
        lines.append(format_sweep_table(dim_results, "dim", ref_mcc))
    if lr_results:
        lines.append(format_sweep_table(lr_results, "lr", ref_mcc))
    if ref_mcc is not None:
        lines.append(f"\n  Reference MSGCA_FV: MCC={ref_mcc:.4f}+/-"
                     f"{refs['MSGCA_FV']['mcc_std']:.4f}")
    if dim_results and lr_results:
        lines.append(format_interpretation(dim_results, lr_results, ref_mcc))
        lines.append(format_latex(dim_results, lr_results))

    table = "\n".join(lines)
    print(table)

    table_path = os.path.join(RESULTS_DIR, "rq5_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table)
    print(f"\nSaved → {table_path}\n         {raw_path}")


if __name__ == "__main__":
    main()