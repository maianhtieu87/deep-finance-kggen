# -*- coding: utf-8 -*-
# baselines/run_rq5.py  — V1
"""
RQ5: Kích thước không gian ẩn và tốc độ học ảnh hưởng như thế nào đến
     hiệu quả dự báo của mô hình đề xuất?

═══════════════════════════════════════════════════════════════════
PHƯƠNG PHÁP:

  Mô hình    : MSGCA_FV (CE loss, không class weights) — đúng với
               "fair comparison" đã báo cáo trong RQ1.

  Grid search: dim × lr  (giữ nguyên dropout=0.1, protocol=fixed_val)
    dim (hidden_dim) : [32, 64, 128, 256]
    lr               : [5e-5, 1e-4, 5e-4, 1e-3]
    → 4 × 4 = 16 combos

  Mỗi combo chạy n_seeds (mặc định 5) với protocol giống hệt
  run_experiments.py MSGCA_FV:
    Phase 1 : train=[0:hval_split], val=[hval_split:inner_T]
              LinearLR warmup → CosineAnnealingLR + early stopping
              → tìm best_epoch
    Phase 2 : train=[0:inner_T] for best_epoch → eval test=[inner_T:T_max]

  Test set  : [inner_T:T_max] — IDENTICAL với main.py, run_ablation.py,
               run_rq3_rq4.py.

  Report    :
    1. Full grid MCC table (dim × lr heatmap text)
    2. Best combo vs reference (MSGCA_FV seed42-saved)
    3. Sensitivity analysis: dim effect, lr effect
    4. LaTeX table

═══════════════════════════════════════════════════════════════════

Usage:
  python baselines/run_rq5.py
  python baselines/run_rq5.py --n-seeds 3 --epochs 150
  python baselines/run_rq5.py --dims 32 64 128 --lrs 1e-4 5e-4
  python baselines/run_rq5.py --skip-existing   # resume từ checkpoint

Outputs:
  baselines/results/rq5_table.txt
  baselines/results/rq5_heatmap.txt
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

SEEDS        = [42, 123, 256, 512, 1024]
# RQ5 dùng patience riêng — KHÔNG kế thừa TrainConfig.early_stop_patience.
# TrainConfig.early_stop_patience có thể = 9999 (tắt early stopping cho main.py),
# nhưng RQ5 cần early stopping để khớp protocol run_experiments.py (avg_ep≈43).
# Giá trị 30 khớp với run_experiments.py _run_msgca_one_seed().
RQ5_PATIENCE = 30
_MOD_DROPOUT = TrainConfig.news_modality_dropout  # default 0.30

# ── RQ5 Search Grid ───────────────────────────────────────────────────────────
DEFAULT_DIMS = [32, 64, 128, 256]
DEFAULT_LRS  = [5e-5, 1e-4, 5e-4, 1e-3]
FIXED_DROPOUT = 0.2   # khớp với best_hparams.json["MSGCA_FV"]["dropout"]
                      # grid search trong run_experiments.py đã chọn 0.2 cho MSGCA_FV
                      # giữ cố định để isolate đúng dim + lr effect


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
    """
    Split logic IDENTICAL với run_ablation.py + run_rq3_rq4.py.
    test = [inner_T : T_max] — cùng test set với main.py.
    """
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
# MODEL FACTORY — MSGCA_FV (CE loss, no class weights)
# =============================================================================

def _build_msgca_fv(
    macro_dim: int,
    news_dim:  int,
    dim:       int,
    lr:        float,
    dropout:   float = FIXED_DROPOUT,
) -> Tuple[StockMovementModel, torch.optim.Optimizer]:
    """
    MSGCA_FV: CE loss, không class weights — "fair comparison" mode.
    dim và lr là hyperparameters cần khảo sát trong RQ5.
    """
    model = StockMovementModel(
        price_dim=1,
        macro_dim=macro_dim,
        news_dim=news_dim,
        dim=dim,                   # ← RQ5 variable
        input_dim=TrainConfig.window_size,
        output_dim=3,
        num_head=_safe_num_head(dim),
        dropout=dropout,
        class_weights=None,        # CE loss — no weighting
        use_focal_loss=False,      # CE loss — fair comparison
        focal_gamma=2.0,
        device=DEVICE,
        n_tickers=N_TICKERS,
        quality_dim=getattr(GlobalConfig, "QUALITY_DIM", 4),
    ).to(DEVICE)
    opt = _make_adamw(model, lr)   # ← RQ5 variable
    return model, opt


def _safe_num_head(dim: int) -> int:
    """
    num_head phải là ước của dim.
    Dùng TrainConfig.num_head nếu hợp lệ, ngược lại fallback xuống giá trị nhỏ hơn.
    """
    nh = TrainConfig.num_head
    if dim % nh == 0:
        return nh
    # tìm ước lớn nhất của dim mà ≤ TrainConfig.num_head
    for candidate in range(nh, 0, -1):
        if dim % candidate == 0:
            return candidate
    return 1


def _make_adamw(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    AdamW với param groups riêng biệt — khớp chính xác main.py + run_experiments.py.
    """
    no_decay_kws = [
        "bias", "LayerNorm.weight", "layernorm.weight",
        "norm.weight", "attn_norm.weight", "out_norm.weight",
    ]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(kw in name for kw in no_decay_kws):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay,    "weight_decay": getattr(TrainConfig, "weight_decay", 1e-4)},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


# =============================================================================
# EVALUATE
# =============================================================================

def evaluate(model: StockMovementModel, data: dict) -> Tuple[float, float]:
    if not data or len(data.get("label", [])) == 0:
        return 0.0, 0.0
    model.eval()
    ds  = StockDataset(data)
    ldr = DataLoader(
        ds, batch_size=256 if DEVICE.type == "cuda" else 64,
        shuffle=False, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
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
    return (
        float(accuracy_score(labels_all, preds_all)),
        float(matthews_corrcoef(labels_all, preds_all)),
    )


# =============================================================================
# TRAINING — 2-PHASE (khớp run_experiments.py MSGCA_FV)
# =============================================================================

def _train_epoch(model, loader, opt) -> float:
    """
    Train 1 epoch với news modality dropout — khớp run_experiments.py.
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        opt.zero_grad(set_to_none=True)

        s_n_in  = batch["s_n"].to(DEVICE)
        mask_in = batch.get("news_mask")
        q_in    = batch.get("news_quality")

        # News Modality Dropout — đọc từ TrainConfig
        if _MOD_DROPOUT > 0.0 and torch.rand(1).item() < _MOD_DROPOUT:
            s_n_in = torch.zeros_like(s_n_in)
            if mask_in is not None:
                mask_in = torch.ones_like(mask_in, dtype=torch.bool)
            q_in = None

        loss = model(
            batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
            batch["s_c"].to(DEVICE), batch["s_m"].to(DEVICE),
            s_n_in, batch["label"].to(DEVICE),
            mode="train",
            ticker_id=batch.get("ticker_id"),
            news_mask=mask_in.to(DEVICE) if mask_in is not None else None,
            news_quality=q_in.to(DEVICE) if q_in is not None else None,
        )
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
    return total_loss


def run_one_seed(
    seed:          int,
    train_hval:    dict,
    val_fixed:     dict,
    train_full:    dict,
    test:          dict,
    macro_dim:     int,
    news_dim:      int,
    dim:           int,
    lr:            float,
    max_epochs:    int   = 200,
    warmup_epochs: int   = 15,
    patience:      Optional[int] = None,
) -> Tuple[float, float, int]:
    """
    2-phase training cho một (dim, lr, seed):
      Phase 1: train_hval + val_fixed → early stopping → best_epoch
      Phase 2: train_full for best_epoch → eval test

    Khớp chính xác với run_experiments.py _run_msgca_one_seed() nhưng
    thay dim và lr bằng tham số truyền vào.

    Returns: (acc, mcc, best_epoch)
    """
    if patience is None:
        patience = RQ5_PATIENCE   # 30 — khớp run_experiments.py, không dùng TrainConfig
    set_seed(seed)

    # ── Phase 1: Tìm best_epoch ─────────────────────────────────────────────
    model, opt = _build_msgca_fv(macro_dim, news_dim, dim, lr)
    ldr = DataLoader(
        StockDataset(train_hval),
        batch_size=getattr(TrainConfig, "batch_size", 32),
        shuffle=True, drop_last=False, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(max_epochs - warmup_epochs, 1), eta_min=1e-6,
    )

    best_mcc, best_epoch, no_improve = -2.0, 1, 0
    min_active = max(warmup_epochs, 40)   # khớp run_experiments.py

    for epoch in range(max_epochs):
        _train_epoch(model, ldr, opt)
        if epoch < warmup_epochs:
            warmup.step()
        else:
            cosine.step()

        if epoch >= min_active:
            _, mcc = evaluate(model, val_fixed)
            if mcc > best_mcc:
                best_mcc   = mcc
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    # ── Phase 2: Retrain full ─────────────────────────────────────────────────
    set_seed(seed)
    model2, opt2 = _build_msgca_fv(macro_dim, news_dim, dim, lr)
    ldr2 = DataLoader(
        StockDataset(train_full),
        batch_size=getattr(TrainConfig, "batch_size", 32),
        shuffle=True, drop_last=False, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=max(best_epoch, 1), eta_min=1e-6,
    )
    for _ in range(best_epoch):
        _train_epoch(model2, ldr2, opt2)
        sched2.step()

    acc, mcc = evaluate(model2, test)
    return acc, mcc, best_epoch


def run_combo(
    dim:        int,
    lr:         float,
    splits:     dict,
    n_seeds:    int,
    max_epochs: int,
    patience:   int  = RQ5_PATIENCE,
    verbose:    bool = True,
) -> dict:
    """Chạy một combo (dim, lr) × n_seeds."""
    acc_list, mcc_list, ep_list = [], [], []
    for seed in SEEDS[:n_seeds]:
        t0 = time.time()
        acc, mcc, ep = run_one_seed(
            seed=seed,
            train_hval=splits["train_hval"],
            val_fixed=splits["val_fixed"],
            train_full=splits["train_full"],
            test=splits["test"],
            macro_dim=splits["macro_dim"],
            news_dim=splits["news_dim"],
            dim=dim, lr=lr,
            max_epochs=max_epochs,
            patience=patience,
        )
        acc_list.append(acc)
        mcc_list.append(mcc)
        ep_list.append(ep)
        if verbose:
            print(f"      seed={seed:5d}  ACC={acc:.4f}  MCC={mcc:.4f}  "
                  f"ep={ep:3d}  ({time.time() - t0:.0f}s)")

    return {
        "dim":      dim,
        "lr":       lr,
        "num_head": _safe_num_head(dim),
        "acc_mean": float(np.mean(acc_list)),
        "acc_std":  float(np.std(acc_list)),
        "mcc_mean": float(np.mean(mcc_list)),
        "mcc_std":  float(np.std(mcc_list)),
        "ep_mean":  float(np.mean(ep_list)),
        "acc_list": [float(x) for x in acc_list],
        "mcc_list": [float(x) for x in mcc_list],
        "n_seeds":  len(acc_list),
    }


# =============================================================================
# REFERENCE LOADING
# =============================================================================

def load_reference_results(results_dir: str) -> dict:
    """Load MSGCA_FV từ raw_results.json (run_experiments.py output)."""
    refs: dict = {}
    raw_path = os.path.join(results_dir, "raw_results.json")
    if not os.path.exists(raw_path):
        print(f"  [Ref] raw_results.json not found. "
              f"Run run_experiments.py first.")
        return refs
    try:
        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)
        if "MSGCA_FV" in raw:
            refs["MSGCA_FV"] = raw["MSGCA_FV"]
            print(f"  [Ref] MSGCA_FV (run_experiments.py): "
                  f"MCC={refs['MSGCA_FV']['mcc_mean']:.4f}+/-"
                  f"{refs['MSGCA_FV']['mcc_std']:.4f}  "
                  f"(dim={TrainConfig.dim}, lr={TrainConfig.learning_rate:.0e})")
    except Exception as e:
        print(f"  [Ref] Error reading raw_results.json: {e}")
    return refs


# =============================================================================
# TABLE FORMATTERS
# =============================================================================

def _lr_label(lr: float) -> str:
    """5e-05 → '5e-5', 1e-04 → '1e-4', ..."""
    s = f"{lr:.0e}"
    # normalise: remove leading zeros in exponent
    parts = s.split("e")
    exp = int(parts[1])
    return f"{parts[0]}e{exp}"


def format_heatmap(results: Dict[str, dict], dims: list, lrs: list) -> str:
    """MCC heatmap: rows=dim, cols=lr."""
    col_w = 18
    sep   = "=" * (10 + col_w * len(lrs))
    lines = [
        sep,
        "  RQ5 — MCC Heatmap  (mean over seeds)",
        "  Model: MSGCA_FV (CE loss, fair comparison)",
        f"  Fixed : dropout={FIXED_DROPOUT}",
        sep,
    ]

    # Header
    header = f"  {'dim':>6} | "
    header += "".join(f"{_lr_label(lr):>{col_w}}" for lr in lrs)
    lines.append(header)
    lines.append("  " + "-" * (8 + col_w * len(lrs)))

    # Find global best
    best_mcc = -999.0
    for r in results.values():
        if r["mcc_mean"] > best_mcc:
            best_mcc = r["mcc_mean"]

    for dim in dims:
        row = f"  {dim:>6} | "
        for lr in lrs:
            key = _combo_key(dim, lr)
            if key not in results:
                row += f"{'N/A':>{col_w}}"
            else:
                r = results[key]
                cell = f"{r['mcc_mean']:.4f}±{r['mcc_std']:.4f}"
                mark = " ✓" if abs(r["mcc_mean"] - best_mcc) < 1e-6 else "  "
                row += f"{cell + mark:>{col_w}}"
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


def format_rq5_table(
    results:  Dict[str, dict],
    dims:     list,
    lrs:      list,
    refs:     dict,
) -> str:
    sep = "=" * 130
    lines = [
        sep,
        "  RQ5 — Hidden Dim × Learning Rate Sensitivity  (MSGCA_FV, CE loss)",
        f"  News dim   : {NEWS_EMB_DIM}D (FinBERT)",
        f"  Fixed param: dropout={FIXED_DROPOUT}",
        f"  Protocol   : 2-phase (hval → best_ep, full → test), warmup=15, patience={RQ5_PATIENCE}",
        f"  Test set   : [inner_T : T_max] — identical to main.py",
        sep,
        f"  {'Key':<22} {'dim':>5} {'lr':>8} {'num_head':>10}"
        f" {'ACC':>16} {'MCC':>16}  {'dMCC(ref)':>10}  {'ep_mean':>8}",
        "-" * 130,
    ]

    ref_mcc = refs.get("MSGCA_FV", {}).get("mcc_mean")
    ref_std = refs.get("MSGCA_FV", {}).get("mcc_std")

    # Sort by MCC descending
    sorted_keys = sorted(
        [k for k in results],
        key=lambda k: results[k]["mcc_mean"],
        reverse=True,
    )
    best_key = sorted_keys[0] if sorted_keys else None

    for key in sorted_keys:
        r    = results[key]
        acc  = f"{r['acc_mean']:.4f}+/-{r['acc_std']:.4f}"
        mcc  = f"{r['mcc_mean']:.4f}+/-{r['mcc_std']:.4f}"
        if ref_mcc is not None:
            d     = r["mcc_mean"] - ref_mcc
            dmcc  = f"{'+' if d >= 0 else ''}{d:.4f}"
        else:
            dmcc = "  N/A  "
        mark = "  ←BEST" if key == best_key else ""
        lines.append(
            f"  {key:<22} {r['dim']:>5} {_lr_label(r['lr']):>8} {r['num_head']:>10}"
            f" {acc:>16} {mcc:>16}  {dmcc:>10}  {r['ep_mean']:>8.1f}{mark}"
        )

    lines.append(sep)

    # ── Reference anchor ────────────────────────────────────────────────────
    if ref_mcc is not None:
        lines.append(
            f"\n  Reference  : MSGCA_FV (run_experiments.py) — "
            f"MCC={ref_mcc:.4f}+/-{ref_std:.4f}  "
            f"(dim={TrainConfig.dim}, lr={TrainConfig.learning_rate:.0e})"
        )
        lines.append(
            f"               Note: Reference trained with same protocol but "
            f"seed=42 loaded from saved .pt — slight variation expected."
        )

    # ── Sensitivity analysis ─────────────────────────────────────────────────
    lines.append("\n  SENSITIVITY ANALYSIS:")
    lines.append(f"  {'─'*60}")

    # Effect of dim (average across all lrs)
    lines.append("  A) Effect of hidden_dim  (averaged over all lrs):")
    for dim in dims:
        dim_results = [results[k] for k in results if results[k]["dim"] == dim]
        if dim_results:
            mean_mcc = float(np.mean([r["mcc_mean"] for r in dim_results]))
            std_mcc  = float(np.std( [r["mcc_mean"] for r in dim_results]))
            lines.append(f"     dim={dim:>4d}  avg_MCC={mean_mcc:.4f}  "
                         f"spread={std_mcc:.4f}  (n={len(dim_results)} lr combos)")

    best_dim = None
    best_dim_mcc = -999.0
    for dim in dims:
        dim_results = [results[k] for k in results if results[k]["dim"] == dim]
        if dim_results:
            mean_mcc = float(np.mean([r["mcc_mean"] for r in dim_results]))
            if mean_mcc > best_dim_mcc:
                best_dim_mcc = mean_mcc
                best_dim = dim
    if best_dim is not None:
        lines.append(f"\n     → Best avg dim: {best_dim}  (avg_MCC={best_dim_mcc:.4f})")
        # Diminishing returns check
        sorted_dims = sorted(dims)
        dim_avgs = []
        for dim in sorted_dims:
            dim_results = [results[k] for k in results if results[k]["dim"] == dim]
            if dim_results:
                dim_avgs.append((dim, float(np.mean([r["mcc_mean"] for r in dim_results]))))
        if len(dim_avgs) >= 2:
            gains = [(dim_avgs[i+1][0], dim_avgs[i+1][1] - dim_avgs[i][1])
                     for i in range(len(dim_avgs)-1)]
            diminishing = all(g[1] <= gains[0][1] for g in gains[1:])
            lines.append(f"     Marginal gains: " +
                         ", ".join(f"{d}→{d2}: {g:+.4f}"
                                   for (d, _), (d2, g) in zip(dim_avgs, gains)))
            if diminishing:
                lines.append("     → Diminishing returns confirmed: larger dim does not "
                              "proportionally improve MCC.")
            else:
                lines.append("     → Non-monotonic: check overfitting at large dim.")

    # Effect of lr (average across all dims)
    lines.append("\n  B) Effect of learning_rate  (averaged over all dims):")
    for lr in lrs:
        lr_results = [results[k] for k in results if abs(results[k]["lr"] - lr) < 1e-10]
        if lr_results:
            mean_mcc = float(np.mean([r["mcc_mean"] for r in lr_results]))
            std_mcc  = float(np.std( [r["mcc_mean"] for r in lr_results]))
            mean_ep  = float(np.mean([r["ep_mean"]  for r in lr_results]))
            lines.append(f"     lr={_lr_label(lr):>6}  avg_MCC={mean_mcc:.4f}  "
                         f"spread={std_mcc:.4f}  avg_ep={mean_ep:.1f}")

    # Best combo
    if best_key:
        br = results[best_key]
        lines.append(f"\n  Best combo : dim={br['dim']}  lr={_lr_label(br['lr'])}  "
                     f"→ MCC={br['mcc_mean']:.4f}+/-{br['mcc_std']:.4f}")
        if ref_mcc is not None:
            delta = br["mcc_mean"] - ref_mcc
            direction = "BETTER" if delta > 0.005 else ("WORSE" if delta < -0.005 else "~SAME")
            lines.append(f"  vs Ref     : delta={delta:+.4f}  [{direction}]")
            if direction == "~SAME":
                lines.append("               → Current TrainConfig is already near-optimal for MSGCA_FV.")
            elif direction == "BETTER":
                lines.append(f"               → Consider updating TrainConfig: "
                             f"dim={br['dim']}, lr={_lr_label(br['lr'])}")

    # ── Interaction effect ──────────────────────────────────────────────────
    if len(dims) >= 2 and len(lrs) >= 2:
        lines.append("\n  C) Interaction (dim × lr) — top-3 combos:")
        for i, key in enumerate(sorted_keys[:3], 1):
            r = results[key]
            lines.append(f"     #{i}: dim={r['dim']}  lr={_lr_label(r['lr'])}  "
                         f"MCC={r['mcc_mean']:.4f}+/-{r['mcc_std']:.4f}  "
                         f"ep≈{r['ep_mean']:.0f}")

    # ── LaTeX ────────────────────────────────────────────────────────────────
    lines.append("\n  [LaTeX — MCC table, sorted by MCC desc]")
    lines.append("  \\begin{tabular}{cccccc}")
    lines.append("  \\hline")
    lines.append("  Key & dim & lr & MCC & ACC & $\\Delta$MCC(ref) \\\\")
    lines.append("  \\hline")
    for key in sorted_keys:
        r = results[key]
        dmcc_str = (f"{r['mcc_mean'] - ref_mcc:+.4f}"
                    if ref_mcc is not None else "--")
        lines.append(
            f"  {key} & {r['dim']} & ${_lr_label(r['lr'])}$ "
            f"& {r['mcc_mean']:.4f}$\\pm${r['mcc_std']:.4f} "
            f"& {r['acc_mean']:.4f}$\\pm${r['acc_std']:.4f} "
            f"& {dmcc_str} \\\\"
        )
    lines.append("  \\hline")
    lines.append("  \\end{tabular}")

    return "\n".join(lines)


def _combo_key(dim: int, lr: float) -> str:
    return f"d{dim}_lr{_lr_label(lr)}"


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="RQ5: Hidden dim × learning rate sensitivity for MSGCA_FV"
    )
    ap.add_argument("--pkl",           default=None,
                    help="Path to unified_dataset_test.pkl")
    ap.add_argument("--dims",          nargs="+", type=int,
                    default=DEFAULT_DIMS,
                    help="Hidden dimensions to test")
    ap.add_argument("--lrs",           nargs="+", type=float,
                    default=DEFAULT_LRS,
                    help="Learning rates to test")
    ap.add_argument("--n-seeds",       type=int, default=5)
    ap.add_argument("--epochs",        type=int, default=200,
                    help="Max epochs per phase (khớp run_experiments.py)")
    ap.add_argument("--patience",      type=int, default=RQ5_PATIENCE,
                    help=f"Early stopping patience (default={RQ5_PATIENCE}, "
                         f"khớp run_experiments.py — KHÔNG dùng TrainConfig.early_stop_patience)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Load checkpoint nếu có, bỏ qua combo đã chạy")
    ap.add_argument("--verbose",       action="store_true", default=True)
    args = ap.parse_args()

    pkl_path = args.pkl or os.path.join(
        GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl"
    )
    if not os.path.exists(pkl_path):
        print(f"Dataset not found: {pkl_path}"); sys.exit(1)

    dims    = sorted(set(args.dims))
    lrs     = sorted(set(args.lrs))
    tickers = GlobalConfig.TICKERS
    n_combos = len(dims) * len(lrs)

    print(f"\n{'='*70}")
    print(f"  RQ5 — Hidden Dim × LR Sensitivity  (MSGCA_FV, CE loss)")
    print(f"  Device  : {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print(f"  Grid    : dim={dims} × lr={[_lr_label(l) for l in lrs]}")
    print(f"  Combos  : {n_combos}  ×  n_seeds={args.n_seeds}  "
          f"= {n_combos * args.n_seeds} training runs")
    print(f"  Epochs  : max={args.epochs}  warmup=15  patience={args.patience}")
    print(f"  Note    : patience={args.patience} (run_experiments.py protocol) "
          f"≠ TrainConfig.early_stop_patience={TrainConfig.early_stop_patience}")
    print(f"  ModDrop : {_MOD_DROPOUT:.0%}  (same as main.py)")
    print(f"  Fixed   : dropout={FIXED_DROPOUT}  loss=CE  class_weights=None")
    print(f"{'='*70}")

    print("\nLoading reference results...")
    refs = load_reference_results(RESULTS_DIR)

    print("\nLoading data splits...")
    splits = load_splits(pkl_path, tickers)

    # ── Checkpoint: nếu đã có raw.json thì load ───────────────────────────────
    raw_path = os.path.join(RESULTS_DIR, "rq5_raw.json")
    all_results: Dict[str, dict] = {}
    if args.skip_existing and os.path.exists(raw_path):
        try:
            with open(raw_path, encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"  Loaded {len(all_results)} existing results from {raw_path}")
        except Exception as e:
            print(f"  [WARN] Could not load checkpoint: {e}")
            all_results = {}

    # ── Main grid loop ─────────────────────────────────────────────────────────
    total_t0 = time.time()
    done = 0

    for dim in dims:
        for lr in lrs:
            key = _combo_key(dim, lr)
            done += 1

            if args.skip_existing and key in all_results:
                r = all_results[key]
                print(f"\n  [{done:02d}/{n_combos}] {key}  "
                      f"[SKIP — already done: MCC={r['mcc_mean']:.4f}]")
                continue

            nh = _safe_num_head(dim)
            print(f"\n  [{done:02d}/{n_combos}] dim={dim}  lr={_lr_label(lr)}  "
                  f"num_head={nh}")
            t0 = time.time()
            result = run_combo(
                dim=dim, lr=lr,
                splits=splits,
                n_seeds=args.n_seeds,
                max_epochs=args.epochs,
                patience=args.patience,
                verbose=args.verbose,
            )
            elapsed = time.time() - t0
            all_results[key] = result
            print(f"  → ACC={result['acc_mean']:.4f}+/-{result['acc_std']:.4f}  "
                  f"MCC={result['mcc_mean']:.4f}+/-{result['mcc_std']:.4f}  "
                  f"ep≈{result['ep_mean']:.0f}  ({elapsed/60:.1f} min)")

            # Save checkpoint sau mỗi combo để resume được
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2)

    total_elapsed = time.time() - total_t0
    print(f"\nTotal: {total_elapsed/60:.1f} min\n")

    if not all_results:
        print("No results to format."); return

    # ── Format và save ─────────────────────────────────────────────────────────
    table  = format_rq5_table(all_results, dims, lrs, refs)
    heatmap = format_heatmap(all_results, dims, lrs)

    print(heatmap)
    print()
    print(table)

    table_path   = os.path.join(RESULTS_DIR, "rq5_table.txt")
    heatmap_path = os.path.join(RESULTS_DIR, "rq5_heatmap.txt")
    with open(table_path,   "w", encoding="utf-8") as f:
        f.write(table)
    with open(heatmap_path, "w", encoding="utf-8") as f:
        f.write(heatmap)
    # raw_path đã được save trong loop
    print(f"\nSaved → {table_path}")
    print(f"         {heatmap_path}")
    print(f"         {raw_path}")


if __name__ == "__main__":
    main()