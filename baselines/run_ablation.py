# baselines/run_ablation.py
"""
End-to-end ablation study cho MSGCA.

Mục tiêu: Trả lời MINH BACH 3 câu hoi:
  Q1: News embeddings co dang hurt model khong?
      -> Test MSGCA_no_news (s_n = zeros)
  Q2: Training protocol co phai la nguyen nhan chinh khong?
      -> Test MSGCA_fixed_val (fixed val split + warmup + 200 epochs)
  Q3: Cac modalities dong gop gi?
      -> Test MSGCA_no_macro (s_m = zeros)

Tat ca variants duoc evaluate tren cung outer test [inner_T:global_T_max].
Protocol khi co the: fixed val split (80%/20% of inner_T) thay vi walk-forward.

Usage:
    python baselines/run_ablation.py
    python baselines/run_ablation.py --n-seeds 3 --variants no_news no_macro fixed_val
    python baselines/run_ablation.py --variants all

Outputs:
    baselines/results/ablation_table.txt
    baselines/results/ablation_raw.json
"""

from typing import Dict, List, Optional, Tuple

import argparse
import json
import os
import sys
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import StockMovementModel
from src.data_loader import data_prepare, N_TICKERS
from configs.config import TrainConfig, GlobalConfig

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS       = [42, 123, 256, 512, 1024]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

ALL_VARIANTS = [
    "baseline_wf",    # MSGCA with walk-forward (reference: matches run_msgca_seeds.py)
    "no_news",        # Q1: s_n = zeros -> does removing news HELP or HURT?
    "no_macro",       # Q3: s_m = zeros -> how much does macro contribute?
    "fixed_val",      # Q2: fixed val split + warmup + 200 epochs (paper protocol)
    "no_news_fixed",  # Q1+Q2: no news + fixed val (combined)
]

VARIANT_DESCRIPTIONS = {
    "baseline_wf":   "MSGCA (walk-forward, 80 epochs) -- reference",
    "no_news":       "No news (s_n=zeros, walk-forward)            -- tests if news hurts",
    "no_macro":      "No macro (s_m=zeros, walk-forward)           -- tests macro contribution",
    "fixed_val":     "Fixed val split + warmup + 200 epochs        -- paper protocol",
    "no_news_fixed": "No news + fixed val + warmup + 200 epochs    -- combined",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class StockDataset(Dataset):
    _KEYS = ["s_o", "s_h", "s_c", "s_m", "s_n", "news_mask", "label", "ticker_id"]
    def __init__(self, d: dict):
        self.d    = d
        self.keys = [k for k in self._KEYS if k in d]
    def __len__(self): return len(self.d["label"])
    def __getitem__(self, i): return {k: self.d[k][i] for k in self.keys}


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
    if not valid: return {}
    m = {}
    for key in valid[0].keys():
        parts = [d[key] for d in valid if key in d]
        if parts and isinstance(parts[0], torch.Tensor):
            m[key] = torch.cat(parts, dim=0)
    if shuffle and "label" in m:
        idx = torch.randperm(len(m["label"]))
        for k in m: m[k] = m[k][idx]
    return m


def load_splits(pkl_path: str, tickers: list) -> dict:
    """
    Load tat ca splits can thiet cho ablation.

    Returns:
      inner_T, global_T_max
      train_wf   : [0 : inner_T]          train cho walk-forward variant
      test       : [inner_T : global_T_max]
      train_hval : [0 : 0.8*inner_T]      train cho fixed-val variant
      val_fixed  : [0.8*inner_T : inner_T] val co dinh (20% of inner)
    """
    dp = data_prepare(pkl_path, include_ticker_id=True)
    valid_T      = [dp.get_max_T(t) for t in tickers if dp.get_max_T(t) > 0]
    global_T_max = min(valid_T)
    inner_T      = int(global_T_max * 0.85)
    hval_split   = int(inner_T * 0.80)   # 80% inner for train, 20% for fixed val

    print(f"  global_T_max={global_T_max}  inner_T={inner_T}  hval_split={hval_split}")
    print(f"  test=[{inner_T}:{global_T_max}] ({global_T_max-inner_T} timesteps)")
    print(f"  fixed val=[{hval_split}:{inner_T}] ({inner_T-hval_split} timesteps per ticker)")

    tr_wf_list, tr_hv_list, va_hv_list, te_list = [], [], [], []
    macro_dim = news_dim = None

    for t in tickers:
        if dp.get_max_T(t) == 0:
            continue
        # Full inner for walk-forward train
        tr_wf, _, te = dp.prepare_data(
            t, train_end=inner_T, val_end=inner_T, test_end=global_T_max
        )
        # Hval splits for fixed-val
        tr_hv, va_hv, _ = dp.prepare_data(
            t, train_end=hval_split, val_end=inner_T, test_end=inner_T
        )

        if macro_dim is None and tr_wf and len(tr_wf.get("label", [])) > 0:
            macro_dim = tr_wf["s_m"].shape[-1]
            news_dim  = tr_wf["s_n"].shape[-1]

        if tr_wf and len(tr_wf.get("label", [])) > 0: tr_wf_list.append(tr_wf)
        if tr_hv and len(tr_hv.get("label", [])) > 0: tr_hv_list.append(tr_hv)
        if va_hv and len(va_hv.get("label", [])) > 0: va_hv_list.append(va_hv)
        if te   and len(te.get("label",   [])) > 0:   te_list.append(te)

    print(f"  macro_dim={macro_dim}  news_dim={news_dim}")

    return {
        "inner_T": inner_T, "global_T_max": global_T_max, "hval_split": hval_split,
        "train_wf":    merge(tr_wf_list, shuffle=True),
        "train_hval":  merge(tr_hv_list, shuffle=True),
        "val_fixed":   merge(va_hv_list, shuffle=False),
        "test":        merge(te_list,    shuffle=False),
        "macro_dim": macro_dim, "news_dim": news_dim,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Strong inverse-freq weights (khop voi run_msgca_seeds.py)."""
    lbl   = labels.numpy()
    cnts  = np.bincount(lbl, minlength=3).astype(float)
    N, n  = cnts.sum(), 3
    w     = np.clip((N / (n * cnts + 1e-8)) ** 1.5, 0.5, 4.0)
    return torch.tensor(w / w.sum() * n, dtype=torch.float32)


def build_model(macro_dim: int, news_dim: int, use_focal: bool = True,
                focal_gamma: float = 2.0, cw: torch.Tensor = None) -> StockMovementModel:
    return StockMovementModel(
        price_dim=1, macro_dim=macro_dim, news_dim=news_dim,
        dim=TrainConfig.dim, input_dim=TrainConfig.window_size,
        output_dim=3, num_head=TrainConfig.num_head, dropout=0.1,
        class_weights=cw, use_focal_loss=use_focal, focal_gamma=focal_gamma,
        device=DEVICE, n_tickers=N_TICKERS, use_ticker_emb=True,
    ).to(DEVICE)


def evaluate(model: StockMovementModel, data: dict,
             zero_news: bool = False, zero_macro: bool = False) -> Tuple[float, float]:
    if not data or len(data.get("label", [])) == 0:
        return 0.0, 0.0
    model.eval()
    ds  = StockDataset(data)
    ldr = DataLoader(ds, batch_size=64, shuffle=False)
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in ldr:
            s_n = batch["s_n"].to(DEVICE)
            s_m = batch["s_m"].to(DEVICE)
            if zero_news:
                s_n = torch.zeros_like(s_n)
            if zero_macro:
                s_m = torch.zeros_like(s_m)
            _, _, preds = model(
                batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                batch["s_c"].to(DEVICE), s_m, s_n,
                batch["label"].to(DEVICE),
                mode="test", return_preds=True,
                ticker_id=batch.get("ticker_id"),
                news_mask=batch.get("news_mask"),
            )
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(batch["label"].numpy())
    if len(set(labels_all)) < 2:
        return float(accuracy_score(labels_all, preds_all)), 0.0
    return (
        float(accuracy_score(labels_all, preds_all)),
        float(matthews_corrcoef(labels_all, preds_all)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD EPOCH SELECTION (subset of run_msgca_seeds.py logic)
# ─────────────────────────────────────────────────────────────────────────────

def find_best_epoch_wf(
    dp, tickers: list, inner_T: int, global_T_max: int,
    macro_dim: int, news_dim: int,
    zero_news: bool = False, zero_macro: bool = False,
    wf_folds: int = 3, max_epochs: int = 80, min_val_size: int = 200,
    use_focal: bool = True, focal_gamma: float = 2.0, seed: int = 42,
) -> int:
    """Walk-forward inner validation để chon best_epoch."""
    chunk_size = inner_T // (wf_folds + 1)
    all_folds_mcc = []

    for k in range(wf_folds):
        train_end = (k + 2) * chunk_size
        val_end   = (k + 3) * chunk_size if k < wf_folds - 1 else inner_T

        list_tr, list_va = [], []
        for t in tickers:
            tr, va, _ = dp.prepare_data(
                t, train_end=train_end, val_end=val_end, test_end=val_end
            )
            if tr and len(tr.get("label", [])) > 0:
                list_tr.append(tr)
                list_va.append(va)

        fold_tr = merge(list_tr, shuffle=True)
        fold_va = merge(list_va, shuffle=False)
        n_val   = len(fold_va.get("label", [])) if fold_va else 0
        if n_val < min_val_size:
            continue

        cw    = compute_class_weights(fold_tr["label"]).to(DEVICE)
        model = build_model(macro_dim, news_dim, use_focal, focal_gamma, cw)
        ds    = StockDataset(fold_tr)
        ldr   = DataLoader(ds, batch_size=32, shuffle=True, drop_last=False)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
        sch   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)

        fold_mcc = []
        for epoch in range(max_epochs):
            model.train()
            for batch in ldr:
                opt.zero_grad()
                s_n = batch["s_n"].to(DEVICE)
                s_m = batch["s_m"].to(DEVICE)
                if zero_news:  s_n = torch.zeros_like(s_n)
                if zero_macro: s_m = torch.zeros_like(s_m)
                loss = model(
                    batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                    batch["s_c"].to(DEVICE), s_m, s_n,
                    batch["label"].to(DEVICE), mode="train",
                    ticker_id=batch.get("ticker_id"),
                    news_mask=batch.get("news_mask"),
                )
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sch.step()
            _, mcc = evaluate(model, fold_va, zero_news, zero_macro)
            fold_mcc.append(mcc)

        all_folds_mcc.append(fold_mcc)

    if not all_folds_mcc:
        return max_epochs // 2
    avg = np.mean(all_folds_mcc, axis=0)
    smoothed = [np.mean(avg[max(0, i-1):min(max_epochs, i+2)]) for i in range(max_epochs)]
    return int(np.argmax(smoothed)) + 1


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE SEED — WALK-FORWARD VARIANT
# ─────────────────────────────────────────────────────────────────────────────

def run_seed_walkforward(
    seed: int, train_data: dict, test_data: dict,
    best_epoch: int, macro_dim: int, news_dim: int,
    zero_news: bool, zero_macro: bool,
    use_focal: bool = True, focal_gamma: float = 2.0,
) -> Tuple[float, float]:
    set_seed(seed)
    cw    = compute_class_weights(train_data["label"]).to(DEVICE)
    model = build_model(macro_dim, news_dim, use_focal, focal_gamma, cw)
    ds    = StockDataset(train_data)
    ldr   = DataLoader(ds, batch_size=32, shuffle=True, drop_last=False)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=30, T_mult=2)

    for epoch in range(best_epoch):
        model.train()
        for batch in ldr:
            opt.zero_grad()
            s_n = batch["s_n"].to(DEVICE)
            s_m = batch["s_m"].to(DEVICE)
            if zero_news:  s_n = torch.zeros_like(s_n)
            if zero_macro: s_m = torch.zeros_like(s_m)
            loss = model(
                batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                batch["s_c"].to(DEVICE), s_m, s_n,
                batch["label"].to(DEVICE), mode="train",
                ticker_id=batch.get("ticker_id"),
                news_mask=batch.get("news_mask"),
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

    return evaluate(model, test_data, zero_news, zero_macro)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE SEED — FIXED VAL VARIANT (paper protocol)
# ─────────────────────────────────────────────────────────────────────────────

def run_seed_fixed_val(
    seed: int, train_data: dict, val_data: dict, test_data: dict,
    macro_dim: int, news_dim: int,
    zero_news: bool = False, zero_macro: bool = False,
    max_epochs: int = 200, patience: int = 30, warmup_epochs: int = 10,
    use_focal: bool = True, focal_gamma: float = 2.0,
) -> Tuple[float, float]:
    """
    Fixed val split + learning rate warmup + early stopping.
    Khop hon voi paper goc (MSGCA, Section V.A.3).
    """
    set_seed(seed)
    cw    = compute_class_weights(train_data["label"]).to(DEVICE)
    model = build_model(macro_dim, news_dim, use_focal, focal_gamma, cw)
    ds    = StockDataset(train_data)
    ldr   = DataLoader(ds, batch_size=32, shuffle=True, drop_last=False)

    opt   = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Warmup: lr starts at 1e-5, reaches 1e-4 after warmup_epochs (paper practice)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max_epochs - warmup_epochs, eta_min=1e-6
    )

    best_mcc, best_state, no_improve = -2.0, None, 0

    for epoch in range(max_epochs):
        model.train()
        for batch in ldr:
            opt.zero_grad()
            s_n = batch["s_n"].to(DEVICE)
            s_m = batch["s_m"].to(DEVICE)
            if zero_news:  s_n = torch.zeros_like(s_n)
            if zero_macro: s_m = torch.zeros_like(s_m)
            loss = model(
                batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                batch["s_c"].to(DEVICE), s_m, s_n,
                batch["label"].to(DEVICE), mode="train",
                ticker_id=batch.get("ticker_id"),
                news_mask=batch.get("news_mask"),
            )
            if torch.isfinite(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        # Schedule
        if epoch < warmup_epochs:
            warmup.step()
        else:
            cosine.step()

        # Early stopping (start after warmup)
        if epoch >= warmup_epochs:
            _, mcc = evaluate(model, val_data, zero_news, zero_macro)
            if mcc > best_mcc:
                best_mcc   = mcc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if best_state:
        model.load_state_dict(best_state)
    return evaluate(model, test_data, zero_news, zero_macro)


# ─────────────────────────────────────────────────────────────────────────────
# RUN ONE VARIANT — 5 SEEDS
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(
    variant: str,
    splits: dict,
    tickers: list,
    pkl_path: str,
    n_seeds: int = 5,
    wf_max_epochs: int = 80,
    fv_max_epochs: int = 200,
    verbose: bool = True,
) -> dict:
    """
    Run one ablation variant across n_seeds.

    Walk-forward variants: per-seed epoch selection (slow but fair to WF).
    Fixed-val variants:    shared protocol, early stopping on val.
    """
    zero_news  = "no_news"  in variant
    zero_macro = "no_macro" in variant
    use_fixval = "fixed"    in variant

    macro_dim = splits["macro_dim"]
    news_dim  = splits["news_dim"]

    train_wf   = splits["train_wf"]
    train_hval = splits["train_hval"]
    val_fixed  = splits["val_fixed"]
    test       = splits["test"]

    acc_list, mcc_list = [], []
    modality_str = (
        "no news" if zero_news and not zero_macro else
        "no macro" if zero_macro and not zero_news else
        "no news+macro" if zero_news and zero_macro else
        "all modalities"
    )
    protocol_str = "fixed_val+warmup+200ep" if use_fixval else "walk-forward+80ep"
    print(f"\n  Variant: {variant}")
    print(f"  Modalities: {modality_str}  |  Protocol: {protocol_str}")

    if not use_fixval:
        # Walk-forward: per-seed epoch selection (same as run_msgca_seeds.py)
        dp = data_prepare(pkl_path, include_ticker_id=True)
        for seed in SEEDS[:n_seeds]:
            set_seed(seed)
            best_ep = find_best_epoch_wf(
                dp=dp, tickers=tickers,
                inner_T=splits["inner_T"], global_T_max=splits["global_T_max"],
                macro_dim=macro_dim, news_dim=news_dim,
                zero_news=zero_news, zero_macro=zero_macro,
                wf_folds=3, max_epochs=wf_max_epochs, min_val_size=200,
            )
            acc, mcc = run_seed_walkforward(
                seed=seed,
                train_data=train_wf, test_data=test,
                best_epoch=best_ep,
                macro_dim=macro_dim, news_dim=news_dim,
                zero_news=zero_news, zero_macro=zero_macro,
            )
            acc_list.append(acc)
            mcc_list.append(mcc)
            if verbose:
                print(f"    Seed {seed}: ep={best_ep:3d}  ACC={acc:.4f}  MCC={mcc:.4f}")
    else:
        # Fixed val: early stopping on fixed val split (paper protocol)
        for seed in SEEDS[:n_seeds]:
            acc, mcc = run_seed_fixed_val(
                seed=seed,
                train_data=train_hval,   # [0 : 0.8*inner_T]
                val_data=val_fixed,      # [0.8*inner_T : inner_T]
                test_data=test,
                macro_dim=macro_dim, news_dim=news_dim,
                zero_news=zero_news, zero_macro=zero_macro,
                max_epochs=fv_max_epochs, patience=30, warmup_epochs=10,
            )
            acc_list.append(acc)
            mcc_list.append(mcc)
            if verbose:
                print(f"    Seed {seed}:            ACC={acc:.4f}  MCC={mcc:.4f}")

    result = {
        "acc_mean": float(np.mean(acc_list)),
        "acc_std":  float(np.std(acc_list)),
        "mcc_mean": float(np.mean(mcc_list)),
        "mcc_std":  float(np.std(mcc_list)),
        "acc_list": acc_list,
        "mcc_list": mcc_list,
        "n_seeds":  len(acc_list),
    }
    print(f"  -> ACC={result['acc_mean']:.4f}+/-{result['acc_std']:.4f}  "
          f"MCC={result['mcc_mean']:.4f}+/-{result['mcc_std']:.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def format_ablation_table(results: dict) -> str:
    lines = []
    sep = "=" * 100
    lines += [sep, "  ABLATION STUDY -- MSGCA Variants  (Mean +/- Std, outer test)", sep]
    lines.append(
        f"{'Variant':<20} {'Description':<46} {'ACC':>14} {'MCC':>14}  vs baseline"
    )
    lines.append("-" * 100)

    # Reference MCC
    ref_mcc = results.get("baseline_wf", {}).get("mcc_mean", 0.0)

    for v in ALL_VARIANTS:
        if v not in results:
            continue
        r    = results[v]
        desc = VARIANT_DESCRIPTIONS.get(v, "")[:45]
        acc  = f"{r['acc_mean']:.4f}+/-{r['acc_std']:.4f}"
        mcc  = f"{r['mcc_mean']:.4f}+/-{r['mcc_std']:.4f}"

        # Delta vs baseline
        if v == "baseline_wf":
            delta = "(reference)"
        else:
            d = r["mcc_mean"] - ref_mcc
            delta = f"+{d:.4f}" if d >= 0 else f"{d:.4f}"

        lines.append(f"{v:<20} {desc:<46} {acc:>14} {mcc:>14}  {delta}")

    lines.append(sep)

    # Interpretation
    lines.append("\nINTERPRETATION:")
    if "no_news" in results and "baseline_wf" in results:
        delta_mcc = results["no_news"]["mcc_mean"] - ref_mcc
        if delta_mcc > 0.005:
            lines.append(f"  Q1 NEWS:  Removing news IMPROVES MCC by {delta_mcc:+.4f}")
            lines.append("            -> News embeddings are hurting the model.")
            lines.append("            -> Consider: drop news channel, or improve embedding quality.")
        elif delta_mcc < -0.005:
            lines.append(f"  Q1 NEWS:  Removing news HURTS MCC by {delta_mcc:.4f}")
            lines.append("            -> News embeddings ARE contributing positively.")
        else:
            lines.append(f"  Q1 NEWS:  News has negligible effect (delta={delta_mcc:+.4f})")
            lines.append("            -> News signal is weak but not harmful.")

    if "no_macro" in results and "baseline_wf" in results:
        delta_mcc = results["no_macro"]["mcc_mean"] - ref_mcc
        if delta_mcc < -0.005:
            lines.append(f"  Q3 MACRO: Removing macro HURTS MCC by {delta_mcc:.4f}")
            lines.append("            -> Macro features are contributing significantly.")
        elif delta_mcc > 0.005:
            lines.append(f"  Q3 MACRO: Removing macro IMPROVES MCC by {delta_mcc:+.4f}")
            lines.append("            -> Macro features may be noisy in cross-attention fusion.")
        else:
            lines.append(f"  Q3 MACRO: Macro has small effect (delta={delta_mcc:+.4f})")

    if "fixed_val" in results and "baseline_wf" in results:
        delta_mcc = results["fixed_val"]["mcc_mean"] - ref_mcc
        delta_std = results["fixed_val"]["mcc_std"] - results["baseline_wf"]["mcc_std"]
        lines.append(f"  Q2 PROTO: Fixed val protocol delta_MCC={delta_mcc:+.4f}  "
                     f"delta_std={delta_std:+.4f}")
        if delta_mcc > 0.005:
            lines.append("            -> Paper protocol significantly better. Training was the bottleneck.")
        if delta_std < -0.003:
            lines.append("            -> Variance reduced. More stable training.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="MSGCA ablation study")
    ap.add_argument("--pkl",          default=None)
    ap.add_argument("--variants",     nargs="+", default=ALL_VARIANTS,
                    choices=ALL_VARIANTS + ["all"],
                    help="Variants to run. Default: all. E.g. --variants no_news fixed_val")
    ap.add_argument("--n-seeds",      type=int, default=5)
    ap.add_argument("--wf-epochs",    type=int, default=80,
                    help="Max epochs for walk-forward variants")
    ap.add_argument("--fv-epochs",    type=int, default=200,
                    help="Max epochs for fixed-val variants")
    ap.add_argument("--load-wf-from", default="output",
                    help="Load baseline_wf results from msgca_results_seed*.json if available")
    ap.add_argument("--verbose",      action="store_true")
    args = ap.parse_args()

    pkl_path = args.pkl or os.path.join(
        GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl"
    )
    if not os.path.exists(pkl_path):
        print(f"Dataset not found: {pkl_path}")
        sys.exit(1)

    variants = ALL_VARIANTS if "all" in args.variants else args.variants
    tickers  = GlobalConfig.TICKERS

    print(f"\n{'='*60}")
    print(f"ABLATION STUDY -- MSGCA")
    print(f"Device  : {DEVICE}")
    print(f"Variants: {variants}")
    print(f"Seeds   : {SEEDS[:args.n_seeds]}")
    print(f"{'='*60}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    splits = load_splits(pkl_path, tickers)

    all_results: Dict[str, dict] = {}

    # ── Try loading baseline_wf from existing JSON (skip if already run) ──────
    if "baseline_wf" in variants:
        acc_list, mcc_list = [], []
        for seed in SEEDS[:args.n_seeds]:
            path = os.path.join(args.load_wf_from, f"msgca_results_seed{seed}.json")
            if os.path.exists(path):
                with open(path) as f:
                    r = json.load(f)
                acc_list.append(r.get("test_acc", 0.0))
                mcc_list.append(r.get("test_mcc", 0.0))

        if len(acc_list) == args.n_seeds:
            all_results["baseline_wf"] = {
                "acc_mean": float(np.mean(acc_list)),
                "acc_std":  float(np.std(acc_list)),
                "mcc_mean": float(np.mean(mcc_list)),
                "mcc_std":  float(np.std(mcc_list)),
                "acc_list": acc_list, "mcc_list": mcc_list, "n_seeds": len(acc_list),
            }
            print(f"\nbaseline_wf: loaded from {args.load_wf_from}")
            print(f"  -> ACC={all_results['baseline_wf']['acc_mean']:.4f}  "
                  f"MCC={all_results['baseline_wf']['mcc_mean']:.4f}")
            variants = [v for v in variants if v != "baseline_wf"]

    # ── Run remaining variants ─────────────────────────────────────────────────
    total_t0 = time.time()
    for variant in variants:
        t0 = time.time()
        result = run_variant(
            variant=variant,
            splits=splits,
            tickers=tickers,
            pkl_path=pkl_path,
            n_seeds=args.n_seeds,
            wf_max_epochs=args.wf_epochs,
            fv_max_epochs=args.fv_epochs,
            verbose=args.verbose,
        )
        all_results[variant] = result
        print(f"  ({(time.time()-t0)/60:.1f} min)")

    # ── Output ────────────────────────────────────────────────────────────────
    print(f"\nTotal: {(time.time()-total_t0)/60:.1f} min")
    print()
    table = format_ablation_table(all_results)
    print(table)

    table_path = os.path.join(RESULTS_DIR, "ablation_table.txt")
    raw_path   = os.path.join(RESULTS_DIR, "ablation_raw.json")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {table_path}")
    print(f"         {raw_path}")


if __name__ == "__main__":
    main()