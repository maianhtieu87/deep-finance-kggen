# -*- coding: utf-8 -*-
# baselines/run_ablation.py — V3 (baseline_wf synced với main.py standard split)
"""
End-to-end ablation study cho MSGCA.

CHANGES vs V2:
  V3 — FIX baseline_wf detection:
    V2 load baseline từ output/msgca_results_seed{seed}.json — các file này
    từ protocol WF cũ (80 epochs) đã bị xóa/deprecated. main.py hiện tại
    KHÔNG save JSON, chỉ save .pt:
      output/best_model_label=rolling_price=vol_adjusted_tid_seed{seed}_standard.pt

    V3 thay bằng:
      1. AUTO-DETECT: glob tìm best_model_*_seed{seed}_standard.pt trong output_dir
         → load weights → evaluate trên cùng test set
         → đảm bảo kết quả 100% nhất quán với main.py (cùng model, cùng test set)
      2. RETRAIN fallback: nếu không tìm thấy .pt, retrain với fixed_val+200ep
         → cùng protocol với main.py

    Chi tiết test set alignment với main.py:
      main.py:   train=[0 : int(T*0.70)]  val=[0.70T : 0.85T]  test=[0.85T : T]
      ablation:  inner_T = int(T*0.85)    → test=[inner_T : T]
      → CÙNG test set, chỉ khác train/val split (ablation dùng hval_split=0.68*inner_T)

  V3 giữ nguyên từ V2:
    - run_seed_fixed_val() — train loop với warmup + early stopping
    - run_seed_walkforward() — giữ cho legacy support
    - Ablation methodology: retrain from scratch với zeros (đúng)
    - zero_news/zero_macro applied BOTH train AND eval (đúng)

  Methodology note — Tại sao retrain từ đầu với zeros là đúng:
    "Train 3 module rồi zero 1 lúc test" = SAI vì:
      H_a = W_a(zeros) + bias_a  → bias_a ≠ 0 (đã học để encode news/macro features)
      H_gated = bias_a * sigmoid(W_b(primary))  → inject learned noise vào primary
    Nhưng khi RETRAIN với zeros:
      Training đẩy gate suppress → H_gated → 0, bias_a → near-zero
      Model converge tới solution tối ưu không có signal đó
    Architecture safety:
      StableGatedCrossAttention: output = norm(primary + dropout(H_gated))
        → residual bảo toàn primary kể cả khi gate bất thường
      v_i skip: cat([H_idm, v_i, v_t_seq]) → price LUÔN có mặt bất kể fusion stage
      → Hai lớp bảo vệ, ablation hoàn toàn an toàn

Usage:
    python baselines/run_ablation.py
    python baselines/run_ablation.py --variants no_news no_macro
    python baselines/run_ablation.py --variants baseline_wf --load-from output/
    python baselines/run_ablation.py --n-seeds 3 --fv-epochs 200 --verbose

Outputs:
    baselines/results/ablation_table.txt
    baselines/results/ablation_raw.json
"""

from typing import Dict, List, Optional, Tuple

import argparse
import glob
import json
import os
import platform
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
from src.data_loader import data_prepare, N_TICKERS, NEWS_EMB_DIM
from configs.config import TrainConfig, GlobalConfig

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_IS_WINDOWS = platform.system() == "Windows"

# TF32 cho Ada/Ampere GPUs — ~10% faster, matches main.py
if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")

SEEDS       = [42, 123, 256, 512, 1024]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Shared training control — đọc từ TrainConfig (configs/config.py) ──────────
# Để vô hiệu hóa early stopping (train đủ fv_epochs): đặt early_stop_patience = 9999
# Để tắt modality dropout: đặt news_modality_dropout = 0.0
_PATIENCE    = TrainConfig.early_stop_patience    # default 30
_MOD_DROPOUT = TrainConfig.news_modality_dropout  # default 0.30


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

ALL_VARIANTS = [
    "baseline_wf",    # Reference A: MSGCA_Best (FocalLoss+weights) — loaded from main.py .pt
    "baseline_fv",    # Reference B: MSGCA_FV (CE, fair comparison) — loaded from run_experiments.py JSON
    "no_news",        # Q1: s_n = zeros train+test → retrain without news
    "no_macro",       # Q3: s_m = zeros train+test → retrain without macro
    "fixed_val",      # Q2: fixed val + warmup + 200ep (protocol sanity check)
    "no_news_fixed",  # Q1+Q2: combined
]

VARIANT_DESCRIPTIONS = {
    "baseline_wf":   "MSGCA_Best (FocalLoss+CW) — load main.py .pt    [Ref-A]",
    "baseline_fv":   "MSGCA_FV (CE, fair) — load run_experiments JSON  [Ref-B]",
    "no_news":       "No news: retrain from scratch, s_n=zeros train+test  [Q1]",
    "no_macro":      "No macro: retrain from scratch, s_m=zeros train+test [Q3]",
    "fixed_val":     "Fixed val + warmup + 200 epochs                      [Q2]",
    "no_news_fixed": "No news + fixed val + warmup + 200 epochs         [Q1+Q2]",
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

class StockDataset(Dataset):
    _KEYS = ["s_o", "s_h", "s_c", "s_m", "s_n", "news_mask", "label",
             "ticker_id", "news_quality"]

    def __init__(self, d: dict):
        self.d    = d
        self.keys = [k for k in self._KEYS if k in d]  # guard: skip absent keys

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
    m = {}
    for key in valid[0].keys():
        parts = [d[key] for d in valid if key in d]
        if parts and isinstance(parts[0], torch.Tensor):
            m[key] = torch.cat(parts, dim=0)
    if shuffle and "label" in m:
        idx = torch.randperm(len(m["label"]))
        for k in m:
            m[k] = m[k][idx]
    return m


def load_splits(pkl_path: str, tickers: list) -> dict:
    """
    Load và split data.

    Test set alignment với main.py (QUAN TRỌNG):
      main.py:  val_end   = int(T_max * 0.85)  →  test = [val_end : T_max]
      ablation: inner_T   = int(T_max * 0.85)  →  test = [inner_T : T_max]
      → CÙNG test set index.

    Train/val split của ablation variants:
      baseline_wf : train=[0:inner_T] (dùng toàn bộ, load .pt từ main.py)
      fixed_val   : train=[0:hval_split], val=[hval_split:inner_T]
      no_news/macro: dùng fixed_val split để nhất quán
    """
    dp = data_prepare(pkl_path, include_ticker_id=True)
    valid_T      = [dp.get_max_T(t) for t in tickers if dp.get_max_T(t) > 0]
    global_T_max = min(valid_T)
    inner_T      = int(global_T_max * 0.85)   # = val_end trong main.py
    hval_split   = int(inner_T * 0.80)

    print(f"  global_T_max={global_T_max}  inner_T={inner_T}  hval_split={hval_split}")
    print(f"  test=[{inner_T}:{global_T_max}] ({global_T_max - inner_T} timesteps)")
    print(f"  Matches main.py: val_end={int(global_T_max*0.85)} → same test set ✓")

    tr_wf_list, tr_hv_list, va_hv_list, te_list = [], [], [], []
    macro_dim = news_dim = None

    for t in tickers:
        if dp.get_max_T(t) == 0:
            continue
        tr_wf, _, te = dp.prepare_data(
            t, train_end=inner_T, val_end=inner_T, test_end=global_T_max
        )
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

    print(f"  macro_dim={macro_dim}  news_dim={news_dim} (NEWS_EMB_DIM={NEWS_EMB_DIM})")

    return {
        "inner_T":      inner_T,
        "global_T_max": global_T_max,
        "hval_split":   hval_split,
        "train_wf":     merge(tr_wf_list,  shuffle=True),
        "train_hval":   merge(tr_hv_list,  shuffle=True),
        "val_fixed":    merge(va_hv_list,  shuffle=False),
        "test":         merge(te_list,     shuffle=False),
        "macro_dim":    macro_dim,
        "news_dim":     news_dim,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MODEL UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Exact copy của main.py: beta=0.9999, sqrt-normalize."""
    lbl  = labels.numpy()
    cnts = np.bincount(lbl, minlength=3).astype(float)
    beta = 0.9999
    eff  = 1.0 - np.power(beta, cnts)
    w    = (1.0 - beta) / (eff + 1e-8)
    w    = np.sqrt(w / w.sum() * 3)
    w    = w / w.sum() * 3
    return torch.tensor(w, dtype=torch.float32)


def build_model(
    macro_dim:   int,
    news_dim:    int,
    use_focal:   bool         = True,
    focal_gamma: float        = 2.0,
    cw:          torch.Tensor = None,
) -> StockMovementModel:
    return StockMovementModel(
        price_dim=1,
        macro_dim=macro_dim,
        news_dim=news_dim,
        dim=TrainConfig.dim,
        input_dim=TrainConfig.window_size,
        output_dim=3,
        num_head=TrainConfig.num_head,
        dropout=0.1,
        class_weights=cw,
        use_focal_loss=use_focal,
        focal_gamma=focal_gamma,
        device=DEVICE,
        n_tickers=N_TICKERS,
        quality_dim=getattr(GlobalConfig, "QUALITY_DIM", 4),
    ).to(DEVICE)


def _make_adamw(model: StockMovementModel, lr: float = 1e-4) -> torch.optim.Optimizer:
    """
    AdamW với param groups riêng biệt — khớp chính xác main.py.
    LayerNorm.weight + bias không có weight_decay (tránh kéo scale LN về 0).
    """
    no_decay_kws = ["bias", "LayerNorm.weight", "layernorm.weight",
                    "norm.weight", "attn_norm.weight", "out_norm.weight"]
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(kw in name for kw in no_decay_kws):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [{"params": decay,    "weight_decay": TrainConfig.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def _make_dataloader(dataset: StockDataset, shuffle: bool, batch_size: int) -> DataLoader:
    """Shared DataLoader factory — num_workers=0 on Windows (avoids subprocess reimport)."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=0,                       # Windows: Triton/multiprocessing issue
        pin_memory=(DEVICE.type == "cuda"),
    )


def evaluate(
    model:      StockMovementModel,
    data:       dict,
    zero_news:  bool = False,
    zero_macro: bool = False,
) -> Tuple[float, float]:
    """
    Evaluate model trên test set.

    zero_news/zero_macro PHẢI nhất quán với training:
      - Nếu train với zero_news=True → eval cũng zero_news=True
      - "Train full + zero lúc test" = SAI (bias_a contamination)

    Architecture safety (không cần lo):
      StableGatedCrossAttention: output = norm(primary + dropout(H_gated))
        → residual bảo toàn primary dù aux=zeros
      v_i skip: cat([H_idm, v_i, v_t_seq])
        → price information luôn có mặt trong final prediction
    """
    if not data or len(data.get("label", [])) == 0:
        return 0.0, 0.0
    model.eval()
    ds    = StockDataset(data)
    bs    = 256 if DEVICE.type == "cuda" else 64
    ldr   = _make_dataloader(ds, shuffle=False, batch_size=bs)
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in ldr:
            s_n = batch["s_n"].to(DEVICE)
            s_m = batch["s_m"].to(DEVICE)
            if zero_news:  s_n = torch.zeros_like(s_n)
            if zero_macro: s_m = torch.zeros_like(s_m)
            q = batch.get("news_quality")
            _, _, preds = model(
                batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                batch["s_c"].to(DEVICE), s_m, s_n,
                batch["label"].to(DEVICE),
                mode="test", return_preds=True,
                ticker_id=batch.get("ticker_id"),
                news_mask=batch.get("news_mask"),
                news_quality=q.to(DEVICE) if (q is not None and not zero_news) else None,
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
# BASELINE_WF — V3: load main.py .pt hoặc retrain
# ─────────────────────────────────────────────────────────────────────────────

def load_msgca_fv_from_experiments(results_dir: str) -> Optional[dict]:
    """
    Load MSGCA_FV results pre-computed by run_experiments.py.

    Source file: baselines/results/raw_results.json
    Key        : "MSGCA_FV"

    Returns dict with acc_mean/acc_std/mcc_mean/mcc_std/acc_list/mcc_list,
    or None if file not found / key absent.

    Academic rationale:
      MSGCA_FV uses CE loss (no class weights) for fair comparison with baselines.
      Showing it alongside MSGCA_Best in the ablation table gives reviewers
      the full picture: Ref-A (production) vs Ref-B (fair) vs ablated variants.
      Delta MCC is reported against Ref-A (FocalLoss variants vs FocalLoss full model)
      for internal consistency; Ref-B is shown as a secondary anchor.
    """
    raw_path = os.path.join(results_dir, "raw_results.json")
    if not os.path.exists(raw_path):
        print(f"  [baseline_fv] raw_results.json not found: {raw_path}")
        print(f"  Run: python baselines/run_experiments.py first.")
        return None

    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        print(f"  [baseline_fv] Error reading {raw_path}: {e}")
        return None

    if "MSGCA_FV" not in raw:
        print(f"  [baseline_fv] 'MSGCA_FV' key not found in {raw_path}")
        print(f"  Keys present: {list(raw.keys())}")
        return None

    fv = raw["MSGCA_FV"]
    # Validate required keys
    required = ["acc_mean", "acc_std", "mcc_mean", "mcc_std"]
    if not all(k in fv for k in required):
        print(f"  [baseline_fv] Missing keys in MSGCA_FV entry: {fv.keys()}")
        return None

    result = {
        "acc_mean":   float(fv["acc_mean"]),
        "acc_std":    float(fv["acc_std"]),
        "mcc_mean":   float(fv["mcc_mean"]),
        "mcc_std":    float(fv["mcc_std"]),
        "acc_list":   fv.get("acc_list", []),
        "mcc_list":   fv.get("mcc_list", []),
        "n_seeds":    int(fv.get("n_seeds", len(fv.get("mcc_list", [])))),
        "source":     "loaded from run_experiments raw_results.json",
        "loss_mode":  "CE (fair comparison)",
    }
    return result


def find_saved_model(output_dir: str, seed: int) -> Optional[str]:
    """
    Tìm model đã save từ main.py cho seed này.

    main.py save path:
      output/best_model_label={label}_price={price}_{tid}_seed{seed}_fixed.pt

    Glob pattern: best_model_*_seed{seed}_fixed.pt
    Trả về file mới nhất nếu có nhiều match (nhiều label/price-mode), None nếu không có.

    KHÔNG tìm msgca_results_seed{seed}.json — file đó từ WF protocol cũ đã deprecated.
    """
    pattern = os.path.join(output_dir, f"best_model_*_seed{seed}_fixed.pt")
    matches = glob.glob(pattern)
    if not matches:
        return None
    # Ưu tiên file mới nhất (chạy gần nhất)
    return max(matches, key=os.path.getmtime)


def load_and_eval_saved_model(
    pt_path:   str,
    test_data: dict,
    macro_dim: int,
    news_dim:  int,
) -> Tuple[float, float]:
    """
    Load .pt đã save từ main.py và evaluate trên test set.
    Đây là cách đúng để lấy baseline_wf kết quả:
      - Cùng model weights với main.py
      - Cùng test set (inner_T : global_T_max)
      → Kết quả 100% nhất quán, không bị confound bởi protocol khác nhau

    strict=False để bỏ qua loss_fn.weight (buffer không cần cho inference).
    """
    model = build_model(macro_dim, news_dim, use_focal=False, cw=None)
    state = torch.load(pt_path, map_location=DEVICE, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # loss_fn.weight là buffer chỉ dùng lúc train → skip là đúng
    real_missing = [k for k in missing if "loss_fn" not in k]
    if real_missing:
        print(f"    [WARN] Unexpected missing keys: {real_missing}")
    return evaluate(model, test_data)


def run_baseline_wf(
    splits:     dict,
    n_seeds:    int,
    output_dir: str,
) -> dict:
    """
    Lấy kết quả baseline theo 2 phương án ưu tiên:

    Phương án 1 (ưu tiên):
      Tìm best_model_*_seed{seed}_fixed.pt trong output_dir
      → load và evaluate → kết quả giống hệt main.py output
      → Source: "loaded from .pt"

    Phương án 2 (fallback):
      Không tìm thấy .pt → retrain với fixed_val+warmup+fv_epochs
      patience và modality dropout lấy từ TrainConfig (configs/config.py)
      → Source: "retrained"

    Không còn đọc msgca_results_seed{seed}.json — deprecated từ V3.
    """
    macro_dim  = splits["macro_dim"]
    news_dim   = splits["news_dim"]
    test       = splits["test"]
    train_hval = splits["train_hval"]
    val_fixed  = splits["val_fixed"]

    acc_list, mcc_list, sources = [], [], []

    for seed in SEEDS[:n_seeds]:
        pt_path = find_saved_model(output_dir, seed)

        if pt_path:
            print(f"    Seed {seed}: found {os.path.basename(pt_path)}")
            acc, mcc = load_and_eval_saved_model(pt_path, test, macro_dim, news_dim)
            sources.append("loaded")
        else:
            print(f"    Seed {seed}: no .pt found in '{output_dir}/' → retrain (fixed_val 200ep)")
            acc, mcc = run_seed_fixed_val(
                seed=seed,
                train_data=train_hval, val_data=val_fixed, test_data=test,
                macro_dim=macro_dim, news_dim=news_dim,
                max_epochs=200, patience=None, warmup_epochs=10,
            )
            sources.append("retrained")

        acc_list.append(acc)
        mcc_list.append(mcc)
        print(f"             ACC={acc:.4f}  MCC={mcc:.4f}  [{sources[-1]}]")

    n_loaded    = sources.count("loaded")
    n_retrained = sources.count("retrained")
    print(f"  Summary: {n_loaded} loaded from main.py .pt  |  {n_retrained} retrained")
    if n_loaded < n_seeds:
        print(f"  TIP: Run main.py first to generate .pt files → auto-load in future runs")

    return {
        "acc_mean": float(np.mean(acc_list)),
        "acc_std":  float(np.std(acc_list)),
        "mcc_mean": float(np.mean(mcc_list)),
        "mcc_std":  float(np.std(mcc_list)),
        "acc_list": acc_list,
        "mcc_list": mcc_list,
        "n_seeds":  len(acc_list),
        "source":   f"{n_loaded} loaded / {n_retrained} retrained",
    }


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE SEED — WALK-FORWARD (giữ lại cho legacy/so sánh)
# ─────────────────────────────────────────────────────────────────────────────

def find_best_epoch_wf(
    dp, tickers: list, inner_T: int, global_T_max: int,
    macro_dim: int, news_dim: int,
    zero_news: bool = False, zero_macro: bool = False,
    wf_folds: int = 3, max_epochs: int = 80, min_val_size: int = 200,
    use_focal: bool = True, focal_gamma: float = 2.0, seed: int = 42,
) -> int:
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
        if len(fold_va.get("label", [])) < min_val_size:
            continue

        cw    = compute_class_weights(fold_tr["label"]).to(DEVICE)
        model = build_model(macro_dim, news_dim, use_focal, focal_gamma, cw)
        ds    = StockDataset(fold_tr)
        ldr   = _make_dataloader(ds, shuffle=True, batch_size=32)
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
                q = batch.get("news_quality")
                loss = model(
                    batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                    batch["s_c"].to(DEVICE), s_m, s_n,
                    batch["label"].to(DEVICE), mode="train",
                    ticker_id=batch.get("ticker_id"),
                    news_mask=batch.get("news_mask"),
                    news_quality=q.to(DEVICE) if (q is not None and not zero_news) else None,
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
    smoothed = [np.mean(avg[max(0, i - 1):min(max_epochs, i + 2)]) for i in range(max_epochs)]
    return int(np.argmax(smoothed)) + 1


def run_seed_walkforward(
    seed: int, train_data: dict, test_data: dict,
    best_epoch: int, macro_dim: int, news_dim: int,
    zero_news: bool = False, zero_macro: bool = False,
    use_focal: bool = True, focal_gamma: float = 2.0,
) -> Tuple[float, float]:
    set_seed(seed)
    cw    = compute_class_weights(train_data["label"]).to(DEVICE)
    model = build_model(macro_dim, news_dim, use_focal, focal_gamma, cw)
    ds    = StockDataset(train_data)
    ldr   = _make_dataloader(ds, shuffle=True, batch_size=32)
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
            q = batch.get("news_quality")
            loss = model(
                batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                batch["s_c"].to(DEVICE), s_m, s_n,
                batch["label"].to(DEVICE), mode="train",
                ticker_id=batch.get("ticker_id"),
                news_mask=batch.get("news_mask"),
                news_quality=q.to(DEVICE) if (q is not None and not zero_news) else None,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sch.step()

    return evaluate(model, test_data, zero_news, zero_macro)


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN ONE SEED — FIXED VAL (khớp với main.py)
# ─────────────────────────────────────────────────────────────────────────────

def run_seed_fixed_val(
    seed:          int,
    train_data:    dict,
    val_data:      dict,
    test_data:     dict,
    macro_dim:     int,
    news_dim:      int,
    zero_news:     bool  = False,
    zero_macro:    bool  = False,
    max_epochs:    int   = 200,
    patience:      int   = None,   # None → dùng _PATIENCE từ TrainConfig
    warmup_epochs: int   = 10,
    use_focal:     bool  = True,
    focal_gamma:   float = 2.0,
) -> Tuple[float, float]:
    """
    Fixed val + LR warmup + early stopping.
    Protocol khớp với main.py: AdamW lr=1e-4, wd=1e-4, CosineAnnealing.
    patience=None → dùng _PATIENCE từ TrainConfig (configs/config.py).
    """
    if patience is None:
        patience = _PATIENCE
    set_seed(seed)
    cw    = compute_class_weights(train_data["label"]).to(DEVICE)
    model = build_model(macro_dim, news_dim, use_focal, focal_gamma, cw)
    ds    = StockDataset(train_data)
    ldr   = _make_dataloader(
        ds, shuffle=True,
        batch_size=getattr(TrainConfig, "batch_size", 32)
    )
    opt    = _make_adamw(model, lr=TrainConfig.learning_rate)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(max_epochs - warmup_epochs, 1), eta_min=1e-6
    )

    best_mcc, best_state, no_improve = -2.0, None, 0

    for epoch in range(max_epochs):
        model.train()
        for batch in ldr:
            opt.zero_grad()
            s_n = batch["s_n"].to(DEVICE)
            s_m = batch["s_m"].to(DEVICE)
            # zero_news / zero_macro: ablation zeros (phải apply ở đây)
            if zero_news:  s_n = torch.zeros_like(s_n)
            if zero_macro: s_m = torch.zeros_like(s_m)
            # News Modality Dropout: chỉ apply khi KHÔNG zero_news
            # (không có ý nghĩa dropout thêm khi đã bỏ hẳn news)
            mask_in = batch.get("news_mask")
            q_in    = batch.get("news_quality")
            if not zero_news and _MOD_DROPOUT > 0.0 and torch.rand(1).item() < _MOD_DROPOUT:
                s_n     = torch.zeros_like(s_n)
                if mask_in is not None:
                    mask_in = torch.ones_like(mask_in, dtype=torch.bool)
                q_in = None   # drop quality together with news
            loss = model(
                batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                batch["s_c"].to(DEVICE), s_m, s_n,
                batch["label"].to(DEVICE), mode="train",
                ticker_id=batch.get("ticker_id"),
                news_mask=mask_in.to(DEVICE) if mask_in is not None else None,
                news_quality=q_in.to(DEVICE) if q_in is not None else None,
            )
            if torch.isfinite(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        if epoch < warmup_epochs:
            warmup.step()
        else:
            cosine.step()

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
# RUN ONE VARIANT — N SEEDS
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(
    variant:       str,
    splits:        dict,
    tickers:       list,
    pkl_path:      str,
    n_seeds:       int  = 5,
    wf_max_epochs: int  = 80,
    fv_max_epochs: int  = 200,
    verbose:       bool = True,
) -> dict:
    """
    Run một ablation variant trên n_seeds.

    Architecture safety reminder (built-in, không cần xử lý manual):
      no_news:  Stage1 gate → suppressed (training), H_id ≈ v_i
                Stage2 chạy bình thường: H_id × v_m → H_idm
                v_i skip luôn bảo toàn price trong cat
      no_macro: Stage2 gate → suppressed (training), H_idm ≈ H_id
                Stage1 chạy bình thường: v_i × v_n → H_id
                v_i skip luôn bảo toàn price trong cat
    """
    zero_news  = "no_news"  in variant
    zero_macro = "no_macro" in variant
    use_fixval = "fixed"    in variant or variant in ("no_news", "no_macro")
    # V3: no_news và no_macro dùng fixed_val để nhất quán với main.py

    macro_dim  = splits["macro_dim"]
    news_dim   = splits["news_dim"]
    train_wf   = splits["train_wf"]
    train_hval = splits["train_hval"]
    val_fixed  = splits["val_fixed"]
    test       = splits["test"]

    modality_str = (
        "no news+macro" if zero_news and zero_macro else
        "no news"       if zero_news else
        "no macro"      if zero_macro else
        "all modalities"
    )
    protocol_str = f"fixed_val+warmup+{fv_max_epochs}ep" if use_fixval else f"walk-forward+{wf_max_epochs}ep"

    print(f"\n  Variant    : {variant}")
    print(f"  Modalities : {modality_str}")
    print(f"  Protocol   : {protocol_str}")
    print(f"  news_dim   : {news_dim}D (NEWS_EMB_DIM={NEWS_EMB_DIM})")

    if zero_news or zero_macro:
        print(f"  [ABLATION] RETRAIN from scratch — zeros in BOTH train AND eval.")
        print(f"  [SAFETY]   fusion residual + v_i skip → degradation safe.")

    acc_list, mcc_list = [], []

    if not use_fixval:
        # Walk-forward: tìm best epoch rồi retrain (legacy, không dùng cho baseline_wf nữa)
        dp = data_prepare(pkl_path, include_ticker_id=True)
        for seed in SEEDS[:n_seeds]:
            set_seed(seed)
            best_ep = find_best_epoch_wf(
                dp=dp, tickers=tickers,
                inner_T=splits["inner_T"],
                global_T_max=splits["global_T_max"],
                macro_dim=macro_dim, news_dim=news_dim,
                zero_news=zero_news, zero_macro=zero_macro,
                wf_folds=3, max_epochs=wf_max_epochs, min_val_size=200,
            )
            acc, mcc = run_seed_walkforward(
                seed=seed, train_data=train_wf, test_data=test,
                best_epoch=best_ep, macro_dim=macro_dim, news_dim=news_dim,
                zero_news=zero_news, zero_macro=zero_macro,
            )
            acc_list.append(acc)
            mcc_list.append(mcc)
            if verbose:
                print(f"    Seed {seed}: ep={best_ep:3d}  ACC={acc:.4f}  MCC={mcc:.4f}")
    else:
        for seed in SEEDS[:n_seeds]:
            acc, mcc = run_seed_fixed_val(
                seed=seed,
                train_data=train_hval, val_data=val_fixed, test_data=test,
                macro_dim=macro_dim, news_dim=news_dim,
                zero_news=zero_news, zero_macro=zero_macro,
                max_epochs=fv_max_epochs, patience=None, warmup_epochs=10,
            )
            acc_list.append(acc)
            mcc_list.append(mcc)
            if verbose:
                print(f"    Seed {seed}: ACC={acc:.4f}  MCC={mcc:.4f}")

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
    sep = "=" * 125
    lines += [
        sep,
        "  ABLATION STUDY -- MSGCA  (Mean +/- Std  |  test=[85%:100%] T_max)",
        f"  News dim    : {NEWS_EMB_DIM}D (FinBERT CLS per-triple weighted-mean)",
        "  Methodology : each non-baseline variant RETRAINED from scratch",
        "               (zeros applied in BOTH train AND eval -- not post-hoc zeroing)",
        "  Ref-A [baseline_wf] : MSGCA_Best (FocalLoss+CW) -- loaded from main.py .pt",
        "  Ref-B [baseline_fv] : MSGCA_FV   (CE, fair)     -- loaded from run_experiments.py",
        "  dRef-A = MCC_variant - MCC_Ref-A  (primary delta, same loss function as variants)",
        "  dRef-B = MCC_variant - MCC_Ref-B  (secondary delta, vs fair-comparison model)",
        sep,
        f"{'Variant':<20} {'Description':<56} {'ACC':>14} {'MCC':>14}  {'dRef-A':>8}  {'dRef-B':>8}",
        "-" * 125,
    ]

    ref_a_mcc = results.get("baseline_wf", {}).get("mcc_mean", None)
    ref_b_mcc = results.get("baseline_fv", {}).get("mcc_mean", None)

    for v in ALL_VARIANTS:
        if v not in results:
            continue
        r    = results[v]
        desc = VARIANT_DESCRIPTIONS.get(v, "")[:55]
        acc  = f"{r['acc_mean']:.4f}+/-{r['acc_std']:.4f}"
        mcc  = f"{r['mcc_mean']:.4f}+/-{r['mcc_std']:.4f}"

        # dRef-A
        if v == "baseline_wf":
            delta_a = "(Ref-A)"
        elif ref_a_mcc is not None:
            d = r["mcc_mean"] - ref_a_mcc
            delta_a = f"{'+' if d >= 0 else ''}{d:.4f}"
        else:
            delta_a = "  N/A  "

        # dRef-B
        if v == "baseline_fv":
            delta_b = "(Ref-B)"
        elif ref_b_mcc is not None:
            d = r["mcc_mean"] - ref_b_mcc
            delta_b = f"{'+' if d >= 0 else ''}{d:.4f}"
        else:
            delta_b = "  N/A  "

        lines.append(
            f"{v:<20} {desc:<56} {acc:>14} {mcc:>14}  {delta_a:>8}  {delta_b:>8}"
        )

    lines.append(sep)
    lines.append("\nINTERPRETATION:")

    # Q1: News channel
    if "no_news" in results and ref_a_mcc is not None:
        d_a = results["no_news"]["mcc_mean"] - ref_a_mcc
        d_b_str = ""
        if ref_b_mcc is not None:
            d_b = results["no_news"]["mcc_mean"] - ref_b_mcc
            d_b_str = f"  dRef-B={d_b:+.4f}"
        tag = f"dRef-A={d_a:+.4f}{d_b_str}"
        if d_a > 0.005:
            lines.append(f"  Q1 NEWS  ({tag}): News HURTS model (Ref-A). "
                         "Consider improving KG quality, coverage, or gating strategy.")
        elif d_a < -0.005:
            lines.append(f"  Q1 NEWS  ({tag}): News HELPS model -- channel contributing positively.")
        else:
            lines.append(f"  Q1 NEWS  ({tag}): Negligible effect (|d|<0.005). "
                         "Signal present but weak or already captured by price.")

    # Q3: Macro channel
    if "no_macro" in results and ref_a_mcc is not None:
        d_a = results["no_macro"]["mcc_mean"] - ref_a_mcc
        d_b_str = ""
        if ref_b_mcc is not None:
            d_b = results["no_macro"]["mcc_mean"] - ref_b_mcc
            d_b_str = f"  dRef-B={d_b:+.4f}"
        tag = f"dRef-A={d_a:+.4f}{d_b_str}"
        if d_a < -0.005:
            lines.append(f"  Q3 MACRO ({tag}): Macro contributes meaningfully to prediction.")
        elif d_a > 0.005:
            lines.append(f"  Q3 MACRO ({tag}): Macro adds noise via Stage2 fusion.")
        else:
            lines.append(f"  Q3 MACRO ({tag}): Small macro effect.")

    # Q2: Protocol sanity
    if "fixed_val" in results and ref_a_mcc is not None:
        d_a = results["fixed_val"]["mcc_mean"] - ref_a_mcc
        if abs(d_a) < 0.01:
            verdict = "Protocol matched. Expected ~0 vs Ref-A."
        else:
            verdict = "Some variance vs Ref-A (different train/val split or seeds)."
        lines.append(f"  Q2 PROTO (dRef-A={d_a:+.4f}): {verdict}")

    # Ref-A vs Ref-B gap note
    if ref_a_mcc is not None and ref_b_mcc is not None:
        gap = ref_b_mcc - ref_a_mcc
        direction = "outperforms" if gap > 0 else "underperforms"
        if abs(gap) > 0.005:
            note = ("CE loss may generalize better on this dataset size (~4.7k train samples). "
                    "Report MSGCA_FV as primary result in paper (fair comparison row)."
                    if gap > 0
                    else "FocalLoss+CW model is stronger here.")
        else:
            note = "Difference within noise -- FocalLoss and CE perform similarly."
        lines.append(
            f"\n  NOTE: Ref-B (MSGCA_FV/CE) {direction} Ref-A (MSGCA_Best/FocalLoss)"
            f" by {gap:+.4f} MCC. {note}"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="MSGCA ablation V3 — baseline_wf loads main.py .pt (not WF JSON)"
    )
    ap.add_argument("--pkl",       default=None,
                    help="Path to unified_dataset_test.pkl")
    ap.add_argument("--variants",  nargs="+", default=ALL_VARIANTS,
                    choices=ALL_VARIANTS + ["all"])
    ap.add_argument("--n-seeds",   type=int, default=5)
    ap.add_argument("--wf-epochs", type=int, default=80,
                    help="Max epochs for walk-forward (legacy)")
    ap.add_argument("--fv-epochs", type=int, default=200,
                    help="Max epochs for fixed_val variants")
    ap.add_argument("--load-from", default="output",
                    help="Dir containing main.py .pt saves for baseline_wf auto-detect")
    ap.add_argument("--verbose",   action="store_true")
    args = ap.parse_args()

    pkl_path = args.pkl or os.path.join(
        GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl"
    )
    if not os.path.exists(pkl_path):
        print(f"Dataset not found: {pkl_path}")
        sys.exit(1)

    variants = ALL_VARIANTS if "all" in args.variants else args.variants
    tickers  = GlobalConfig.TICKERS

    print(f"\n{'='*65}")
    print(f"ABLATION STUDY — MSGCA V3")
    print(f"Device  : {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)}, TF32 on)" if DEVICE.type == "cuda" else ""))
    print(f"OS      : {platform.system()} (torch.compile {'disabled' if _IS_WINDOWS else 'enabled'})")
    print(f"News dim: {NEWS_EMB_DIM}D (FinBERT CLS)")
    print(f"Variants: {variants}")
    print(f"Seeds   : {SEEDS[:args.n_seeds]}")
    print(f"Epochs  : fixed_val={args.fv_epochs}  wf={args.wf_epochs}")
    print(f"Load dir: {args.load_from}/  (baseline_wf .pt files)")
    print(f"{'='*65}")

    print("\nLoading data splits...")
    splits = load_splits(pkl_path, tickers)

    all_results: Dict[str, dict] = {}
    total_t0 = time.time()

    # ── baseline_wf: load .pt từ main.py hoặc retrain ─────────────────────────
    if "baseline_wf" in variants:
        print(f"\n  Variant    : baseline_wf  [Ref-A: MSGCA_Best, FocalLoss+CW]")
        print(f"  Strategy   : auto-detect best_model_*_seed*_fixed.pt in '{args.load_from}/'")
        t0 = time.time()
        all_results["baseline_wf"] = run_baseline_wf(
            splits=splits, n_seeds=args.n_seeds, output_dir=args.load_from,
        )
        r = all_results["baseline_wf"]
        print(f"  -> ACC={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}  "
              f"MCC={r['mcc_mean']:.4f}+/-{r['mcc_std']:.4f}  "
              f"({(time.time() - t0) / 60:.1f} min)")
        variants = [v for v in variants if v != "baseline_wf"]

    # ── baseline_fv: load MSGCA_FV từ run_experiments.py JSON — zero compute ──
    if "baseline_fv" in variants:
        experiments_results_dir = os.path.join(
            os.path.dirname(__file__), "results"
        )
        print(f"\n  Variant    : baseline_fv  [Ref-B: MSGCA_FV, CE loss]")
        print(f"  Strategy   : load from {experiments_results_dir}/raw_results.json (no retraining)")
        fv_result = load_msgca_fv_from_experiments(experiments_results_dir)
        if fv_result is not None:
            all_results["baseline_fv"] = fv_result
            print(f"  -> ACC={fv_result['acc_mean']:.4f}+/-{fv_result['acc_std']:.4f}  "
                  f"MCC={fv_result['mcc_mean']:.4f}+/-{fv_result['mcc_std']:.4f}  "
                  f"[n_seeds={fv_result['n_seeds']}]  (0.0 min — loaded)")
        else:
            print(f"  -> SKIPPED (raw_results.json not found or MSGCA_FV key absent)")
            print(f"     Run: python baselines/run_experiments.py first")
        variants = [v for v in variants if v != "baseline_fv"]

    # Các variants còn lại
    for variant in variants:
        t0 = time.time()
        all_results[variant] = run_variant(
            variant=variant, splits=splits, tickers=tickers, pkl_path=pkl_path,
            n_seeds=args.n_seeds,
            wf_max_epochs=args.wf_epochs,
            fv_max_epochs=args.fv_epochs,
            verbose=args.verbose,
        )
        print(f"  ({(time.time() - t0) / 60:.1f} min)")

    print(f"\nTotal: {(time.time() - total_t0) / 60:.1f} min\n")

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