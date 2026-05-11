"""
MSGCA Trainer — 2-phase training logic for MSGCA_FV (CE loss mode).

Phase 1: train on [0:hval_split] with early stopping → find best_epoch
Phase 2: retrain on [0:inner_T]  for best_epoch    → evaluate on test

This module is self-contained: all imports are local.
"""

import random
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.utils.data import DataLoader, Dataset

from model.stock_model import StockMovementModel, N_TICKERS


SEEDS: List[int] = [42, 123, 256, 512, 1024]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MSGCADataset(Dataset):
    def __init__(self, d: dict):
        self.d    = d
        self.keys = [k for k in d if isinstance(d[k], torch.Tensor)]

    def __len__(self):
        return len(self.d["label"])

    def __getitem__(self, i):
        return {k: self.d[k][i] for k in self.keys}


# ─────────────────────────────────────────────────────────────────────────────
# Model builder — CE loss (MSGCA_FV / fair comparison mode)
# ─────────────────────────────────────────────────────────────────────────────

def build_model_ce(
    macro_dim:   int,
    news_dim:    int,
    lr:          float,
    dropout:     float,
    device:      torch.device,
    dim:         int   = 64,
    num_head:    int   = 2,
    window_size: int   = 20,
    n_tickers:   int   = N_TICKERS,
    quality_dim: int   = 4,
) -> Tuple[StockMovementModel, torch.optim.Optimizer]:
    """
    Build MSGCA model in CE (fair comparison) mode — no focal loss, no class weights.
    Matches the MSGCA_FV configuration from run_experiments.py.
    """
    model = StockMovementModel(
        price_dim=1,
        macro_dim=macro_dim,
        news_dim=news_dim,
        dim=dim,
        input_dim=window_size,
        output_dim=3,
        num_head=num_head,
        dropout=dropout,
        class_weights=None,
        use_focal_loss=False,
        device=device,
        n_tickers=n_tickers,
        quality_dim=quality_dim,
    ).to(device)

    opt = make_adamw(model, lr)
    return model, opt


def make_adamw(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """
    AdamW with separate param groups:
      - LayerNorm weights and biases: no weight decay (prevents LN scale collapse)
      - All other params: weight_decay=1e-4
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
        [{"params": decay, "weight_decay": 1e-4},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:          StockMovementModel,
    loader:         DataLoader,
    opt:            torch.optim.Optimizer,
    device:         torch.device,
    mod_dropout:    float = 0.30,
) -> float:
    """
    Train one epoch with optional news modality dropout.

    mod_dropout: probability of zeroing the entire news stream for a batch.
    Set to 0.0 to disable.
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        opt.zero_grad(set_to_none=True)

        s_n_in  = batch["s_n"].to(device)
        mask_in = batch.get("news_mask")
        q_in    = batch.get("news_quality")

        if mod_dropout > 0.0 and torch.rand(1).item() < mod_dropout:
            s_n_in = torch.zeros_like(s_n_in)
            if mask_in is not None:
                mask_in = torch.ones_like(mask_in, dtype=torch.bool)
            q_in = None

        loss = model(
            batch["s_o"].to(device), batch["s_h"].to(device),
            batch["s_c"].to(device), batch["s_m"].to(device),
            s_n_in, batch["label"].to(device),
            mode="train",
            ticker_id=batch.get("ticker_id"),
            news_mask=mask_in.to(device) if mask_in is not None else None,
            news_quality=q_in.to(device) if q_in is not None else None,
        )

        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

    return total_loss


def eval_model(
    model:      StockMovementModel,
    data:       dict,
    device:     torch.device,
    batch_size: int = 64,
) -> Tuple[float, float]:
    """Evaluate model → (accuracy, MCC)."""
    if not data or len(data.get("label", [])) == 0:
        return 0.0, 0.0

    model.eval()
    ldr = DataLoader(MSGCADataset(data), batch_size=batch_size, shuffle=False)
    preds_all, labels_all = [], []

    with torch.no_grad():
        for batch in ldr:
            q = batch.get("news_quality")
            _, _, preds = model(
                batch["s_o"].to(device), batch["s_h"].to(device),
                batch["s_c"].to(device), batch["s_m"].to(device),
                batch["s_n"].to(device), batch["label"].to(device),
                mode="test", return_preds=True,
                ticker_id=batch.get("ticker_id"),
                news_mask=batch.get("news_mask"),
                news_quality=q.to(device) if q is not None else None,
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
# 2-phase training for one seed
# ─────────────────────────────────────────────────────────────────────────────

def run_one_seed(
    seed:        int,
    train_hval:  dict,
    val_hval:    dict,
    train_full:  dict,
    test:        dict,
    macro_dim:   int,
    news_dim:    int,
    hp:          dict,
    device:      torch.device,
    max_epochs:  int   = 150,
    patience:    int   = 30,
    warmup_epochs: int = 15,
    mod_dropout: float = 0.30,
    verbose:     bool  = False,
    dim:         int   = 64,
    num_head:    int   = 2,
    window_size: int   = 20,
    n_tickers:   int   = N_TICKERS,
    quality_dim: int   = 4,
) -> Tuple[float, float, int]:
    """
    2-phase training for one random seed.

    Phase 1: train on train_hval → early stopping on val_hval MCC → find best_epoch
    Phase 2: retrain on train_full for best_epoch → evaluate on test

    Returns: (acc, mcc, best_epoch)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    build_kwargs = dict(
        macro_dim=macro_dim, news_dim=news_dim,
        lr=hp["lr"], dropout=hp["dropout"], device=device,
        dim=dim, num_head=num_head, window_size=window_size,
        n_tickers=n_tickers, quality_dim=quality_dim,
    )

    # ── Phase 1: find best epoch ──────────────────────────────────────────
    model, opt = build_model_ce(**build_kwargs)
    ldr = DataLoader(MSGCADataset(train_hval), batch_size=32, shuffle=True, drop_last=False)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(max_epochs - warmup_epochs, 1), eta_min=1e-6
    )

    best_mcc, best_epoch, no_improve = -2.0, 1, 0
    min_active = max(warmup_epochs, 40)   # don't start early-stopping before epoch 40

    for epoch in range(max_epochs):
        train_epoch(model, ldr, opt, device, mod_dropout=mod_dropout)
        if epoch < warmup_epochs:
            warmup.step()
        else:
            cosine.step()

        if epoch >= min_active:
            _, mcc = eval_model(model, val_hval, device)
            if mcc > best_mcc:
                best_mcc = mcc
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if verbose and (epoch + 1) % 20 == 0:
            _, cur_mcc = eval_model(model, val_hval, device)
            print(f"      ep {epoch+1:3d}: val_MCC={cur_mcc:.4f}")

    if verbose:
        print(f"    Phase1 done: best_ep={best_epoch}  val_MCC={best_mcc:.4f}")

    # ── Phase 2: retrain on full inner split ──────────────────────────────
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model2, opt2 = build_model_ce(**build_kwargs)
    ldr2  = DataLoader(MSGCADataset(train_full), batch_size=32, shuffle=True, drop_last=False)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=best_epoch, eta_min=1e-6)

    for _ in range(best_epoch):
        train_epoch(model2, ldr2, opt2, device, mod_dropout=mod_dropout)
        sched2.step()

    acc, mcc = eval_model(model2, test, device)
    return acc, mcc, best_epoch


# ─────────────────────────────────────────────────────────────────────────────
# Grid search
# ─────────────────────────────────────────────────────────────────────────────

def grid_search(
    train_hval:  dict,
    val_hval:    dict,
    macro_dim:   int,
    news_dim:    int,
    device:      torch.device,
    grid:        Dict[str, list] = None,
    dim:         int   = 64,
    num_head:    int   = 2,
    window_size: int   = 20,
    n_tickers:   int   = N_TICKERS,
    quality_dim: int   = 4,
    max_epochs:  int   = 100,
    patience:    int   = 20,
    warmup_epochs: int = 15,
    mod_dropout: float = 0.30,
    verbose:     bool  = True,
    resume_results: list = None,
) -> dict:
    """
    Grid search over hyperparameters using seed=42 for reproducibility.

    grid: dict of lists, e.g. {"lr": [5e-5, 1e-4], "dropout": [0.1, 0.2]}
    resume_results: list of previously computed results (for --resume mode)

    Returns: {"best_hparams": dict, "best_mcc": float, "all_results": list}
    """
    default_grid = {
        "lr":      [5e-5, 1e-4, 3e-4, 5e-4],
        "dropout": [0.1, 0.2, 0.3],
    }
    grid = grid or default_grid

    keys   = list(grid.keys())
    combos = [dict(zip(keys, v)) for v in product(*[grid[k] for k in keys])]

    # Build set of already-completed HP combos (for resume)
    done_hps = set()
    all_results = list(resume_results or [])
    best_mcc    = max((r["val_mcc"] for r in all_results), default=-2.0)
    best_hp     = next((r for r in all_results if r["val_mcc"] == best_mcc), combos[0])

    for r in all_results:
        done_hps.add(tuple(r[k] for k in keys))

    torch.manual_seed(42)
    np.random.seed(42)

    for i, hp in enumerate(combos):
        hp_key = tuple(hp[k] for k in keys)
        if hp_key in done_hps:
            if verbose:
                print(f"    [{i+1:02d}/{len(combos)}] SKIP (already done): "
                      + " ".join(f"{k}={hp[k]}" for k in keys))
            continue

        _, mcc, ep = run_one_seed(
            seed=42,
            train_hval=train_hval, val_hval=val_hval,
            train_full=train_hval, test=val_hval,   # search on val
            macro_dim=macro_dim, news_dim=news_dim,
            hp=hp, device=device,
            max_epochs=max_epochs, patience=patience,
            warmup_epochs=warmup_epochs, mod_dropout=mod_dropout,
            dim=dim, num_head=num_head, window_size=window_size,
            n_tickers=n_tickers, quality_dim=quality_dim,
        )

        is_best = mcc > best_mcc
        result  = {**hp, "val_mcc": mcc, "best_epoch": ep}
        all_results.append(result)

        if verbose:
            flag = " ←best" if is_best else ""
            print(f"    [{i+1:02d}/{len(combos)}] "
                  + " ".join(f"{k}={hp[k]:.0e}" if isinstance(hp[k], float) else f"{k}={hp[k]}"
                              for k in keys)
                  + f" → val_MCC={mcc:.4f}  ep={ep}{flag}")

        if is_best:
            best_mcc = mcc
            best_hp  = hp.copy()

    return {"best_hparams": best_hp, "best_mcc": best_mcc, "all_results": all_results}


# ─────────────────────────────────────────────────────────────────────────────
# Final multi-seed evaluation
# ─────────────────────────────────────────────────────────────────────────────

def final_eval(
    train_hval:  dict,
    val_hval:    dict,
    train_full:  dict,
    test:        dict,
    best_hparams: dict,
    macro_dim:   int,
    news_dim:    int,
    device:      torch.device,
    n_seeds:     int   = 5,
    dim:         int   = 64,
    num_head:    int   = 2,
    window_size: int   = 20,
    n_tickers:   int   = N_TICKERS,
    quality_dim: int   = 4,
    max_epochs:  int   = 150,
    patience:    int   = 30,
    warmup_epochs: int = 15,
    mod_dropout: float = 0.30,
    verbose:     bool  = True,
) -> dict:
    """
    Final evaluation with n_seeds using best_hparams.
    Reports mean±std ACC and MCC across seeds.
    """
    acc_list, mcc_list, ep_list = [], [], []

    for seed in SEEDS[:n_seeds]:
        acc, mcc, ep = run_one_seed(
            seed=seed,
            train_hval=train_hval, val_hval=val_hval,
            train_full=train_full, test=test,
            macro_dim=macro_dim, news_dim=news_dim,
            hp=best_hparams, device=device,
            max_epochs=max_epochs, patience=patience,
            warmup_epochs=warmup_epochs, mod_dropout=mod_dropout,
            dim=dim, num_head=num_head, window_size=window_size,
            n_tickers=n_tickers, quality_dim=quality_dim,
            verbose=verbose,
        )
        acc_list.append(acc)
        mcc_list.append(mcc)
        ep_list.append(ep)
        if verbose:
            print(f"    Seed {seed}: ep={ep:3d}  ACC={acc:.4f}  MCC={mcc:.4f}")

    return {
        "acc_mean":  float(np.mean(acc_list)),
        "acc_std":   float(np.std(acc_list)),
        "mcc_mean":  float(np.mean(mcc_list)),
        "mcc_std":   float(np.std(mcc_list)),
        "acc_list":  acc_list,
        "mcc_list":  mcc_list,
        "ep_mean":   float(np.mean(ep_list)),
        "n_seeds":   len(acc_list),
        "hparams":   best_hparams,
    }
