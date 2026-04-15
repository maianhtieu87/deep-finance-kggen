# baselines/hparam_search.py
"""
Hyperparameter search + final 5-seed evaluation cho flat baselines.

━━━ FACTORY CONTRACT ━━━
  model_factory: Callable nhận KEYWORD args (hidden_dim, macro_dim, dropout) → nn.Module
  Ví dụ hợp lệ:
    factory = lambda hidden_dim, macro_dim, dropout: LSTMBaseline(hidden_dim, 3, dropout)
    factory(hidden_dim=64, macro_dim=5, dropout=0.1)  ← OK

━━━ SEARCH SPACE ━━━
  lr:         [5e-5, 1e-4, 5e-4]
  hidden_dim: [32, 64, 128]
  dropout:    [0.1, 0.2]
  → 3×3×2 = 18 combos

━━━ TIMELINE ━━━
  Grid search : train=[0:hval_split], val=[hval_split:inner_T]
  Final 5-seed: train=[0:inner_T],    test=[inner_T:global_T_max]
"""

from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from itertools import product
from sklearn.metrics import accuracy_score, matthews_corrcoef
from torch.utils.data import DataLoader, TensorDataset


# ─────────────────────────────────────────────────────────────────────────────
# SEEDS & SEARCH SPACE
# ─────────────────────────────────────────────────────────────────────────────

SEEDS: List[int] = [42, 123, 256, 512, 1024]

HPARAM_GRID: Dict[str, List] = {
    "lr":         [5e-5, 1e-4, 5e-4],
    "hidden_dim": [32, 64, 128],
    "dropout":    [0.1, 0.2],
}


def get_all_combinations(grid: Dict[str, List] = None) -> List[Dict]:
    """Trả về list tất cả tổ hợp hyperparameters."""
    grid = grid or HPARAM_GRID
    keys = list(grid.keys())
    return [dict(zip(keys, vals)) for vals in product(*[grid[k] for k in keys])]


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class FlatDataset(TensorDataset):
    """
    TensorDataset cho flat baselines.
    Trả về (indicators, s_n, s_m, label) — thứ tự cố định.
    """
    def __init__(self, d: dict):
        super().__init__(
            d["indicators"],   # (N, W, 3)
            d["s_n"],          # (N, W, 1024)
            d["s_m"],          # (N, W, M)
            d["label"],        # (N,)
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _forward_flat(model: nn.Module, batch, device: torch.device) -> torch.Tensor:
    """
    Gọi forward cho flat baselines.
    Tất cả flat baseline forward() nhận (indicators, s_n, s_m) — tên tham số cố định.
    """
    indicators, s_n, s_m, _ = batch
    return model(
        indicators=indicators.to(device),
        s_n=s_n.to(device),
        s_m=s_m.to(device),
    )


def _eval_flat_mcc(
    model: nn.Module,
    data: dict,
    device: torch.device,
    batch_size: int = 256,
) -> float:
    """Evaluate → val MCC (dùng trong grid search)."""
    if not data or len(data.get("label", [])) == 0:
        return 0.0
    model.eval()
    loader = DataLoader(FlatDataset(data), batch_size=batch_size, shuffle=False)
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            logits = _forward_flat(model, batch, device)
            preds_all.extend(logits.argmax(1).cpu().numpy())
            labels_all.extend(batch[3].numpy())
    if len(set(labels_all)) < 2:
        return 0.0
    return float(matthews_corrcoef(labels_all, preds_all))


def _eval_flat_both(
    model: nn.Module,
    data: dict,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[float, float]:
    """Evaluate → (acc, MCC) (dùng trong final eval)."""
    if not data or len(data.get("label", [])) == 0:
        return 0.0, 0.0
    model.eval()
    loader = DataLoader(FlatDataset(data), batch_size=batch_size, shuffle=False)
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in loader:
            logits = _forward_flat(model, batch, device)
            preds_all.extend(logits.argmax(1).cpu().numpy())
            labels_all.extend(batch[3].numpy())
    if len(set(labels_all)) < 2:
        return float(accuracy_score(labels_all, preds_all)), 0.0
    return (
        float(accuracy_score(labels_all, preds_all)),
        float(matthews_corrcoef(labels_all, preds_all)),
    )


def _train_one_combo(
    model: nn.Module,
    train_data: dict,
    val_data: dict,
    hparams: dict,
    device: torch.device,
    max_epochs: int = 100,
    patience: int = 20,
    batch_size: int = 256,
) -> float:
    """
    Train 1 hparam combo, trả về best val_MCC.
    Early stopping sau epoch 30.
    """
    model = model.to(device)
    opt   = torch.optim.Adam(
        model.parameters(),
        lr=hparams["lr"],
        weight_decay=1e-5,
        eps=1e-6,
    )
    criterion = nn.CrossEntropyLoss()
    loader    = DataLoader(
        FlatDataset(train_data), batch_size=batch_size, shuffle=True, drop_last=False
    )

    best_mcc, no_improve = -2.0, 0

    for epoch in range(max_epochs):
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            logits = _forward_flat(model, batch, device)
            loss   = criterion(logits, batch[3].long().to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch < 30:
            continue

        mcc = _eval_flat_mcc(model, val_data, device, batch_size)
        if mcc > best_mcc:
            best_mcc   = mcc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_mcc


# ─────────────────────────────────────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_flat(
    model_factory: Callable,
    train_hval_data: dict,
    val_hval_data: dict,
    macro_dim: int,
    grid: Dict[str, List] = None,
    device: torch.device = None,
    verbose: bool = True,
    seed: int = 42,
) -> Dict:
    """
    Grid search cho flat baselines.

    Parameters
    ----------
    model_factory : fn(hidden_dim=int, macro_dim=int, dropout=float) → nn.Module
                    Phải nhận ĐÚNG 3 keyword args này.
    train_hval_data, val_hval_data : splits cho hparam search (không chạm outer test)
    macro_dim : lấy từ data["macro_dim"]

    Returns
    -------
    {"best_hparams": dict, "best_mcc": float, "all_results": list}
    """
    device  = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    combos  = get_all_combinations(grid)
    results = []
    best_mcc, best_hparams = -2.0, None

    torch.manual_seed(seed)
    np.random.seed(seed)

    for i, hp in enumerate(combos):
        # ── Khởi tạo model với keyword args (tên cố định) ─────────────────────
        model = model_factory(
            hidden_dim=hp["hidden_dim"],
            macro_dim=macro_dim,
            dropout=hp["dropout"],
        )
        mcc = _train_one_combo(model, train_hval_data, val_hval_data, hp, device)
        results.append({**hp, "val_mcc": mcc})

        is_best = mcc > best_mcc
        if verbose:
            mark = " ←best" if is_best else ""
            print(f"    [{i+1:02d}/{len(combos)}] lr={hp['lr']:.0e} "
                  f"dim={hp['hidden_dim']:3d} drop={hp['dropout']:.1f} "
                  f"→ val_MCC={mcc:.4f}{mark}")

        if is_best:
            best_mcc, best_hparams = mcc, hp.copy()

    return {"best_hparams": best_hparams, "best_mcc": best_mcc, "all_results": results}


# ─────────────────────────────────────────────────────────────────────────────
# FINAL 5-SEED EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def final_eval_flat(
    model_factory: Callable,
    train_data: dict,
    test_data: dict,
    best_hparams: dict,
    macro_dim: int,
    n_seeds: int = 5,
    max_epochs: int = 200,
    batch_size: int = 256,
    device: torch.device = None,
    verbose: bool = True,
) -> Dict:
    """
    Retrain với best_hparams × n_seeds, evaluate trên outer test.

    Parameters
    ----------
    model_factory : fn(hidden_dim=int, macro_dim=int, dropout=float) → nn.Module
    train_data    : FULL inner range [0:inner_T]
    test_data     : outer test [inner_T:global_T_max]
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_list, mcc_list = [], []

    for seed in SEEDS[:n_seeds]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # ── Khởi tạo model với keyword args ──────────────────────────────────
        model = model_factory(
            hidden_dim=best_hparams["hidden_dim"],
            macro_dim=macro_dim,
            dropout=best_hparams["dropout"],
        ).to(device)

        opt   = torch.optim.Adam(
            model.parameters(),
            lr=best_hparams["lr"],
            weight_decay=1e-5,
            eps=1e-6,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max_epochs, eta_min=1e-6,
        )
        criterion = nn.CrossEntropyLoss()
        loader    = DataLoader(
            FlatDataset(train_data), batch_size=batch_size, shuffle=True, drop_last=False
        )

        for epoch in range(max_epochs):
            model.train()
            for batch in loader:
                opt.zero_grad(set_to_none=True)
                logits = _forward_flat(model, batch, device)
                loss   = criterion(logits, batch[3].long().to(device))
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            sched.step()

        acc, mcc = _eval_flat_both(model, test_data, device, batch_size)
        acc_list.append(acc)
        mcc_list.append(mcc)

        if verbose:
            print(f"    Seed {seed}: ACC={acc:.4f}  MCC={mcc:.4f}")

    return {
        "acc_mean": float(np.mean(acc_list)),
        "acc_std":  float(np.std(acc_list)),
        "mcc_mean": float(np.mean(mcc_list)),
        "mcc_std":  float(np.std(mcc_list)),
        "acc_list": acc_list,
        "mcc_list": mcc_list,
    }