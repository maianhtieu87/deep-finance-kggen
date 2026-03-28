# baselines/trainer.py
"""
Unified trainer cho tất cả baseline models và MSGCA variants.

Hỗ trợ 2 loại model:
  1. Flat baselines (LSTM, ALSTM, ESTIMATE, DTML, ALSTM-W, SLOT, LLM-Stock):
     - Forward: model(indicators, s_n, s_m, ...) → logits
     - DataLoader: TensorDataset của indicators, s_n, s_m, label

  2. MSGCA-style models (MSGCANoGate, MSGCAWithGLU, StockMovementModel):
     - Forward: model(s_o, s_h, s_c, s_m, s_n, label, mode=...) → loss/metrics
     - DataLoader: giống StockDataset trong main.py

Hyperparameters khớp với paper MSGCA (Section 5.1):
  lr=1e-4, epochs=200, warmup=10, Adam, clip_grad=1.0
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef
from typing import Callable, Optional


# ──────────────────────────────────────────────────────────────────────
# Dataset classes
# ──────────────────────────────────────────────────────────────────────

class FlatDataset(TensorDataset):
    """Dataset cho flat baselines — trả về (indicators, s_n, s_m, label)."""

    def __init__(self, data: dict):
        super().__init__(
            data["indicators"],  # (N, W, 3)
            data["s_n"],         # (N, W, 1024)
            data["s_m"],         # (N, W, M)
            data["label"],       # (N,)
        )


class MSGCADataset(Dataset):
    """Dataset cho MSGCA-style models — trả về dict."""

    def __init__(self, data: dict):
        self.data = data

    def __len__(self):
        return len(self.data["label"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# ──────────────────────────────────────────────────────────────────────
# Seeds
# ──────────────────────────────────────────────────────────────────────

SEEDS = [42, 123, 256, 512, 1024]

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_flat(model: nn.Module, data: dict, batch_size: int = 512,
                  device=None) -> tuple[float, float]:
    """Evaluate flat baseline → (acc, mcc)."""
    if not data or len(data.get("label", [])) == 0:
        return 0.0, 0.0
    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(FlatDataset(data), batch_size=batch_size)
    preds_all, labels_all = [], []
    for indicators, s_n, s_m, labels in loader:
        logits = model(
            indicators=indicators.to(device),
            s_n=s_n.to(device),
            s_m=s_m.to(device),
        )
        preds_all.extend(logits.argmax(1).cpu().numpy())
        labels_all.extend(labels.numpy())
    acc = accuracy_score(labels_all, preds_all)
    mcc = matthews_corrcoef(labels_all, preds_all) if len(set(labels_all)) > 1 else 0.0
    return acc, mcc


@torch.no_grad()
def evaluate_msgca(model: nn.Module, data: dict, batch_size: int = 64,
                   device=None) -> tuple[float, float]:
    """Evaluate MSGCA-style model → (acc, mcc)."""
    if not data or len(data.get("label", [])) == 0:
        return 0.0, 0.0
    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(MSGCADataset(data), batch_size=batch_size)
    preds_all, labels_all = [], []
    for batch in loader:
        preds = model(
            batch["s_o"].to(device), batch["s_h"].to(device),
            batch["s_c"].to(device), batch["s_m"].to(device),
            batch["s_n"].to(device), batch["label"].to(device),
            mode="test", return_preds=True,
        )
        if isinstance(preds, tuple):  # (acc, mcc, preds)
            preds = preds[2]
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(batch["label"].numpy())
    acc = accuracy_score(labels_all, preds_all)
    mcc = matthews_corrcoef(labels_all, preds_all) if len(set(labels_all)) > 1 else 0.0
    return acc, mcc


# ──────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────

def train_flat(
    model: nn.Module,
    train_data: dict,
    valid_data: dict,
    test_data: dict,
    epochs: int = 200,
    lr: float = 1e-4,
    batch_size: int = 512,
    warmup_epochs: int = 10,
    device=None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Train flat baseline và return (test_acc, test_mcc)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(FlatDataset(train_data), batch_size=batch_size, shuffle=True)

    best_mcc, best_state = -2.0, None
    for epoch in range(epochs):
        model.train()
        for indicators, s_n, s_m, labels in loader:
            opt.zero_grad()
            logits = model(
                indicators=indicators.to(device),
                s_n=s_n.to(device),
                s_m=s_m.to(device),
            )
            criterion(logits, labels.to(device)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if epoch < warmup_epochs:
            sched.step()
        if epoch >= 40:
            _, val_mcc = evaluate_flat(model, valid_data, batch_size, device)
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if verbose and (epoch + 1) % 50 == 0:
                print(f"    epoch {epoch+1}: val_mcc={val_mcc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return evaluate_flat(model, test_data, batch_size, device)


def train_msgca_variant(
    model: nn.Module,
    train_data: dict,
    valid_data: dict,
    test_data: dict,
    epochs: int = 200,
    lr: float = 1e-4,
    batch_size: int = 32,
    warmup_epochs: int = 10,
    device=None,
    verbose: bool = False,
) -> tuple[float, float]:
    """Train MSGCA-style model (MSGCANoGate, MSGCAWithGLU) và return (test_acc, test_mcc)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    loader = DataLoader(MSGCADataset(train_data), batch_size=batch_size, shuffle=True)

    best_mcc, best_state = -2.0, None
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            opt.zero_grad()
            loss = model(
                batch["s_o"].to(device), batch["s_h"].to(device),
                batch["s_c"].to(device), batch["s_m"].to(device),
                batch["s_n"].to(device), batch["label"].to(device),
                mode="train",
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if epoch < warmup_epochs:
            sched.step()
        if epoch >= 40:
            _, val_mcc = evaluate_msgca(model, valid_data, batch_size, device)
            if val_mcc > best_mcc:
                best_mcc = val_mcc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            if verbose and (epoch + 1) % 50 == 0:
                print(f"    epoch {epoch+1}: val_mcc={val_mcc:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return evaluate_msgca(model, test_data, batch_size, device)


# ──────────────────────────────────────────────────────────────────────
# Multi-seed runner
# ──────────────────────────────────────────────────────────────────────

def run_multi_seed(
    model_factory: Callable[[], nn.Module],
    train_data: dict,
    valid_data: dict,
    test_data: dict,
    model_name: str = "Model",
    is_msgca_style: bool = False,
    n_runs: int = 5,
    epochs: int = 200,
    lr: float = 1e-4,
    batch_size: int = 512,
    device=None,
    verbose: bool = False,
) -> dict:
    """
    Chạy N lần với random seeds khác nhau → mean ± std.

    Parameters
    ----------
    is_msgca_style : bool
        True  → dùng train_msgca_variant (model nhận s_o,s_h,s_c,s_m,s_n,label)
        False → dùng train_flat (model nhận indicators,s_n,s_m)

    Returns
    -------
    dict với acc_mean, acc_std, mcc_mean, mcc_std, acc_list, mcc_list
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_list, mcc_list = [], []
    train_fn = train_msgca_variant if is_msgca_style else train_flat

    for i, seed in enumerate(SEEDS[:n_runs]):
        set_seed(seed)
        model = model_factory()
        acc, mcc = train_fn(
            model, train_data, valid_data, test_data,
            epochs=epochs, lr=lr, batch_size=batch_size, device=device,
            verbose=verbose,
        )
        acc_list.append(acc)
        mcc_list.append(mcc)
        print(f"  [{model_name}] Run {i+1}/{n_runs} | ACC={acc:.4f} | MCC={mcc:.4f}")

    result = {
        "acc_mean": float(np.mean(acc_list)),
        "acc_std":  float(np.std(acc_list)),
        "mcc_mean": float(np.mean(mcc_list)),
        "mcc_std":  float(np.std(mcc_list)),
        "acc_list": acc_list,
        "mcc_list": mcc_list,
    }
    print(
        f"  ✓ [{model_name}] ACC={result['acc_mean']:.4f}±{result['acc_std']:.4f} "
        f"MCC={result['mcc_mean']:.4f}±{result['mcc_std']:.4f}"
    )
    return result