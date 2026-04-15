# baselines/trainer.py
"""
Unified trainer cho flat baselines và MSGCA variants.

CHANGES vs previous:
  - evaluate_msgca: pass news_mask + ticker_id vào model.forward()
  - train_msgca_variant: pass news_mask + ticker_id vào model.forward()
  (Model đã được fix sequential fusion + news_mask bug, trainer phải khớp)
"""

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef
from typing import Callable, Dict, List, Optional, Tuple


SEEDS = [42, 123, 256, 512, 1024]

FLAT_OPTIMIZER_CLS  = torch.optim.Adam
MSGCA_OPTIMIZER_CLS = torch.optim.Adam


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ─────────────────────────────────────────────────────────────────────────────

class FlatDataset(TensorDataset):
    def __init__(self, data: dict):
        super().__init__(
            data["indicators"],
            data["s_n"],
            data["s_m"],
            data["label"],
        )


class MSGCADataset(Dataset):
    def __init__(self, data: dict):
        self.data = data
        self.keys = [k for k in data if isinstance(data[k], torch.Tensor)]

    def __len__(self) -> int:
        return len(self.data["label"])

    def __getitem__(self, idx: int) -> dict:
        return {k: self.data[k][idx] for k in self.keys}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_flat(
    model: nn.Module,
    data: dict,
    batch_size: int = 512,
    device=None,
) -> Tuple[float, float]:
    if not data or len(data.get("label", [])) == 0:
        return 0.0, 0.0

    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(FlatDataset(data), batch_size=batch_size, shuffle=False)
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
def evaluate_msgca(
    model: nn.Module,
    data: dict,
    batch_size: int = 64,
    device=None,
) -> Tuple[float, float]:
    """
    Evaluate MSGCA-style model → (acc, mcc).

    FIX: truyền news_mask và ticker_id vào model.forward().
    Model đã được fix sequential fusion + news_mask, trainer phải khớp.
    """
    if not data or len(data.get("label", [])) == 0:
        return 0.0, 0.0

    device = device or next(model.parameters()).device
    model.eval()
    loader = DataLoader(MSGCADataset(data), batch_size=batch_size, shuffle=False)
    preds_all, labels_all = [], []

    for batch in loader:
        result = model(
            batch["s_o"].to(device),
            batch["s_h"].to(device),
            batch["s_c"].to(device),
            batch["s_m"].to(device),
            batch["s_n"].to(device),
            batch["label"].to(device),
            mode="test",
            return_preds=True,
            ticker_id=batch.get("ticker_id"),          # FIX: pass ticker_id
            news_mask=batch.get("news_mask"),           # FIX: pass news_mask
        )
        # StockMovementModel returns (acc, mcc, preds); ablations return preds only
        if isinstance(result, tuple):
            preds = result[2]
        else:
            preds = result
        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(batch["label"].numpy())

    acc = accuracy_score(labels_all, preds_all)
    mcc = matthews_corrcoef(labels_all, preds_all) if len(set(labels_all)) > 1 else 0.0
    return acc, mcc


# ─────────────────────────────────────────────────────────────────────────────
# Training functions
# ─────────────────────────────────────────────────────────────────────────────

def train_flat(
    model: nn.Module,
    train_data: dict,
    valid_data: dict,
    test_data: dict,
    epochs: int = 200,
    lr: float = 1e-4,
    batch_size: int = 512,
    warmup_epochs: int = 10,
    weight_decay: float = 1e-5,
    patience: int = 30,
    device=None,
    verbose: bool = False,
) -> Tuple[float, float]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt = FLAT_OPTIMIZER_CLS(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-6,
    )
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
    )
    criterion = nn.CrossEntropyLoss()
    loader    = DataLoader(FlatDataset(train_data), batch_size=batch_size, shuffle=True)

    best_mcc, best_state = -2.0, None
    epochs_no_improve    = 0

    for epoch in range(epochs):
        model.train()
        for indicators, s_n, s_m, labels in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(
                indicators=indicators.to(device),
                s_n=s_n.to(device),
                s_m=s_m.to(device),
            )
            loss = criterion(logits, labels.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch < warmup_epochs:
            sched.step()

        if epoch >= 40:
            _, val_mcc = evaluate_flat(model, valid_data, batch_size, device)
            if val_mcc > best_mcc:
                best_mcc   = val_mcc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"      Early stop at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 50 == 0:
                print(f"      epoch {epoch+1}: val_mcc={val_mcc:.4f}")

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
    weight_decay: float = 1e-4,
    patience: int = 30,
    device=None,
    verbose: bool = False,
    optimizer_cls=None,
) -> Tuple[float, float]:
    """
    Train MSGCA-style model.

    FIX: truyền news_mask và ticker_id vào model.forward() ở cả train lẫn eval.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    opt_cls = optimizer_cls or MSGCA_OPTIMIZER_CLS
    opt = opt_cls(
        model.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-6,
    )
    sched = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
    )
    loader = DataLoader(
        MSGCADataset(train_data), batch_size=batch_size, shuffle=True,
    )

    best_mcc, best_state = -2.0, None
    epochs_no_improve    = 0

    for epoch in range(epochs):
        model.train()
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss = model(
                batch["s_o"].to(device),
                batch["s_h"].to(device),
                batch["s_c"].to(device),
                batch["s_m"].to(device),
                batch["s_n"].to(device),
                batch["label"].to(device),
                mode="train",
                ticker_id=batch.get("ticker_id"),      # FIX: pass ticker_id
                news_mask=batch.get("news_mask"),       # FIX: pass news_mask
            )
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if epoch < warmup_epochs:
            sched.step()

        if epoch >= 40:
            _, val_mcc = evaluate_msgca(model, valid_data, batch_size, device)
            if val_mcc > best_mcc:
                best_mcc   = val_mcc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    if verbose:
                        print(f"      Early stop at epoch {epoch + 1}")
                    break

            if verbose and (epoch + 1) % 50 == 0:
                print(f"      epoch {epoch+1}: val_mcc={val_mcc:.4f}")

    if best_state:
        model.load_state_dict(best_state)

    return evaluate_msgca(model, test_data, batch_size, device)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed runner
# ─────────────────────────────────────────────────────────────────────────────

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
    batch_size: Optional[int] = None,
    device=None,
    verbose: bool = False,
    optimizer_cls=None,
) -> dict:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if batch_size is None:
        batch_size = 32 if is_msgca_style else 512

    train_fn = train_msgca_variant if is_msgca_style else train_flat
    acc_list, mcc_list = [], []

    for i, seed in enumerate(SEEDS[:n_runs]):
        set_seed(seed)
        model = model_factory()

        train_kwargs = dict(
            epochs=epochs, lr=lr, batch_size=batch_size,
            device=device, verbose=verbose,
        )
        if is_msgca_style and optimizer_cls is not None:
            train_kwargs["optimizer_cls"] = optimizer_cls

        acc, mcc = train_fn(
            model, train_data, valid_data, test_data,
            **train_kwargs,
        )
        acc_list.append(acc)
        mcc_list.append(mcc)
        print(f"    [{model_name}] Run {i+1}/{n_runs} | ACC={acc:.4f} | MCC={mcc:.4f}")

    result = {
        "acc_mean": float(np.mean(acc_list)),
        "acc_std":  float(np.std(acc_list)),
        "mcc_mean": float(np.mean(mcc_list)),
        "mcc_std":  float(np.std(mcc_list)),
        "acc_list": acc_list,
        "mcc_list": mcc_list,
    }
    print(
        f"  [OK] [{model_name}] "
        f"ACC={result['acc_mean']:.4f}+/-{result['acc_std']:.4f}  "
        f"MCC={result['mcc_mean']:.4f}+/-{result['mcc_std']:.4f}"
    )
    return result