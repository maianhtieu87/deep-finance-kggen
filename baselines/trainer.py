# baselines/trainer.py
"""
BaselineTrainer — Trainer chung cho tất cả baseline models.

Features:
  - Chạy N_RUNS lần với random seeds khác nhau → báo mean ± std
  - Cùng hyperparameters với paper MSGCA (lr=1e-4, warmup, Adam)
  - Evaluate bằng ACC + MCC (sklearn)
  - Hỗ trợ cả flat-input baselines lẫn MSGCA-NAF (cần graph list)
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, matthews_corrcoef


# ──────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def build_loader(data_dict: dict, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Tạo DataLoader từ data_dict chứa các Tensor.
    Chỉ sử dụng cho flat-input baselines (không phải MSGCA-NAF).
    """
    keys   = ["indicators", "s_news_per_day", "s_graph_emb", "s_m", "label"]
    arrays = [data_dict[k] for k in keys]
    ds     = TensorDataset(*arrays)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ──────────────────────────────────────────────────────────────────────
# Flat baseline: models nhận tensor trực tiếp
# ──────────────────────────────────────────────────────────────────────

def train_flat_baseline(
    model: nn.Module,
    train_data: dict,
    valid_data: dict,
    test_data: dict,
    epochs: int       = 200,
    lr: float         = 1e-4,
    batch_size: int   = 1024,
    warmup_epochs: int = 10,
    device: torch.device = None,
) -> tuple:
    """
    Train một flat baseline (LSTM, ALSTM, ESTIMATE, DTML, ALSTM-W, SLOT, LLM-Stock).

    Returns
    -------
    (test_acc, test_mcc) : float, float
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    criterion  = nn.CrossEntropyLoss()
    train_loader = build_loader(train_data, batch_size, shuffle=True)

    best_val_mcc = -2.0
    best_state   = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            indicators, s_news, s_graph, s_m, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            logits = model(
                indicators    = indicators,
                s_news_per_day = s_news,
                s_graph_emb   = s_graph,
                s_m           = s_m,
            )
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch < warmup_epochs:
            scheduler.step()

        # Validate
        val_acc, val_mcc = eval_flat_baseline(model, valid_data, batch_size, device)
        if val_mcc > best_val_mcc and epoch >= 50:
            best_val_mcc = val_mcc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # Test với best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return eval_flat_baseline(model, test_data, batch_size, device)


def eval_flat_baseline(
    model: nn.Module,
    data_dict: dict,
    batch_size: int = 512,
    device: torch.device = None,
) -> tuple:
    """Evaluate flat baseline → (acc, mcc)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not data_dict or len(data_dict.get("label", [])) == 0:
        return 0.0, 0.0

    model.eval()
    loader = build_loader(data_dict, batch_size, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            indicators, s_news, s_graph, s_m, labels = [b.to(device) for b in batch]
            logits = model(
                indicators     = indicators,
                s_news_per_day = s_news,
                s_graph_emb    = s_graph,
                s_m            = s_m,
            )
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    try:
        mcc = matthews_corrcoef(all_labels, all_preds)
    except Exception:
        mcc = 0.0
    return acc, mcc


# ──────────────────────────────────────────────────────────────────────
# MSGCA-NAF: cần graph list (PyG Data objects)
# ──────────────────────────────────────────────────────────────────────

def train_msgca_naf(
    model,
    train_data_full: dict,   # output của data_prepare.prepare_data (gốc)
    valid_data_full: dict,
    test_data_full: dict,
    epochs: int       = 200,
    lr: float         = 1e-4,
    batch_size: int   = 128,
    device: torch.device = None,
) -> tuple:
    """
    Train MSGCA-NAF model (dùng s_n_graphs — giống main.py).
    Tái sử dụng collate_fn và StockGraphDataset từ main.py.
    """
    from main import StockGraphDataset, collate_graph_batch, evaluate as eval_main

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=10
    )
    criterion  = nn.CrossEntropyLoss()

    train_ds     = StockGraphDataset(train_data_full)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_graph_batch, drop_last=False,
    )

    best_val_mcc = -2.0
    best_state   = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(
                batch["s_o"].to(device),
                batch["s_h"].to(device),
                batch["s_c"].to(device),
                batch["s_m"].to(device),
                batch["s_n_graphs"],
                batch["label"].to(device),
                mode="train",
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if epoch < 10:
            scheduler.step()

        val_acc, val_mcc = eval_main(model, valid_data_full)
        if val_mcc > best_val_mcc and epoch >= 50:
            best_val_mcc = val_mcc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, test_mcc = eval_main(model, test_data_full)
    return test_acc, test_mcc


# ──────────────────────────────────────────────────────────────────────
# Multi-seed runner
# ──────────────────────────────────────────────────────────────────────

SEEDS = [42, 123, 256, 512, 1024]


def run_baseline_multi_seed(
    model_factory,           # callable() → nn.Module (mới mỗi run)
    train_data: dict,
    valid_data: dict,
    test_data: dict,
    model_name: str   = "Model",
    n_runs: int       = 5,
    epochs: int       = 200,
    lr: float         = 1e-4,
    batch_size: int   = 1024,
    device: torch.device = None,
    is_naf: bool      = False,   # True nếu là MSGCA-NAF
) -> dict:
    """
    Chạy baseline N lần với random seeds khác nhau.

    Returns
    -------
    dict với keys:
      acc_mean, acc_std, mcc_mean, mcc_std,
      acc_list, mcc_list
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    acc_list, mcc_list = [], []
    seeds = SEEDS[:n_runs]

    for run_idx, seed in enumerate(seeds):
        set_seed(seed)
        model = model_factory()

        if is_naf:
            acc, mcc = train_msgca_naf(
                model, train_data, valid_data, test_data,
                epochs=epochs, lr=lr, batch_size=batch_size, device=device,
            )
        else:
            acc, mcc = train_flat_baseline(
                model, train_data, valid_data, test_data,
                epochs=epochs, lr=lr, batch_size=batch_size, device=device,
            )

        acc_list.append(acc)
        mcc_list.append(mcc)
        print(f"  [{model_name}] Run {run_idx+1}/{n_runs} | ACC={acc:.4f} | MCC={mcc:.4f}")

    result = {
        "acc_mean": float(np.mean(acc_list)),
        "acc_std":  float(np.std(acc_list)),
        "mcc_mean": float(np.mean(mcc_list)),
        "mcc_std":  float(np.std(mcc_list)),
        "acc_list": acc_list,
        "mcc_list": mcc_list,
    }
    print(
        f"  ✅ [{model_name}] ACC: {result['acc_mean']:.4f}±{result['acc_std']:.4f} | "
        f"MCC: {result['mcc_mean']:.4f}±{result['mcc_std']:.4f}"
    )
    return result