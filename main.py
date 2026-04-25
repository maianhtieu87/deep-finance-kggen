# main.py
"""
Training script — Standard Time Series Split (Train/Val/Test)

Phase 2 changes:
  [PHASE2-A] StockDataset._BASE_KEYS: added "news_quality"
    - news_quality is always present in data_loader V5.7 output
      (zeros when no quality file; real 4D vectors when available)

  [PHASE2-B] evaluate(): pass news_quality from batch to model
    - batch.get("news_quality") → .to(device) or None

  [PHASE2-C] train_and_evaluate():
    - StockMovementModel instantiation: pass quality_dim from GlobalConfig
    - Modality dropout: quality_in=None when dropping news
      (conceptually clean: full news drop = zero embeddings + no quality signal)
    - Training loss call: pass quality_in to model

Previous fixes retained:
  [FIX-1] CosineAnnealingLR (no restarts)
  [FIX-2] AdamW with separate param groups (no weight_decay on LN/bias)
  [FIX-3] News Modality Dropout
  [FIX-4] Early Stopping with patience
"""

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef

from src.model import StockMovementModel
from src.data_loader import data_prepare, N_TICKERS
from configs.config import TrainConfig, GlobalConfig

# ── Shared hyperparams ────────────────────────────────────────────────────────
NEWS_MODALITY_DROPOUT = TrainConfig.news_modality_dropout   # default 0.30
EARLY_STOP_PATIENCE   = TrainConfig.early_stop_patience     # default 30


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


device = torch.device(
    "cuda" if TrainConfig.use_cuda and torch.cuda.is_available() else "cpu"
)


class StockDataset(Dataset):
    # [PHASE2-A] "news_quality" added — always present from data_loader V5.7
    _BASE_KEYS = [
        "s_o", "s_h", "s_c", "s_m", "s_n",
        "news_mask", "label",
        "news_quality",   # (T, window_size, quality_dim) — zeros if no quality file
    ]

    def __init__(self, d: dict):
        self.d    = d
        self.keys = list(self._BASE_KEYS)
        if "ticker_id" in d:
            self.keys.append("ticker_id")
        # Guard: drop keys not present in d (backward-compat with old PKL)
        self.keys = [k for k in self.keys if k in d]

    def __len__(self) -> int:
        return len(self.d["label"])

    def __getitem__(self, i: int) -> dict:
        return {k: self.d[k][i] for k in self.keys}


def merge_datasets(dicts: list, shuffle: bool = False) -> dict:
    valid_dicts = [d for d in dicts if d and "label" in d and len(d["label"]) > 0]
    if not valid_dicts:
        return {}
    merged = {}
    all_keys = set(valid_dicts[0].keys())
    for key in all_keys:
        parts = [d[key] for d in valid_dicts if key in d]
        if parts:
            merged[key] = torch.cat(parts, dim=0)
    if shuffle and "label" in merged:
        idx = torch.randperm(len(merged["label"]))
        for k in merged:
            merged[k] = merged[k][idx]
    return merged


def compute_class_weights(labels: torch.Tensor, suppress_print=False) -> torch.Tensor:
    lbl  = labels.cpu().numpy()
    cnts = np.bincount(lbl, minlength=3)
    beta = 0.9999
    eff  = 1.0 - np.power(beta, cnts)
    w    = (1.0 - beta) / (eff + 1e-8)
    w    = np.sqrt(w / w.sum() * 3)
    w    = w / w.sum() * 3
    wt   = torch.tensor(w, dtype=torch.float32)
    if not suppress_print:
        print("  Class Weights:")
        for i, (cls, wi) in enumerate(zip(["DOWN", "FLAT", "UP"], w)):
            print(f"    {cls}: count={int(cnts[i])}  weight={wi:.4f}")
    return wt


def _batch_quality(batch: dict) -> "torch.Tensor | None":
    """
    Extract news_quality from batch safely.
    Returns tensor on CPU (caller moves to device), or None if absent.
    """
    q = batch.get("news_quality")
    return q if (q is not None and isinstance(q, torch.Tensor)) else None


def evaluate(model: StockMovementModel, data_dict: dict) -> tuple:
    if not data_dict or len(data_dict.get("label", [])) == 0:
        return 0.0, 0.0
    ds  = StockDataset(data_dict)
    ldr = DataLoader(ds, batch_size=64, shuffle=False)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in ldr:
            # [PHASE2-B] pass news_quality from batch
            q = _batch_quality(batch)
            _, _, preds = model(
                batch["s_o"].to(device),
                batch["s_h"].to(device),
                batch["s_c"].to(device),
                batch["s_m"].to(device),
                batch["s_n"].to(device),
                batch["label"].to(device),
                mode="test",
                return_preds=True,
                ticker_id=batch.get("ticker_id"),
                news_mask=batch["news_mask"].to(device),
                news_quality=q.to(device) if q is not None else None,  # [PHASE2-B]
            )
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
    return (accuracy_score(all_labels, all_preds),
            matthews_corrcoef(all_labels, all_preds))


def _make_optimizer(model: StockMovementModel, lr: float, weight_decay: float):
    """
    [FIX-2] AdamW với separate param groups: LayerNorm + bias không có weight_decay.
    """
    no_decay_keywords = [
        "bias", "LayerNorm.weight", "layernorm.weight",
        "norm.weight", "attn_norm.weight", "out_norm.weight",
    ]
    decay_params    = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(kw in name for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def train_and_evaluate(
    train_data: dict, val_data: dict, test_data: dict,
    include_ticker_id: bool, max_epochs: int, save_path: str,
):
    print(f"\n{'-'*60}")
    print(f"TRAINING (Epochs: {max_epochs}, "
          f"N_Train={len(train_data['label'])}, "
          f"N_Val={len(val_data['label'])}, "
          f"N_Test={len(test_data['label'])})")
    print(f"  News Modality Dropout : {NEWS_MODALITY_DROPOUT:.0%} of steps")
    print(f"  Early Stop Patience   : {EARLY_STOP_PATIENCE} epochs")

    s_m_dim = train_data["s_m"].shape[-1]
    s_n_dim = train_data["s_n"].shape[-1]
    cw      = compute_class_weights(train_data["label"]).to(device)

    # [PHASE2-C] Read quality_dim from config.
    # Default MUST match data_loader.py default (both use 4).
    # data_loader builds quality tensor with dim=4 regardless of config,
    # so model must also be built with dim=4 to avoid shape mismatch.
    quality_dim = getattr(GlobalConfig, "QUALITY_DIM", 4)
    print(f"  Quality dim           : {quality_dim}D "
          f"({'pipeline' if quality_dim > 1 else 'norm proxy'})")

    ds_train  = StockDataset(train_data)
    ldr_train = DataLoader(
        ds_train,
        batch_size=getattr(TrainConfig, "batch_size", 32),
        shuffle=True,
        drop_last=False,
    )

    # [PHASE2-C] pass quality_dim so MultimodalSourceEncoding → NewsEncoder built correctly
    model = StockMovementModel(
        price_dim=1, macro_dim=s_m_dim, news_dim=s_n_dim,
        dim=TrainConfig.dim, input_dim=TrainConfig.window_size,
        output_dim=3, num_head=TrainConfig.num_head, dropout=0.1,
        class_weights=cw,
        use_focal_loss=getattr(TrainConfig, "use_focal_loss", True),
        focal_gamma=getattr(TrainConfig, "focal_gamma", 2.0),
        device=device, n_tickers=N_TICKERS,
        quality_dim=quality_dim,   # [PHASE2-C]
    ).to(device)

    optimizer = _make_optimizer(
        model,
        lr=getattr(TrainConfig, "learning_rate", 1e-4),
        weight_decay=getattr(TrainConfig, "weight_decay", 1e-4),
    )

    # [FIX-1] CosineAnnealingLR — no restarts, monotone decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=1e-6
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    best_val_mcc     = -1.0
    best_epoch       = 0
    patience_counter = 0

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0.0

        for batch in ldr_train:
            optimizer.zero_grad()

            # ── [FIX-3] News Modality Dropout ────────────────────────────────
            # [PHASE2-C] When dropping news: also set quality_in=None so
            # NewsEncoder falls back to norm proxy (norm of zero vector = 0 → gate≈0).
            # Keeps the conceptual contract: "no news" = zero embedding + no quality.
            if torch.rand(1).item() < NEWS_MODALITY_DROPOUT:
                s_n_in     = torch.zeros_like(batch["s_n"])
                mask_in    = torch.ones(batch["news_mask"].shape, dtype=torch.bool)
                quality_in = None                       # drop quality together with news
            else:
                s_n_in     = batch["s_n"]
                mask_in    = batch["news_mask"]
                quality_in = _batch_quality(batch)      # real quality or None

            loss = model(
                batch["s_o"].to(device),
                batch["s_h"].to(device),
                batch["s_c"].to(device),
                batch["s_m"].to(device),
                s_n_in.to(device),
                batch["label"].to(device),
                mode="train",
                ticker_id=(
                    batch["ticker_id"]
                    if include_ticker_id and "ticker_id" in batch
                    else None
                ),
                news_mask=mask_in.to(device),
                news_quality=(                          # [PHASE2-C]
                    quality_in.to(device) if quality_in is not None else None
                ),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        val_acc, val_mcc = evaluate(model, val_data)

        if (epoch + 1) % 5 == 0 or epoch == max_epochs - 1:
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Ep {epoch+1:03d}/{max_epochs} | "
                  f"Loss {total_loss/len(ldr_train):.4f} | "
                  f"Val ACC: {val_acc:.4f} | Val MCC: {val_mcc:.4f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Patience: {patience_counter}/{EARLY_STOP_PATIENCE}")

        # ── [FIX-4] Early Stopping ────────────────────────────────────────────
        if val_mcc > best_val_mcc:
            best_val_mcc     = val_mcc
            best_epoch       = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"\n  [Early Stop] Val MCC không cải thiện sau "
                      f"{EARLY_STOP_PATIENCE} epochs. Dừng tại epoch {epoch+1}.")
                break

    print(f"\n{'-'*60}")
    print(f"LOADING BEST MODEL (Epoch {best_epoch} - Val MCC: {best_val_mcc:.4f})")

    model.load_state_dict(torch.load(save_path))
    test_acc, test_mcc = evaluate(model, test_data)

    print(f"  Test ACC : {test_acc:.4f}")
    print(f"  Test MCC : {test_mcc:.4f}")
    print(f"{'-'*60}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl",          default=None)
    ap.add_argument("--price-mode",   default="vol_adjusted",
                    choices=["vol_adjusted", "pct_first", "absolute"])
    ap.add_argument("--label-mode",   default="rolling",
                    choices=["rolling", "fixed", "volatility"])
    ap.add_argument("--tickers",      nargs="+", default=None)
    ap.add_argument("--no-ticker-id", action="store_true")
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--epochs",       type=int, default=200)
    args = ap.parse_args()

    set_seed(args.seed)

    pkl = args.pkl or os.path.join(
        GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl"
    )
    if not os.path.exists(pkl):
        print(f"PKL not found: {pkl}")
        return

    tickers     = ([t.upper() for t in args.tickers]
                   if args.tickers else GlobalConfig.TICKERS)
    include_tid = not args.no_ticker_id

    dp = data_prepare(
        pkl,
        price_mode=args.price_mode,
        label_mode=args.label_mode,
        include_ticker_id=include_tid,
    )

    valid_T      = [dp.get_max_T(t) for t in tickers if dp.get_max_T(t) > 0]
    if not valid_T:
        print("No valid data found.")
        return
    global_T_max = min(valid_T)

    train_ratio = getattr(TrainConfig, "train_ratio", 0.7)
    valid_ratio = getattr(TrainConfig, "valid_ratio", 0.15)
    train_end   = int(global_T_max * train_ratio)
    val_end     = int(global_T_max * (train_ratio + valid_ratio))

    print(f"\n{'-'*60}")
    print(f"TIME SERIES SPLIT (T_max={global_T_max})")
    print(f"  Train : [0:{train_end}]  ({train_ratio*100:.0f}%)")
    print(f"  Val   : [{train_end}:{val_end}]  ({valid_ratio*100:.0f}%)")
    print(f"  Test  : [{val_end}:{global_T_max}]  (15%)")
    print(f"{'-'*60}")

    list_train, list_val, list_test = [], [], []
    for t in tickers:
        tr, va, te = dp.prepare_data(
            t, train_end=train_end, val_end=val_end, test_end=global_T_max
        )
        if tr and len(tr.get("label", [])) > 0: list_train.append(tr)
        if va and len(va.get("label", [])) > 0: list_val.append(va)
        if te and len(te.get("label", [])) > 0: list_test.append(te)

    train_data = merge_datasets(list_train, shuffle=True)
    val_data   = merge_datasets(list_val,   shuffle=False)
    test_data  = merge_datasets(list_test,  shuffle=False)

    tag = (f"label={args.label_mode}_price={args.price_mode}"
           f"{'_notid' if args.no_ticker_id else '_tid'}"
           f"_seed{args.seed}_fixed")
    save_path = f"output/best_model_{tag}.pt"

    train_and_evaluate(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        include_ticker_id=include_tid,
        max_epochs=args.epochs,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()