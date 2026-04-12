# main_v3.py
"""
Training script V3 — Walk-Forward Validation Edition
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

# ĐỒNG BỘ IMPORT VỚI FILE CỦA BẠN
from src.model import StockMovementModel
from src.data_loader import data_prepare, N_TICKERS
from configs.config import TrainConfig, GlobalConfig

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
    _BASE_KEYS = ["s_o", "s_h", "s_c", "s_m", "s_n", "news_mask", "label"]

    def __init__(self, d: dict):
        self.d    = d
        self.keys = list(self._BASE_KEYS)
        # [TICKER EMBEDDING LOGIC] Bắt ID từ loader
        if "ticker_id" in d:
            self.keys.append("ticker_id")

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
    lbl   = labels.cpu().numpy()
    cnts  = np.bincount(lbl, minlength=3)
    beta  = 0.9999
    eff   = 1.0 - np.power(beta, cnts)
    w     = (1.0 - beta) / (eff + 1e-8)
    w     = np.sqrt(w / w.sum() * 3)
    w     = w / w.sum() * 3
    wt    = torch.tensor(w, dtype=torch.float32)
    if not suppress_print:
        print("  Class Weights:")
        for i, (cls, wi) in enumerate(zip(["DOWN", "FLAT", "UP"], w)):
            print(f"    {cls}: count={int(cnts[i])}  weight={wi:.4f}")
    return wt


def evaluate(model: StockMovementModel, data_dict: dict) -> tuple:
    if not data_dict or len(data_dict.get("label", [])) == 0:
        return 0.0, 0.0

    ds  = StockDataset(data_dict)
    ldr = DataLoader(ds, batch_size=64, shuffle=False)
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in ldr:
            _, _, preds = model(
                batch["s_o"].to(device),
                batch["s_h"].to(device),
                batch["s_c"].to(device),
                batch["s_m"].to(device),
                batch["s_n"].to(device),
                batch["label"].to(device),
                mode="test",
                return_preds=True,
                # [TICKER EMBEDDING LOGIC] Truyền ID vào khi Test
                ticker_id=batch.get("ticker_id"),
                news_mask=batch["news_mask"].to(device),
            )
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())

    from sklearn.metrics import accuracy_score, matthews_corrcoef
    return (accuracy_score(all_labels, all_preds),
            matthews_corrcoef(all_labels, all_preds))


def train_inner_fold(
    fold_idx: int,
    train_data: dict,
    val_data: dict,
    include_ticker_id: bool,
    max_epochs: int = 60
) -> list:
    print(f"\n[{fold_idx}] INNER FOLD TRAINING (N_Train={len(train_data['label'])}, N_Val={len(val_data['label'])})")
    
    s_m_dim = train_data["s_m"].shape[-1]
    s_n_dim = train_data["s_n"].shape[-1]
    cw = compute_class_weights(train_data["label"], suppress_print=True).to(device)
    
    ds  = StockDataset(train_data)
    ldr = DataLoader(ds, batch_size=getattr(TrainConfig, "batch_size", 32), shuffle=True, drop_last=False)

    model = StockMovementModel(
        price_dim=1, macro_dim=s_m_dim, news_dim=s_n_dim,
        dim=TrainConfig.dim, input_dim=TrainConfig.window_size,
        output_dim=3, num_head=TrainConfig.num_head, dropout=0.1,
        class_weights=cw, use_focal_loss=getattr(TrainConfig, "use_focal_loss", True),
        focal_gamma=getattr(TrainConfig, "focal_gamma", 2.0),
        device=device, n_tickers=N_TICKERS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=getattr(TrainConfig, "learning_rate", 1e-4), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    mcc_history = []
    
    for epoch in range(max_epochs):
        model.train()
        for batch in ldr:
            optimizer.zero_grad()
            loss = model(
                batch["s_o"].to(device), batch["s_h"].to(device), batch["s_c"].to(device),
                batch["s_m"].to(device), batch["s_n"].to(device), batch["label"].to(device),
                mode="train",
                # [TICKER EMBEDDING LOGIC] Truyền ID vào khi Train
                ticker_id=(batch["ticker_id"] if include_ticker_id and "ticker_id" in batch else None),
                news_mask=batch["news_mask"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        _, val_mcc = evaluate(model, val_data)
        mcc_history.append(val_mcc)
        
        if (epoch + 1) % 20 == 0 or epoch == max_epochs - 1:
            print(f"  Fold {fold_idx} | Ep {epoch+1:03d}/{max_epochs} | Val MCC {val_mcc:.4f}")

    return mcc_history


def retrain_final_model(
    train_data: dict,
    test_data: dict,
    best_epochs: int,
    include_ticker_id: bool,
    save_path: str
):
    print(f"\n{'-'*60}\nRETRAINING FINAL MODEL (Epochs: {best_epochs}, N_Train={len(train_data['label'])})")
    
    s_m_dim = train_data["s_m"].shape[-1]
    s_n_dim = train_data["s_n"].shape[-1]
    cw = compute_class_weights(train_data["label"]).to(device)
    
    ds  = StockDataset(train_data)
    ldr = DataLoader(ds, batch_size=getattr(TrainConfig, "batch_size", 32), shuffle=True, drop_last=False)

    model = StockMovementModel(
        price_dim=1, macro_dim=s_m_dim, news_dim=s_n_dim,
        dim=TrainConfig.dim, input_dim=TrainConfig.window_size,
        output_dim=3, num_head=TrainConfig.num_head, dropout=0.1,
        class_weights=cw, use_focal_loss=getattr(TrainConfig, "use_focal_loss", True),
        focal_gamma=getattr(TrainConfig, "focal_gamma", 2.0),
        device=device, n_tickers=N_TICKERS,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=getattr(TrainConfig, "learning_rate", 1e-4), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=1e-6)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    for epoch in range(best_epochs):
        model.train()
        total_loss = 0.0
        for batch in ldr:
            optimizer.zero_grad()
            loss = model(
                batch["s_o"].to(device), batch["s_h"].to(device), batch["s_c"].to(device),
                batch["s_m"].to(device), batch["s_n"].to(device), batch["label"].to(device),
                mode="train",
                # [TICKER EMBEDDING LOGIC] Truyền ID vào khi Retrain
                ticker_id=(batch["ticker_id"] if include_ticker_id and "ticker_id" in batch else None),
                news_mask=batch["news_mask"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Retrain | Ep {epoch+1:03d} | Train Loss {total_loss/len(ldr):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\n  Final model saved to {save_path}")

    ta, tm = evaluate(model, test_data)
    print(f"\n{'-'*60}")
    print(f"FINAL OUTER TEST PERFORMANCE (MÙ HOÀN TOÀN TỪ ĐẦU ĐẾN CUỐI)")
    print(f"  Test ACC : {ta:.4f}")
    print(f"  Test MCC : {tm:.4f}")
    print(f"{'-'*60}\n")


def main():
    ap = argparse.ArgumentParser(description="WF-Train StockMovementModel V3 (P1+P3+P4)")
    ap.add_argument("--pkl",         default=None)
    ap.add_argument("--price-mode",  default="vol_adjusted",
                    choices=["vol_adjusted", "pct_first", "absolute"])
    ap.add_argument("--label-mode",  default="rolling",
                    choices=["rolling", "fixed", "volatility"])
    ap.add_argument("--tickers",     nargs="+", default=None)
    ap.add_argument("--no-ticker-id", action="store_true")
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--wf-folds",    type=int, default=3, help="Số lượng Walk-forward Inner Folds")
    ap.add_argument("--wf-epochs",   type=int, default=60, help="Số max epochs để dò tìm ở Inner Folds")
    args = ap.parse_args()

    set_seed(args.seed)

    pkl = args.pkl or os.path.join(
        GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl"
    )
    if not os.path.exists(pkl):
        print(f"PKL not found: {pkl}")
        return

    tickers = ([t.upper() for t in args.tickers] if args.tickers else GlobalConfig.TICKERS)
    include_tid = not args.no_ticker_id

    dp = data_prepare(
        pkl, price_mode=args.price_mode, label_mode=args.label_mode, include_ticker_id=include_tid,
    )

    valid_T = [dp.get_max_T(t) for t in tickers if dp.get_max_T(t) > 0]
    if not valid_T:
        print("No valid data found.")
        return
    global_T_max = min(valid_T)
    
    outer_test_start = int(global_T_max * 0.85)  
    inner_T = outer_test_start
    chunk_size = inner_T // (args.wf_folds + 1)

    print(f"\n{'-'*60}")
    print(f"WALK-FORWARD LAYOUT (T_max = {global_T_max})")
    print(f"Outer Test range: [{outer_test_start} : {global_T_max}] (15%)")
    print(f"Inner Data range: [0 : {outer_test_start}] (85%) divided into {args.wf_folds} anchored folds")
    print(f"{'-'*60}")

    all_folds_mcc = np.zeros((args.wf_folds, args.wf_epochs))

    for k in range(args.wf_folds):
        train_end = (k + 2) * chunk_size
        val_end   = (k + 3) * chunk_size
        if k == args.wf_folds - 1:
            val_end = inner_T

        print(f"\nBuilding Fold {k+1}/{args.wf_folds} | Train:[0:{train_end}], Val:[{train_end}:{val_end}]")
        
        list_train, list_val = [], []
        for t in tickers:
            tr, va, _ = dp.prepare_data(t, train_end=train_end, val_end=val_end, test_end=val_end)
            if tr and len(tr.get("label", [])) > 0:
                list_train.append(tr)
                list_val.append(va)

        fold_train = merge_datasets(list_train, shuffle=True)
        fold_val   = merge_datasets(list_val, shuffle=False)

        fold_mcc = train_inner_fold(
            fold_idx=k+1, train_data=fold_train, val_data=fold_val,
            include_ticker_id=include_tid, max_epochs=args.wf_epochs
        )
        all_folds_mcc[k, :] = fold_mcc

    avg_mcc_history = np.mean(all_folds_mcc, axis=0)
    smoothed_mcc = [np.mean(avg_mcc_history[max(0, i-1):min(args.wf_epochs, i+2)]) for i in range(args.wf_epochs)]
    best_epoch = np.argmax(smoothed_mcc) + 1  

    print(f"\n{'-'*60}")
    print(f"INNER LOOP COMPLETE.")
    print(f"Selected Optimal Epoch: {best_epoch} (Avg Smoothed MCC: {smoothed_mcc[best_epoch-1]:.4f})")
    print(f"{'-'*60}")

    print("\nBuilding Final Retrain & Outer Test Datasets...")
    list_retrain, list_test = [], []
    for t in tickers:
        tr, _, te = dp.prepare_data(t, train_end=inner_T, val_end=inner_T, test_end=global_T_max)
        if tr and len(tr.get("label", [])) > 0:
            list_retrain.append(tr)
            list_test.append(te)

    retrain_data = merge_datasets(list_retrain, shuffle=True)
    outer_test_data = merge_datasets(list_test, shuffle=False)

    tag = f"label={args.label_mode}_price={args.price_mode}{'_notid' if args.no_ticker_id else '_tid'}_seed{args.seed}_wf"
    save_path = f"output/best_model_{tag}.pt"

    retrain_final_model(
        train_data=retrain_data,
        test_data=outer_test_data,
        best_epochs=best_epoch,
        include_ticker_id=include_tid,
        save_path=save_path
    )

if __name__ == "__main__":
    main()