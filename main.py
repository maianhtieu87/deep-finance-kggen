# main.py — V4 (Voyage embedding, no GATv2)
"""
Training script V4.

Thay đổi so với V3:
  - s_n là tensor (B, T, 1024) thay vì list of PyG Data
  - Không cần custom collate_fn (standard DataLoader works)
  - StockMovementModel nhận s_n trực tiếp
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig, GlobalConfig


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device(
    "cuda" if TrainConfig.use_cuda and torch.cuda.is_available() else "cpu"
)
set_seed(TrainConfig.seed)


# ─────────────────────────────────────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────────────────────────────────────

class StockDataset(Dataset):
    """Simple dataset — all tensors, no PyG objects needed."""

    def __init__(self, data_dict):
        self.s_o   = data_dict["s_o"]
        self.s_h   = data_dict["s_h"]
        self.s_c   = data_dict["s_c"]
        self.s_m   = data_dict["s_m"]
        self.s_n   = data_dict["s_n"]    # (N, T, 1024)
        self.label = data_dict["label"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            "s_o":   self.s_o[idx],
            "s_h":   self.s_h[idx],
            "s_c":   self.s_c[idx],
            "s_m":   self.s_m[idx],
            "s_n":   self.s_n[idx],
            "label": self.label[idx],
        }


def merge_datasets(list_of_dicts, shuffle: bool = False):
    if not list_of_dicts:
        return {}
    merged = {}
    for key in ["s_o", "s_h", "s_c", "s_m", "s_n", "label"]:
        parts = [d[key] for d in list_of_dicts if d and key in d]
        if parts:
            merged[key] = torch.cat(parts, dim=0)

    if shuffle and "label" in merged:
        idx = torch.randperm(len(merged["label"]))
        for key in merged:
            merged[key] = merged[key][idx]
    return merged


def compute_class_weights(labels_tensor: torch.Tensor) -> torch.Tensor:
    labels       = labels_tensor.detach().cpu().numpy()
    class_counts = np.bincount(labels, minlength=3)
    beta         = 0.9999
    effective_num = 1.0 - np.power(beta, class_counts)
    weights       = (1.0 - beta) / (effective_num + 1e-8)
    weights       = weights / np.sum(weights) * 3
    weights       = np.sqrt(weights)
    weights       = weights / np.sum(weights) * 3
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    print("\n  Class Weights:")
    for i, (cls, w) in enumerate(zip(["DOWN", "FLAT", "UP"], weights)):
        print(f"    {cls}: count={int(class_counts[i])}  weight={w:.4f}")
    return weights_tensor


def evaluate(model: torch.nn.Module, data_dict: dict):
    if not data_dict or "label" not in data_dict or len(data_dict["label"]) == 0:
        return 0.0, 0.0

    dataset = StockDataset(data_dict)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            acc, mcc, preds = model(
                batch["s_o"].to(device),
                batch["s_h"].to(device),
                batch["s_c"].to(device),
                batch["s_m"].to(device),
                batch["s_n"].to(device),
                batch["label"].to(device),
                mode="test",
                return_preds=True,
            )
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())

    from sklearn.metrics import accuracy_score, matthews_corrcoef
    return (
        accuracy_score(all_labels, all_preds),
        matthews_corrcoef(all_labels, all_preds),
    )


def train_model(train_data: dict, valid_data: dict, test_data: dict):
    if not train_data:
        return

    s_m_dim = train_data["s_m"].shape[-1]
    s_n_dim = train_data["s_n"].shape[-1]  # should be 1024

    print(f"\n  Macro dim : {s_m_dim}")
    print(f"  News dim  : {s_n_dim} (Voyage-3-large)")

    class_weights = compute_class_weights(train_data["label"]).to(device)

    train_dataset = StockDataset(train_data)
    train_loader  = DataLoader(
        train_dataset,
        batch_size=getattr(TrainConfig, "batch_size", 32),
        shuffle=True,
        drop_last=False,
    )

    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m_dim,
        news_dim=s_n_dim,      # 1024
        dim=TrainConfig.dim,
        input_dim=TrainConfig.window_size,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        dropout=0.1,
        class_weights=class_weights,
        use_focal_loss=getattr(TrainConfig, "use_focal_loss", True),
        focal_gamma=getattr(TrainConfig, "focal_gamma", 2.0),
        device=device,
        use_gnn=False,  # V4: no GATv2
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=getattr(TrainConfig, "learning_rate", 1e-4),
        weight_decay=getattr(TrainConfig, "weight_decay", 1e-4),
    )

    # Cosine LR schedule with warm restarts
    # Important for sequential fusion with conservative gate init (converges slowly)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    best_val_mcc = -1.0
    best_val_acc = -1.0
    save_dir  = "output"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pt")

    print(f"\n  Training on {device} ...")
    print(f"  LR schedule: CosineAnnealingWarmRestarts(T_0=30, T_mult=2)")

    # --- THÊM BIẾN EARLY STOPPING ---
    epochs_since_improvement = 0

    for epoch in range(int(TrainConfig.epoch_num)):
        model.train()
        total_loss  = 0
        num_batches = 0
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(
                batch["s_o"].to(device),
                batch["s_h"].to(device),
                batch["s_c"].to(device),
                batch["s_m"].to(device),
                batch["s_n"].to(device),
                batch["label"].to(device),
                mode="train",
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss  += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        scheduler.step()  # Step LR scheduler each epoch

        if (epoch + 1) % 10 == 0:
            val_acc, val_mcc = evaluate(model, valid_data)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:03d} | Loss {avg_loss:.4f} | "
                  f"Val ACC {val_acc:.4f} | Val MCC {val_mcc:.4f} | "
                  f"LR {current_lr:.2e}")

            is_best = val_mcc > best_val_mcc or (
                val_mcc == best_val_mcc and val_acc > best_val_acc
            )
            
            if is_best:
                best_val_mcc = val_mcc
                best_val_acc = val_acc
                epochs_since_improvement = 0  # Reset early stopping
                
                if epoch >= 9:  # Save from first eval checkpoint
                    torch.save(model.state_dict(), save_path)
                    print(f"    >>> Best model saved (MCC={val_mcc:.4f} ACC={val_acc:.4f})")
            else:
                epochs_since_improvement += 10 # Cộng 10 vì check mỗi 10 epoch
            # --- KIỂM TRA EARLY STOPPING ---
            if epochs_since_improvement > 30:
                    print(f"    >>> Early stopping at epoch {epoch+1} (No improvement for > 30 epochs)")
                    break

    print("\n  Final evaluation...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device,
                                          weights_only=True))
        val_acc, val_mcc = evaluate(model, valid_data)
        print(f"  Valid: ACC={val_acc:.4f}  MCC={val_mcc:.4f}")
        test_acc, test_mcc = evaluate(model, test_data)
        print(f"  Test : ACC={test_acc:.4f}  MCC={test_mcc:.4f}")
    else:
        print("  No best model saved.")


# if __name__ == "__main__":
#     pkl_path = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
#     print(f"Loading: {pkl_path}")

#     if not os.path.exists(pkl_path):
#         print("unified_dataset_test.pkl not found. Run main_test.py first.")
#         raise SystemExit(1)

#     dp = data_prepare(pkl_path)

#     list_train, list_valid, list_test = [], [], []
#     for ticker in GlobalConfig.TICKERS:
#         try:
#             tr, val, te = dp.prepare_data(ticker)
#             if tr and len(tr.get("label", [])) > 0:
#                 list_train.append(tr)
#                 list_valid.append(val)
#                 list_test.append(te)
#                 print(f"  {ticker}: Train={len(tr['label'])} "
#                       f"Valid={len(val.get('label',[]))} "
#                       f"Test={len(te.get('label',[]))}")
#         except Exception as e:
#             print(f"  Skip {ticker}: {e}")

#     final_train = merge_datasets(list_train, shuffle=True)
#     final_valid = merge_datasets(list_valid, shuffle=False)
#     final_test  = merge_datasets(list_test,  shuffle=False)

#     if final_train:
#         # Kiểm tra an toàn trước khi train
#         for k in ["s_o", "s_h", "s_c", "s_m", "s_n"]:
#             if torch.isnan(final_train[k]).any():
#                 print(f"[CẢNH BÁO] Phát hiện NaN trong tensor {k}. Hãy kiểm tra lại data_loader!")
                
#         train_model(final_train, final_valid, final_test)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Stock Movement Model")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    # THÊM CƠ CHẾ --ticker VÀO ĐÂY:
    parser.add_argument("--ticker", nargs="+", default=None, help="List of tickers to train on, e.g. TSLA AAPL")
    args = parser.parse_args()

    pkl_path = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    print(f"Loading: {pkl_path}")

    if not os.path.exists(pkl_path):
        print("unified_dataset_test.pkl not found. Run main_test.py first.")
        raise SystemExit(1)

    dp = data_prepare(pkl_path)

    # XỬ LÝ LOGIC TICKER: Nếu người dùng truyền --ticker thì dùng list đó, nếu không thì dùng GlobalConfig
    if args.ticker:
        tickers_to_run = [t.upper() for t in args.ticker]
    else:
        tickers_to_run = GlobalConfig.TICKERS

    print(f"Running for tickers: {tickers_to_run}")

    list_train, list_valid, list_test = [], [], []
    for ticker in tickers_to_run:
        try:
            tr, val, te = dp.prepare_data(ticker)
            if tr and len(tr.get("label", [])) > 0:
                list_train.append(tr)
                list_valid.append(val)
                list_test.append(te)
                print(f"  {ticker}: Train={len(tr['label'])} "
                      f"Valid={len(val.get('label',[]))} "
                      f"Test={len(te.get('label',[]))}")
        except Exception as e:
            print(f"  Skip {ticker}: {e}")

    if not list_train:
        print("Không có dữ liệu để train cho các Ticker đã chọn.")
        raise SystemExit(1)

    final_train = merge_datasets(list_train, shuffle=True)
    final_valid = merge_datasets(list_valid, shuffle=False)
    final_test  = merge_datasets(list_test,  shuffle=False)

    if final_train:
        # Kiểm tra an toàn trước khi train
        for k in ["s_o", "s_h", "s_c", "s_m", "s_n"]:
            if torch.isnan(final_train[k]).any():
                print(f"[CẢNH BÁO] Phát hiện NaN trong tensor {k}. Hãy kiểm tra lại data_loader!")
                
        train_model(final_train, final_valid, final_test)

