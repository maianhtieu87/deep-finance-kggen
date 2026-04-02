# #!/usr/bin/env python3
# """
# main_single_ticker.py
# ----------------------
# Train và evaluate mô hình cho 1 hoặc nhiều ticker chỉ định.
# Dùng khi chưa có đầy đủ news embedding cho toàn bộ 9 tickers.

# Usage:
#     python main_single_ticker.py --ticker TSLA
#     python main_single_ticker.py --ticker TSLA AAPL WMT
#     python main_single_ticker.py --ticker TSLA --epochs 100

# Logic:
#     - Giống main.py nhưng chỉ process các ticker được chỉ định
#     - Các ticker không có news embedding sẽ có s_n = zeros (không crash)
#     - Model vẫn train bình thường với price + macro + news(zeros nếu thiếu)
# """

# import os
# import argparse
# import random
# import numpy as np
# import torch
# from torch.utils.data import DataLoader, Dataset

# from src.model import StockMovementModel
# from src.data_loader import data_prepare
# from configs.config import TrainConfig, GlobalConfig


# def set_seed(seed: int):
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     random.seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False


# class StockDataset(Dataset):
#     def __init__(self, data_dict):
#         self.s_o   = data_dict["s_o"]
#         self.s_h   = data_dict["s_h"]
#         self.s_c   = data_dict["s_c"]
#         self.s_m   = data_dict["s_m"]
#         self.s_n   = data_dict["s_n"]
#         self.label = data_dict["label"]

#     def __len__(self):
#         return len(self.label)

#     def __getitem__(self, idx):
#         return {
#             "s_o":   self.s_o[idx],
#             "s_h":   self.s_h[idx],
#             "s_c":   self.s_c[idx],
#             "s_m":   self.s_m[idx],
#             "s_n":   self.s_n[idx],
#             "label": self.label[idx],
#         }


# def merge_datasets(list_of_dicts, shuffle: bool = False):
#     if not list_of_dicts:
#         return {}
#     merged = {}
#     for key in ["s_o", "s_h", "s_c", "s_m", "s_n", "label"]:
#         parts = [d[key] for d in list_of_dicts if d and key in d]
#         if parts:
#             merged[key] = torch.cat(parts, dim=0)
#     if shuffle and "label" in merged:
#         idx = torch.randperm(len(merged["label"]))
#         for key in merged:
#             merged[key] = merged[key][idx]
#     return merged


# def compute_class_weights(labels_tensor: torch.Tensor) -> torch.Tensor:
#     labels       = labels_tensor.detach().cpu().numpy()
#     class_counts = np.bincount(labels, minlength=3)
#     beta         = 0.9999
#     effective_num = 1.0 - np.power(beta, class_counts)
#     weights       = (1.0 - beta) / (effective_num + 1e-8)
#     weights       = weights / np.sum(weights) * 3
#     weights       = np.sqrt(weights)
#     weights       = weights / np.sum(weights) * 3
#     print("\n  Class Weights:")
#     for i, (cls, w) in enumerate(zip(["DOWN", "FLAT", "UP"], weights)):
#         print(f"    {cls}: count={int(class_counts[i])}  weight={w:.4f}")
#     return torch.tensor(weights, dtype=torch.float32)


# def evaluate(model, data_dict, device):
#     if not data_dict or "label" not in data_dict or len(data_dict["label"]) == 0:
#         return 0.0, 0.0
#     dataset = StockDataset(data_dict)
#     loader  = DataLoader(dataset, batch_size=64, shuffle=False)
#     model.eval()
#     all_preds, all_labels = [], []
#     with torch.no_grad():
#         for batch in loader:
#             acc, mcc, preds = model(
#                 batch["s_o"].to(device), batch["s_h"].to(device),
#                 batch["s_c"].to(device), batch["s_m"].to(device),
#                 batch["s_n"].to(device), batch["label"].to(device),
#                 mode="test", return_preds=True,
#             )
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(batch["label"].cpu().numpy())
#     from sklearn.metrics import accuracy_score, matthews_corrcoef
#     return accuracy_score(all_labels, all_preds), matthews_corrcoef(all_labels, all_preds)


# def train_model(train_data, valid_data, test_data, device, epochs, save_tag=""):
#     if not train_data:
#         return
#     s_m_dim = train_data["s_m"].shape[-1]
#     s_n_dim = train_data["s_n"].shape[-1]
#     print(f"\n  Macro dim : {s_m_dim}")
#     print(f"  News dim  : {s_n_dim} (Voyage-3-large)")

#     class_weights = compute_class_weights(train_data["label"]).to(device)
#     train_dataset = StockDataset(train_data)
#     train_loader  = DataLoader(train_dataset, batch_size=TrainConfig.batch_size,
#                                shuffle=True, drop_last=False)

#     model = StockMovementModel(
#         price_dim=1, macro_dim=s_m_dim, news_dim=s_n_dim,
#         dim=TrainConfig.dim, input_dim=TrainConfig.window_size,
#         output_dim=TrainConfig.output_dim, num_head=TrainConfig.num_head,
#         dropout=0.1, class_weights=class_weights,
#         use_focal_loss=TrainConfig.use_focal_loss,
#         focal_gamma=TrainConfig.focal_gamma,
#         device=device, use_gnn=False,
#     ).to(device)

#     optimizer = torch.optim.Adam(
#         model.parameters(),
#         lr=TrainConfig.learning_rate,
#         weight_decay=TrainConfig.weight_decay,
#     )

#     best_val_mcc = -1.0
#     best_val_acc = -1.0
#     save_dir  = "output"
#     os.makedirs(save_dir, exist_ok=True)
#     tag = f"_{save_tag}" if save_tag else ""
#     save_path = os.path.join(save_dir, f"best_model{tag}.pt")

#     print(f"\n  Training on {device}  ({epochs} epochs)...")

#     for epoch in range(epochs):
#         model.train()
#         total_loss, num_batches = 0, 0
#         for batch in train_loader:
#             optimizer.zero_grad()
#             loss = model(
#                 batch["s_o"].to(device), batch["s_h"].to(device),
#                 batch["s_c"].to(device), batch["s_m"].to(device),
#                 batch["s_n"].to(device), batch["label"].to(device),
#                 mode="train",
#             )
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#             optimizer.step()
#             total_loss += loss.item()
#             num_batches += 1

#         avg_loss = total_loss / max(num_batches, 1)
#         val_acc, val_mcc = evaluate(model, valid_data, device)

#         if (epoch + 1) % 10 == 0:
#             print(f"  Epoch {epoch+1:03d} | Loss {avg_loss:.4f} | "
#                   f"Val ACC {val_acc:.4f} | Val MCC {val_mcc:.4f}")

#         is_best = val_mcc > best_val_mcc or (
#             val_mcc == best_val_mcc and val_acc > best_val_acc)
#         if is_best and epoch >= 40:
#             best_val_mcc = val_mcc
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), save_path)
#             print(f"    >>> Best saved (MCC={val_mcc:.4f} ACC={val_acc:.4f})")

#     print("\n  Final evaluation...")
#     if os.path.exists(save_path):
#         model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
#         val_acc,  val_mcc  = evaluate(model, valid_data, device)
#         test_acc, test_mcc = evaluate(model, test_data,  device)
#         print(f"  Valid: ACC={val_acc:.4f}  MCC={val_mcc:.4f}")
#         print(f"  Test : ACC={test_acc:.4f}  MCC={test_mcc:.4f}")
#     else:
#         print("  No best model saved.")


# def main():
#     parser = argparse.ArgumentParser(description="Train model cho ticker(s) chỉ định")
#     parser.add_argument("--ticker", nargs="+", default=["TSLA"],
#                         help="Ticker(s) để train (default: TSLA)")
#     parser.add_argument("--epochs", type=int, default=TrainConfig.epoch_num,
#                         help=f"Số epochs (default: {TrainConfig.epoch_num})")
#     parser.add_argument("--pkl", default=None,
#                         help="Path đến unified_dataset.pkl (auto nếu không chỉ định)")
#     parser.add_argument("--seed", type=int, default=TrainConfig.seed)
#     args = parser.parse_args()

#     set_seed(args.seed)
#     device = torch.device(
#         "cuda" if TrainConfig.use_cuda and torch.cuda.is_available() else "cpu"
#     )

#     pkl_path = args.pkl or os.path.join(
#         GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl"
#     )
#     if not os.path.exists(pkl_path):
#         print(f"Không tìm thấy dataset: {pkl_path}")
#         print("Chạy main_test.py trước.")
#         return

#     tickers = [t.upper() for t in args.ticker]
#     print(f"\nLoading: {pkl_path}")
#     print(f"Tickers: {tickers}")

#     dp = data_prepare(pkl_path)

#     list_train, list_valid, list_test = [], [], []
#     for ticker in tickers:
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

#     if not list_train:
#         print("Không có dữ liệu để train.")
#         return

#     final_train = merge_datasets(list_train, shuffle=True)
#     final_valid = merge_datasets(list_valid, shuffle=False)
#     final_test  = merge_datasets(list_test,  shuffle=False)

#     save_tag = "_".join(tickers).lower()
#     train_model(final_train, final_valid, final_test,
#                 device=device, epochs=args.epochs, save_tag=save_tag)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
main_single_ticker.py
----------------------
Train và evaluate mô hình cho 1 hoặc nhiều ticker chỉ định.
Dùng khi chưa có đầy đủ news embedding cho toàn bộ 9 tickers.

Usage:
    python main_single_ticker.py --ticker TSLA
    python main_single_ticker.py --ticker TSLA AAPL WMT
    python main_single_ticker.py --ticker TSLA --epochs 100

Logic:
    - Giống main.py nhưng chỉ process các ticker được chỉ định
    - Các ticker không có news embedding sẽ có s_n = zeros (không crash)
    - Model vẫn train bình thường với price + macro + news(zeros nếu thiếu)
"""

import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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


class StockDataset(Dataset):
    def __init__(self, data_dict):
        self.s_o   = data_dict["s_o"]
        self.s_h   = data_dict["s_h"]
        self.s_c   = data_dict["s_c"]
        self.s_m   = data_dict["s_m"]
        self.s_n   = data_dict["s_n"]
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
    labels = labels_tensor.detach().cpu().numpy()
    class_counts = np.bincount(labels, minlength=3)
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / np.sum(weights) * 3
    weights = np.sqrt(weights)
    weights = weights / np.sum(weights) * 3

    print("\n  Class Weights:")
    for i, (cls, w) in enumerate(zip(["DOWN", "FLAT", "UP"], weights)):
        print(f"    {cls}: count={int(class_counts[i])}  weight={w:.4f}")

    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, data_dict, device):
    if not data_dict or "label" not in data_dict or len(data_dict["label"]) == 0:
        return 0.0, 0.0

    dataset = StockDataset(data_dict)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    all_preds, all_labels = [], []

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
    return accuracy_score(all_labels, all_preds), matthews_corrcoef(all_labels, all_preds)


def train_model(train_data, valid_data, test_data, device, epochs, save_tag=""):
    if not train_data:
        return

    s_m_dim = train_data["s_m"].shape[-1]
    s_n_dim = train_data["s_n"].shape[-1]

    print(f"\n  Macro dim : {s_m_dim}")
    print(f"  News dim  : {s_n_dim} (Voyage-3-large)")

    class_weights = compute_class_weights(train_data["label"]).to(device)

    train_dataset = StockDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=TrainConfig.batch_size,
        shuffle=True,
        drop_last=False
    )

    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m_dim,
        news_dim=s_n_dim,
        dim=TrainConfig.dim,
        input_dim=TrainConfig.window_size,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        dropout=0.1,
        class_weights=class_weights,
        use_focal_loss=TrainConfig.use_focal_loss,
        focal_gamma=TrainConfig.focal_gamma,
        device=device,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TrainConfig.learning_rate,
        weight_decay=TrainConfig.weight_decay,
        eps=1e-6,
    )

    # best_smooth_mcc = -1.0
    best_val_mcc = -1.0
    best_val_acc = -1.0
    
    patience = 40
    # epochs_no_improve = 0
    no_improve = 0
    
    # mcc_history = []
    # smooth_window = 5      # Tính trung bình 5 epochs gần nhất

    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    tag = f"_{save_tag}" if save_tag else ""
    save_path = os.path.join(save_dir, f"best_model{tag}.pt")

    print(f"\n  Training on {device}  ({epochs} epochs)...")

    torch.autograd.set_detect_anomaly(True)


    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)

            loss = model(
                batch["s_o"].to(device),
                batch["s_h"].to(device),
                batch["s_c"].to(device),
                batch["s_m"].to(device),
                batch["s_n"].to(device),
                batch["label"].to(device),
                mode="train",
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    f"Loss became NaN/Inf at epoch={epoch+1}, batch={batch_idx}"
                )

            loss.backward()

            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise ValueError(
                        f"Gradient NaN/Inf at epoch={epoch+1}, batch={batch_idx}, param={name}"
                    )

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if not torch.isfinite(torch.tensor(float(grad_norm))):
                raise ValueError(
                    f"Gradient norm became NaN/Inf at epoch={epoch+1}, batch={batch_idx}"
                )

            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        val_acc, val_mcc = evaluate(model, valid_data, device)

        # mcc_history.append(val_mcc)
        # smooth_mcc = float(np.mean(mcc_history[-smooth_window:]))
        
        if (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:03d} | Loss {avg_loss:.4f} | "
                f"Val ACC {val_acc:.4f} | Val MCC {val_mcc:.4f}"
            )

        # if (epoch + 1) % 10 == 0:
        #     print(
        #         f"  Epoch {epoch+1:03d} | Loss {avg_loss:.4f} | "
        #         f"Val ACC {val_acc:.4f} | Val MCC {val_mcc:.4f} | Smooth MCC {smooth_mcc:.4f}"
        #     )

        
        is_best = val_mcc > best_val_mcc or (
            val_mcc == best_val_mcc and val_acc > best_val_acc
        )

        is_best = val_mcc > best_val_mcc or (
            val_mcc == best_val_mcc and val_acc > best_val_acc)
        if is_best and epoch >= 40:
            best_val_mcc = val_mcc
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"    >>> Best saved (MCC={val_mcc:.4f} ACC={val_acc:.4f})")

    print("\n  Final evaluation...")
    if os.path.exists(save_path):
        model.load_state_dict(
            torch.load(save_path, map_location=device, weights_only=True)
        )
        val_acc, val_mcc = evaluate(model, valid_data, device)
        test_acc, test_mcc = evaluate(model, test_data, device)
        print(f"  Valid: ACC={val_acc:.4f}  MCC={val_mcc:.4f}")
        print(f"  Test : ACC={test_acc:.4f}  MCC={test_mcc:.4f}")
    else:
        print("  No best model saved.")


def main():
    parser = argparse.ArgumentParser(description="Train model cho ticker(s) chỉ định")
    parser.add_argument(
        "--ticker",
        nargs="+",
        default=["TSLA"],
        help="Ticker(s) để train (default: TSLA)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TrainConfig.epoch_num,
        help=f"Số epochs (default: {TrainConfig.epoch_num})"
    )
    parser.add_argument(
        "--pkl",
        default=None,
        help="Path đến unified_dataset.pkl (auto nếu không chỉ định)"
    )
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if TrainConfig.use_cuda and torch.cuda.is_available() else "cpu"
    )

    pkl_path = args.pkl or os.path.join(
        GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl"
    )

    if not os.path.exists(pkl_path):
        print(f"Không tìm thấy dataset: {pkl_path}")
        print("Chạy main_test.py trước.")
        return

    tickers = [t.upper() for t in args.ticker]
    print(f"\nLoading: {pkl_path}")
    print(f"Tickers: {tickers}")

    dp = data_prepare(pkl_path)

    list_train, list_valid, list_test = [], [], []
    for ticker in tickers:
        try:
            tr, val, te = dp.prepare_data(ticker)
            if tr and len(tr.get("label", [])) > 0:
                list_train.append(tr)
                list_valid.append(val)
                list_test.append(te)
                print(
                    f"  {ticker}: Train={len(tr['label'])} "
                    f"Valid={len(val.get('label',[]))} "
                    f"Test={len(te.get('label',[]))}"
                )
        except Exception as e:
            print(f"  Skip {ticker}: {e}")

    if not list_train:
        print("Không có dữ liệu để train.")
        return

    final_train = merge_datasets(list_train, shuffle=True)
    final_valid = merge_datasets(list_valid, shuffle=False)
    final_test = merge_datasets(list_test, shuffle=False)

    save_tag = "_".join(tickers).lower()
    train_model(
        final_train,
        final_valid,
        final_test,
        device=device,
        epochs=args.epochs,
        save_tag=save_tag,
    )


if __name__ == "__main__":
    main()