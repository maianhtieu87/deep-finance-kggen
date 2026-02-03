# =========================================================
# FILE: main.py
# =========================================================

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig, GlobalConfig


# --- 1. SETUP ---
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if TrainConfig.use_cuda and torch.cuda.is_available() else "cpu")
set_seed(TrainConfig.seed)


# --- 2. HELPER: DATASET & MERGE ---
class StockDataset(Dataset):
    def __init__(self, data_dict):
        self.s_o = data_dict["s_o"]
        self.s_h = data_dict["s_h"]
        self.s_c = data_dict["s_c"]
        self.s_m = data_dict["s_m"]
        self.s_n = data_dict["s_n"]  # news / KG features
        self.label = data_dict["label"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            "s_o": self.s_o[idx],
            "s_h": self.s_h[idx],
            "s_c": self.s_c[idx],
            "s_m": self.s_m[idx],
            "s_n": self.s_n[idx],
            "label": self.label[idx],
        }


def merge_datasets(list_of_dicts, shuffle: bool = False):
    """
    Gộp nhiều ticker lại thành 1 dict duy nhất.
    Mỗi dict trong list_of_dicts có key: s_o, s_h, s_c, s_m, s_n, label
    """
    if not list_of_dicts:
        return {}

    keys = list(list_of_dicts[0].keys())
    merged = {}

    for k in keys:
        parts = [d[k] for d in list_of_dicts if d and k in d and isinstance(d[k], torch.Tensor)]
        if parts:
            merged[k] = torch.cat(parts, dim=0)

    if shuffle and "label" in merged and isinstance(merged["label"], torch.Tensor):
        idx = torch.randperm(len(merged["label"]))
        for k in merged:
            merged[k] = merged[k][idx]

    return merged


# --- 3. HELPER: WEIGHTS (STABLE TIER 1) ---
def compute_class_weights(labels_tensor: torch.Tensor) -> torch.Tensor:
    """
    [TIER 1] Balanced Class Weights (Effective-Sqrt)
    """
    labels = labels_tensor.detach().cpu().numpy()
    class_counts = np.bincount(labels, minlength=3)  # ensure 3 classes
    num_classes = len(class_counts)

    beta = 0.9999
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)

    # normalize
    weights = weights / np.sum(weights) * num_classes

    # sqrt smoothing
    weights = np.sqrt(weights)
    weights = weights / np.sum(weights) * num_classes

    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("\n⚖️  [TIER 1] Balanced Class Weights (Effective-Sqrt):")
    classes = ["DOWN", "FLAT", "UP"]
    for i, w in enumerate(weights):
        print(f"   ► {classes[i]:<4}: Count={int(class_counts[i]):<4} | Weight={w:.4f}")

    return weights_tensor


# --- 4. EVALUATE ---
def evaluate(model: torch.nn.Module, data_dict: dict):
    if not data_dict or "label" not in data_dict or len(data_dict["label"]) == 0:
        return 0.0, 0.0

    model.eval()
    with torch.no_grad():
        acc, mcc = model(
            data_dict["s_o"].to(device),
            data_dict["s_h"].to(device),
            data_dict["s_c"].to(device),
            data_dict["s_m"].to(device),
            data_dict["s_n"].to(device),
            data_dict["label"].to(device),
            mode="test",
        )
    return float(acc), float(mcc)


# --- 5. TRAIN ---
def train_model(train_data: dict, valid_data: dict, test_data: dict):
    if not train_data or "label" not in train_data or len(train_data["label"]) == 0:
        print("⚠️ train_data rỗng. Dừng.")
        return

    # dims
    s_m_dim = train_data["s_m"].shape[-1]

    # weights
    print("\n  Calculating Class Weights (Balancing Strategy)...")
    class_weights = compute_class_weights(train_data["label"]).to(device)

    # batch size: FIX đồng bộ config
    real_batch_size = getattr(TrainConfig, "batch_size", None)
    if real_batch_size is None:
        real_batch_size = getattr(TrainConfig, "train_batch_size", 128)

    print(f"   ► Batch Size: {real_batch_size}")

    train_loader = DataLoader(
        StockDataset(train_data),
        batch_size=real_batch_size,
        shuffle=True,
        drop_last=False,
    )

    print(f"\n🚀 Initializing Model on {device}...")
    print("   ► Strategy: [TIER 2] Improved Focal Loss (Temp=1.5, Smooth=0.1)")

    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m_dim,
        news_dim=TrainConfig.news_embed_dim,  # (T, 1024) hiện tại
        dim=TrainConfig.dim,                  # hidden dim của model
        input_dim=TrainConfig.window_size,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        dropout=0.1,
        class_weights=class_weights,
        use_focal_loss=True,
        device=device,
    ).to(device)

    lr = getattr(TrainConfig, "learning_rate", 1e-4)
    wd = getattr(TrainConfig, "weight_decay", 1e-5)  # FIX: dùng config

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    best_val_mcc = -1.0
    best_val_acc = -1.0

    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pt")

    print("\n⚔️  STARTING TRAINING...")

    for epoch in range(int(TrainConfig.epoch_num)):
        model.train()
        total_loss = 0.0
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

            total_loss += float(loss.item())
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        val_acc, val_mcc = evaluate(model, valid_data)

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:03d} | Loss {avg_loss:.4f} | "
                f"Val ACC {val_acc:.4f} | Val MCC {val_mcc:.4f}"
            )

        # Save best: MCC priority then ACC
        is_best = False
        if val_mcc > best_val_mcc:
            is_best = True
        elif val_mcc == best_val_mcc and val_acc > best_val_acc:
            is_best = True

        if is_best:
            best_val_mcc = val_mcc
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   >>> New Best Model Saved! (MCC: {val_mcc:.4f} - Acc: {val_acc:.4f})")

    print("\n🏁 FINAL TEST & SANITY CHECK...")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))

        print("🔍 Sanity Check on VALID SET:")
        val_acc_check, val_mcc_check = evaluate(model, valid_data)
        print(f"   VALID RESULT -> ACC: {val_acc_check:.4f}, MCC: {val_mcc_check:.4f}")

        print("\n🔍 Run on TEST SET:")
        test_acc, test_mcc = evaluate(model, test_data)
        print(f"🏆 TEST RESULT  -> ACC: {test_acc:.4f}, MCC: {test_mcc:.4f}")
    else:
        print("⚠️ No best model saved.")


if __name__ == "__main__":
    # Dataset do main_test.py build ra
    pkl_path = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    print(f"📦 Loading processed dataset from: {pkl_path}")

    if not os.path.exists(pkl_path):
        print("❌ Không thấy unified_dataset_test.pkl. Hãy chạy main_test.py trước.")
        raise SystemExit(1)

    dp = data_prepare(pkl_path)

    target_tickers = getattr(GlobalConfig, "TICKERS", ["TSLA", "AMZN", "MSFT", "NFLX"])

    list_train, list_valid, list_test = [], [], []

    for ticker in target_tickers:
        try:
            tr, val, te = dp.prepare_data(ticker)
            if tr and len(tr.get("label", [])) > 0:
                list_train.append(tr)
                list_valid.append(val)
                list_test.append(te)
                print(f"✅ Loaded {ticker}: Train={len(tr['label'])} Valid={len(val.get('label', []))} Test={len(te.get('label', []))}")
            else:
                print(f"⚠️ {ticker}: empty train set, skip.")
        except Exception as e:
            print(f"⚠️ Skip ticker {ticker} vì lỗi: {e}")

    final_train = merge_datasets(list_train, shuffle=True)
    final_valid = merge_datasets(list_valid, shuffle=False)
    final_test = merge_datasets(list_test, shuffle=False)

    if final_train and len(final_train.get("label", [])) > 0:
        train_model(final_train, final_valid, final_test)
    else:
        print("⚠️ Không có dữ liệu train sau khi merge.")