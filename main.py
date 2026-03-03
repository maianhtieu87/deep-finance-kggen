# main.py - FIXED VERSION WITH GRAPH BATCHING
"""
Training script với Graph Data Support

CHANGES:
- Custom collate_fn để xử lý PyG Data objects
- DataLoader nhận s_n_graphs thay vì s_n tensors
- Updated TrainConfig với GNN parameters
"""

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


# --- 2. CUSTOM DATASET FOR GRAPHS ---
class StockGraphDataset(Dataset):
    """
    Dataset that handles graph data properly.
    """
    def __init__(self, data_dict):
        self.s_o = data_dict["s_o"]
        self.s_h = data_dict["s_h"]
        self.s_c = data_dict["s_c"]
        self.s_m = data_dict["s_m"]
        self.s_n_graphs = data_dict["s_n_graphs"]  # List of PyG Data objects
        self.label = data_dict["label"]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return {
            "s_o": self.s_o[idx],
            "s_h": self.s_h[idx],
            "s_c": self.s_c[idx],
            "s_m": self.s_m[idx],
            "s_n_graph": self.s_n_graphs[idx],  # Single graph
            "label": self.label[idx],
        }


# --- 3. CUSTOM COLLATE FUNCTION ---
def collate_graph_batch(batch):
    """
    Custom collate function for graph batching.
    
    Args:
        batch: List of dicts from __getitem__
    
    Returns:
        Batched dict with:
            - s_o, s_h, s_c, s_m: Stacked tensors (B, T, D)
            - s_n_graphs: List of Data objects (length B)
            - label: Stacked tensor (B,)
    """
    s_o_batch = torch.stack([item["s_o"] for item in batch])
    s_h_batch = torch.stack([item["s_h"] for item in batch])
    s_c_batch = torch.stack([item["s_c"] for item in batch])
    s_m_batch = torch.stack([item["s_m"] for item in batch])
    
    # Graphs remain as list (batching handled in model)
    s_n_graphs = [item["s_n_graph"] for item in batch]
    
    label_batch = torch.stack([item["label"] for item in batch])
    
    return {
        "s_o": s_o_batch,
        "s_h": s_h_batch,
        "s_c": s_c_batch,
        "s_m": s_m_batch,
        "s_n_graphs": s_n_graphs,
        "label": label_batch,
    }


# --- 4. MERGE DATASETS ---
def merge_datasets(list_of_dicts, shuffle: bool = False):
    """
    Merge multiple ticker datasets.
    
    Note: s_n_graphs is list of lists, needs flattening.
    """
    if not list_of_dicts:
        return {}
    
    merged = {}
    
    # Merge tensors
    for key in ["s_o", "s_h", "s_c", "s_m", "label"]:
        parts = [d[key] for d in list_of_dicts if d and key in d]
        if parts:
            merged[key] = torch.cat(parts, dim=0)
    
    # Merge graph lists (flatten)
    graph_parts = [d["s_n_graphs"] for d in list_of_dicts if d and "s_n_graphs" in d]
    if graph_parts:
        merged["s_n_graphs"] = [g for sublist in graph_parts for g in sublist]
    
    # Shuffle if needed
    if shuffle and "label" in merged:
        idx = torch.randperm(len(merged["label"]))
        for key in ["s_o", "s_h", "s_c", "s_m", "label"]:
            merged[key] = merged[key][idx]
        
        # Shuffle graphs accordingly
        merged["s_n_graphs"] = [merged["s_n_graphs"][i] for i in idx.tolist()]
    
    return merged


# --- 5. COMPUTE CLASS WEIGHTS ---
def compute_class_weights(labels_tensor: torch.Tensor) -> torch.Tensor:
    """Balanced Class Weights (Effective-Sqrt Strategy)."""
    labels = labels_tensor.detach().cpu().numpy()
    class_counts = np.bincount(labels, minlength=3)
    num_classes = len(class_counts)

    # Effective Number
    beta = 0.9999
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-8)
    weights = weights / np.sum(weights) * num_classes

    # Sqrt Smoothing
    weights = np.sqrt(weights)
    weights = weights / np.sum(weights) * num_classes

    weights_tensor = torch.tensor(weights, dtype=torch.float32)

    print("\n⚖️ [TIER 1] Balanced Class Weights:")
    classes = ["DOWN", "FLAT", "UP"]
    for i, w in enumerate(weights):
        count = int(class_counts[i])
        print(f"   ► {classes[i]:<4}: Count={count:<4} | Weight={w:.4f}")

    return weights_tensor


# --- 6. EVALUATE ---
def evaluate(model: torch.nn.Module, data_dict: dict):
    """Evaluate model on a dataset."""
    if not data_dict or "label" not in data_dict or len(data_dict["label"]) == 0:
        return 0.0, 0.0
    
    # Create DataLoader
    dataset = StockGraphDataset(data_dict)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_graph_batch
    )
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            acc, mcc, preds = model(
                batch["s_o"].to(device),
                batch["s_h"].to(device),
                batch["s_c"].to(device),
                batch["s_m"].to(device),
                batch["s_n_graphs"],  # Pass list of graphs
                batch["label"].to(device),
                mode="test",
                return_preds=True
            )
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
    
    # Compute overall metrics
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    final_acc = accuracy_score(all_labels, all_preds)
    final_mcc = matthews_corrcoef(all_labels, all_preds)
    
    return final_acc, final_mcc


# --- 7. TRAIN ---
def train_model(train_data: dict, valid_data: dict, test_data: dict):
    """Train model with graph data."""
    if not train_data:
        return

    s_m_dim = train_data["s_m"].shape[-1]
    
    # Calculate class weights
    print("\n🔢 Calculating Class Weights...")
    train_labels = train_data["label"]
    class_weights = compute_class_weights(train_labels).to(device)
    
    # Create DataLoader with custom collate
    batch_size = getattr(TrainConfig, "batch_size", 128)
    print(f"   ► Batch Size: {batch_size}")
    
    train_dataset = StockGraphDataset(train_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_graph_batch,
        drop_last=False
    )

    print(f"\n🚀 Initializing Model on {device}...")
    print(f"   ► Strategy: BALANCED FOCAL LOSS + GNN")
    
    # ✅ GNN Configuration from TrainConfig
    use_gnn = getattr(TrainConfig, "use_gnn", True)
    gnn_type = getattr(TrainConfig, "gnn_type", "sage")
    gnn_hidden = getattr(TrainConfig, "gnn_hidden_dim", 256)
    gnn_layers = getattr(TrainConfig, "gnn_num_layers", 2)
    gnn_heads = getattr(TrainConfig, "gnn_heads", 4)
    gnn_pool = getattr(TrainConfig, "gnn_pool", "attention")
    
    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m_dim,
        news_dim=TrainConfig.news_embed_dim,
        dim=TrainConfig.dim,
        input_dim=TrainConfig.window_size,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        dropout=0.1,
        class_weights=class_weights,
        use_focal_loss=TrainConfig.use_focal_loss,
        focal_gamma=TrainConfig.focal_gamma,  
        device=device,
        # GNN params
        use_gnn=use_gnn,
        gnn_type=gnn_type,
        gnn_hidden_dim=gnn_hidden,
        gnn_num_layers=gnn_layers,
        gnn_heads=gnn_heads,
        gnn_pool=gnn_pool,
    ).to(device)

    lr = getattr(TrainConfig, "learning_rate", 1e-4)
    wd = getattr(TrainConfig, "weight_decay", 1e-5)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=wd
    )

    best_val_mcc = -1.0
    best_val_acc = -1.0
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pt")

    print("\n⚡️ STARTING TRAINING...")

    for epoch in range(int(TrainConfig.epoch_num)):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            loss = model(
                batch["s_o"].to(device),
                batch["s_h"].to(device),
                batch["s_c"].to(device),
                batch["s_m"].to(device),
                batch["s_n_graphs"],  # Pass graph list
                batch["label"].to(device),
                mode="train"
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)

        # Validate
        val_acc, val_mcc = evaluate(model, valid_data)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss {avg_loss:.4f} | Val ACC {val_acc:.4f} | Val MCC {val_mcc:.4f}")

        # Save best model
        is_best = False
        if val_mcc > best_val_mcc:
            is_best = True
        elif val_mcc == best_val_mcc and val_acc > best_val_acc:
            is_best = True
        
        if is_best and (epoch + 10) >= 50:
            best_val_mcc = val_mcc
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   >>> New Best Model Saved! (MCC: {val_mcc:.4f} - Acc: {val_acc:.4f})")

    print("\n🏆 FINAL TEST & SANITY CHECK...")
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
        except Exception as e:
            print(f"⚠️ Skip ticker {ticker} vì lỗi: {e}")

    final_train = merge_datasets(list_train, shuffle=True)
    final_valid = merge_datasets(list_valid, shuffle=False)
    final_test = merge_datasets(list_test, shuffle=False)

    if len(final_train) > 0:
        train_model(final_train, final_valid, final_test)