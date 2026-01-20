import torch
import random
import numpy as np
import os
from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig

# --- 1. SETUP ---
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if TrainConfig.use_cuda and torch.cuda.is_available() else "cpu")
set_seed(TrainConfig.seed)

# --- 2. HELPER: MERGE ---
def merge_datasets(list_of_dicts, shuffle=False):
    if not list_of_dicts: return {}
    keys = list_of_dicts[0].keys()
    merged_data = {}
    for key in keys:
        tensors = [d[key] for d in list_of_dicts if d and key in d]
        if tensors:
            merged_data[key] = torch.cat(tensors, dim=0)
    
    if shuffle and "label" in merged_data:
        indices = torch.randperm(len(merged_data["label"]))
        for key in merged_data:
            merged_data[key] = merged_data[key][indices]
    return merged_data

def compute_class_weights(labels_tensor):
    """
    Tính weights theo phương pháp Inverse Class Frequency (Sklearn style).
    Weight_class_i = Total_Samples / (Num_Classes * Count_class_i)
    """
    # Chuyển về CPU numpy để tính toán
    labels = labels_tensor.cpu().numpy()
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # Tránh chia cho 0 nếu lỡ có class nào rỗng (dù khó xảy ra với Z-score)
    class_counts = np.maximum(class_counts, 1) 
    
    weights = total_samples / (num_classes * class_counts)
    
    # Chuyển về Tensor
    return torch.tensor(weights, dtype=torch.float32)

# --- 3. EVALUATE ---
def evaluate(model, data_dict):
    if not data_dict: return 0.0, 0.0
    model.eval()
    with torch.no_grad():
        s_o = data_dict["s_o"].to(device)
        s_h = data_dict["s_h"].to(device)
        s_c = data_dict["s_c"].to(device)
        s_m = data_dict["s_m"].to(device)
        s_n = data_dict["s_n"].to(device)
        label = data_dict["label"].to(device)
        
        acc, mcc = model(s_o, s_h, s_c, s_m, s_n, label, mode="test")
    return acc, mcc

# --- 4. TRAIN ---
def train_model(train_data, valid_data, test_data):
    if not train_data: return

    s_m_dim = train_data["s_m"].shape[-1]
    
    print("\n  Calculating Class Weights (Balancing Strategy)...")
    train_labels = train_data["label"]
    class_weights = compute_class_weights(train_labels).to(device)
    
    print(f"   ► Class Counts: {np.bincount(train_labels.cpu().numpy())}")
    print(f"   ► Computed Alpha: {class_weights.cpu().numpy()}")
    # Ví dụ output: [1.2, 0.6, 1.2] -> Lớp Flat (giữa) weight thấp, 2 bên weight cao
    
    print(f"\n🚀 Initializing Model on {device}...")
    print(f"   ► Strategy: FOCAL LOSS (Gamma=2.0) + ALPHA BALANCING")
    
    # KHỞI TẠO MODEL VỚI FOCAL LOSS & KHÔNG WEIGHTS
    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m_dim,
        news_dim=TrainConfig.news_embed_dim,
        dim=TrainConfig.dim,                 # Giảm về 64 nếu cần
        input_dim=TrainConfig.window_size,   
        output_dim=TrainConfig.output_dim,   
        num_head=TrainConfig.num_head,
        dropout=0.1,                         # Dropout vừa phải
        class_weights=class_weights,                  # [IMPORTANT] Không dùng Weights thủ công
        use_focal_loss=True,                   # [IMPORTANT] Bật Focal Loss
        device=device
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TrainConfig.learning_rate, # 1e-3
        weight_decay=1e-4             # Regularization
    )

    best_val_mcc = -1.0 # Theo dõi MCC thay vì ACC
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pt")

    print("\n⚔️  STARTING TRAINING...")

    for epoch in range(TrainConfig.epoch_num):
        model.train()
        optimizer.zero_grad()
        
        loss = model(
            train_data["s_o"].to(device), train_data["s_h"].to(device),
            train_data["s_c"].to(device), train_data["s_m"].to(device),
            train_data["s_n"].to(device), train_data["label"].to(device),
            mode="train"
        )
        
        loss.backward()
        if not torch.isfinite(loss): break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        val_acc, val_mcc = evaluate(model, valid_data)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss {loss.item():.4f} | Val ACC {val_acc:.4f} | Val MCC {val_mcc:.4f}")

        # Ưu tiên lưu model có MCC cao nhất (tránh lưu model đoán bừa Mode Collapse)
        is_best = False
        if val_mcc > best_val_mcc:
            is_best = True
        elif val_mcc == best_val_mcc and val_acc > best_val_acc:
            is_best = True
            
        if is_best:
            best_val_mcc = val_mcc
            best_val_acc = val_acc # Cập nhật best ACC
            torch.save(model.state_dict(), save_path)
            print(f"   >>> New Best Model Saved! (MCC: {val_mcc:.4f} - Acc: {val_acc:.4f})")

    # =========================================================
    # [UPDATED] FINAL TEST & SANITY CHECK BLOCK
    # =========================================================
    print("\n🏁 FINAL TEST & SANITY CHECK...")
    
    if os.path.exists(save_path):
        # Load lại model tốt nhất
        model.load_state_dict(torch.load(save_path))
        
        # --- BƯỚC 1: KIỂM TRA LẠI TRÊN VALID (Nơi ta biết chắc chắn MCC > 0) ---
        print("🔍 Sanity Check on VALID SET:")
        # Lưu ý: Trong hàm này biến tên là 'valid_data', không phải 'final_valid'
        val_acc_check, val_mcc_check = evaluate(model, valid_data) 
        print(f"   VALID RESULT -> ACC: {val_acc_check:.4f}, MCC: {val_mcc_check:.4f}")
        

        # --- BƯỚC 2: CHẠY TRÊN TEST ---
        print("\n🔍 Run on TEST SET:")
        # Lưu ý: Trong hàm này biến tên là 'test_data', không phải 'final_test'
        test_acc, test_mcc = evaluate(model, test_data)
        print(f"🏆 TEST RESULT  -> ACC: {test_acc:.4f}, MCC: {test_mcc:.4f}")
        
        # --- BƯỚC 3: IN RA DỰ BÁO CỤ THỂ (DEBUG) ---
        model.eval()
        with torch.no_grad():
            # Lấy 10 mẫu đầu tiên để xem thử nó đoán cái gì
            if "s_o" in test_data and len(test_data["s_o"]) > 0:
                print("   (Debug) Checking raw predictions on first batch...")
                # Đoạn này để giữ chỗ, nếu bạn muốn in chi tiết prediction thì cần sửa hàm evaluate
                # để trả về logits, hoặc dùng file debug riêng.
    else:
        print("⚠️ No best model saved.")

if __name__ == "__main__":
    # Cập nhật đường dẫn pkl của bạn ở đây
    pkl_path = r"D:\DeepFinance\data\processed\unified_dataset_test.pkl" 
    dp = data_prepare(pkl_path)
    
    target_tickers = ["TSLA", "AMZN", "MSFT", "NFLX", "AAPL", "GOOGL", "NVDA", "META"] 
    
    list_train, list_valid, list_test = [], [], []
    for ticker in target_tickers:
        try:
            tr, val, te = dp.prepare_data(ticker)
            if tr and len(tr.get("label", [])) > 0:
                list_train.append(tr); list_valid.append(val); list_test.append(te)
        except: pass

    final_train = merge_datasets(list_train, shuffle=True)
    final_valid = merge_datasets(list_valid, shuffle=False)
    final_test  = merge_datasets(list_test,  shuffle=False)

    if len(final_train) > 0:
        train_model(final_train, final_valid, final_test)