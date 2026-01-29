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

# ================================================================
# CHIẾN LƯỢC 1: SQRT BALANCING (KHUYẾN NGHỊ)
# ================================================================
def compute_class_weights_sqrt(labels_tensor):
    """
    Sử dụng căn bậc hai của inverse frequency.
    Công thức: Weight_i = sqrt(Total / Count_i) / mean(sqrt(...))
    
    ƯU ĐIỂM: 
    - Phạt nhẹ hơn inverse frequency thông thường
    - Vẫn giữ được xu hướng cân bằng
    - Tránh weights quá cực đoan
    
    VÍ DỤ:
    Class counts: [50, 100, 50] 
    -> Inverse: [4.0, 2.0, 4.0] (phạt quá nặng!)
    -> SQRT:    [1.41, 1.0, 1.41] (vừa phải)
    """
    labels = labels_tensor.cpu().numpy()
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    
    # Tính sqrt của inverse frequency
    sqrt_weights = np.sqrt(total_samples / (class_counts + 1e-6))
    
    # Normalize về mean = 1
    normalized_weights = sqrt_weights / sqrt_weights.mean()
    
    return torch.tensor(normalized_weights, dtype=torch.float32)

# ================================================================
# CHIẾN LƯỢC 2: EFFECTIVE NUMBER OF SAMPLES (ENS)
# ================================================================
def compute_class_weights_ens(labels_tensor, beta=0.99):
    """
    Class-Balanced Loss Based on Effective Number of Samples (CVPR 2019).
    Paper: https://arxiv.org/abs/1901.05555
    
    Công thức: E_n = (1 - β^n) / (1 - β)
              Weight_i = (1 - β) / E_n_i
    
    THAM SỐ:
    - beta=0.0: Không cân bằng (giống vanilla CE)
    - beta=0.9: Cân bằng nhẹ
    - beta=0.99: Cân bằng vừa (mặc định)
    - beta=0.999: Cân bằng mạnh
    
    Ý NGHĨA: 
    - Khi số lượng mẫu tăng, hiệu quả học tập giảm dần
    - Mẫu thứ 100 không còn quan trọng bằng mẫu đầu tiên
    """
    labels = labels_tensor.cpu().numpy()
    class_counts = np.bincount(labels)
    
    effective_num = 1.0 - np.power(beta, class_counts)
    weights = (1.0 - beta) / (effective_num + 1e-6)
    
    # Normalize về tổng = số lượng classes
    weights = weights / weights.sum() * len(class_counts)
    
    return torch.tensor(weights, dtype=torch.float32)

# ================================================================
# CHIẾN LƯỢC 3: INVERSE FREQUENCY CLIPPED (An toàn hơn)
# ================================================================
def compute_class_weights_clipped(labels_tensor, max_ratio=10.0):
    """
    Inverse Frequency nhưng giới hạn tỷ lệ max/min.
    
    THAM SỐ:
    - max_ratio: Tỷ lệ tối đa giữa weight lớn nhất và nhỏ nhất
    
    VÍ DỤ:
    Class counts: [10, 100, 10]
    -> Inverse thông thường: [10, 1, 10] (ratio = 10x)
    -> Clipped (max_ratio=5): [5, 1, 5] (ratio = 5x)
    """
    labels = labels_tensor.cpu().numpy()
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    
    # Inverse frequency
    weights = total_samples / (num_classes * class_counts + 1e-6)
    
    # Clip để tránh quá cực đoan
    min_weight = weights.min()
    weights = np.minimum(weights, min_weight * max_ratio)
    
    return torch.tensor(weights, dtype=torch.float32)

# --- 3. EVALUATE ---
def evaluate(model, data_dict, return_details=False):
    """
    Đánh giá model với option trả về chi tiết predictions.
    
    Args:
        return_details: Nếu True, trả về (acc, mcc, preds, targets)
    """
    if not data_dict: 
        return (0.0, 0.0, None, None) if return_details else (0.0, 0.0)
    
    model.eval()
    with torch.no_grad():
        s_o = data_dict["s_o"].to(device)
        s_h = data_dict["s_h"].to(device)
        s_c = data_dict["s_c"].to(device)
        s_m = data_dict["s_m"].to(device)
        s_n = data_dict["s_n"].to(device)
        label = data_dict["label"].to(device)
        
        if return_details:
            # Thử dùng return_preds nếu model hỗ trợ, không thì fallback
            try:
                acc, mcc, preds = model(s_o, s_h, s_c, s_m, s_n, label, mode="test", return_preds=True)
            except TypeError:
                # Model cũ không hỗ trợ return_preds
                acc, mcc = model(s_o, s_h, s_c, s_m, s_n, label, mode="test")
                # Tính preds thủ công
                with torch.no_grad():
                    v_m, v_i, v_n = model.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)
                    fused_news = model.fusion_news(primary=v_i, aux=v_n)
                    fused_macro = model.fusion_macro(primary=v_i, aux=v_m)
                    v_fused_total = (fused_news + fused_macro) / 2.0
                    logits = model.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
                    preds = torch.argmax(logits, dim=1)
            
            return acc, mcc, preds, label
        else:
            acc, mcc = model(s_o, s_h, s_c, s_m, s_n, label, mode="test")
            return acc, mcc

# --- 4. DETAILED PREDICTION ANALYSIS ---
def analyze_predictions(preds, targets, class_names=["Down", "Flat", "Up"]):
    """
    Phân tích chi tiết predictions để debug class imbalance.
    """
    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    print("\n" + "="*60)
    print("📊 DETAILED PREDICTION ANALYSIS")
    print("="*60)
    
    # 1. Overall distribution
    pred_counts = np.bincount(preds_np, minlength=3)
    true_counts = np.bincount(targets_np, minlength=3)
    
    print(f"\n📉 Ground Truth Distribution: {true_counts}")
    print(f"🔮 Prediction Distribution:   {pred_counts}")
    print(f"   Δ Difference:               {pred_counts - true_counts}")
    
    # 2. Per-class metrics
    print(f"\n{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(
        targets_np, preds_np, labels=[0, 1, 2], zero_division=0
    )
    
    for i, name in enumerate(class_names):
        print(f"{name:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # 3. Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets_np, preds_np, labels=[0, 1, 2])
    
    print("\n📊 Confusion Matrix:")
    print("      Pred Down  Pred Flat  Pred Up")
    for i, name in enumerate(class_names):
        print(f"{name:>8}  {cm[i, 0]:>9}  {cm[i, 1]:>9}  {cm[i, 2]:>7}")
    
    # 4. Critical warnings
    print("\n⚠️  CRITICAL CHECKS:")
    for i, name in enumerate(class_names):
        if pred_counts[i] == 0:
            print(f"   ❌ Class {i} ({name}): NEVER PREDICTED!")
        elif pred_counts[i] < true_counts[i] * 0.3:
            print(f"   ⚠️  Class {i} ({name}): Severely under-predicted ({pred_counts[i]}/{true_counts[i]})")
        elif recall[i] < 0.1:
            print(f"   ⚠️  Class {i} ({name}): Recall too low ({recall[i]:.4f})")
    
    print("="*60 + "\n")
    
    return precision, recall, f1

# --- 5. TRAIN ---
def train_model(train_data, valid_data, test_data):
    if not train_data: return

    s_m_dim = train_data["s_m"].shape[-1]
    train_labels = train_data["label"]
    
    # ============================================================
    # 🎯 CHỌN CHIẾN LƯỢC CÂN BẰNG
    # ============================================================
    # Uncomment 1 trong 4 options sau:
    
    # OPTION 1: SQRT Balancing (Nhẹ nhàng nhất - KHUYẾN NGHỊ THỬ ĐẦU TIÊN)
    class_weights = compute_class_weights_sqrt(train_labels).to(device)
    strategy_name = "SQRT BALANCING"
    
    # OPTION 2: Effective Number of Samples (Có nền tảng lý thuyết)
    # class_weights = compute_class_weights_ens(train_labels, beta=0.95).to(device)
    # strategy_name = "EFFECTIVE NUMBER (beta=0.95)"
    
    # OPTION 3: Clipped Inverse Frequency (An toàn hơn inverse thuần)
    # class_weights = compute_class_weights_clipped(train_labels, max_ratio=5.0).to(device)
    # strategy_name = "CLIPPED INVERSE (max_ratio=5.0)"
    
    # OPTION 4: Tắt hoàn toàn Alpha, chỉ dùng Focal Loss
    # class_weights = None
    # strategy_name = "FOCAL ONLY (No Alpha)"
    
    # ============================================================
    
    class_counts = np.bincount(train_labels.cpu().numpy())
    print("\n" + "="*60)
    print(f"🎯 BALANCING STRATEGY: {strategy_name}")
    print("="*60)
    print(f"   ► Class Distribution: {class_counts}")
    if class_weights is not None:
        print(f"   ► Computed Weights:   {class_weights.cpu().numpy()}")
    else:
        print(f"   ► Weights: None (Uniform)")
    
    # ============================================================
    # 🔧 CHỌN LOSS FUNCTION
    # ============================================================
    # Uncomment 1 trong 3 options:
    
    # OPTION A: Focal Loss với gamma thấp (KHUYẾN NGHỊ)
    use_focal = True
    focal_gamma = 1.0  # Giảm từ 2.0 → 1.0 để phạt nhẹ hơn
    use_smoothing = False
    
    # OPTION B: Label Smoothing (Alternative tốt)
    # use_focal = False
    # use_smoothing = True
    # focal_gamma = 2.0  # Không dùng
    
    # OPTION C: Vanilla Cross Entropy
    # use_focal = False
    # use_smoothing = False
    # focal_gamma = 2.0  # Không dùng
    
    # ============================================================
    
    print(f"\n🚀 Initializing Model on {device}...")
    
    # ============================================================
    # QUAN TRỌNG: Nếu model.py chưa có focal_gamma, comment dòng đó đi
    # ============================================================
    try:
        # Thử khởi tạo với parameters mới
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
            use_focal_loss=use_focal,
            focal_gamma=focal_gamma,        # [KEY] Gamma điều chỉnh được
            use_label_smoothing=use_smoothing,
            smoothing=0.1,
            device=device
        ).to(device)
        print("✅ Using UPDATED model.py with flexible gamma")
    except TypeError:
        # Fallback: Dùng model cũ (chỉ hỗ trợ gamma=2.0 cố định)
        print("⚠️  Using OLD model.py (gamma fixed at 2.0)")
        print("   💡 To enable flexible gamma, update src/model.py with the artifact code")
        
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
            use_focal_loss=use_focal,
            device=device
        ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TrainConfig.learning_rate,
        weight_decay=1e-4
    )

    best_val_mcc = -1.0
    best_val_acc = 0.0
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_model.pt")

    print("\n⚔️  STARTING TRAINING...")
    print("="*60)

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
        if not torch.isfinite(loss): 
            print(f"⚠️  Training stopped: Loss became {loss.item()}")
            break
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        val_acc, val_mcc = evaluate(model, valid_data)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:03d} | Loss {loss.item():.4f} | Val ACC {val_acc:.4f} | Val MCC {val_mcc:.4f}")

        # Lưu model tốt nhất theo MCC (ưu tiên), ACC (phụ)
        is_best = False
        if val_mcc > best_val_mcc:
            is_best = True
        elif val_mcc == best_val_mcc and val_acc > best_val_acc:
            is_best = True
            
        if is_best:
            best_val_mcc = val_mcc
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"   >>> ✨ New Best Model! (MCC: {val_mcc:.4f}, Acc: {val_acc:.4f})")

    # =========================================================
    # 🏁 FINAL EVALUATION & DEEP ANALYSIS
    # =========================================================
    print("\n" + "="*60)
    print("🏁 FINAL EVALUATION")
    print("="*60)
    
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        
        # 1. Validation sanity check
        print("\n🔍 VALIDATION SET (Sanity Check):")
        val_acc, val_mcc, val_preds, val_targets = evaluate(model, valid_data, return_details=True)
        print(f"   ACC: {val_acc:.4f} | MCC: {val_mcc:.4f}")
        if val_mcc > 0.1:
            print("   ✅ Model is learning meaningful patterns")
        else:
            print("   ⚠️  MCC too low - model might be guessing randomly")
        
        # 2. Test set evaluation
        print("\n🔍 TEST SET (Final Performance):")
        test_acc, test_mcc, test_preds, test_targets = evaluate(model, test_data, return_details=True)
        print(f"   ACC: {test_acc:.4f} | MCC: {test_mcc:.4f}")
        
        # 3. Deep analysis of test predictions
        if test_preds is not None:
            analyze_predictions(test_preds, test_targets)
        
        # 4. Confidence analysis (optional - chỉ chạy nếu model hỗ trợ)
        try:
            print("\n🔍 CONFIDENCE ANALYSIS (Sample):")
            model.eval()
            with torch.no_grad():
                probs, preds, confidence = model.get_prediction_confidence(
                    test_data["s_o"][:20].to(device),
                    test_data["s_h"][:20].to(device),
                    test_data["s_c"][:20].to(device),
                    test_data["s_m"][:20].to(device),
                    test_data["s_n"][:20].to(device)
                )
                
                print(f"   Average Confidence: {confidence.mean():.4f}")
                print(f"   Min Confidence: {confidence.min():.4f}")
                print(f"   Max Confidence: {confidence.max():.4f}")
                
                # Check if model is overconfident
                if confidence.mean() > 0.9:
                    print("   ⚠️  Model might be overconfident!")
                    print("   💡 Consider: Label Smoothing or Temperature Scaling")
        except AttributeError:
            print("\n   (Skipping confidence analysis - update model.py to enable)")
    else:
        print("⚠️ No best model saved.")

if __name__ == "__main__":
    pkl_path = r"D:\ProjectNCKH\deep_finance\data\processed\unified_dataset_test.pkl"
    dp = data_prepare(pkl_path)
    
    target_tickers = ["TSLA", "AMZN", "MSFT", "NFLX"]
    
    list_train, list_valid, list_test = [], [], []
    for ticker in target_tickers:
        try:
            tr, val, te = dp.prepare_data(ticker)
            if tr and len(tr.get("label", [])) > 0:
                list_train.append(tr)
                list_valid.append(val)
                list_test.append(te)
        except Exception as e:
            print(f"⚠️  Failed to load {ticker}: {e}")

    final_train = merge_datasets(list_train, shuffle=True)
    final_valid = merge_datasets(list_valid, shuffle=False)
    final_test  = merge_datasets(list_test, shuffle=False)

    if len(final_train) > 0:
        train_model(final_train, final_valid, final_test)
    else:
        print("❌ No training data available!")