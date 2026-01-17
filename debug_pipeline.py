import torch
import numpy as np
import os
import sys
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

# Import project modules
from src.model import StockMovementModel
from src.data_loader import data_prepare
from configs.config import TrainConfig, GlobalConfig

# --- CONFIG ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join("output", "best_model.pt")
DATA_PATH = r"D:\DeepFinance\data\processed\unified_dataset_test.pkl" # Đảm bảo đường dẫn đúng

def print_header(title):
    print(f"\n{'='*60}")
    print(f"🔎 {title}")
    print(f"{'='*60}")

def load_data_per_ticker(tickers):
    """
    Load dữ liệu Test riêng biệt cho từng mã để phân tích behavior.
    """
    dp = data_prepare(DATA_PATH)
    ticker_datasets = {}
    
    print(f"📥 Loading TEST data for: {tickers}")
    for t in tickers:
        try:
            # Chỉ lấy tập Test (index 2)
            _, _, test_data = dp.prepare_data(
                stock_name=t,
                window_size=TrainConfig.window_size,
                # Các tham số khác lấy từ config mặc định trong data_loader
            )
            
            if test_data and len(test_data.get("label", [])) > 0:
                ticker_datasets[t] = test_data
                print(f"   ✅ {t}: {len(test_data['label'])} samples")
            else:
                print(f"   ⚠️ {t}: No data")
        except Exception as e:
            print(f"   ❌ {t}: Error {e}")
            
    return ticker_datasets

def run_forward_pass_manually(model, data_dict):
    """
    Chạy forward pass thủ công để lấy Logits (vì hàm forward của model.py trả về loss/acc)
    """
    model.eval()
    with torch.no_grad():
        s_o = data_dict["s_o"].to(DEVICE)
        s_h = data_dict["s_h"].to(DEVICE)
        s_c = data_dict["s_c"].to(DEVICE)
        s_m = data_dict["s_m"].to(DEVICE)
        s_n = data_dict["s_n"].to(DEVICE)
        
        # 1. Encoder
        v_m, v_i, v_n = model.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)
        
        # 2. Fusion
        fused_news = model.fusion_news(primary=v_i, aux=v_n)
        fused_macro = model.fusion_macro(primary=v_i, aux=v_m)
        v_fused_total = (fused_news + fused_macro) / 2.0
        
        # 3. Predictor -> Logits
        logits = model.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
        
        # Get Predictions & Probs
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
    return preds.cpu().numpy(), data_dict["label"].numpy(), probs.cpu().numpy()

def analyze_performance():
    # 1. Load Model
    print_header("1. LOADING MODEL")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Cannot find model at {MODEL_PATH}")
        return

    # Khởi tạo model architecture (Dummy params để load weights)
    # Lưu ý: Cần biết dimension của macro từ dữ liệu thật, ở đây giả định lấy từ config hoặc hardcode nếu cần
    # Để an toàn, ta load 1 mẫu data trước để lấy macro_dim
    dp = data_prepare(DATA_PATH)
    dummy_train, _, _ = dp.prepare_data("TSLA") 
    macro_dim = dummy_train["s_m"].shape[-1] if dummy_train else 6 # Fallback
    
    model = StockMovementModel(
        price_dim=1,
        macro_dim=macro_dim,
        news_dim=TrainConfig.news_embed_dim,
        dim=TrainConfig.dim,
        input_dim=TrainConfig.window_size,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        dropout=0.0, # Dropout không quan trọng khi eval
        device=DEVICE
    ).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Weights loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading weights: {e}")
        return

    # 2. Load Data
    print_header("2. LOADING DATA")
    # List các mã bạn đã train thành công
    target_tickers = ["TSLA", "AMZN", "MSFT", "NFLX"] 
    datasets = load_data_per_ticker(target_tickers)
    
    if not datasets:
        print("❌ No datasets loaded.")
        return

    # 3. Deep Dive Analysis
    print_header("3. DEEP DIVE ANALYSIS")
    
    all_preds = []
    all_labels = []
    
    print(f"{'TICKER':<10} | {'SAMPLES':<8} | {'ACTUAL DIST (0/1/2)':<25} | {'PRED DIST (0/1/2)':<25} | {'ACC':<8} | {'MCC':<8}")
    print("-" * 110)

    for ticker, data in datasets.items():
        preds, labels, probs = run_forward_pass_manually(model, data)
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        
        # Calculate Stats per Ticker
        from sklearn.metrics import accuracy_score, matthews_corrcoef
        acc = accuracy_score(labels, preds)
        mcc = matthews_corrcoef(labels, preds)
        
        # Count distributions
        act_counts = Counter(labels)
        pred_counts = Counter(preds)
        
        act_dist = f"{act_counts.get(0,0)}/{act_counts.get(1,0)}/{act_counts.get(2,0)}"
        pred_dist = f"{pred_counts.get(0,0)}/{pred_counts.get(1,0)}/{pred_counts.get(2,0)}"
        
        print(f"{ticker:<10} | {len(labels):<8} | {act_dist:<25} | {pred_dist:<25} | {acc:.4f}   | {mcc:.4f}")

    # 4. Global Analysis
    print_header("4. GLOBAL SUMMARY (ALL TICKERS COMBINED)")
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    unique_act, counts_act = np.unique(all_labels, return_counts=True)
    unique_pred, counts_pred = np.unique(all_preds, return_counts=True)
    
    print("📉 ACTUAL Labels Distribution (Ground Truth):")
    print(f"   {dict(zip(unique_act, counts_act))}")
    
    print("\n🔮 PREDICTED Labels Distribution:")
    print(f"   {dict(zip(unique_pred, counts_pred))}")
    
    # Check Mode Collapse
    if len(unique_pred) == 1:
        print("\n⚠️  CRITICAL WARNING: MODE COLLAPSE DETECTED!")
        print(f"   Mô hình chỉ dự đoán duy nhất lớp {unique_pred[0]} cho toàn bộ dữ liệu.")
        print("   -> Đây là lý do MCC = 0.0000")
    
    print("\n📊 Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
    print(f"      Pred 0  Pred 1  Pred 2")
    print(f"Act 0   {cm[0][0]:<7} {cm[0][1]:<7} {cm[0][2]:<7}")
    print(f"Act 1   {cm[1][0]:<7} {cm[1][1]:<7} {cm[1][2]:<7}")
    print(f"Act 2   {cm[2][0]:<7} {cm[2][1]:<7} {cm[2][2]:<7}")

if __name__ == "__main__":
    analyze_performance()