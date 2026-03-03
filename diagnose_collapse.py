# diagnose_collapse.py - Comprehensive Diagnostic Tool
"""
Mục tiêu: Tìm nguyên nhân gốc rễ của model collapse (predict 100% DOWN)

Kiểm tra theo thứ tự:
1. Data Quality - Feature statistics, missing values
2. Graph Quality - Structure, node features, edge connectivity
3. Feature Scale - Magnitude mismatch giữa các modalities
4. Gate Analysis - Fusion gate values
5. Gradient Flow - Xem gradients có chảy về News/Macro không
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"🔍 {title}")
    print(f"{'='*70}")


def print_subsection(title: str):
    print(f"\n--- {title} ---")


# ============================================================
# SECTION 1: DATA QUALITY ANALYSIS
# ============================================================

def analyze_data_quality(data_path: str):
    """Phân tích chất lượng dữ liệu tổng thể"""
    print_section("1. DATA QUALITY ANALYSIS")
    
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found: {data_path}")
        return None
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"📦 Total trading days: {len(data)}")
    
    # Sample dates
    dates = sorted(data.keys())
    sample_date = dates[len(dates)//2]
    day_data = data[sample_date]
    
    print(f"\n📊 Sample date: {sample_date}")
    print(f"   Available keys: {list(day_data.keys())}")
    
    # Check each modality
    stats = {
        'price_available': 0,
        'macro_available': 0,
        'news_embedding_available': 0,
        'kg_tensor_available': 0,
        'kg_tensor_exists_on_disk': 0,
    }
    
    tickers = ['TSLA', 'AMZN', 'MSFT', 'NFLX']
    
    for date_key in dates:
        dd = data[date_key]
        
        # Price
        if 'price' in dd and dd['price']:
            for t in tickers:
                if t in dd['price'] and dd['price'][t]:
                    stats['price_available'] += 1
                    break
        
        # Macro
        if 'macro' in dd and dd['macro']:
            stats['macro_available'] += 1
        
        # News embedding (old format)
        if 'news_embedding' in dd and dd['news_embedding']:
            for t in tickers:
                if t in dd['news_embedding'] and dd['news_embedding'][t]:
                    stats['news_embedding_available'] += 1
                    break
        
        # KG tensor (new format)
        if 'kg_tensor' in dd and dd['kg_tensor']:
            for t in tickers:
                if t in dd['kg_tensor']:
                    stats['kg_tensor_available'] += 1
                    path = dd['kg_tensor'][t]
                    if path and os.path.exists(path):
                        stats['kg_tensor_exists_on_disk'] += 1
                    break
    
    print(f"\n📈 Data Availability (out of {len(dates)} days):")
    for key, val in stats.items():
        pct = val / len(dates) * 100
        status = "✅" if pct > 90 else ("⚠️" if pct > 50 else "❌")
        print(f"   {status} {key}: {val}/{len(dates)} ({pct:.1f}%)")
    
    return data, dates


# ============================================================
# SECTION 2: FEATURE SCALE ANALYSIS
# ============================================================

def analyze_feature_scales(data_path: str):
    """Phân tích scale của các features"""
    print_section("2. FEATURE SCALE ANALYSIS")
    
    from src.data_loader import data_prepare
    from configs.config import GlobalConfig
    
    dp = data_prepare(data_path)
    
    # Load one ticker
    train, valid, test = dp.prepare_data("TSLA")
    
    if not train:
        print("❌ No data loaded")
        return
    
    print_subsection("Price Features (after Z-score)")
    for name, tensor in [('s_o', train['s_o']), ('s_h', train['s_h']), ('s_c', train['s_c'])]:
        print(f"   {name}: mean={tensor.mean():.4f}, std={tensor.std():.4f}, "
              f"min={tensor.min():.4f}, max={tensor.max():.4f}")
    
    print_subsection("Macro Features (after Z-score)")
    s_m = train['s_m']
    print(f"   s_m: mean={s_m.mean():.4f}, std={s_m.std():.4f}, "
          f"min={s_m.min():.4f}, max={s_m.max():.4f}")
    print(f"   s_m shape: {s_m.shape}")
    
    print_subsection("News/KG Graph Features")
    graphs = train.get('s_n_graphs', [])
    if graphs:
        # Analyze graph statistics
        node_counts = []
        edge_counts = []
        node_magnitudes = []
        
        for g in graphs[:100]:  # Sample 100
            if g is not None and hasattr(g, 'x'):
                node_counts.append(g.x.shape[0])
                if g.x.numel() > 0:
                    node_magnitudes.append(g.x.norm(dim=-1).mean().item())
                if hasattr(g, 'edge_index'):
                    edge_counts.append(g.edge_index.shape[1])
        
        if node_counts:
            print(f"   Nodes per graph: mean={np.mean(node_counts):.1f}, "
                  f"min={np.min(node_counts)}, max={np.max(node_counts)}")
        if edge_counts:
            print(f"   Edges per graph: mean={np.mean(edge_counts):.1f}, "
                  f"min={np.min(edge_counts)}, max={np.max(edge_counts)}")
        if node_magnitudes:
            print(f"   Node feature L2 norm: mean={np.mean(node_magnitudes):.4f}, "
                  f"std={np.std(node_magnitudes):.4f}")
            
            # ⚠️ Critical check
            if np.mean(node_magnitudes) > 10:
                print(f"   ⚠️ WARNING: Node features have LARGE magnitude!")
                print(f"      This can cause scale mismatch with Z-scored price/macro!")
            elif np.mean(node_magnitudes) < 0.01:
                print(f"   ⚠️ WARNING: Node features are near-ZERO!")
    else:
        print("   ❌ No graph data found in train set")
    
    print_subsection("Label Distribution")
    labels = train['label'].numpy()
    for i, name in enumerate(['DOWN', 'FLAT', 'UP']):
        count = (labels == i).sum()
        pct = count / len(labels) * 100
        print(f"   {name}: {count} ({pct:.1f}%)")
    
    return train


# ============================================================
# SECTION 3: GRAPH QUALITY DEEP DIVE
# ============================================================

def analyze_graph_quality(data_path: str, interim_path: str):
    """Phân tích chi tiết chất lượng Knowledge Graph"""
    print_section("3. KNOWLEDGE GRAPH QUALITY ANALYSIS")
    
    kg_tensor_dir = os.path.join(interim_path, "kg", "tensors")
    kg_stable_dir = os.path.join(interim_path, "kg", "window_graph_stable")
    
    # Check directories exist
    if not os.path.exists(kg_tensor_dir):
        print(f"❌ KG tensor directory not found: {kg_tensor_dir}")
        return
    
    print_subsection("3.1 Tensor File Analysis")
    
    tickers = ['TSLA', 'AMZN', 'MSFT', 'NFLX']
    
    for ticker in tickers:
        ticker_dir = os.path.join(kg_tensor_dir, ticker)
        if not os.path.exists(ticker_dir):
            print(f"   ❌ {ticker}: No tensor directory")
            continue
        
        files = [f for f in os.listdir(ticker_dir) if f.endswith('.pt')]
        print(f"\n   📊 {ticker}: {len(files)} tensor files")
        
        # Sample some files
        sample_files = files[:5] if len(files) >= 5 else files
        
        empty_graphs = 0
        total_nodes = 0
        total_edges = 0
        emb_dims = set()
        zero_emb_count = 0
        
        for f in files:
            fpath = os.path.join(ticker_dir, f)
            try:
                tensor_data = torch.load(fpath, map_location='cpu')
                
                if isinstance(tensor_data, dict):
                    x = tensor_data.get('node_x', tensor_data.get('x'))
                    edge_index = tensor_data.get('edge_index')
                    graph_emb = tensor_data.get('graph_emb')
                    
                    if x is not None:
                        total_nodes += x.shape[0]
                        if x.shape[0] == 0 or x.sum() == 0:
                            empty_graphs += 1
                    
                    if edge_index is not None:
                        total_edges += edge_index.shape[1]
                    
                    if graph_emb is not None:
                        emb_dims.add(len(graph_emb) if isinstance(graph_emb, list) else graph_emb.shape[-1])
                        if isinstance(graph_emb, torch.Tensor) and graph_emb.abs().sum() < 1e-6:
                            zero_emb_count += 1
                        elif isinstance(graph_emb, list) and sum(abs(x) for x in graph_emb) < 1e-6:
                            zero_emb_count += 1
                            
            except Exception as e:
                print(f"      ❌ Error loading {f}: {e}")
        
        avg_nodes = total_nodes / len(files) if files else 0
        avg_edges = total_edges / len(files) if files else 0
        
        print(f"      Avg nodes/graph: {avg_nodes:.1f}")
        print(f"      Avg edges/graph: {avg_edges:.1f}")
        print(f"      Empty graphs: {empty_graphs}/{len(files)} ({empty_graphs/len(files)*100:.1f}%)")
        print(f"      Zero embeddings: {zero_emb_count}/{len(files)} ({zero_emb_count/len(files)*100:.1f}%)")
        print(f"      Embedding dims: {emb_dims}")
        
        # ⚠️ Critical warnings
        if empty_graphs / len(files) > 0.5:
            print(f"      ⚠️ WARNING: Over 50% graphs are empty!")
        if zero_emb_count / len(files) > 0.5:
            print(f"      ⚠️ WARNING: Over 50% graph embeddings are zero!")
        if avg_edges < 1:
            print(f"      ⚠️ WARNING: Very few edges - GNN cannot propagate information!")
    
    print_subsection("3.2 Triple Quality Analysis")
    
    if os.path.exists(kg_stable_dir):
        for ticker in tickers[:1]:  # Just analyze one ticker
            ticker_stable = os.path.join(kg_stable_dir, ticker)
            if not os.path.exists(ticker_stable):
                continue
            
            files = [f for f in os.listdir(ticker_stable) if f.endswith('.json')][:10]
            
            print(f"\n   📋 {ticker} - Sample triples from stable graphs:")
            
            for f in files[:3]:
                fpath = os.path.join(ticker_stable, f)
                try:
                    with open(fpath, 'r', encoding='utf-8') as fp:
                        data = json.load(fp)
                    
                    triples = data.get('triples', [])
                    print(f"\n      Date: {data.get('date')}")
                    print(f"      Num triples: {len(triples)}")
                    
                    # Show sample triples
                    for t in triples[:3]:
                        if isinstance(t, (list, tuple)) and len(t) == 3:
                            print(f"         ({t[0][:30]}..., {t[1]}, {t[2][:30]}...)")
                            
                except Exception as e:
                    print(f"      ❌ Error: {e}")


# ============================================================
# SECTION 4: FUSION GATE ANALYSIS
# ============================================================

def analyze_fusion_gates(model_path: str, data_path: str):
    """Phân tích giá trị của fusion gates"""
    print_section("4. FUSION GATE ANALYSIS")
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    from src.model import StockMovementModel
    from src.data_loader import data_prepare
    from configs.config import TrainConfig
    
    device = torch.device('cpu')
    
    # Load data
    dp = data_prepare(data_path)
    train, _, test = dp.prepare_data("TSLA")
    
    if not test:
        print("❌ No test data")
        return
    
    s_m_dim = train['s_m'].shape[-1]
    
    # Load model
    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m_dim,
        news_dim=TrainConfig.news_embed_dim,
        dim=TrainConfig.dim,
        input_dim=TrainConfig.window_size,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        dropout=0.0,
        class_weights=torch.tensor([1.0, 1.0, 1.0]),
        use_focal_loss=True,
        device=device,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print_subsection("4.1 Gate Weight Statistics")
    
    # Extract gate weights
    news_gate_weight = model.fusion_news.gate_linear.weight.data
    news_gate_bias = model.fusion_news.gate_linear.bias.data
    
    macro_gate_weight = model.fusion_macro.gate_linear.weight.data
    macro_gate_bias = model.fusion_macro.gate_linear.bias.data
    
    print(f"   News Gate Linear:")
    print(f"      Weight: mean={news_gate_weight.mean():.4f}, std={news_gate_weight.std():.4f}")
    print(f"      Bias: mean={news_gate_bias.mean():.4f}, std={news_gate_bias.std():.4f}")
    
    print(f"\n   Macro Gate Linear:")
    print(f"      Weight: mean={macro_gate_weight.mean():.4f}, std={macro_gate_weight.std():.4f}")
    print(f"      Bias: mean={macro_gate_bias.mean():.4f}, std={macro_gate_bias.std():.4f}")
    
    print_subsection("4.2 Gate Activation Values (on test data)")
    
    # Hook to capture gate values
    gate_values_news = []
    gate_values_macro = []
    
    def hook_news_gate(module, input, output):
        # output is after sigmoid
        pass
    
    # Manual forward to capture gates
    with torch.no_grad():
        # Take a batch
        batch_size = min(32, len(test['label']))
        
        s_o = test['s_o'][:batch_size]
        s_h = test['s_h'][:batch_size]
        s_c = test['s_c'][:batch_size]
        s_m = test['s_m'][:batch_size]
        graphs = test['s_n_graphs'][:batch_size]
        
        # Forward through encoder
        v_m, v_i, _ = model.multimodal_encoder(s_o, s_h, s_c, s_m, None)
        v_n = model._encode_graphs(graphs)
        
        # Manually compute gates
        # News gate
        news_gate_input = model.fusion_news.gate_linear(v_i)
        news_gate = torch.sigmoid(news_gate_input)
        
        # Macro gate
        macro_gate_input = model.fusion_macro.gate_linear(v_i)
        macro_gate = torch.sigmoid(macro_gate_input)
        
        print(f"\n   News Gate Values (sampled):")
        print(f"      Mean: {news_gate.mean():.4f}")
        print(f"      Std:  {news_gate.std():.4f}")
        print(f"      Min:  {news_gate.min():.4f}")
        print(f"      Max:  {news_gate.max():.4f}")
        
        # ⚠️ Critical check
        if news_gate.mean() < 0.1:
            print(f"      ⚠️ WARNING: News gate is mostly CLOSED (mean < 0.1)!")
            print(f"         → Model is ignoring news information!")
        elif news_gate.mean() > 0.9:
            print(f"      ⚠️ WARNING: News gate is mostly OPEN (mean > 0.9)!")
            print(f"         → Model might be over-relying on potentially noisy news!")
        
        print(f"\n   Macro Gate Values (sampled):")
        print(f"      Mean: {macro_gate.mean():.4f}")
        print(f"      Std:  {macro_gate.std():.4f}")
        print(f"      Min:  {macro_gate.min():.4f}")
        print(f"      Max:  {macro_gate.max():.4f}")
        
        if macro_gate.mean() < 0.1:
            print(f"      ⚠️ WARNING: Macro gate is mostly CLOSED!")


# ============================================================
# SECTION 5: GRADIENT FLOW ANALYSIS
# ============================================================

def analyze_gradient_flow(model_path: str, data_path: str):
    """Kiểm tra gradient có chảy về tất cả các modalities không"""
    print_section("5. GRADIENT FLOW ANALYSIS")
    
    from src.model import StockMovementModel
    from src.data_loader import data_prepare
    from configs.config import TrainConfig
    
    device = torch.device('cpu')
    
    dp = data_prepare(data_path)
    train, _, _ = dp.prepare_data("TSLA")
    
    if not train:
        print("❌ No data")
        return
    
    s_m_dim = train['s_m'].shape[-1]
    
    model = StockMovementModel(
        price_dim=1,
        macro_dim=s_m_dim,
        news_dim=TrainConfig.news_embed_dim,
        dim=TrainConfig.dim,
        input_dim=TrainConfig.window_size,
        output_dim=TrainConfig.output_dim,
        num_head=TrainConfig.num_head,
        dropout=0.1,
        class_weights=torch.tensor([1.0, 1.0, 1.0]),
        use_focal_loss=True,
        device=device,
    ).to(device)
    
    model.train()
    
    # Take a small batch
    batch_size = 4
    s_o = train['s_o'][:batch_size].requires_grad_(True)
    s_h = train['s_h'][:batch_size].requires_grad_(True)
    s_c = train['s_c'][:batch_size].requires_grad_(True)
    s_m = train['s_m'][:batch_size].requires_grad_(True)
    graphs = train['s_n_graphs'][:batch_size]
    labels = train['label'][:batch_size]
    
    # Forward
    loss = model(s_o, s_h, s_c, s_m, graphs, labels, mode='train')
    loss.backward()
    
    print_subsection("5.1 Input Gradients")
    
    print(f"   Price (s_o) gradient: {'✅ EXISTS' if s_o.grad is not None and s_o.grad.abs().sum() > 0 else '❌ NONE/ZERO'}")
    print(f"   Price (s_h) gradient: {'✅ EXISTS' if s_h.grad is not None and s_h.grad.abs().sum() > 0 else '❌ NONE/ZERO'}")
    print(f"   Price (s_c) gradient: {'✅ EXISTS' if s_c.grad is not None and s_c.grad.abs().sum() > 0 else '❌ NONE/ZERO'}")
    print(f"   Macro (s_m) gradient: {'✅ EXISTS' if s_m.grad is not None and s_m.grad.abs().sum() > 0 else '❌ NONE/ZERO'}")
    
    if s_o.grad is not None:
        print(f"\n   Price gradient magnitude: {s_o.grad.abs().mean():.6f}")
    if s_m.grad is not None:
        print(f"   Macro gradient magnitude: {s_m.grad.abs().mean():.6f}")
    
    print_subsection("5.2 Module Gradients")
    
    modules_to_check = [
        ('Price Encoder', model.multimodal_encoder.indicator_encoder),
        ('Macro Encoder', model.multimodal_encoder.macro_encoder),
        ('KG Encoder', model.kg_encoder if hasattr(model, 'kg_encoder') else None),
        ('News Fusion', model.fusion_news),
        ('Macro Fusion', model.fusion_macro),
        ('Predictor', model.movement_predictor),
    ]
    
    for name, module in modules_to_check:
        if module is None:
            print(f"   {name}: N/A")
            continue
        
        has_grad = False
        grad_sum = 0.0
        param_count = 0
        
        for param in module.parameters():
            if param.grad is not None:
                has_grad = True
                grad_sum += param.grad.abs().sum().item()
                param_count += param.numel()
        
        avg_grad = grad_sum / param_count if param_count > 0 else 0
        status = "✅" if has_grad and avg_grad > 1e-8 else "⚠️"
        print(f"   {status} {name}: grad_mean={avg_grad:.8f}")


# ============================================================
# SECTION 6: RECOMMENDATIONS
# ============================================================

def generate_recommendations(findings: dict):
    """Generate actionable recommendations based on findings"""
    print_section("6. RECOMMENDATIONS")
    
    print("""
Based on the analysis, here are the prioritized fixes:

🔴 CRITICAL (Fix immediately):
─────────────────────────────
1. FEATURE SCALE NORMALIZATION
   - Graph embeddings (L2 norm ~1.0) vs Price (Z-score ~0-1)
   - Solution: Add LayerNorm after graph encoding
   
2. GRAPH QUALITY CHECK
   - Many graphs may be empty or have zero embeddings
   - Solution: Add fallback mechanisms, improve triple extraction

🟡 IMPORTANT (Fix after critical):
──────────────────────────────────
3. FUSION GATE INITIALIZATION
   - Gates may be biased to close (block aux info)
   - Solution: Initialize gate bias to positive value (e.g., 0.5)
   
4. LOSS FUNCTION TUNING
   - Focal loss γ may be too high
   - Solution: Try γ=1.0 instead of 2.0

🟢 NICE TO HAVE:
────────────────
5. Learning rate scheduling
6. Gradient clipping adjustment
7. More sophisticated graph pooling
""")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("🔬 COMPREHENSIVE MODEL COLLAPSE DIAGNOSTIC")
    print("=" * 70)
    
    # Paths
    from configs.config import GlobalConfig
    
    data_path = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    model_path = os.path.join("output", "best_model.pt")
    interim_path = GlobalConfig.INTERIM_PATH
    
    print(f"\n📁 Data path: {data_path}")
    print(f"📁 Model path: {model_path}")
    print(f"📁 Interim path: {interim_path}")
    
    # Run diagnostics
    analyze_data_quality(data_path)
    analyze_feature_scales(data_path)
    analyze_graph_quality(data_path, interim_path)
    
    if os.path.exists(model_path):
        analyze_fusion_gates(model_path, data_path)
        analyze_gradient_flow(model_path, data_path)
    else:
        print("\n⚠️ Model not found, skipping gate and gradient analysis")
    
    generate_recommendations({})
    
    print("\n" + "=" * 70)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()