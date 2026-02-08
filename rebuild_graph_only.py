# rebuild_graph_only.py - FIXED VERSION
"""
Rebuild KG graphs from cached article triples (NO LLM calls).

This script:
1. Loads cached article-level triples (from previous LLM extraction)
2. Rebuilds window graphs (rolling 20-day aggregation)
3. Performs entity resolution with Voyage embeddings
4. Encodes graphs with GNN (GraphSAGE)
5. Generates embedded_kg.json index

⚠️ PREREQUISITES:
- Article-level triples cache must exist at: data/interim/kg_article_cache/
- Voyage API key must be set (for entity embeddings)
- PyTorch Geometric must be installed

✅ NO LLM CALLS - only uses cached triples
✅ Voyage API calls - for entity resolution and node features
"""

import os
import shutil
import time

from configs.config import GlobalConfig


def main():
    print("=" * 60)
    print("🚀 REBUILD GRAPH-ONLY (NO LLM)")
    print("=" * 60)
    
    # ==========================================
    # 1. BACKUP OLD KG INDEX
    # ==========================================
    kg_index = os.path.join(GlobalConfig.INTERIM_PATH, "kg_embeddings", "embedded_kg.json")
    
    if os.path.exists(kg_index):
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(GlobalConfig.INTERIM_PATH, "kg_embeddings")
        os.makedirs(backup_dir, exist_ok=True)
        
        bak_path = os.path.join(backup_dir, f"embedded_kg.backup_{ts}.json")
        shutil.copy2(kg_index, bak_path)
        print(f"\n🧷 Backup old KG index:")
        print(f"   {bak_path}")
    else:
        print("\n⚠️  No existing KG index found (first run)")
    
    # ==========================================
    # 2. CHECK PREREQUISITES
    # ==========================================
    print("\n🔍 Checking prerequisites...")
    
    # Check cache directory
    cache_dir = os.path.join(GlobalConfig.INTERIM_PATH, "kg_article_cache")
    if not os.path.exists(cache_dir):
        print(f"\n❌ ERROR: Article cache not found at: {cache_dir}")
        print("   You need to extract triples first:")
        print("   → python build_kg.py")
        return
    
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.json')]
    num_cached = len(cache_files)
    print(f"   ✅ Article cache: {num_cached} files")
    
    if num_cached == 0:
        print(f"\n❌ ERROR: Cache directory is empty!")
        print("   Extract triples first: python build_kg.py")
        return
    
    # Check news data
    news_path = os.path.join(GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet")
    if not os.path.exists(news_path):
        print(f"\n❌ ERROR: News data not found at: {news_path}")
        return
    
    print(f"   ✅ News data: {news_path}")
    
    # Check Voyage API key
    voyage_key = os.getenv("VOYAGE_API_KEY", GlobalConfig.VOYAGE_API_KEY)
    if not voyage_key or voyage_key in ["---", "YOUR_API_KEY"]:
        print(f"\n❌ ERROR: VOYAGE_API_KEY not set!")
        print("   Set environment variable:")
        print("   → $env:VOYAGE_API_KEY='your_key_here'  (Windows PowerShell)")
        print("   → export VOYAGE_API_KEY='your_key_here'  (Linux/Mac)")
        return
    
    print(f"   ✅ Voyage API key: {voyage_key[:8]}...{voyage_key[-4:]}")
    
    # ==========================================
    # 3. INITIALIZE EMBEDDER WITH GNN
    # ==========================================
    print("\n🔧 Initializing KG Embedder with GNN...")
    
    from data_pipeline.processors.news_processor import KGGenNewsEmbedder
    
    embedder = KGGenNewsEmbedder(
        interim_root=GlobalConfig.INTERIM_PATH,
        
        # Window & Aggregation
        window_days=20,                      # Rolling window size
        max_triples_cap_per_day=None,       # No limit per day
        top_triples_per_article=5,          # Top-5 triples per article
        
        # Entity Resolution
        kmeans_k=128,                        # Number of clusters
        use_voyage_resolution=True,          # ✅ Use Voyage for entity resolution
        
        # Node Features
        use_voyage_node_features=True,       # ✅ Use Voyage embeddings as node features
        
        # LLM Control
        allow_llm_when_missing=False,        # ✅ CRITICAL: NO LLM calls
        enable_cache=True,                   # Use article cache
        
        # ✅ GNN ENCODER CONFIGURATION (CRITICAL!)
        use_graph_encoder_embedding=True,    # Enable GNN encoding
        graph_out_dim=128,                   # Output dimension (match config)
        graph_hidden_dim=128,                # Hidden dimension
        graph_num_layers=2,                  # Number of GraphSAGE layers
        graph_dropout=0.1,                   # Dropout rate
        graph_use_gat=False,                 # Use GraphSAGE (not GAT)
        
        # Debug
        debug_print_samples=False,           # Set True to see graph details
    )
    
    print("   ✅ Embedder initialized")
    print(f"      Window: {embedder.window_days} days")
    print(f"      GNN: {'GAT' if embedder.use_graph_encoder_embedding and embedder.graph_encoder and hasattr(embedder.graph_encoder, 'use_gat') and embedder.graph_encoder.use_gat else 'GraphSAGE'}")
    print(f"      GNN layers: {len(embedder.graph_encoder.sage_layers) if embedder.graph_encoder else 'N/A'}")
    print(f"      Output dim: {embedder.graph_out_dim}")
    
    # ==========================================
    # 4. REBUILD GRAPHS
    # ==========================================
    print("\n🔄 Rebuilding graphs from cached triples...")
    print("   (This will take 5-15 minutes depending on data size)")
    print("")
    
    start_time = time.time()
    
    try:
        result_path = embedder.rebuild_graph_only()
        
        elapsed = time.time() - start_time
        print(f"\n⏱️  Time elapsed: {elapsed/60:.1f} minutes")
        
        # ==========================================
        # 5. VERIFY OUTPUTS
        # ==========================================
        print("\n🔍 Verifying outputs...")
        
        if os.path.exists(result_path):
            print(f"   ✅ KG index: {result_path}")
            
            import json
            with open(result_path, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
            
            num_dates = len(kg_data)
            print(f"   ✅ Generated {num_dates} date entries")
            
            # Check sample
            if num_dates > 0:
                sample_date = list(kg_data.keys())[min(10, num_dates-1)]
                sample_records = kg_data[sample_date]
                
                print(f"\n   📊 Sample date: {sample_date}")
                print(f"      Records: {len(sample_records)}")
                
                if sample_records:
                    rec = sample_records[0]
                    print(f"      Ticker: {rec.get('equity')}")
                    
                    if 'kg_tensor_path' in rec:
                        path = rec['kg_tensor_path']
                        if os.path.exists(path):
                            import torch
                            tensor = torch.load(path, map_location='cpu')
                            print(f"      ✅ Tensor file exists")
                            print(f"         Nodes: {tensor['node_x'].shape[0] if 'node_x' in tensor else 'N/A'}")
                            print(f"         Edges: {tensor['edge_index'].shape[1] if 'edge_index' in tensor else 'N/A'}")
                            print(f"         Graph emb dim: {len(tensor['graph_emb']) if 'graph_emb' in tensor else 'N/A'}")
                        else:
                            print(f"      ⚠️  Tensor file missing: {path}")
                    else:
                        print(f"      ⚠️  No kg_tensor_path in record")
        else:
            print(f"   ❌ Result path not found: {result_path}")
        
        print("\n" + "=" * 60)
        print("✅ GRAPH REBUILD COMPLETED!")
        print("=" * 60)
        print("\n📋 Next steps:")
        print("   1. Verify dataset: python verify_dataset.py")
        print("   2. Build unified dataset: python build_dataset.py")
        print("   3. Train model: python main.py")
        
    except Exception as e:
        print(f"\n❌ ERROR during rebuild: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n💡 Troubleshooting:")
        print("   1. Check if article cache has valid triples")
        print("   2. Verify Voyage API key is valid")
        print("   3. Check if news data is properly formatted")
        print("   4. Try: python verify_cache.py")


if __name__ == "__main__":
    main()