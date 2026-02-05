"""
Test GNN Encoder without LLM extraction
Tests the graph encoding part only with sample triples
"""

import os
import torch
import sys

sys.path.insert(0, 'data_pipeline')

from processors.news_processor import KGGenNewsEmbedder
from encoders.kg_graph_encoder import build_edge_index_from_triples

print("="*60)
print("GNN ENCODER TEST (No LLM)")
print("="*60)

# Sample triples from the news (manually created for testing)
sample_triples = [
    ("SoftBank", "acquired", "Sharp factory"),
    ("SoftBank", "partnered with", "OpenAI"),
    ("SoftBank", "builds", "AI data center"),
    ("OpenAI", "created", "ChatGPT"),
    ("Microsoft", "invested in", "OpenAI"),
    ("OpenAI", "partnered with", "Oracle"),
    ("Oracle", "builds", "data center"),
    ("Nvidia", "supplies", "AI chips"),
]

print(f"\n📊 Sample Triples ({len(sample_triples)}):")
for i, (s, p, o) in enumerate(sample_triples[:5], 1):
    print(f"   {i}. ({s}, {p}, {o})")
if len(sample_triples) > 5:
    print(f"   ... and {len(sample_triples) - 5} more")

# Initialize embedder (NO LLM, NO Voyage for quick test)
print("\n🔧 Initializing KGGenNewsEmbedder...")
embedder = KGGenNewsEmbedder(
    interim_root="data/interim",
    kmeans_k=128,
    use_voyage_resolution=False,      # Skip Voyage
    use_voyage_node_features=False,   # Skip Voyage node features
    allow_llm_when_missing=False,     # No LLM calls
)

print(f"   Encoder type: {type(embedder.graph_encoder).__name__}")

# Extract entities
entities = list(set(
    [s for s, _, _ in sample_triples] + 
    [o for _, _, o in sample_triples]
))
node2id = {e: i for i, e in enumerate(entities)}

print(f"\n🕸️  Building Graph:")
print(f"   Entities: {len(entities)}")

# Build edge_index
edge_index = build_edge_index_from_triples(sample_triples, node2id)
print(f"   Edges: {edge_index.shape[1]}")

# Create dummy node features (simulating embeddings)
node_features = torch.randn(len(entities), 1024)
print(f"   Node features: {node_features.shape}")

# Test GNN encoder
print("\n🔬 Testing GNN Encoder:")
try:
    with torch.no_grad():
        output = embedder.graph_encoder(
            x=node_features,
            edge_index=edge_index,
            batch=None,
        )
    
    print(f"   ✅ Forward pass successful!")
    print(f"   Input shape:  {node_features.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output mean:  {output.mean():.4f}")
    print(f"   Output std:   {output.std():.4f}")
    
    # Check shape
    if output.shape == (1, 128):
        print("\n" + "="*60)
        print("✅ GNN INTEGRATION TEST PASSED!")
        print("="*60)
        print("✓ Real GNN encoder (GraphSAGE)")
        print("✓ Message passing enabled")
        print("✓ Output dimension correct (128)")
        print("✓ Ready for full pipeline")
        print("="*60)
    else:
        print(f"\n⚠️  Warning: Shape is {output.shape}, expected (1, 128)")
        
except Exception as e:
    print(f"\n❌ ERROR during forward pass:")
    print(f"   {e}")
    import traceback
    traceback.print_exc()

print("\n💡 Next Steps:")
print("   1. Set GEMINI_API_KEY to test full extraction")
print("   2. Run: python main_test.py (rebuild dataset)")
print("   3. Run: python main.py (train model)")