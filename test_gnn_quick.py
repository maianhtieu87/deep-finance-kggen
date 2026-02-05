import os
import torch
import sys

sys.path.append('data_pipeline')

from processors.news_processor import KGGenNewsEmbedder
from encoders.kg_graph_encoder import build_edge_index_from_triples

print("="*60)
print("QUICK GNN INTEGRATION TEST")
print("="*60)

# Init embedder
embedder = KGGenNewsEmbedder(
    interim_root="data/interim",
    kmeans_k=128,
    use_voyage_resolution=False,
    allow_llm_when_missing=False,
)

print("\n✅ KGGenNewsEmbedder initialized")
print(f"   Encoder type: {type(embedder.graph_encoder).__name__}")

# Test data
triples = [
    ("Tesla", "announced", "factory"),
    ("Tesla", "builds", "facility"),
    ("Elon Musk", "leads", "Tesla"),
]

entities = list(set([s for s,_,_ in triples] + [o for _,_,o in triples]))
node2id = {e: i for i, e in enumerate(entities)}
edge_index = build_edge_index_from_triples(triples, node2id)

print(f"\n📊 Test Graph:")
print(f"   Nodes: {len(entities)}")
print(f"   Edges: {edge_index.shape[1]}")

# Test forward pass
node_features = torch.randn(len(entities), 1024)
output = embedder.graph_encoder(node_features, edge_index)

print(f"\n🔬 Forward Pass:")
print(f"   Input:  {node_features.shape}")
print(f"   Output: {output.shape}")

if output.shape == (1, 128):
    print("\n✅ GNN INTEGRATION SUCCESSFUL!")
else:
    print(f"\n❌ Shape mismatch! Expected (1, 128), got {output.shape}")

print("="*60)
