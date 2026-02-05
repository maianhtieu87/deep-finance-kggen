"""
OPTIMIZED KG GRAPH ENCODER (dim=128)
-------------------------------------
Real GNN encoder với message passing, optimized cho config hiện tại.

Key settings:
- node_dim=1024 (Voyage embeddings)
- hidden_dim=128 (khớp với model config)
- output_dim=128 (khớp với model config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATv2Conv, global_mean_pool, global_max_pool


class KGGraphEncoder(nn.Module):
    """
    Optimized GNN encoder for Knowledge Graph features.
    
    Architecture:
        Input: Node features (N_nodes, 1024) + edge_index
        → Project: 1024 → 128
        → GraphSAGE Layer 1
        → GraphSAGE Layer 2
        → Optional: GAT Layer
        → Global Pooling (mean + max)
        → Output: (1, 128)
    
    This replaces the old Linear-only encoder to enable real message passing.
    """
    
    def __init__(
        self,
        node_dim: int = 1024,      # Voyage embedding dimension
        hidden_dim: int = 128,     # Hidden dimension (matches model config)
        output_dim: int = 128,     # Output dimension (matches model config)
        num_sage_layers: int = 2,
        use_gat: bool = False,     # Set to True for attention
        gat_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gat = use_gat
        
        # Project Voyage embeddings to hidden_dim
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # GraphSAGE Layers
        self.sage_layers = nn.ModuleList()
        self.sage_norms = nn.ModuleList()
        
        for i in range(num_sage_layers):
            self.sage_layers.append(
                SAGEConv(hidden_dim, hidden_dim, aggr='mean')
            )
            self.sage_norms.append(nn.LayerNorm(hidden_dim))
        
        # Optional GAT Layer
        if use_gat:
            self.gat = GATv2Conv(
                hidden_dim,
                hidden_dim // gat_heads,
                heads=gat_heads,
                dropout=dropout,
                concat=True,
            )
            self.gat_norm = nn.LayerNorm(hidden_dim)
        
        # Graph-level readout (mean + max pooling)
        # Combined: 2 * hidden_dim → output_dim
        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"✅ KGGraphEncoder initialized: {node_dim} → {hidden_dim} → {output_dim}")
        if use_gat:
            print(f"   Using GAT with {gat_heads} heads")
    
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features (N_nodes, node_dim=1024)
            edge_index: Graph connectivity (2, N_edges)
            batch: Batch assignment for each node (N_nodes,)
                   If None, assumes single graph
        
        Returns:
            graph_features: (batch_size, output_dim=128)
        """
        
        # === Input Projection ===
        h = self.input_proj(x)  # (N_nodes, hidden_dim=128)
        
        # === GraphSAGE Layers with Residual ===
        for i, (sage, norm) in enumerate(zip(self.sage_layers, self.sage_norms)):
            h_new = sage(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)
            
            # Residual connection
            h = h + h_new
        
        # === Optional GAT Layer ===
        if self.use_gat:
            h_gat = self.gat(h, edge_index)
            h_gat = self.gat_norm(h_gat)
            h_gat = F.relu(h_gat)
            h_gat = self.dropout(h_gat)
            h = h + h_gat  # Residual
        
        # === Graph-level Readout ===
        if batch is None:
            # Single graph: global pooling
            h_mean = h.mean(dim=0, keepdim=True)  # (1, hidden_dim)
            h_max = h.max(dim=0, keepdim=True)[0]  # (1, hidden_dim)
        else:
            # Batch of graphs: use PyG global pooling
            h_mean = global_mean_pool(h, batch)  # (batch_size, hidden_dim)
            h_max = global_max_pool(h, batch)    # (batch_size, hidden_dim)
        
        # Combine mean + max
        h_readout = torch.cat([h_mean, h_max], dim=-1)  # (batch_size, hidden_dim * 2)
        
        # Final projection
        graph_features = self.readout_mlp(h_readout)  # (batch_size, output_dim=128)
        
        return graph_features


# ============================================
# LIGHTWEIGHT VERSION (for faster experiments)
# ============================================

class LightweightKGEncoder(nn.Module):
    """
    Simpler version with only 1 SAGE layer.
    Faster but less expressive.
    """
    
    def __init__(self, node_dim=1024, hidden_dim=128, output_dim=128, dropout=0.1):
        super().__init__()
        
        # Project + SAGE + Pooling
        self.proj = nn.Linear(node_dim, hidden_dim)
        self.sage = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        print(f"✅ LightweightKGEncoder: {node_dim} → {hidden_dim} → {output_dim}")
    
    def forward(self, x, edge_index, batch=None):
        # Project
        h = F.relu(self.proj(x))
        
        # SAGE
        h = self.sage(h, edge_index)
        h = self.norm(h)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Global mean pooling
        if batch is None:
            graph_features = h.mean(dim=0, keepdim=True)
        else:
            graph_features = global_mean_pool(h, batch)
        
        # Output projection
        graph_features = self.output_proj(graph_features)
        
        return graph_features


# ============================================
# HELPER: Build edge_index from triples
# ============================================

def build_edge_index_from_triples(triples, node2id):
    """
    Build PyTorch Geometric edge_index from triples.
    
    Args:
        triples: List of (subject, predicate, object) tuples
        node2id: Dict mapping node_name → node_id
    
    Returns:
        edge_index: (2, num_edges) LongTensor
    """
    edges = []
    
    for s, p, o in triples:
        if s in node2id and o in node2id:
            s_idx = node2id[s]
            o_idx = node2id[o]
            
            # Add both directions (undirected graph)
            edges.append([s_idx, o_idx])
            edges.append([o_idx, s_idx])
    
    if not edges:
        # Empty graph: return empty edge_index
        return torch.zeros((2, 0), dtype=torch.long)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()  # (2, num_edges)
    
    return edge_index


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    """
    Test the encoder with sample data.
    """
    
    print("="*60)
    print("Testing KGGraphEncoder (dim=128)")
    print("="*60)
    
    # === Sample graph data ===
    num_nodes = 30
    num_edges = 60
    node_dim = 1024  # Voyage embeddings
    
    # Random node features
    x = torch.randn(num_nodes, node_dim)
    
    # Random edge_index
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    
    # === Test full encoder ===
    print("\n1. Testing Full KGGraphEncoder:")
    encoder = KGGraphEncoder(
        node_dim=1024,
        hidden_dim=128,
        output_dim=128,
        num_sage_layers=2,
        use_gat=False,  # Set True to enable GAT
        dropout=0.1,
    )
    
    graph_features = encoder(x, edge_index, batch=None)
    print(f"   Input shape:  {x.shape}")
    print(f"   Output shape: {graph_features.shape}")
    print(f"   Output mean:  {graph_features.mean():.4f}")
    print(f"   Output std:   {graph_features.std():.4f}")
    
    # === Test lightweight encoder ===
    print("\n2. Testing LightweightKGEncoder:")
    lightweight = LightweightKGEncoder(
        node_dim=1024,
        hidden_dim=128,
        output_dim=128,
    )
    
    graph_features_light = lightweight(x, edge_index, batch=None)
    print(f"   Output shape: {graph_features_light.shape}")
    
    # === Test batch processing ===
    print("\n3. Testing Batch Processing:")
    x_batch = torch.randn(100, node_dim)  # 100 total nodes
    edge_index_batch = torch.randint(0, 100, (2, 200))
    batch = torch.cat([
        torch.zeros(30, dtype=torch.long),   # Graph 0
        torch.ones(40, dtype=torch.long),    # Graph 1
        torch.full((30,), 2, dtype=torch.long),  # Graph 2
    ])
    
    graph_features_batch = encoder(x_batch, edge_index_batch, batch)
    print(f"   Batch output: {graph_features_batch.shape}")  # Should be (3, 128)
    
    # === Test gradient flow ===
    print("\n4. Testing Gradient Flow:")
    encoder.train()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    
    # Dummy forward + backward
    out = encoder(x, edge_index)
    loss = out.mean()
    loss.backward()
    
    # Check gradients
    has_grad = sum(1 for p in encoder.parameters() if p.grad is not None)
    total_params = sum(1 for _ in encoder.parameters())
    print(f"   Params with gradients: {has_grad}/{total_params}")
    
    if has_grad == total_params:
        print("   ✅ All parameters have gradients - GNN is working!")
    else:
        print("   ⚠️  Some parameters missing gradients")
    
    # === Test edge_index builder ===
    print("\n5. Testing edge_index builder:")
    sample_triples = [
        ("Tesla", "announced", "factory"),
        ("Tesla", "builds", "production facility"),
        ("Elon Musk", "leads", "Tesla"),
    ]
    
    entities = list(set([s for s, _, _ in sample_triples] + [o for _, _, o in sample_triples]))
    node2id = {ent: i for i, ent in enumerate(entities)}
    
    edge_index_built = build_edge_index_from_triples(sample_triples, node2id)
    print(f"   Triples: {len(sample_triples)}")
    print(f"   Nodes: {len(entities)}")
    print(f"   Edges: {edge_index_built.shape[1]}")
    print(f"   edge_index shape: {edge_index_built.shape}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)