import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional

from torch_geometric.nn import (
    SAGEConv,
    GATv2Conv,
    global_mean_pool,
    global_max_pool,
)

# ============================================
# RELATION & ENTITY DEFINITIONS
# ============================================

VALID_RELATIONS = [
    "ANNOUNCES", "RAISES", "CUTS", "INVESTS_IN", "DIVESTS", "APPOINTS",
    "POS_IMPACTS", "NEG_IMPACTS", "COMPETES_WITH", "REGULATES", "SUPPLIES_TO",
    "CONTROLS", "SIGNALS", "RELATES_TO",
]
RELATION_TO_IDX = {r: i for i, r in enumerate(VALID_RELATIONS)}
NUM_RELATIONS   = len(VALID_RELATIONS)   # 14
DEFAULT_RELATION_IDX = RELATION_TO_IDX["RELATES_TO"]

VALID_ENTITY_TYPES = [
    "COMP", "PERSON", "ORG_GOV", "ORG_REG",
    "PRODUCT", "ECON_IND", "FIN_ASSET", "CONCEPT",
]

# Edge attr dimension:
# relation_one_hot(14) + confidence(1) + price_impact(1) + relevance(1) = 17D
EDGE_ATTR_DIM = NUM_RELATIONS + 3  # 17

# Node feature dimension:
# Voyage embedding (1024) + entity_type_onehot (8) + target_flag (1) = 1033
NODE_FEATURE_DIM = 1024 + len(VALID_ENTITY_TYPES) + 1  # 1033


# ============================================
# HELPER: build_node_info
# ============================================

def build_node_info(rich_triples: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Collect all entity names and their types from rich triples.

    Returns:
        Dict[entity_name → entity_type_code]
        (last seen type wins if an entity appears with multiple types)
    """
    node_info: Dict[str, str] = {}
    for t in rich_triples:
        subj = t.get("subject", {})
        obj  = t.get("object",  {})
        s_name = (subj.get("name") or "").strip()
        s_type = subj.get("type", "CONCEPT")
        o_name = (obj.get("name")  or "").strip()
        o_type = obj.get("type",  "CONCEPT")
        if s_name:
            node_info[s_name] = s_type
        if o_name:
            node_info[o_name] = o_type
    return node_info


# ============================================
# HELPER: build_node_features
# ============================================

def build_node_features(
    node_info: Dict[str, str],
    voyage_embedder,              # VoyageEmbedder instance
    ticker: str,
) -> Tuple[List[str], Dict[str, int], torch.Tensor]:
    """
    Build node feature matrix: Voyage embedding + entity-type one-hot + target flag.

    Args:
        node_info       : Dict[entity_name → entity_type_code]  (from build_node_info)
        voyage_embedder : VoyageEmbedder instance
        ticker          : Target stock ticker (used to set target_flag)

    Returns:
        nodes   : List[str]          — ordered node names
        node2id : Dict[str, int]     — name → index
        x       : (N, 1033) FloatTensor
    """
    nodes   = sorted(node_info.keys())
    node2id = {n: i for i, n in enumerate(nodes)}

    if not nodes:
        return nodes, node2id, torch.zeros(0, NODE_FEATURE_DIM)

    # Voyage embeddings (N, 1024)
    emb_list    = voyage_embedder.embed_texts(nodes)
    voyage_embs = torch.tensor(emb_list, dtype=torch.float32)

    # Entity-type one-hot (N, 8)
    type_feats = []
    for n in nodes:
        etype = node_info.get(n, "CONCEPT")
        idx   = VALID_ENTITY_TYPES.index(etype) if etype in VALID_ENTITY_TYPES else 7
        oh    = [0.0] * len(VALID_ENTITY_TYPES)
        oh[idx] = 1.0
        type_feats.append(oh)
    type_tensor = torch.tensor(type_feats, dtype=torch.float32)  # (N, 8)

    # Target-stock indicator (N, 1)
    ticker_lower = ticker.lower()
    target_flag  = torch.tensor(
        [
            1.0 if (ticker_lower in n.lower() or n.lower() in ticker_lower) else 0.0
            for n in nodes
        ],
        dtype=torch.float32,
    ).unsqueeze(-1)  # (N, 1)

    x = torch.cat([voyage_embs, type_tensor, target_flag], dim=-1)  # (N, 1033)
    return nodes, node2id, x


# ============================================
# HELPER: build_rich_edge_data   (single authoritative version)
# ============================================

def build_rich_edge_data(
    rich_triples: List[Dict[str, Any]],
    node2id: Dict[str, int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge_index and edge_attr from rich triples.

    Edge attr layout (17D):
        [0:14]  relation_one_hot   (14D)
        [14]    confidence         (1D)
        [15]    price_impact_score (1D)
        [16]    relevance_to_ticker(1D)

    Returns:
        edge_index : (2, E) LongTensor
        edge_attr  : (E, 17) FloatTensor
    """
    edge_src: List[int] = []
    edge_dst: List[int] = []
    edge_attrs: List[List[float]] = []

    for t in rich_triples:
        try:
            s_name = t["subject"]["name"]
            o_name = t["object"]["name"]
        except (KeyError, TypeError):
            continue

        if s_name not in node2id or o_name not in node2id:
            continue

        src = node2id[s_name]
        dst = node2id[o_name]

        relation  = t.get("relation", "RELATES_TO")
        rel_idx   = RELATION_TO_IDX.get(relation, DEFAULT_RELATION_IDX)

        rel_onehot = [0.0] * NUM_RELATIONS
        rel_onehot[rel_idx] = 1.0

        conf      = float(t.get("confidence",          0.5))
        impact    = float(t.get("price_impact_score",  0.0))
        relevance = float(t.get("relevance_to_ticker", 0.5))

        fwd_attr = rel_onehot + [conf, impact, relevance]          # 17D forward
        rev_attr = rel_onehot + [conf * 0.7, impact * -0.3, relevance * 0.7]  # reverse

        edge_src.append(src);  edge_dst.append(dst)
        edge_attrs.append(fwd_attr)

        edge_src.append(dst);  edge_dst.append(src)
        edge_attrs.append(rev_attr)

    if not edge_src:
        return (
            torch.zeros(2, 0, dtype=torch.long),
            torch.zeros(0, EDGE_ATTR_DIM, dtype=torch.float32),
        )

    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    edge_attr  = torch.tensor(edge_attrs,           dtype=torch.float32)
    return edge_index, edge_attr


# ============================================
# ORIGINAL HOMOGENEOUS GRAPH ENCODER
# ============================================

class KGGraphEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int = 1024,
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_sage_layers: int = 2,
        use_gat: bool = False,
        gat_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.node_dim   = node_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_gat    = use_gat

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.sage_layers = nn.ModuleList()
        self.sage_norms  = nn.ModuleList()
        for _ in range(num_sage_layers):
            self.sage_layers.append(SAGEConv(hidden_dim, hidden_dim, aggr="mean"))
            self.sage_norms.append(nn.LayerNorm(hidden_dim))

        if use_gat:
            self.gat = GATv2Conv(
                hidden_dim, hidden_dim // gat_heads,
                heads=gat_heads, dropout=dropout, concat=True,
            )
            self.gat_norm = nn.LayerNorm(hidden_dim)

        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(dropout)
        print(f"✅ KGGraphEncoder initialized: {node_dim} → {hidden_dim} → {output_dim}")

    def forward(self, x, edge_index, batch=None):
        h = self.input_proj(x)
        for sage, norm in zip(self.sage_layers, self.sage_norms):
            h_new = norm(F.relu(sage(h, edge_index)))
            h_new = self.dropout(h_new)
            h = h + h_new

        if self.use_gat:
            h_gat = F.relu(self.gat_norm(self.gat(h, edge_index)))
            h = h + self.dropout(h_gat)

        if batch is None:
            h_mean = h.mean(0, keepdim=True)
            h_max  = h.max(0, keepdim=True)[0]
        else:
            h_mean = global_mean_pool(h, batch)
            h_max  = global_max_pool(h, batch)

        return self.readout_mlp(torch.cat([h_mean, h_max], dim=-1))


# ============================================
# RELATION-AWARE KG ENCODER — GATv2 (V2)
# ============================================

class KGGraphEncoderGATv2(nn.Module):
    """
    GATv2-based Knowledge Graph Encoder với relation-aware attention.

    Node features: Voyage(1024) + entity_type_onehot(8) + target_flag(1) = 1033D
    Edge features: relation_onehot(14) + confidence(1) + price_impact(1) + relevance(1) = 17D

    Attention uses dims [0:16] (relation+conf+relevance, NOT price_impact).
    Impact-weighted pooling uses dim [15] (price_impact_score).
    """

    def __init__(
        self,
        node_dim: int = NODE_FEATURE_DIM,   # 1033
        hidden_dim: int = 128,
        output_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        edge_attr_dim: int = EDGE_ATTR_DIM,  # 17
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.residual   = residual
        # edge dims seen by attention: all except price_impact (dim 15)
        # We project edge attrs [0:14] + [16] → 15D for GATv2
        self._attn_edge_dim = NUM_RELATIONS + 2   # 16  (rel_onehot + conf + relevance)

        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        head_dim = hidden_dim // num_heads

        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.gat_layers = nn.ModuleList()
        self.gat_norms  = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=head_dim,
                    heads=num_heads,
                    edge_dim=self._attn_edge_dim,  # 16D
                    dropout=dropout,
                    concat=True,
                    add_self_loops=True,
                )
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))

        self.readout_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def _extract_attn_edge(self, edge_attr: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Strip price_impact (dim 15) from edge_attr so attention sees 16D:
        rel_onehot(14) + confidence(1) + relevance(1).
        Layout in edge_attr: [0:14]=rel_onehot, [14]=conf, [15]=impact, [16]=relevance
        → keep [0:15] + [16]  → cat → 16D
        """
        if edge_attr is None:
            return None
        # [0:15] = rel_onehot + conf  (15 dims)
        # [16]   = relevance          (1 dim)
        return torch.cat([edge_attr[:, :15], edge_attr[:, 16:17]], dim=-1)  # (E, 16)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Build attention edge features (16D) and extract impact scores
        attn_edge = self._extract_attn_edge(edge_attr)          # (E, 16) or None
        impact_scores = edge_attr[:, 15] if edge_attr is not None and edge_attr.shape[1] > 15 else None

        h = self.input_proj(x)

        for gat, norm in zip(self.gat_layers, self.gat_norms):
            h_new = gat(h, edge_index, edge_attr=attn_edge)
            h_new = norm(h_new)
            h_new = F.elu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new  # residual

        # Impact-weighted pooling
        if impact_scores is not None and edge_index.shape[1] > 0:
            dst_nodes    = edge_index[1]
            node_impact  = torch.zeros(x.size(0), device=x.device)
            node_impact.scatter_reduce_(
                0, dst_nodes, impact_scores.abs(),
                reduce="amax", include_self=True,
            )
            h_weighted = h * torch.sigmoid(node_impact).unsqueeze(-1)
        else:
            h_weighted = h

        if batch is None:
            h_mean     = h.mean(0, keepdim=True)
            h_imp_mean = h_weighted.mean(0, keepdim=True)
        else:
            h_mean     = global_mean_pool(h,          batch)
            h_imp_mean = global_mean_pool(h_weighted, batch)

        return self.readout_mlp(torch.cat([h_mean, h_imp_mean], dim=-1))


# ============================================
# LIGHTWEIGHT VERSION
# ============================================

class LightweightKGEncoder(nn.Module):
    def __init__(self, node_dim=1024, hidden_dim=128, output_dim=128, dropout=0.1):
        super().__init__()
        self.proj       = nn.Linear(node_dim, hidden_dim)
        self.sage       = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.norm       = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        print(f"✅ LightweightKGEncoder: {node_dim} → {hidden_dim} → {output_dim}")

    def forward(self, x, edge_index, batch=None):
        h = F.relu(self.proj(x))
        h = F.relu(self.norm(self.sage(h, edge_index)))
        h = self.dropout(h)
        if batch is None:
            graph_features = h.mean(0, keepdim=True)
        else:
            graph_features = global_mean_pool(h, batch)
        return self.output_proj(graph_features)


# ============================================
# LEGACY HELPER — simple edge index
# ============================================

def build_edge_index_from_triples(triples, node2id):
    """Old-style builder for homogeneous graphs (list of (s, p, o) tuples)."""
    edges = []
    for s, p, o in triples:
        if s in node2id and o in node2id:
            edges.append([node2id[s], node2id[o]])
            edges.append([node2id[o], node2id[s]])
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t()


# ============================================
# USAGE EXAMPLE
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing KGGraphEncoderGATv2")
    print("=" * 60)

    rich_triples = [
        {
            "subject":  {"name": "Tesla", "type": "COMP"},
            "relation": "ANNOUNCES",
            "object":   {"name": "Cybertruck launch", "type": "PRODUCT"},
            "confidence": 0.9, "price_impact_score": 0.5, "relevance_to_ticker": 1.0,
        },
        {
            "subject":  {"name": "Elon Musk", "type": "PERSON"},
            "relation": "POS_IMPACTS",
            "object":   {"name": "Tesla", "type": "COMP"},
            "confidence": 0.8, "price_impact_score": 0.3, "relevance_to_ticker": 0.9,
        },
    ]

    node_info = build_node_info(rich_triples)
    print(f"\nnode_info: {node_info}")

    nodes   = sorted(node_info.keys())
    node2id = {n: i for i, n in enumerate(nodes)}

    # Fake 1033-dim features (no Voyage needed for unit test)
    x = torch.randn(len(nodes), NODE_FEATURE_DIM)

    edge_index, edge_attr = build_rich_edge_data(rich_triples, node2id)
    print(f"edge_index: {edge_index.shape}  edge_attr: {edge_attr.shape}")

    encoder = KGGraphEncoderGATv2(
        node_dim=NODE_FEATURE_DIM,
        hidden_dim=128, output_dim=128, num_heads=4, num_layers=2,
    )
    encoder.eval()
    with torch.no_grad():
        out = encoder(x, edge_index, edge_attr=edge_attr)
    print(f"Output shape: {out.shape}")
    print("\n✅ ALL TESTS PASSED!")