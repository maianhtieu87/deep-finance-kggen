# src/model.py - FIXED VERSION WITH GNN INTEGRATION
"""
Stock Movement Model với Graph Neural Network

CHANGES:
- Thay MultimodalEncoding bằng KGGraphEncoder cho news
- Forward nhận graph data (x, edge_index) thay vì flat vector
- Hỗ trợ dynamic loading từ paths
- L2 normalization cho graph embeddings
"""

import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F
import os
from torch_geometric.data import Batch

from encoders.kg_graph_encoder import KGGraphEncoder  # ✅ Import GNN
from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction


class FocalLoss(nn.Module):
    """
    Focal Loss: Giải quyết mất cân bằng bằng cách tập trung vào mẫu khó (Hard Examples).
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross Entropy với Label Smoothing."""
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight = weight

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        if self.weight is not None:
            weight_expanded = self.weight.unsqueeze(0).expand_as(log_probs)
            loss = -(true_dist * log_probs * weight_expanded).sum(dim=-1)
        else:
            loss = -(true_dist * log_probs).sum(dim=-1)
        
        return loss.mean()


class StockMovementModel(nn.Module):
    """
    MSGCA Framework với Graph Neural Network Integration.
    
    Architecture:
    - Price/Macro: Temporal encoding (LSTM/Transformer)
    - News: KG Graph Encoder (GNN: GraphSAGE + GAT)
    - Fusion: Stable Gated Cross Attention
    - Prediction: MLP-based movement classifier
    """
    def __init__(
        self,
        price_dim,
        macro_dim,
        news_dim,  # Node feature dimension
        dim,       # Hidden dimension
        input_dim,  # Window size
        output_dim, # Num classes
        num_head,
        device,
        dropout=0.1,
        class_weights=None,
        use_focal_loss=True,
        focal_gamma=2.0,
        use_label_smoothing=False,
        smoothing=0.1,
        # ===== GNN Params =====
        use_gnn=True,              # Enable GNN encoder
        gnn_type="sage",           # "sage" or "gat"
        gnn_hidden_dim=256,        # GNN hidden dimension
        gnn_num_layers=2,          # Number of GNN layers
        gnn_heads=4,               # GAT heads (if use_gat)
        gnn_pool="attention",      # Pooling: "mean", "max", "attention"
    ):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.use_gnn = use_gnn
        self.news_dim = news_dim
        
        # ===== 1. Encoders =====
        if use_gnn:
            # ✅ Use GNN for news encoding
            self.kg_encoder = KGGraphEncoder(
                node_dim=news_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=dim,
                num_sage_layers=gnn_num_layers,  # ✅ FIXED
                use_gat=(gnn_type == "gat"),
                gat_heads=gnn_heads,
                dropout=dropout
            ).to(device)
            print(f"🔧 KG Encoder: {gnn_type.upper()} (layers={gnn_num_layers}, pool={gnn_pool})")
        else:
            # Fallback: Simple linear projection
            self.kg_encoder = nn.Sequential(
                nn.Linear(news_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ).to(device)
            print("⚠️ Using Linear Projection for News (GNN disabled)")
        
        # Price & Macro encoder (keep existing)
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim,
            macro_dim=macro_dim,
            news_dim=dim,  # Will be replaced by GNN output
            dim=dim
        )

        # ===== 2. Fusion =====
        self.fusion_news = StableGatedCrossAttention(dim=dim, num_head=num_head)
        self.fusion_macro = StableGatedCrossAttention(dim=dim, num_head=num_head)

        # ===== 3. Predictor =====
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim,
            window_size=input_dim,
            num_classes=output_dim,
            dropout=dropout
        )

        # ===== 4. Loss =====
        self.use_focal_loss = use_focal_loss
        self.use_label_smoothing = use_label_smoothing
        
        if use_label_smoothing:
            self.loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)
            print(f"🔧 Loss: LABEL SMOOTHING (ε={smoothing})")
        elif use_focal_loss:
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            print(f"🔧 Loss: FOCAL LOSS (γ={focal_gamma})")
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print(f"🔧 Loss: STANDARD CE")

    def _process_graph_batch(self, graph_list):
        """
        Process list of PyG Data objects into batch.
        
        Args:
            graph_list: List of Data objects (length B)
        
        Returns:
            batched_graph: Batch object or None if all empty
        """
        if not graph_list:
            return None
        
        # Filter out None/empty graphs
        valid_graphs = [g for g in graph_list if g is not None and g.x.size(0) > 0]
        
        if len(valid_graphs) == 0:
            # All graphs are empty, return dummy
            return None
        
        # Batch graphs using PyG
        try:
            batched = Batch.from_data_list(valid_graphs)
            return batched
        except Exception as e:
            print(f"❌ Error batching graphs: {e}")
            return None

    def _encode_graphs(self, graph_list):
        """
        Encode batch of graphs to fixed-size embeddings.
        
        Args:
            graph_list: List of PyG Data objects
        
        Returns:
            v_n: Graph embeddings (B, T, dim)
        """
        B = len(graph_list)
        
        if not self.use_gnn:
            # Fallback: Use mean of node features
            embeddings = []
            for graph in graph_list:
                if graph is None or graph.x.size(0) == 0:
                    emb = torch.zeros(self.news_dim).to(self.device)
                else:
                    emb = graph.x.mean(dim=0)  # Mean pooling
                embeddings.append(emb)
            
            v_n_flat = torch.stack(embeddings).to(self.device)  # (B, news_dim)
            v_n_projected = self.kg_encoder(v_n_flat)  # (B, dim)
            
            # Expand to sequence
            v_n = v_n_projected.unsqueeze(1)  # (B, 1, dim)
            return v_n
        
        # ✅ Use GNN
        batched_graph = self._process_graph_batch(graph_list)
        
        if batched_graph is None:
            # All empty, return zeros
            v_n = torch.zeros(B, 1, self.kg_encoder.output_dim).to(self.device)
            return v_n
        
        # Move to device
        batched_graph = batched_graph.to(self.device)
        
        # Run GNN
        graph_embeddings = self.kg_encoder(
            batched_graph.x, 
            batched_graph.edge_index,
            batched_graph.batch
        )  # (B, dim)
        
        # L2 Normalize to preserve geometry
        graph_embeddings = F.normalize(graph_embeddings, p=2, dim=-1)
        
        # Expand to sequence (B, 1, dim)
        v_n = graph_embeddings.unsqueeze(1)
        
        return v_n

    def forward(self, s_o, s_h, s_c, s_m, s_n_graphs, label=None, mode="train", 
                return_preds=False, return_logits=False):
        """
        Forward pass với Graph Neural Network.
        
        Args:
            s_o, s_h, s_c: Price features (B, T, 1)
            s_m: Macro features (B, T, macro_dim)
            s_n_graphs: List of PyG Data objects (length B)
            label: Ground truth labels (B,)
            mode: "train" or "test"
        
        Returns:
            - mode="train": loss
            - mode="test": (acc, mcc) or with predictions/logits
        """
        # 1. Encode Price & Macro (existing logic)
        v_m, v_i, _ = self.multimodal_encoder(s_o, s_h, s_c, s_m, None)
        
        # 2. Encode News Graphs using GNN
        v_n = self._encode_graphs(s_n_graphs)  # (B, 1, dim)
        
        # 3. Stable Fusion (Guided by Indicator v_i)
        fused_news = self.fusion_news(primary=v_i, aux=v_n)
        fused_macro = self.fusion_macro(primary=v_i, aux=v_m)
        
        # 4. Combine Fused Features
        v_fused_total = (fused_news + fused_macro) / 2.0
        
        # 5. Prediction
        logits = self.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
        logits = torch.clamp(logits, -15, 15)
        
        # ===== TRAIN MODE =====
        if mode == "train":
            if isinstance(label, list):
                target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
            else:
                target = label.long().to(self.device)
            
            loss = self.loss_fn(logits, target)
            return loss
        
        # ===== TEST MODE =====
        elif mode == "test":
            if isinstance(label, list):
                target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
            else:
                target = label.long().to(self.device)
            
            preds = torch.argmax(logits, dim=1)
            acc = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
            mcc = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
            
            if return_logits:
                return acc, mcc, preds, logits
            elif return_preds:
                return acc, mcc, preds
            else:
                return acc, mcc
        
        # ===== LOGITS MODE =====
        elif mode == "logits":
            return logits

    def get_prediction_confidence(self, s_o, s_h, s_c, s_m, s_n_graphs):
        """Utility method to analyze prediction confidence."""
        with torch.no_grad():
            v_m, v_i, _ = self.multimodal_encoder(s_o, s_h, s_c, s_m, None)
            v_n = self._encode_graphs(s_n_graphs)
            
            fused_news = self.fusion_news(primary=v_i, aux=v_n)
            fused_macro = self.fusion_macro(primary=v_i, aux=v_m)
            v_fused_total = (fused_news + fused_macro) / 2.0
            
            logits = self.movement_predictor(fused_seq=v_fused_total, orig_seq=v_i)
            probs = F.softmax(logits, dim=-1)
            confidence, preds = torch.max(probs, dim=-1)
            
            return probs, preds, confidence