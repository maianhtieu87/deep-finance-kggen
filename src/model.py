# =========================================================
# FILE: src/model.py - MSGCA SEQUENTIAL 2-STAGE FUSION
# =========================================================

import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F
import os
from torch_geometric.data import Batch

from encoders.kg_graph_encoder import KGGraphEncoder
from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
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
    """Cross Entropy with Label Smoothing"""
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
    def __init__(
        self,
        price_dim,
        macro_dim,
        news_dim,
        dim,
        input_dim,
        output_dim,
        num_head,
        device,
        dropout=0.1,
        class_weights=None,
        use_focal_loss=True,
        focal_gamma=2.0,
        use_label_smoothing=False,
        smoothing=0.1,
        # GNN Params
        use_gnn=True,
        gnn_type="sage",
        gnn_hidden_dim=256,
        gnn_num_layers=2,
        gnn_heads=4,
        gnn_pool="attention",
    ):
        super().__init__()
        self.device = device
        self.output_dim = output_dim
        self.use_gnn = use_gnn
        self.news_dim = news_dim
        
        # ===== 1. ENCODERS =====
        if use_gnn:
            self.kg_encoder = KGGraphEncoder(
                node_dim=news_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=dim,
                num_sage_layers=gnn_num_layers,
                use_gat=(gnn_type == "gat"),
                gat_heads=gnn_heads,
                dropout=dropout
            ).to(device)
            print(f"🔧 KG Encoder: {gnn_type.upper()} ({gnn_num_layers} layers)")
        else:
            self.kg_encoder = nn.Sequential(
                nn.Linear(news_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ).to(device)
        
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim,
            macro_dim=macro_dim,
            news_dim=dim,
            dim=dim
        )

        # ===== 2. MSGCA SEQUENTIAL 2-STAGE FUSION =====
        
        # STAGE 1: Price (Primary) + News/Graph (Auxiliary)
        # Purpose: Filter noisy news using reliable price signal
        self.fusion_stage1 = StableGatedCrossAttention(
            dim=dim,
            num_head=num_head,
            dropout=dropout
        )
        
        # STAGE 2: (Price + News/Graph) + Macro
        # Purpose: Integrate macro info using cleaned Stage 1 output as the new Primary
        self.fusion_stage2 = StableGatedCrossAttention(
            dim=dim,
            num_head=num_head,
            dropout=dropout
        )
        
        print("🔧 Fusion Strategy: SEQUENTIAL 2-STAGE (MSGCA Paper)")
        print("   ► Stage 1: Price (Primary) + News/Graph (Auxiliary)")
        print("   ► Stage 2: Fused Stage_1 (New Primary) + Macro (Auxiliary)")

        # ===== 3. PREDICTOR =====
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim,
            window_size=input_dim,
            num_classes=output_dim,
            dropout=dropout
        )

        # ===== 4. LOSS =====
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
        """Batch PyG Data objects"""
        if not graph_list:
            return None
        
        valid_graphs = [g for g in graph_list if g is not None and g.x.size(0) > 0]
        
        if len(valid_graphs) == 0:
            return None
        
        try:
            batched = Batch.from_data_list(valid_graphs)
            return batched
        except Exception as e:
            print(f"❌ Error batching graphs: {e}")
            return None

    def _encode_graphs(self, graph_list):
        """Encode graphs to embeddings"""
        B = len(graph_list)
        
        if not self.use_gnn:
            embeddings = []
            for graph in graph_list:
                if graph is None or graph.x.size(0) == 0:
                    emb = torch.zeros(self.news_dim).to(self.device)
                else:
                    emb = graph.x.mean(dim=0)
                embeddings.append(emb)
            
            v_n_flat = torch.stack(embeddings).to(self.device)
            v_n_projected = self.kg_encoder(v_n_flat)
            v_n = v_n_projected.unsqueeze(1)
            return v_n
        
        batched_graph = self._process_graph_batch(graph_list)
        
        if batched_graph is None:
            v_n = torch.zeros(B, 1, self.kg_encoder.output_dim).to(self.device)
            return v_n
        
        batched_graph = batched_graph.to(self.device)
        
        graph_embeddings = self.kg_encoder(
            batched_graph.x, 
            batched_graph.edge_index,
            batched_graph.batch
        )
        
        graph_embeddings = F.normalize(graph_embeddings, p=2, dim=-1)
        v_n = graph_embeddings.unsqueeze(1) # Transform to (Batch, Sequence=1, Dim) for Cross-Attention
        
        return v_n

    def forward(self, s_o, s_h, s_c, s_m, s_n_graphs, label=None, mode="train", 
                return_preds=False, return_logits=False):
        
        # 1. Encode Modalities
        v_m, v_i, _ = self.multimodal_encoder(s_o, s_h, s_c, s_m, None)
        v_n = self._encode_graphs(s_n_graphs)
        
        # 2. Sequential Fusion
        H_stage1 = self.fusion_stage1(
            primary=v_i,    # Price indicators (T, D)
            aux=v_n         # Graph/News (1, D)
        )  # → (B, T, D)
        
        H_final = self.fusion_stage2(
            primary=H_stage1,  # Fused Stage 1 (T, D)
            aux=v_m            # Macro (T, D)
        )  # → (B, T, D)
        
        # 3. Predictor
        logits = self.movement_predictor(
            fused_seq=H_final, 
            orig_seq=v_i       # Cung cấp Price gốc để tạo Residual connection 
        )
        logits = torch.clamp(logits, -15, 15)
        
        # 4. Mode Routing
        if mode == "train":
            if isinstance(label, list):
                target = torch.tensor([item[0] for item in label], dtype=torch.long, device=self.device)
            else:
                target = label.long().to(self.device)
            
            loss = self.loss_fn(logits, target)
            return loss
        
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
        
        elif mode == "logits":
            return logits

    def get_prediction_confidence(self, s_o, s_h, s_c, s_m, s_n_graphs):
        """Get prediction probabilities and confidence"""
        self.eval() # Ensure model is in eval mode for inference
        with torch.no_grad():
            v_m, v_i, _ = self.multimodal_encoder(s_o, s_h, s_c, s_m, None)
            v_n = self._encode_graphs(s_n_graphs)
            
            # Sequential 2-stage fusion
            H_stage1 = self.fusion_stage1(primary=v_i, aux=v_n)
            H_final = self.fusion_stage2(primary=H_stage1, aux=v_m)
            
            logits = self.movement_predictor(fused_seq=H_final, orig_seq=v_i)
            probs = F.softmax(logits, dim=-1)
            confidence, preds = torch.max(probs, dim=-1)
            
            return probs, preds, confidence