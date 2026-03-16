# src/model.py - MSGCA Sequential 2-Stage Fusion with GATv2 KG Encoder
"""
Key fixes vs previous version:
- KGGraphEncoder → KGGraphEncoderGATv2  (node_dim 1033, edge_attr 17D)
- graph_norm added before fusion
- _encode_graphs passes edge_attr to GATv2
- node_dim sourced from TrainConfig.kg_node_dim (1033) not news_dim (128)
"""

import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F
from torch_geometric.data import Batch

from configs.config import TrainConfig
from encoders.kg_graph_encoder import KGGraphEncoderGATv2
from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce    = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt    = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean() if self.reduction == "mean" else focal.sum()


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.weight    = weight

    def forward(self, inputs, targets):
        n   = inputs.size(-1)
        lp  = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            td = torch.zeros_like(lp).fill_(self.smoothing / (n - 1))
            td.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        if self.weight is not None:
            loss = -(td * lp * self.weight.unsqueeze(0)).sum(dim=-1)
        else:
            loss = -(td * lp).sum(dim=-1)
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# STOCK MOVEMENT MODEL
# ─────────────────────────────────────────────────────────────────────────────

class StockMovementModel(nn.Module):
    def __init__(
        self,
        price_dim,
        macro_dim,
        news_dim,       # GATv2 graph output dim = TrainConfig.news_embed_dim = 128
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
        # GNN params (kept for API compat but values sourced from TrainConfig)
        use_gnn=True,
        gnn_type="gat",
        gnn_hidden_dim=128,
        gnn_num_layers=2,
        gnn_heads=4,
        gnn_pool="mean",
    ):
        super().__init__()
        self.device   = device
        self.output_dim = output_dim
        self.use_gnn  = use_gnn
        self.news_dim = news_dim   # = graph_out_dim = 128

        # ── KG Encoder: GATv2 ────────────────────────────────────────────────
        # node_dim = 1033 (Voyage 1024 + entity_type 8 + target_flag 1)
        # output_dim = news_dim = 128 (feeds into fusion as graph embedding)
        kg_node_dim = getattr(TrainConfig, "kg_node_dim", 1033)

        if use_gnn:
            self.kg_encoder = KGGraphEncoderGATv2(
                node_dim=kg_node_dim,
                hidden_dim=gnn_hidden_dim,
                output_dim=news_dim,        # → 128D graph embedding
                num_heads=gnn_heads,
                num_layers=gnn_num_layers,
                dropout=dropout,
            ).to(device)
            print(f"🔧 KG Encoder: GATv2 "
                  f"({kg_node_dim}D → {gnn_hidden_dim}D hidden → {news_dim}D out, "
                  f"{gnn_num_layers} layers, {gnn_heads} heads)")
        else:
            # Fallback: simple linear projection from raw node features
            self.kg_encoder = nn.Sequential(
                nn.Linear(kg_node_dim, news_dim),
                nn.LayerNorm(news_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ).to(device)

        # Normalize graph output to same scale as price features
        self.graph_norm = nn.LayerNorm(news_dim)

        # ── Multimodal encoder ────────────────────────────────────────────────
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim,
            macro_dim=macro_dim,
            news_dim=news_dim,
            dim=dim,
        )

        # ── MSGCA Sequential 2-Stage Fusion ──────────────────────────────────
        # Stage 1: Price (Primary) + Graph/News (Aux) → filter noisy news
        self.fusion_stage1 = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        # Stage 2: Fused Stage1 (Primary) + Macro (Aux) → integrate macro
        self.fusion_stage2 = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)

        print("🔧 Fusion: SEQUENTIAL 2-STAGE MSGCA")
        print("   Stage 1: Price (Primary) + Graph (Aux)")
        print("   Stage 2: Fused1 (Primary) + Macro (Aux)")

        # ── Predictor ─────────────────────────────────────────────────────────
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=input_dim, num_classes=output_dim, dropout=dropout,
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        self.use_focal_loss      = use_focal_loss
        self.use_label_smoothing = use_label_smoothing
        if use_label_smoothing:
            self.loss_fn = LabelSmoothingCrossEntropy(smoothing=smoothing, weight=class_weights)
            print(f"🔧 Loss: Label Smoothing (ε={smoothing})")
        elif use_focal_loss:
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            print(f"🔧 Loss: Focal Loss (γ={focal_gamma})")
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print("🔧 Loss: Standard CE")

    def _encode_graphs(self, graph_list) -> torch.Tensor:
        """
        Encode a list of PyG Data objects → (B, 1, news_dim) tensor.
        Handles edge_attr for GATv2 when present.
        """
        B = len(graph_list)
        zero = torch.zeros(B, 1, self.news_dim, device=self.device)

        if not self.use_gnn:
            # Simple mean-pool of raw node features → linear projection
            embeddings = []
            for g in graph_list:
                if g is None or g.x is None or g.x.size(0) == 0:
                    embeddings.append(torch.zeros(self.news_dim, device=self.device))
                else:
                    embeddings.append(g.x.mean(0).to(self.device))
            v = torch.stack(embeddings)                   # (B, kg_node_dim)
            return self.kg_encoder(v).unsqueeze(1)        # (B, 1, news_dim)

        # Filter valid graphs
        valid = [
            (i, g) for i, g in enumerate(graph_list)
            if g is not None and g.x is not None and g.x.size(0) > 0
        ]
        if not valid:
            return zero

        try:
            valid_graphs = [g for _, g in valid]
            batched = Batch.from_data_list(valid_graphs).to(self.device)
        except Exception as e:
            print(f"❌ Graph batching error: {e}")
            return zero

        # GATv2 forward — pass edge_attr if present
        edge_attr = getattr(batched, "edge_attr", None)
        try:
            embs = self.kg_encoder(
                x=batched.x,
                edge_index=batched.edge_index,
                edge_attr=edge_attr,
                batch=batched.batch,
            )                                             # (n_valid, news_dim)
        except Exception as e:
            print(f"❌ GATv2 encode error: {e}")
            return zero

        embs = F.normalize(embs, p=2, dim=-1)

        out = zero.clone()
        for out_idx, (orig_idx, _) in enumerate(valid):
            out[orig_idx, 0, :] = embs[out_idx]
        return out                                        # (B, 1, news_dim)

    def forward(
        self,
        s_o, s_h, s_c, s_m,
        s_n_graphs,
        label=None,
        mode="train",
        return_preds=False,
        return_logits=False,
    ):
        # 1. Encode modalities
        v_m, v_i, _ = self.multimodal_encoder(s_o, s_h, s_c, s_m, None)

        v_n_raw = self._encode_graphs(s_n_graphs)   # (B, 1, news_dim)
        v_n     = self.graph_norm(v_n_raw)           # normalize to same scale as v_i

        # 2. Sequential 2-stage fusion
        H1     = self.fusion_stage1(primary=v_i,  aux=v_n)   # (B, T, dim)
        H_final= self.fusion_stage2(primary=H1,   aux=v_m)   # (B, T, dim)

        # 3. Predict
        logits = self.movement_predictor(fused_seq=H_final, orig_seq=v_i)
        logits = torch.clamp(logits, -15, 15)

        # 4. Route by mode
        def _target(label):
            if isinstance(label, list):
                return torch.tensor([x[0] for x in label],
                                    dtype=torch.long, device=self.device)
            return label.long().to(self.device)

        if mode == "train":
            return self.loss_fn(logits, _target(label))

        elif mode == "test":
            target = _target(label)
            preds  = torch.argmax(logits, dim=1)
            acc    = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
            mcc    = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
            if return_logits:
                return acc, mcc, preds, logits
            if return_preds:
                return acc, mcc, preds
            return acc, mcc

        elif mode == "logits":
            return logits

    def get_prediction_confidence(self, s_o, s_h, s_c, s_m, s_n_graphs):
        self.eval()
        with torch.no_grad():
            v_m, v_i, _ = self.multimodal_encoder(s_o, s_h, s_c, s_m, None)
            v_n = self.graph_norm(self._encode_graphs(s_n_graphs))
            H1      = self.fusion_stage1(primary=v_i, aux=v_n)
            H_final = self.fusion_stage2(primary=H1,  aux=v_m)
            logits  = self.movement_predictor(fused_seq=H_final, orig_seq=v_i)
            probs   = F.softmax(logits, dim=-1)
            conf, preds = torch.max(probs, dim=-1)
        return probs, preds, conf