# src/model.py
"""
V4 — StockMovementModel (Voyage embedding, no GATv2)

Thay đổi so với V3:
  - Bỏ KGGraphEncoderGATv2 và kg_encoder hoàn toàn
  - s_n đầu vào là tensor (B, T, 1024) thay vì list of PyG Data
  - NewsEncoder(1024 → dim) project xuống dim=256 rồi vào MSGCA
  - Fusion order: price → news → macro → predict (như thiết kế gốc)
  - _encode_graphs() được thay bằng _encode_news() — đơn giản hơn nhiều

Interface:
    model(s_o, s_h, s_c, s_m, s_n, label, mode="train")
    # s_n: (B, T, 1024) — Voyage embedding của news theo window
"""

import torch
from torch import nn
from sklearn.metrics import accuracy_score, matthews_corrcoef
import torch.nn.functional as F

from configs.config import TrainConfig
from encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction


# ─────────────────────────────────────────────────────────────────────────────
# LOSS FUNCTIONS (unchanged)
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
        n  = inputs.size(-1)
        lp = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            td = torch.zeros_like(lp).fill_(self.smoothing / (n - 1))
            td.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        if self.weight is not None:
            loss = -(td * lp * self.weight.unsqueeze(0)).sum(dim=-1)
        else:
            loss = -(td * lp).sum(dim=-1)
        return loss.mean()


# ─────────────────────────────────────────────────────────────────────────────
# STOCK MOVEMENT MODEL V4
# ─────────────────────────────────────────────────────────────────────────────

class StockMovementModel(nn.Module):
    def __init__(
        self,
        price_dim,
        macro_dim,
        news_dim,         # = 1024 (Voyage) — NewsEncoder projects to dim internally
        dim,              # = 256 hidden dim
        input_dim,        # = window_size = 20
        output_dim,       # = 3 (DOWN/FLAT/UP)
        num_head,         # = 4
        device,
        dropout=0.1,
        class_weights=None,
        use_focal_loss=True,
        focal_gamma=2.0,
        use_label_smoothing=False,
        smoothing=0.1,
        # Legacy GNN params — ignored in V4 but kept for API compat
        use_gnn=False,
        gnn_type="gat",
        gnn_hidden_dim=128,
        gnn_num_layers=2,
        gnn_heads=4,
        gnn_pool="mean",
    ):
        super().__init__()
        self.device   = device
        self.news_dim = news_dim  # 1024

        if use_gnn:
            print("  Note: use_gnn=True is ignored in V4 "
                  "(GATv2 removed, using Voyage embedding directly)")

        # ── Multimodal encoder (price + macro + news) ─────────────────────────
        # MultimodalSourceEncoding has NewsEncoder(news_dim=1024, dim=256)
        # which projects 1024 → 256 via Linear + GELU
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim,
            macro_dim=macro_dim,
            news_dim=news_dim,   # 1024
            dim=dim,
        )

        # ── MSGCA Sequential 2-Stage Fusion ───────────────────────────────────
        # Stage 1: Price (Primary) + News (Aux)
        self.fusion_stage1 = StableGatedCrossAttention(
            dim=dim, num_head=num_head, dropout=dropout
        )
        # Stage 2: Fused1 (Primary) + Macro (Aux)
        self.fusion_stage2 = StableGatedCrossAttention(
            dim=dim, num_head=num_head, dropout=dropout
        )

        print("  Fusion: 2-stage MSGCA")
        print("    Stage 1: Price (primary) + News/Voyage (aux)")
        print("    Stage 2: Fused1 (primary) + Macro (aux)")

        # ── Predictor ─────────────────────────────────────────────────────────
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=input_dim, num_classes=output_dim, dropout=dropout,
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        if use_label_smoothing:
            self.loss_fn = LabelSmoothingCrossEntropy(
                smoothing=smoothing, weight=class_weights
            )
            print(f"  Loss: Label Smoothing (ε={smoothing})")
        elif use_focal_loss:
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
            print(f"  Loss: Focal Loss (γ={focal_gamma})")
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            print("  Loss: Standard CE")

    def forward(
        self,
        s_o, s_h, s_c, s_m,
        s_n,           # (B, T, 1024) — Voyage news embedding
        label=None,
        mode="train",
        return_preds=False,
        return_logits=False,
        # Legacy kwarg — ignored
        s_n_graphs=None,
    ):
        """
        Forward pass.

        Args:
            s_o, s_h, s_c : (B, T, 1)   price OHLC
            s_m           : (B, T, M)   macro indicators
            s_n           : (B, T, 1024) Voyage news embedding
            label         : (B,) long tensor
            mode          : "train" | "test" | "logits"
        """
        # Handle legacy s_n_graphs kwarg (from V3 main.py collate_fn)
        # If s_n is None but s_n_graphs is provided, use zeros
        if s_n is None:
            if s_n_graphs is not None:
                B = s_o.shape[0]
                T = s_o.shape[1]
                s_n = torch.zeros(B, T, self.news_dim, device=self.device)
            else:
                B = s_o.shape[0]
                T = s_o.shape[1]
                s_n = torch.zeros(B, T, self.news_dim, device=self.device)
        else:
            s_n = s_n.to(self.device)

        # 1. Encode all modalities
        #    MultimodalSourceEncoding returns (v_m, v_i, v_n)
        #    v_m: (B, T, dim)  macro encoded
        #    v_i: (B, T, dim)  price encoded
        #    v_n: (B, T, dim)  news encoded (NewsEncoder: 1024 → dim)
        v_m, v_i, v_n = self.multimodal_encoder(
            s_o.to(self.device),
            s_h.to(self.device),
            s_c.to(self.device),
            s_m.to(self.device),
            s_n,
        )

        # v_n might be None if MultimodalSourceEncoding receives None
        # but we ensure s_n is always a tensor above
        if v_n is None:
            v_n = torch.zeros_like(v_i)

        # 2. Sequential 2-stage MSGCA fusion
        H1      = self.fusion_stage1(primary=v_i, aux=v_n)   # Price + News
        H_final = self.fusion_stage2(primary=H1,  aux=v_m)   # Fused + Macro

        # 3. Predict
        logits = self.movement_predictor(fused_seq=H_final, orig_seq=v_i)
        logits = torch.clamp(logits, -15, 15)

        # 4. Route by mode
        def _target(label):
            if isinstance(label, list):
                return torch.tensor(
                    [x[0] if isinstance(x, (list, tuple)) else x for x in label],
                    dtype=torch.long, device=self.device
                )
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

    def get_prediction_confidence(self, s_o, s_h, s_c, s_m, s_n):
        self.eval()
        with torch.no_grad():
            v_m, v_i, v_n = self.multimodal_encoder(
                s_o.to(self.device),
                s_h.to(self.device),
                s_c.to(self.device),
                s_m.to(self.device),
                s_n.to(self.device) if s_n is not None else
                torch.zeros(s_o.shape[0], s_o.shape[1], self.news_dim, device=self.device),
            )
            if v_n is None:
                v_n = torch.zeros_like(v_i)
            H1      = self.fusion_stage1(primary=v_i, aux=v_n)
            H_final = self.fusion_stage2(primary=H1,  aux=v_m)
            logits  = self.movement_predictor(fused_seq=H_final, orig_seq=v_i)
            probs   = F.softmax(logits, dim=-1)
            conf, preds = torch.max(probs, dim=-1)
        return probs, preds, conf