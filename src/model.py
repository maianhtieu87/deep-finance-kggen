# src/model.py
"""
StockMovementModel — ticker-aware
"""

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef

from configs.config import TrainConfig
from src.data_loader import N_TICKERS # Đã sửa import đúng version
from encoders.mutil_encoder import MultimodalSourceEncoding
from src.fusion import StableGatedCrossAttention
from src.predictor import FinegrainedMovementPrediction


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma; self.alpha = alpha; self.reduction = reduction

    def forward(self, inputs, targets):
        ce    = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt    = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean() if self.reduction == "mean" else focal.sum()


class StockMovementModel(nn.Module):
    def __init__(
        self,
        price_dim, macro_dim, news_dim, dim, input_dim, output_dim, num_head,
        device, dropout=0.1, class_weights=None,
        use_focal_loss=True, focal_gamma=2.0,
        use_gnn=False,   # legacy compat
        n_tickers=N_TICKERS,
        ticker_emb_dim=16,
    ):
        super().__init__()
        self.device   = device
        self.news_dim = news_dim
        self.dim      = dim

        # ── Ticker embedding ──────────────────────────────────────────────────
        self.ticker_emb = nn.Embedding(n_tickers, ticker_emb_dim, padding_idx=None)
        self.ticker_proj = nn.Sequential(
            nn.Linear(ticker_emb_dim, dim),
            nn.LayerNorm(dim),
            nn.Tanh(),   # bounded activation — keeps ticker signal in stable range
        )

        # ── Encoders ──────────────────────────────────────────────────────────
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim, macro_dim=macro_dim, news_dim=news_dim,
            dim=dim, dropout=dropout,
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion_stage1 = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        self.fusion_stage2 = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)

        # Modality gate: price context + ticker identity
        self.modality_gate = nn.Linear(2 * dim, dim)

        # ── Predictor ─────────────────────────────────────────────────────────
        self.pre_predict_proj = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim),
            nn.LayerNorm(2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=input_dim, num_classes=output_dim, dropout=dropout,
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        if use_focal_loss:
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # ── Init ticker embedding ──────────────────────────────────────────────
        nn.init.normal_(self.ticker_emb.weight, mean=0, std=0.02)

    def forward(
        self,
        s_o, s_h, s_c, s_m, s_n,
        label=None,
        mode="train",
        return_preds=False,
        ticker_id=None,    
        news_mask=None, **kwargs
    ):
        B = s_o.shape[0]
        T = s_o.shape[1]

        # ── Move to device ────────────────────────────────────────────────────
        s_o = s_o.to(self.device); s_h = s_h.to(self.device)
        s_c = s_c.to(self.device); s_m = s_m.to(self.device)
        s_n = (s_n.to(self.device) if s_n is not None
               else torch.zeros(B, T, self.news_dim, device=self.device))

        # ======================================================================
        # [TICKER EMBEDDING] - VỊ TRÍ ĐỂ BẬT/TẮT KIỂM CHỨNG (ABLATION STUDY)
        # ======================================================================
        if ticker_id is not None:
            tid = ticker_id.to(self.device)
        else:
            tid = torch.zeros(B, dtype=torch.long, device=self.device)

        # 1. BẢN FULL (Chạy bình thường để lấy kết quả cao nhất):
        # v_t = self.ticker_proj(self.ticker_emb(tid))   # (B, dim)
        v_t = torch.zeros(B, self.dim, device=self.device)
        # 2. BẢN ABLATION (Tắt Ticker Emb để thầy thấy sự khác biệt)
        # Bôi đen dòng v_t ở trên, và mở comment dòng dưới đây:
        # v_t = torch.zeros(B, self.dim, device=self.device)
        # ======================================================================

        # ── Encode modalities ─────────────────────────────────────────────────
        v_m, v_i, v_n = self.multimodal_encoder(s_o, s_h, s_c, s_m, s_n)
        if v_n is None:
            v_n = torch.zeros_like(v_i)

        # ── Gated fusion ──────────────────────────────────────────────────────
        H_news  = self.fusion_stage1(primary=v_i, aux=v_n)
        H_macro = self.fusion_stage2(primary=v_i, aux=v_m)

        # Gate: condition on price context AND ticker identity
        price_ctx = v_i.mean(dim=1)                          # (B, dim)
        gate_input = torch.cat([price_ctx, v_t], dim=-1)    # (B, 2*dim)
        w = torch.sigmoid(self.modality_gate(gate_input)).unsqueeze(1)  # (B, 1, dim)
        H_fused = w * H_news + (1.0 - w) * H_macro           # (B, T, dim)

        # ── Ticker-conditioned prediction ─────────────────────────────────────
        v_t_seq = v_t.unsqueeze(1).expand(-1, T, -1)         # (B, T, dim)
        combined = torch.cat([H_fused, v_i, v_t_seq], dim=-1)  # (B, T, 3*dim)
        H_final  = self.pre_predict_proj(combined)               # (B, T, 2*dim)

        H_pred_fused = H_final[:, :, :self.dim]   # (B, T, dim)
        H_pred_orig  = H_final[:, :, self.dim:]   # (B, T, dim)

        logits = self.movement_predictor(fused_seq=H_pred_fused, orig_seq=H_pred_orig)
        logits = torch.clamp(logits, -15, 15)

        # ── Output ────────────────────────────────────────────────────────────
        def _target(lbl):
            if isinstance(lbl, list):
                return torch.tensor(
                    [x[0] if isinstance(x, (list, tuple)) else x for x in lbl],
                    dtype=torch.long, device=self.device)
            return lbl.long().to(self.device)

        if mode == "train":
            return self.loss_fn(logits, _target(label))

        preds = torch.argmax(logits, dim=1)
        if mode == "test":
            target = _target(label)
            acc = accuracy_score(target.cpu().numpy(), preds.cpu().numpy())
            mcc = matthews_corrcoef(target.cpu().numpy(), preds.cpu().numpy())
            if return_preds:
                return acc, mcc, preds
            return acc, mcc

        return logits