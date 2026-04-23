# src/model.py
"""
StockMovementModel — Sequential Gated Cross-Attention

CHANGES vs V3:
  [FIX-P2] Sửa lỗi H_pred_orig không thực sự là "original price signal".

  Vấn đề cũ:
    combined = cat([H_idm, v_i, v_t_seq])  → Linear(3dim, 2dim)
    H_pred_fused = H_final[:, :, :dim]     # = projection của mix(H_idm+v_i+v_t)
    H_pred_orig  = H_final[:, :, dim:]     # = CŨNG projection của mix(H_idm+v_i+v_t)
    → Cả 2 path vào predictor đều từ cùng 1 input → feat_agg nhận redundant info
    → Predictor không có "shortcut" đến price signal thuần → gradient kém

  Fix:
    fused_input  = cat([H_idm, v_t_seq])   → fused_proj(Linear 2dim → dim)
    H_pred_fused = fused_proj(fused_input)  # multimodal + ticker context
    H_pred_orig  = v_i                      # PURE price signal (BiGRU output)
    → feat_agg nhận: (1) multimodal summary, (2) raw price features
    → price signal có direct path đến logits, không bị pha trộn với news/macro
    → Gradient từ loss chảy trực tiếp về IndicatorEncoder qua orig path

  Param count change:
    Cũ: Linear(3*dim, 2*dim) = 3*64*2*64 = 24,576
    Mới: Linear(2*dim, dim)  = 2*64*64   = 8,192   (nhỏ hơn, ít overfit hơn)
"""

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef

from configs.config import TrainConfig
from src.data_loader import N_TICKERS
from encoders.mutil_encoder import MultimodalSourceEncoding
from src.fusion import StableGatedCrossAttention
from src.predictor import FinegrainedMovementPrediction


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.alpha     = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce    = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt    = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        return focal.mean() if self.reduction == "mean" else focal.sum()


class StockMovementModel(nn.Module):
    def __init__(
        self,
        price_dim:   int,
        macro_dim:   int,
        news_dim:    int,
        dim:         int,
        input_dim:   int,
        output_dim:  int,
        num_head:    int,
        device,
        dropout:     float = 0.1,
        class_weights=None,
        use_focal_loss: bool  = True,
        focal_gamma:    float = 2.0,
        use_gnn:        bool  = False,
        n_tickers:      int   = N_TICKERS,
        ticker_emb_dim: int   = 16,
    ):
        super().__init__()
        self.device   = device
        self.news_dim = news_dim
        self.dim      = dim

        # ── Ticker embedding ─────────────────────────────────────────────────
        self.ticker_emb  = nn.Embedding(n_tickers, ticker_emb_dim, padding_idx=None)
        self.ticker_proj = nn.Sequential(
            nn.Linear(ticker_emb_dim, dim),
            nn.LayerNorm(dim),
            nn.Tanh(),
        )

        # ── Multimodal encoders ───────────────────────────────────────────────
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim,
            macro_dim=macro_dim,
            news_dim=news_dim,
            dim=dim,
            dropout=dropout,
        )

        # ── Sequential Gated Cross-Attention Fusion ───────────────────────────
        self.fusion_stage1 = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        self.fusion_stage2 = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)

        # ── [FIX-P2] Predictor input projection ──────────────────────────────
        #
        # Cũ: pre_predict_proj = Linear(3*dim → 2*dim)
        #     Input: cat([H_idm, v_i, v_t_seq]) — mixed signal
        #     Output split thành fused/orig — cả 2 cùng nguồn → redundant
        #
        # Mới: fused_proj = Linear(2*dim → dim)
        #     Input: cat([H_idm, v_t_seq]) — multimodal + ticker ONLY
        #     orig path nhận v_i trực tiếp — PURE price, không mixing
        #
        # Rationale:
        #   FinegrainedMovementPrediction đang làm dual-path prediction:
        #     path1 (fused): "what does the multimodal model think?"
        #     path2 (orig):  "what does raw price alone think?"
        #   Để dual-path có ý nghĩa, 2 path phải có information khác nhau.
        #   Với fix này, gradient từ loss có 2 distinct paths:
        #     - Qua fused_proj → H_idm → fusion → news/price/macro encoders
        #     - Qua v_i        → IndicatorEncoder (direct, không qua fusion)
        self.fused_proj = nn.Sequential(
            nn.Linear(2 * dim, dim),     # cat([H_idm, v_t_seq]) → dim
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Movement Predictor ────────────────────────────────────────────────
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim,
            window_size=input_dim,
            num_classes=output_dim,
            dropout=dropout,
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        if use_focal_loss:
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        nn.init.normal_(self.ticker_emb.weight, mean=0.0, std=0.02)

    def forward(
        self,
        s_o, s_h, s_c, s_m, s_n,
        label=None,
        mode:         str  = "train",
        return_preds: bool = False,
        ticker_id=None,
        news_mask=None,
        **kwargs,
    ):
        B = s_o.shape[0]
        T = s_o.shape[1]

        s_o = s_o.to(self.device)
        s_h = s_h.to(self.device)
        s_c = s_c.to(self.device)
        s_m = s_m.to(self.device)
        s_n = (s_n.to(self.device) if s_n is not None
               else torch.zeros(B, T, self.news_dim, device=self.device))

        # Ticker embedding
        tid = (ticker_id.to(self.device) if ticker_id is not None
               else torch.zeros(B, dtype=torch.long, device=self.device))
        v_t = self.ticker_proj(self.ticker_emb(tid))   # (B, dim)

        news_mask_dev = (news_mask.to(self.device) if news_mask is not None else None)

        # ── Encode modalities ─────────────────────────────────────────────────
        v_m, v_i, v_n = self.multimodal_encoder(
            s_o, s_h, s_c, s_m, s_n,
            news_mask=news_mask_dev,
        )
        if v_n is None:
            v_n = torch.zeros_like(v_i)

        # ── Fusion ────────────────────────────────────────────────────────────
        # Stage 1: price × news
        H_id  = self.fusion_stage1(primary=v_i, aux=v_n, aux_mask=news_mask_dev)
        # Stage 2: H_id × macro
        H_idm = self.fusion_stage2(primary=H_id, aux=v_m)

        # ── [FIX-P2] Dual-path prediction ────────────────────────────────────
        #
        # fused path: multimodal output H_idm, conditioned on ticker
        #   → tells the predictor what the full model (price+news+macro) thinks
        v_t_seq      = v_t.unsqueeze(1).expand(-1, T, -1)   # (B, T, dim)
        fused_input  = torch.cat([H_idm, v_t_seq], dim=-1)  # (B, T, 2*dim)
        H_pred_fused = self.fused_proj(fused_input)          # (B, T, dim)

        # orig path: pure price BiGRU output v_i, NO mixing with news/macro
        #   → tells the predictor what price pattern alone suggests
        #   → acts as a skip connection from IndicatorEncoder to logits
        #   → direct gradient path: loss → time_agg_orig → v_i → BiGRU
        H_pred_orig  = v_i                                   # (B, T, dim)

        logits = self.movement_predictor(
            fused_seq=H_pred_fused,
            orig_seq=H_pred_orig,
        )
        logits = torch.clamp(logits, -15, 15)

        def _target(lbl):
            if isinstance(lbl, list):
                return torch.tensor(
                    [x[0] if isinstance(x, (list, tuple)) else x for x in lbl],
                    dtype=torch.long, device=self.device,
                )
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