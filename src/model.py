# src/model.py
"""
StockMovementModel — Sequential Gated Cross-Attention

Phase 2 changes (vs previous):
  [PHASE2-1] quality_dim parameter in __init__
    - Passed through to MultimodalSourceEncoding → NewsEncoder → quality_mlp
    - quality_dim=1 : Phase 1 (norm proxy, backward-compat default)
    - quality_dim=4 : Phase 2 (pipeline quality from KG triples)
    - Read from GlobalConfig.QUALITY_DIM at call site (main.py)

  [PHASE2-2] news_quality in forward()
    - Optional (B, T, quality_dim) tensor from batch
    - Sent to device + passed to multimodal_encoder → news_encoder
    - None when no quality data or during modality dropout

Previous fixes retained:
  [FIX-P2] Dual-path predictor: H_pred_fused = fused_proj(H_idm + ticker),
           H_pred_orig = v_i (pure price BiGRU, direct gradient path)
"""

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef

from configs.config import TrainConfig, GlobalConfig
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
        price_dim:      int,
        macro_dim:      int,
        news_dim:       int,
        dim:            int,
        input_dim:      int,
        output_dim:     int,
        num_head:       int,
        device,
        dropout:        float = 0.1,
        class_weights=None,
        use_focal_loss: bool  = True,
        focal_gamma:    float = 2.0,
        use_gnn:        bool  = False,
        n_tickers:      int   = N_TICKERS,
        ticker_emb_dim: int   = 16,
        quality_dim:    int   = 1,    # [PHASE2-1] 1=norm proxy; 4=pipeline quality
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
        # [PHASE2-1] quality_dim passed to MultimodalSourceEncoding → NewsEncoder
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim,
            macro_dim=macro_dim,
            news_dim=news_dim,
            dim=dim,
            dropout=dropout,
            quality_dim=quality_dim,
        )

        # ── Sequential Gated Cross-Attention Fusion ───────────────────────────
        self.fusion_stage1 = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        self.fusion_stage2 = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)

        # ── [FIX-P2] Predictor input projection ──────────────────────────────
        # fused_proj: cat([H_idm, v_t_seq]) → dim  (multimodal + ticker only)
        # orig path : v_i directly             (pure price, no mixing)
        self.fused_proj = nn.Sequential(
            nn.Linear(2 * dim, dim),
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
        mode:          str  = "train",
        return_preds:  bool = False,
        ticker_id=None,
        news_mask=None,
        news_quality=None,   # [PHASE2-2] (B, T, quality_dim) or None
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

        news_mask_dev    = (news_mask.to(self.device)
                            if news_mask is not None else None)

        # [PHASE2-2] Move quality tensor to device; None = norm proxy fallback
        news_quality_dev = (news_quality.to(self.device)
                            if news_quality is not None else None)

        # ── Encode modalities ─────────────────────────────────────────────────
        v_m, v_i, v_n, g_n = self.multimodal_encoder(
            s_o, s_h, s_c, s_m, s_n,
            news_mask=news_mask_dev,
            news_quality=news_quality_dev,   # [PHASE2-2]
        )
        if v_n is None:
            v_n = torch.zeros_like(v_i)

        # ── Fusion ────────────────────────────────────────────────────────────
        # Stage 1: price × news (news_mask gates out no-news positions)
        H_id  = self.fusion_stage1(primary=v_i, aux=v_n, aux_mask=news_mask_dev)
        # Stage 2: (price+news) × macro
        H_idm = self.fusion_stage2(primary=H_id, aux=v_m)

        # ── [FIX-P2] Dual-path prediction ────────────────────────────────────
        v_t_seq      = v_t.unsqueeze(1).expand(-1, T, -1)   # (B, T, dim)
        fused_input  = torch.cat([H_idm, v_t_seq], dim=-1)  # (B, T, 2*dim)
        H_pred_fused = self.fused_proj(fused_input)          # (B, T, dim)
        H_pred_orig  = v_i                                   # (B, T, dim) — pure price

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