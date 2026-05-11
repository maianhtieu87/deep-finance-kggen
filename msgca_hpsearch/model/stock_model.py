"""
StockMovementModel — Sequential Gated Cross-Attention (MSGCA)

Standalone version: all imports are local (no project-level configs).
Config is passed as constructor parameters instead of reading from TrainConfig.

Architecture:
  price  → IndicatorSequenceEncoder (BiGRU)     → v_i (B,T,dim)
  macro  → MacroIndicatorEncoder   (Linear)     → v_m (B,T,dim)
  news   → NewsEncoder (quality-gated)          → v_n (B,T,dim)
  ticker → Embedding + proj                     → v_t (B,dim)

  Stage 1: GatedCrossAttn(primary=v_i, aux=v_n) → H_id
  Stage 2: GatedCrossAttn(primary=H_id, aux=v_m) → H_idm

  Dual-path predictor:
    fused = fused_proj(cat[H_idm, v_t_seq])
    orig  = v_i  (pure price, direct gradient path)
    logits = FinegrainedMovementPrediction(fused, orig)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef

from .encoders.mutil_encoder import MultimodalSourceEncoding
from .fusion import StableGatedCrossAttention
from .predictor import FinegrainedMovementPrediction


# ─────────────────────────────────────────────────────────────────────────────
# Default ticker list — must match the dataset you're using
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_TICKERS = ["TSLA", "AAPL", "AMZN", "MSFT", "GOOGL", "META", "BA", "JPM", "WMT"]
TICKER_TO_ID    = {t: i for i, t in enumerate(DEFAULT_TICKERS)}
N_TICKERS       = len(DEFAULT_TICKERS)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
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
        input_dim:      int,          # = window_size (T)
        output_dim:     int,          # = num_classes (3)
        num_head:       int,
        device,
        dropout:        float = 0.1,
        class_weights          = None,
        use_focal_loss: bool   = False,
        focal_gamma:    float  = 2.0,
        n_tickers:      int    = N_TICKERS,
        ticker_emb_dim: int    = 16,
        quality_dim:    int    = 4,
    ):
        super().__init__()
        self.device   = device
        self.news_dim = news_dim
        self.dim      = dim

        # ── Ticker embedding ──────────────────────────────────────────────
        self.ticker_emb  = nn.Embedding(n_tickers, ticker_emb_dim, padding_idx=None)
        self.ticker_proj = nn.Sequential(
            nn.Linear(ticker_emb_dim, dim),
            nn.LayerNorm(dim),
            nn.Tanh(),
        )
        nn.init.normal_(self.ticker_emb.weight, mean=0.0, std=0.02)

        # ── Multimodal encoders ───────────────────────────────────────────
        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim,
            macro_dim=macro_dim,
            news_dim=news_dim,
            dim=dim,
            dropout=dropout,
            quality_dim=quality_dim,
        )

        # ── Sequential Gated Cross-Attention ─────────────────────────────
        self.fusion_stage1 = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)
        self.fusion_stage2 = StableGatedCrossAttention(dim=dim, num_head=num_head, dropout=dropout)

        # ── Dual-path predictor input projection ─────────────────────────
        self.fused_proj = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ── Movement predictor ────────────────────────────────────────────
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim,
            window_size=input_dim,
            num_classes=output_dim,
            dropout=dropout,
        )

        # ── Loss function ─────────────────────────────────────────────────
        if use_focal_loss:
            self.loss_fn = FocalLoss(alpha=class_weights, gamma=focal_gamma)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def forward(
        self,
        s_o, s_h, s_c, s_m, s_n,
        label        = None,
        mode:  str   = "train",
        return_preds: bool = False,
        ticker_id    = None,
        news_mask    = None,
        news_quality = None,
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

        tid = (ticker_id.to(self.device) if ticker_id is not None
               else torch.zeros(B, dtype=torch.long, device=self.device))
        v_t = self.ticker_proj(self.ticker_emb(tid))   # (B, dim)

        news_mask_dev    = (news_mask.to(self.device) if news_mask is not None else None)
        news_quality_dev = (news_quality.to(self.device) if news_quality is not None else None)

        # ── Encode ────────────────────────────────────────────────────────
        v_m, v_i, v_n, g_n = self.multimodal_encoder(
            s_o, s_h, s_c, s_m, s_n,
            news_mask=news_mask_dev,
            news_quality=news_quality_dev,
        )
        if v_n is None:
            v_n = torch.zeros_like(v_i)

        # ── Fusion ────────────────────────────────────────────────────────
        H_id  = self.fusion_stage1(primary=v_i, aux=v_n, aux_mask=news_mask_dev)
        H_idm = self.fusion_stage2(primary=H_id, aux=v_m)

        # ── Dual-path prediction ──────────────────────────────────────────
        v_t_seq      = v_t.unsqueeze(1).expand(-1, T, -1)
        fused_input  = torch.cat([H_idm, v_t_seq], dim=-1)
        H_pred_fused = self.fused_proj(fused_input)
        H_pred_orig  = v_i

        logits = self.movement_predictor(fused_seq=H_pred_fused, orig_seq=H_pred_orig)
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
