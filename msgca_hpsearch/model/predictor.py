"""
FinegrainedMovementPrediction — dual-path movement prediction head.

Dual path:
  fused_seq : multimodal features (price + news + macro + ticker)
  orig_seq  : pure price BiGRU output (direct gradient path)

Both paths aggregate time dimension T→1 via ThreeLayerMLP,
then concatenate and project to num_classes logits.
"""

import torch
import torch.nn as nn
from .modules.layers import ThreeLayerMLP


class FinegrainedMovementPrediction(nn.Module):

    def __init__(self, dim: int, window_size: int, num_classes: int = 3, dropout: float = 0.0):
        super().__init__()

        self.time_agg_fused = ThreeLayerMLP(
            d_in=window_size, d_out=1,
            d_h1=window_size // 2, d_h2=window_size // 4,
            final_activation=True, dropout=dropout,
        )
        self.time_agg_orig = ThreeLayerMLP(
            d_in=window_size, d_out=1,
            d_h1=window_size // 2, d_h2=window_size // 4,
            final_activation=True, dropout=dropout,
        )
        self.feat_agg = ThreeLayerMLP(
            d_in=2 * dim, d_out=num_classes,
            d_h1=dim, d_h2=dim // 2,
            final_activation=False, dropout=dropout,
        )

    def forward(self, fused_seq: torch.Tensor, orig_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_seq : (B, T, dim) — multimodal fused features
            orig_seq  : (B, T, dim) — pure price encoding
        Returns:
            logits : (B, num_classes)
        """
        fused_t = fused_seq.transpose(1, 2)   # (B, dim, T)
        orig_t  = orig_seq.transpose(1, 2)

        h_fused = self.time_agg_fused(fused_t).squeeze(-1)   # (B, dim)
        h_orig  = self.time_agg_orig(orig_t).squeeze(-1)     # (B, dim)

        h_final = torch.cat([h_fused, h_orig], dim=-1)       # (B, 2*dim)
        return self.feat_agg(h_final)                         # (B, num_classes)
