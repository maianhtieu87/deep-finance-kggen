"""
StableGatedCrossAttention — MSGCA Gated Cross-Attention Fusion.

Implements MSGCA equations 8-11:
  Eq 8-9: H_unstable = MultiheadCrossAttention(Q=primary, K=V=aux)
  Eq 10:  H_gated    = H_a ⊙ H_b
  Eq 11:  H_a = W_a · H_unstable + b
          H_b = sigmoid(W_b · primary + b')
  output  = LayerNorm(primary + dropout(H_gated))
"""

import torch
import torch.nn as nn
from typing import Optional


class StableGatedCrossAttention(nn.Module):

    def __init__(self, dim: int, num_head: int, dropout: float = 0.1):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_head,
            batch_first=True,
            dropout=dropout,
        )

        self.W_a    = nn.Linear(dim, dim)
        self.bias_a = nn.Parameter(torch.zeros(dim))
        self.W_b    = nn.Linear(dim, dim, bias=True)
        self.bias_b = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.W_b.bias, 1.0)   # gate starts open

        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        primary:  torch.Tensor,
        aux:      torch.Tensor,
        aux_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            primary  : (B, T, dim) — price features (query + gate signal)
            aux      : (B, T, dim) — news or macro (key/value)
            aux_mask : (B, T) BoolTensor — True = exclude this timestep from softmax
                       Used for price×news stage; None for price×macro stage.
        Returns:
            (B, T, dim)
        """
        # Safe masking: fully-masked samples cause NaN in MHA softmax
        if aux_mask is not None:
            fully_masked = aux_mask.all(dim=1, keepdim=True)
            if fully_masked.any():
                aux_mask = aux_mask & ~fully_masked

        # Eq 8-9: unstable cross-attention
        H_unstable, _ = self.cross_attn(
            query=primary,
            key=aux,
            value=aux,
            key_padding_mask=aux_mask,
            need_weights=False,
        )

        # Eq 10-11: gated selection
        H_a     = self.W_a(H_unstable) + self.bias_a
        H_b     = torch.sigmoid(self.W_b(primary))
        H_gated = H_a * H_b

        return self.norm(primary + self.dropout(H_gated))
