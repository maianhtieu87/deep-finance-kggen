# # src/fusion.py - MSGCA PARALLEL GATED CROSS-ATTENTION
# """
# MSGCA-compliant Gated Cross-Attention

# KEY CHANGES from old code:
# 1. ✅ Gating formula: g * new (NOT Highway: (1-g)*old + g*new)
# 2. ✅ Single transform layer (W_a only)
# 3. ✅ Element-wise gating (Eq. 10-11 from MSGCA paper)

# BACKWARD COMPATIBLE:
# - Class name: StableGatedCrossAttention (unchanged)
# - Method signature: forward(primary, aux) (unchanged)
# """

# import torch
# import torch.nn as nn


# class StableGatedCrossAttention(nn.Module):
#     """
#     MSGCA Gated Cross-Attention Mechanism
    
#     Paper Reference: MSGCA Equations 8-11
#     - Multi-head cross-attention for unstable fusion
#     - Primary modality guides stable selection via gating
    
#     Args:
#         dim: Hidden dimension
#         num_head: Number of attention heads
#         dropout: Dropout rate (default: 0.1)
#     """
    
#     def __init__(self, dim, num_head, dropout=0.1): # Add clamp_value and debug_nan
#         super().__init__()
        
#         # ===== STEP 1: Multi-Head Cross-Attention (Eq. 8-9) =====
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=dim,
#             num_heads=num_head,
#             batch_first=True,
#             dropout=dropout
#         )
        
#         # ===== STEP 2: Gating Mechanism (Eq. 10-11) =====
#         # W_a: Transform unstable features from cross-attention
#         self.W_a = nn.Linear(dim, dim)
#         self.bias_a = nn.Parameter(torch.zeros(dim))
        
#         # W_b: Generate gate signal from primary (stable) modality
#         self.W_b = nn.Linear(dim, dim, bias = True)
#         self.bias_b = nn.Parameter(torch.zeros(dim))

#         nn.init.constant_(self.W_b.bias, 1.0)
        
#         # ===== STEP 3: Normalization =====
#         self.norm = nn.LayerNorm(dim)
#         self.dropout = nn.Dropout(dropout)

    
#     def forward(self, primary, aux):
#         # STEP 1: UNSTABLE FUSION (Eq. 8-9)
#         H_unstable, _ = self.cross_attn(
#             query=primary,
#             key=aux,
#             value=aux,
#             need_weights=False
#         )

#         # STEP 2: STABLE GATING (Eq. 10-11)
        
#         H_a = self.W_a(H_unstable) + self.bias_a
#         # H_a = torch.clamp(H_a, -10.0, 10.0)        
#         H_b = torch.sigmoid(self.W_b(primary)) #+ self.bias_b
#         H_gated = H_a * H_b
        
#         # STEP 3: RESIDUAL + NORMALIZATION
        
#         output = self.norm(primary + self.dropout(H_gated))
#         return output

# src/fusion.py
"""
P4 — StableGatedCrossAttention with news attention masking.

Change vs previous:
  forward() accepts aux_mask (B, T) BoolTensor.
  When True for a position, that key is excluded from cross-attention softmax.
  Applied to Stage 1 (price × news) to prevent zero-embedding days from
  diluting the attention output.
  Not applied to Stage 2 (price × macro): macro is always available.

Edge case handled:
  If all T keys in a sample are masked (window with no news at all),
  softmax would produce NaN. We detect fully-masked samples and unmask
  them so attention runs over zero vectors → near-zero output.
  The gated selection (H_b) then naturally down-weights this contribution.

Original MSGCA equations preserved exactly:
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

        # Eq 8-9: multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_head,
            batch_first=True,
            dropout=dropout,
        )

        # Eq 11: gating layers
        self.W_a   = nn.Linear(dim, dim)
        self.bias_a = nn.Parameter(torch.zeros(dim))

        self.W_b   = nn.Linear(dim, dim, bias=True)
        self.bias_b = nn.Parameter(torch.zeros(dim))
        nn.init.constant_(self.W_b.bias, 1.0)   # gate starts open

        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        primary:  torch.Tensor,            # (B, T, dim) — price features (query)
        aux:      torch.Tensor,            # (B, T, dim) — news or macro (key/value)
        aux_mask: Optional[torch.Tensor] = None,  # (B, T) bool: True = exclude key
    ) -> torch.Tensor:
        """
        Args:
            primary  : price encoder output, used as Q and as gate signal.
            aux      : news or macro encoder output, used as K and V.
            aux_mask : (B, T) BoolTensor.
                       True  = this timestep has no valid embedding (P4).
                               Excluded from softmax via key_padding_mask.
                       None  = no masking (used for macro, always available).

        Returns:
            output : (B, T, dim)
        """
        # ── P4: safe masking ──────────────────────────────────────────────────
        # If all keys in a sample are masked, PyTorch MHA produces NaN.
        # Detect fully-masked samples and unmask them: the result is uniform
        # attention over zero vectors → near-zero output, which is correct
        # (no news → gate down-weights the contribution automatically).
        if aux_mask is not None:
            fully_masked = aux_mask.all(dim=1, keepdim=True)  # (B, 1)
            if fully_masked.any():
                # ~fully_masked broadcasts to (B, T)
                aux_mask = aux_mask & ~fully_masked

        # ── Eq 8-9: unstable cross-attention ─────────────────────────────────
        H_unstable, _ = self.cross_attn(
            query=primary,
            key=aux,
            value=aux,
            key_padding_mask=aux_mask,   # None for macro fusion
            need_weights=False,
        )

        # ── Eq 10-11: gated selection ─────────────────────────────────────────
        H_a     = self.W_a(H_unstable) + self.bias_a
        H_b     = torch.sigmoid(self.W_b(primary))
        H_gated = H_a * H_b

        # ── Residual + norm ───────────────────────────────────────────────────
        return self.norm(primary + self.dropout(H_gated))