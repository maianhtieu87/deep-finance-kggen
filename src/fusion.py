# src/fusion.py - MSGCA PARALLEL GATED CROSS-ATTENTION
"""
MSGCA-compliant Gated Cross-Attention

KEY CHANGES from old code:
1. ✅ Gating formula: g * new (NOT Highway: (1-g)*old + g*new)
2. ✅ Single transform layer (W_a only)
3. ✅ Element-wise gating (Eq. 10-11 from MSGCA paper)

BACKWARD COMPATIBLE:
- Class name: StableGatedCrossAttention (unchanged)
- Method signature: forward(primary, aux) (unchanged)
"""

import torch
import torch.nn as nn


class StableGatedCrossAttention(nn.Module):
    """
    MSGCA Gated Cross-Attention Mechanism
    
    Paper Reference: MSGCA Equations 8-11
    - Multi-head cross-attention for unstable fusion
    - Primary modality guides stable selection via gating
    
    Args:
        dim: Hidden dimension
        num_head: Number of attention heads
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(self, dim, num_head, dropout=0.1): # Add clamp_value and debug_nan
        super().__init__()
        
        # ===== STEP 1: Multi-Head Cross-Attention (Eq. 8-9) =====
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_head,
            batch_first=True,
            dropout=dropout
        )
        
        # ===== STEP 2: Gating Mechanism (Eq. 10-11) =====
        # W_a: Transform unstable features from cross-attention
        self.W_a = nn.Linear(dim, dim)
        self.bias_a = nn.Parameter(torch.zeros(dim))
        
        # W_b: Generate gate signal from primary (stable) modality
        self.W_b = nn.Linear(dim, dim, bias = True)
        self.bias_b = nn.Parameter(torch.zeros(dim))

        nn.init.constant_(self.W_b.bias, 1.0)
        
        # ===== STEP 3: Normalization =====
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, primary, aux):
        # STEP 1: UNSTABLE FUSION (Eq. 8-9)
        H_unstable, _ = self.cross_attn(
            query=primary,
            key=aux,
            value=aux,
            need_weights=False
        )

        # STEP 2: STABLE GATING (Eq. 10-11)
        
        H_a = self.W_a(H_unstable) + self.bias_a
        # H_a = torch.clamp(H_a, -10.0, 10.0)        
        H_b = torch.sigmoid(self.W_b(primary)) #+ self.bias_b
        H_gated = H_a * H_b
        
        # STEP 3: RESIDUAL + NORMALIZATION
        
        output = self.norm(primary + self.dropout(H_gated))
        return output

