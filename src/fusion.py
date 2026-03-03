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
    
    def __init__(self, dim, num_head, dropout=0.1):
        super().__init__()
        
        # ===== STEP 1: Multi-Head Cross-Attention (Eq. 8-9) =====
        # Internally processes M heads in PARALLEL:
        # for m in range(num_heads):
        #     Q_m = primary @ W_Q[m]
        #     K_m = aux @ W_K[m]
        #     V_m = aux @ W_V[m]
        #     attn_m = softmax(Q_m @ K_m^T / √d) @ V_m
        # output = concat(attn_1, ..., attn_M) @ W_O
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
        self.W_b = nn.Linear(dim, dim)
        self.bias_b = nn.Parameter(torch.zeros(dim))
        
        # ===== STEP 3: Normalization =====
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, primary, aux):
        """
        MSGCA Gated Fusion Forward Pass
        
        Args:
            primary: Primary/stable modality (B, T, D)
                    e.g., Price indicators (complete, reliable)
            aux: Auxiliary/unstable modality (B, T, D)
                 e.g., News/Macro (sparse, noisy)
        
        Returns:
            output: Gated fusion result (B, T, D)
        
        Process:
            1. Unstable fusion via multi-head cross-attention
            2. Stable selection via gating from primary
            3. Residual connection + normalization
        """
        
        # ========================================
        # STEP 1: UNSTABLE FUSION (Eq. 8-9)
        # ========================================
        # Multi-head cross-attention:
        # - Query: from Primary (what we trust)
        # - Key/Value: from Auxiliary (what we want to filter)
        # 
        # Output: H^l_{unstable} = MHA(Q=primary, K=aux, V=aux)
        H_unstable, _ = self.cross_attn(
            query=primary,      # (B, T, D) - Reliable signal
            key=aux,            # (B, T, D) - Noisy signal
            value=aux,          # (B, T, D)
            need_weights=False  # Don't return attention weights
        )  # → (B, T, D)
        
        # ========================================
        # STEP 2: STABLE GATING (Eq. 10-11)
        # ========================================
        
        # Eq. 10a: Transform unstable features
        # H_a = H_unstable @ W_a + b_a
        H_a = self.W_a(H_unstable) + self.bias_a  # (B, T, D)
        
        # Eq. 10b: Generate gate from PRIMARY modality
        # H_b = Sigmoid(Primary @ W_b + b_b)
        # 
        # Gate interpretation:
        # - H_b → 1: Auxiliary is reliable, keep it
        # - H_b → 0: Auxiliary is noisy, filter it out
        H_b = torch.sigmoid(self.W_b(primary) + self.bias_b)  # (B, T, D)
        
        # Eq. 11: Gated selection (Element-wise product)
        # H_gated = H_a ⊙ H_b
        #
        # ⚠️ KEY DIFFERENCE from Highway Networks:
        # - Highway: output = (1-gate)*old + gate*new
        # - MSGCA:   output = gate * new  (only keep gated part)
        #
        # Why MSGCA is better:
        # - Explicitly filters noise via primary guidance
        # - Prevents "leaky residual" from noisy modality
        H_gated = H_a * H_b  # (B, T, D)
        
        # ========================================
        # STEP 3: RESIDUAL + NORMALIZATION
        # ========================================
        # Add residual from primary (stable baseline)
        # Normalize for training stability
        output = self.norm(primary + self.dropout(H_gated))
        
        return output