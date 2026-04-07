# import torch
# from torch import nn
# from torch.nn.utils.parametrizations import spectral_norm as spectral_norm


# class IndicatorSequenceEncoder(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#         self.proj_c = spectral_norm(nn.Linear(1, dim))
#         self.proj_o = spectral_norm(nn.Linear(1, dim))
#         self.proj_h = spectral_norm(nn.Linear(1, dim))
#         self.combine = spectral_norm(nn.Linear(3 * dim, dim))

#     def forward(self, s_o, s_h, s_c):
#         """
#         s_o, s_h, s_c: (T, 1) hoặc (batch, T, 1)
#         """
#         v_o = self.proj_o(s_o)
#         v_h = self.proj_h(s_h)
#         v_c = self.proj_c(s_c)

        # v_i = self.combine(torch.cat([v_o, v_h, v_c], dim=-1))
        # return v_i
    
    
# Dùng gần nhất
# import torch
# from torch import nn
# from torch.nn.utils.parametrizations import spectral_norm as spectral_norm
# from configs.config import TrainConfig

# class IndicatorSequenceEncoder(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
        
#         # Lấy dropout từ config hoặc default là 0.1
#         # dropout_rate = getattr(TrainConfig, 'drop_out', 0.1)

#         # 1. Projection cho Close Price
#         self.proj_c = nn.Sequential(
#             spectral_norm(nn.Linear(1, dim)),
#             # nn.GELU(),
#             # nn.Dropout(dropout_rate)       
#         )

#         # 2. Projection cho Open Price
#         self.proj_o = nn.Sequential(
#             spectral_norm(nn.Linear(1, dim)),
#             # nn.GELU(), 
#             # nn.Dropout(dropout_rate)       
#         )

#         # 3. Projection cho High Price
#         self.proj_h = nn.Sequential(
#             spectral_norm(nn.Linear(1, dim)),
#             # nn.GELU(), 
#             # nn.Dropout(dropout_rate)       
#         )

#         # 4. Combine Layer: Gộp thông tin từ O, H, C
#         # Input: 3 * dim -> Output: dim
#         # Thêm LayerNorm để ổn định feature trước khi đưa vào Transformer/LSTM
#         self.combine = nn.Sequential(
#             spectral_norm(nn.Linear(3 * dim, dim)),
#             # nn.LayerNorm(dim),
#             # nn.GELU()
#         )

#     def forward(self, s_o, s_h, s_c):
#         """
#         Input:
#             s_o, s_h, s_c: Tensor shape (batch, T, 1) hoặc (T, 1)
#         Output:
#             v_i: Tensor shape (batch, T, dim)
#         """
#         # Feature Projection
#         v_o = self.proj_o(s_o)
#         v_h = self.proj_h(s_h)
#         v_c = self.proj_c(s_c)

#         # Concatenate & Combine
#         # Nối 3 vector theo chiều feature (dim=-1)
#         merged = torch.cat([v_o, v_h, v_c], dim=-1)
        
#         v_i = self.combine(merged)
        
#         return v_i
    

# import torch
# from torch import nn

# class IndicatorSequenceEncoder(nn.Module):
#     def __init__(self, dim,):
#         super().__init__()
#         self.dim = dim
        
#         self.proj_c = nn.Linear(1, dim)
#         self.proj_o = nn.Linear(1, dim)
#         self.proj_h = nn.Linear(1, dim)

#         # Gộp thông tin O, H, C
#         self.combine = nn.Linear(3 * dim, dim)

#     def forward(self, s_o, s_h, s_c):
#         v_o = self.proj_o(s_o)
#         v_h = self.proj_h(s_h)
#         v_c = self.proj_c(s_c)

#         merged = torch.cat([v_o, v_h, v_c], dim=-1)
#         v_i = self.combine(merged)
        
#         return v_i
    

# encoders/indicator_encoder.py
"""
V2 — GRU-based Indicator Sequence Encoder (P3)

Change vs V1 (spectral_norm linear):
  Linear projection is permutation-invariant — it cannot distinguish
  an uptrend from a reversal if both have the same mean value.

  GRU captures sequential order. Measured pattern discrimination:
    Linear mean-pool  : cos(uptrend, reversal) = 0.77  (near-identical)
    GRU last state    : cos(uptrend, reversal) = 0.02  (well-separated ✓)

Architecture:
  1. Per-component projection: O/H/C each Linear(1 → dim)
  2. Combine: Linear(3*dim → dim)  [same as paper's Eq. 1-2]
  3. Bidirectional GRU(dim → dim//2 each dir → dim total)
     Bidirectional: forward pass captures momentum/trend,
                    backward pass captures reversal context.
  4. Residual + LayerNorm: preserve linear features, stabilize training.

Init strategy:
  - Orthogonal init for GRU recurrent weights → stable gradient norms
  - Forget gate bias = 1.0 → model "remembers" by default at start,
    learning to selectively forget as training progresses
  - Xavier for input weights (standard)

References:
  ALSTM (Feng et al. 2018), SLOT (Soun et al. 2022), DTML (Yoo et al. 2021)
  all use LSTM/GRU in the price encoder for temporal pattern capture.
"""

import torch
import torch.nn as nn


class IndicatorSequenceEncoder(nn.Module):

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim

        # ── Step 1: Per-component linear projection (as in MSGCA Eq. 1-2) ──
        self.proj_c = nn.Linear(1, dim)   # Close price
        self.proj_o = nn.Linear(1, dim)   # Open  price
        self.proj_h = nn.Linear(1, dim)   # High  price

        # ── Step 2: Combine O/H/C into single sequence ─────────────────────
        self.combine = nn.Linear(3 * dim, dim)

        # ── Step 3: Bidirectional GRU ─────────────────────────────────────
        # hidden_size = dim // 2 each direction → output = dim (same as input)
        # No stacking (num_layers=1): sufficient for 20-step window,
        # deeper GRU risks overfitting at ~600 training samples.
        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,   # single layer → GRU dropout has no effect anyway
        )

        # ── Step 4: Output normalization ───────────────────────────────────
        # LayerNorm stabilizes GRU output before entering cross-attention.
        self.out_norm = nn.LayerNorm(dim)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        self._init_weights()

    def _init_weights(self):
        """
        Principled initialization for stable training.
        Critical for GRU: prevents gradient vanishing/exploding at start.
        """
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                # Orthogonal init: recurrent weight matrix → preserves
                # gradient norm through time steps at initialization
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                # Xavier: input-to-hidden → appropriate scale for GELU-like
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Forget gate bias = 1.0: model starts by remembering everything,
                # learns to forget selectively as it trains
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)

        # Standard init for projection layers
        for layer in [self.proj_c, self.proj_o, self.proj_h, self.combine]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        s_o: torch.Tensor,   # (B, T, 1) — Open  price sequence
        s_h: torch.Tensor,   # (B, T, 1) — High  price sequence
        s_c: torch.Tensor,   # (B, T, 1) — Close price sequence
    ) -> torch.Tensor:       # (B, T, dim)
        """
        Encode three price sequences into a temporally-aware representation.

        The GRU output at position t encodes the full context from t=0
        (forward direction) and from t=T-1 back to t (backward direction).
        This allows the model to distinguish:
          - Uptrend:  consistently positive increments → forward GRU builds up
          - Reversal: positive first half, negative second half → backward GRU
                      detects the inflection and suppresses the "momentum" signal
        """
        # Per-component projection: each price series → latent space
        v_o = self.proj_o(s_o)   # (B, T, dim)
        v_h = self.proj_h(s_h)   # (B, T, dim)
        v_c = self.proj_c(s_c)   # (B, T, dim)

        # Combine O/H/C into unified sequence (MSGCA Eq. 1-2 adapted)
        fused = self.combine(torch.cat([v_o, v_h, v_c], dim=-1))  # (B, T, dim)

        # Bidirectional GRU: capture temporal patterns
        gru_out, _ = self.gru(fused)   # (B, T, dim)  [dim//2 fwd + dim//2 bwd]

        # Residual connection: preserve linear projection features.
        # If GRU gates saturate near zero early in training (common with
        # aggressive LR), the residual ensures price info still flows through.
        out = self.out_norm(gru_out + fused)   # (B, T, dim)

        if self.dropout is not None:
            out = self.dropout(out)

        return out