# encoders/news_encoder.py
"""
V3 — FinBERT-compatible NewsEncoder with Temporal Self-Attention.

V3 changes vs V2 (Voyage 1024D):
  - input_dim: 1024 → 768 (FinBERT [CLS] hidden state dimension)
  - Compression ratio: 768→256→64 (12x) instead of 1024→256→64 (16x)
    → Less aggressive compression, less information loss
  - Added temporal self-attention (from At-LSTM, Paper 1):
    Learns WHICH DAYS in the window matter, instead of fixed exponential decay.
    Fixed decay cannot distinguish: earnings announcement vs routine analyst update
    at the same temporal position. Self-attention learns this dynamically.
  - news_mask propagated into temporal_attn to exclude no-news timesteps
    from attention keys (consistent with fusion.py safe-masking logic).

Architecture flow:
  (B, T, 768)  →  temporal_decay  →  projector(768→256→64)
  →  temporal_self_attn(q=v_n, k=v_n, v=v_n, mask=news_mask)
  →  residual + LayerNorm  →  (B, T, 64)

Rationale for temporal self-attention:
  At-LSTM (Paper 1) found explicit day-level attention was a key component.
  Within the news window [t-19, ..., t], the model learns:
    - Days with high-impact triples (earnings, regulatory) → high attention weight
    - Days with zero vectors (no news) → masked out
    - Days with analyst noise → low attention weight
  This is learned end-to-end, not hand-crafted.
"""

import torch
import torch.nn as nn
from typing import Optional


class NewsEncoder(nn.Module):

    def __init__(self, input_dim: int, dim: int, dropout: float = 0.1):
        super().__init__()

        # 768 → 256 → dim (FinBERT → shared latent space)
        # mid = max(dim*4, 256): ensures at least 256 even if dim < 64
        mid = max(dim * 4, 256)

        self.projector = nn.Sequential(
            nn.Linear(input_dim, mid),    # 768 → 256
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, dim),           # 256 → 64
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal self-attention: which days in the window are most informative?
        # num_heads=2: dim=64 → 32D per head (sufficient for 20-step window)
        # Small model → single layer, no stacking (avoids overfitting at ~600 samples)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=2,
            batch_first=True,
            dropout=dropout,
        )
        self.attn_norm    = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for layer in self.projector:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(
        self,
        s_n: torch.Tensor,                         # (B, T, input_dim)
        news_mask: Optional[torch.Tensor] = None,  # (B, T) bool: True = no news
    ) -> torch.Tensor:                              # (B, T, dim)
        """
        Args:
            s_n       : News embedding sequence. Zero vectors for no-news days.
            news_mask : BoolTensor, True where there is no news (matches fusion.py convention).
                        These timesteps are excluded from temporal attention keys
                        to prevent zero-embedding days from diluting the attention output.
        """
        # Step 1: Temporal recency decay (prior: recent news matters more)
        # decay[-1] = 1.0 (most recent), decay[0] ≈ exp(-0.1*T) (oldest)
        T = s_n.shape[1]
        decay = torch.exp(
            -0.1 * torch.arange(T, 0, -1, dtype=torch.float32)
        ).to(s_n.device)
        s_n = s_n * decay.unsqueeze(0).unsqueeze(-1)  # (B, T, D)

        # Step 2: Project to shared latent space
        v_n = self.projector(s_n)  # (B, T, dim)

        # Step 3: Temporal self-attention with news masking
        # Safe masking: if ALL timesteps are masked (window with no news),
        # PyTorch MHA produces NaN. Detect and unmask fully-masked samples.
        attn_mask = news_mask
        if attn_mask is not None:
            fully_masked = attn_mask.all(dim=1, keepdim=True)  # (B, 1)
            if fully_masked.any():
                # For fully-masked samples, allow attention over zero vectors
                # → near-zero output, gate in fusion.py down-weights this
                attn_mask = attn_mask & ~fully_masked  # (B, T)

        attn_out, _ = self.temporal_attn(
            query=v_n,
            key=v_n,
            value=v_n,
            key_padding_mask=attn_mask,  # True = exclude from softmax
            need_weights=False,
        )

        # Step 4: Residual connection
        # If temporal_attn saturates early, residual ensures decay-weighted
        # features still flow through to the cross-attention fusion stage.
        v_n = self.attn_norm(v_n + self.attn_dropout(attn_out))

        return v_n  # (B, T, dim)