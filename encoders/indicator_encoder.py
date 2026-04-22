# encoders/indicator_encoder.py

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
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
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