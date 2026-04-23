# encoders/news_encoder.py
"""
V4 — FinBERT NewsEncoder với temporal_attn optional.

CHANGES vs V3:
  [FIX-P3a] Thêm flag use_temporal_attn (default=False).

  Lý do temporal_attn có thể gây hại với dataset nhỏ:
    - MHA(dim=64, heads=2) ≈ 4 * dim^2 = 16,384 params chỉ cho news encoder
    - Với N_Train=4752 và ~40% news_mask=True (no news), effective keys per sample
      chỉ còn ~12 positions (0.6 * 20). Attention weights với 12 keys và dim=64
      → model học "which of 12 positions matters" → dễ overfit pattern cụ thể
      của training tickers
    - Temporal decay (fixed) đã encode inductive bias "recent news matters more"
      một cách ổn định mà không cần học thêm params

  Khuyến nghị:
    - Bắt đầu với use_temporal_attn=False (default, ít params, ít overfit)
    - Nếu sau khi fix scheduler + modality dropout mà val_mcc vẫn tốt, thử True
    - So sánh bằng ablation study (2 runs, same seed)

  Architecture (use_temporal_attn=False):
    (B, T, 768) → temporal_decay → projector(768→256→64) → (B, T, 64)

  Architecture (use_temporal_attn=True):
    (B, T, 768) → temporal_decay → projector → temporal_attn → residual → (B, T, 64)
"""

import torch
import torch.nn as nn
from typing import Optional


class NewsEncoder(nn.Module):

    def __init__(
        self,
        input_dim:          int,
        dim:                int,
        dropout:            float = 0.1,
        use_temporal_attn:  bool  = False,   # [FIX-P3a] default OFF
    ):
        super().__init__()
        self.use_temporal_attn = use_temporal_attn

        # 768 → 256 → dim (2-step compression, tránh info collapse 1-step)
        mid = max(dim * 4, 256)

        self.projector = nn.Sequential(
            nn.Linear(input_dim, mid),   # 768 → 256
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, dim),          # 256 → 64
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal self-attention (optional)
        if use_temporal_attn:
            # num_heads=2: dim=64 → 32D/head, đủ cho window 20 bước
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
        s_n:       torch.Tensor,                        # (B, T, input_dim)
        news_mask: Optional[torch.Tensor] = None,       # (B, T) bool: True = no news
    ) -> torch.Tensor:                                   # (B, T, dim)
        """
        Args:
            s_n       : News embedding sequence. Zero vectors for no-news days.
            news_mask : True = no news at that timestep (excluded from attn keys).
        """
        # ── Step 1: Temporal recency decay ───────────────────────────────────
        # decay[-1]=1.0 (most recent), decay[0]≈exp(-2.0) (oldest in T=20 window)
        # Fixed prior: recent news có impact lớn hơn → không cần học
        T = s_n.shape[1]
        decay = torch.exp(
            -0.1 * torch.arange(T, 0, -1, dtype=torch.float32)
        ).to(s_n.device)
        s_n = s_n * decay.unsqueeze(0).unsqueeze(-1)   # (B, T, D)

        # ── Step 2: Project to shared latent space ────────────────────────────
        v_n = self.projector(s_n)   # (B, T, dim)

        # ── Step 3: Optional temporal self-attention ──────────────────────────
        if self.use_temporal_attn:
            # Safe masking: nếu ALL timesteps masked → PyTorch MHA → NaN
            attn_mask = news_mask
            if attn_mask is not None:
                fully_masked = attn_mask.all(dim=1, keepdim=True)   # (B, 1)
                if fully_masked.any():
                    attn_mask = attn_mask & ~fully_masked

            attn_out, _ = self.temporal_attn(
                query=v_n,
                key=v_n,
                value=v_n,
                key_padding_mask=attn_mask,
                need_weights=False,
            )
            # Residual: nếu attn saturate sớm, decay-weighted features vẫn chảy qua
            v_n = self.attn_norm(v_n + self.attn_dropout(attn_out))

        return v_n   # (B, T, dim)