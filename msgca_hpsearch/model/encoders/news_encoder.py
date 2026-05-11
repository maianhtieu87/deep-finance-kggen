"""
NewsEncoder — Quality-aware news encoding with dual gate.

Architecture:
  (B, T, input_dim)
    → quality signal (external pipeline stats OR embedding-norm proxy)
    → temporal decay
    → projector (input_dim → mid → dim)
    → explicit zero-out for no-news days
    → dual gate: q_gate (quality_mlp) × f_gate (feature_gate)
    → optional temporal self-attention
    → (v_n: B,T,dim), (g_n: B,T,1)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class NewsEncoder(nn.Module):

    def __init__(
        self,
        input_dim:         int,
        dim:               int,
        dropout:           float = 0.1,
        use_temporal_attn: bool  = False,
        learnable_decay:   bool  = False,
        quality_dim:       int   = 4,
    ):
        super().__init__()
        self.use_temporal_attn = use_temporal_attn
        self.learnable_decay   = learnable_decay
        self.quality_dim       = quality_dim

        mid = max(dim * 4, 256)

        self.projector = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.quality_mlp = nn.Sequential(
            nn.Linear(quality_dim, max(dim // 4, 16)),
            nn.GELU(),
            nn.Linear(max(dim // 4, 16), 1),
        )

        self.feature_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

        if use_temporal_attn:
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=dim, num_heads=2, batch_first=True, dropout=dropout,
            )
            self.attn_norm    = nn.LayerNorm(dim)
            self.attn_dropout = nn.Dropout(dropout)

        if learnable_decay:
            self.decay_logit = nn.Parameter(torch.tensor(-2.3))

        self._init_weights()

    def _init_weights(self):
        for module in [self.projector, self.quality_mlp, self.feature_gate]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def _apply_temporal_decay(self, s_n: torch.Tensor) -> torch.Tensor:
        T = s_n.shape[1]
        if self.learnable_decay:
            alpha = torch.nn.functional.softplus(self.decay_logit)
        else:
            alpha = torch.tensor(0.1, device=s_n.device, dtype=s_n.dtype)
        time_idx = torch.arange(T, 0, -1, device=s_n.device, dtype=s_n.dtype)
        decay = torch.exp(-alpha * time_idx)
        return s_n * decay.view(1, T, 1)

    def forward(
        self,
        s_n:          torch.Tensor,
        news_mask:    Optional[torch.Tensor] = None,
        news_quality: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = s_n.shape

        # ── Step 1: Quality signal ─────────────────────────────────────────
        if news_quality is not None:
            if news_quality.shape[-1] != self.quality_dim:
                raise ValueError(
                    f"news_quality dim mismatch: got {news_quality.shape[-1]}, "
                    f"expected {self.quality_dim}."
                )
            q_signal = news_quality
        else:
            input_norm = s_n.norm(dim=-1, keepdim=True)
            max_norm   = input_norm.detach().max().clamp(min=1e-8)
            norm_proxy = input_norm / max_norm
            if self.quality_dim == 1:
                q_signal = norm_proxy
            else:
                q_signal = torch.cat([
                    norm_proxy,
                    torch.zeros(B, T, self.quality_dim - 1,
                                device=s_n.device, dtype=s_n.dtype),
                ], dim=-1)

        # ── Step 2: Temporal decay ─────────────────────────────────────────
        s_n = self._apply_temporal_decay(s_n)

        # ── Step 3: Project to latent space ──────────────────────────────
        v_n = self.projector(s_n)

        # ── Step 4: Zero-out no-news days ─────────────────────────────────
        if news_mask is not None:
            v_n = v_n.masked_fill(news_mask.unsqueeze(-1), 0.0)

        # ── Step 5: Dual gate ─────────────────────────────────────────────
        q_gate = torch.sigmoid(self.quality_mlp(q_signal))
        f_gate = torch.sigmoid(self.feature_gate(v_n))
        g_n    = q_gate * f_gate

        if news_mask is not None:
            g_n = g_n.masked_fill(news_mask.unsqueeze(-1), 0.0)

        v_n = v_n * g_n

        # ── Step 6: Optional temporal self-attention ──────────────────────
        if self.use_temporal_attn:
            attn_mask = news_mask
            if attn_mask is not None:
                fully_masked = attn_mask.all(dim=1, keepdim=True)
                if fully_masked.any():
                    attn_mask = attn_mask & ~fully_masked
            attn_out, _ = self.temporal_attn(
                query=v_n, key=v_n, value=v_n,
                key_padding_mask=attn_mask, need_weights=False,
            )
            v_n = self.attn_norm(v_n + self.attn_dropout(attn_out))
            if news_mask is not None:
                v_n = v_n.masked_fill(news_mask.unsqueeze(-1), 0.0)

        return v_n, g_n
