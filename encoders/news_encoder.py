# encoders/news_encoder.py
# """
# V4 — FinBERT NewsEncoder với temporal_attn optional.

# CHANGES vs V3:
#   [FIX-P3a] Thêm flag use_temporal_attn (default=False).

#   Lý do temporal_attn có thể gây hại với dataset nhỏ:
#     - MHA(dim=64, heads=2) ≈ 4 * dim^2 = 16,384 params chỉ cho news encoder
#     - Với N_Train=4752 và ~40% news_mask=True (no news), effective keys per sample
#       chỉ còn ~12 positions (0.6 * 20). Attention weights với 12 keys và dim=64
#       → model học "which of 12 positions matters" → dễ overfit pattern cụ thể
#       của training tickers
#     - Temporal decay (fixed) đã encode inductive bias "recent news matters more"
#       một cách ổn định mà không cần học thêm params

#   Khuyến nghị:
#     - Bắt đầu với use_temporal_attn=False (default, ít params, ít overfit)
#     - Nếu sau khi fix scheduler + modality dropout mà val_mcc vẫn tốt, thử True
#     - So sánh bằng ablation study (2 runs, same seed)

#   Architecture (use_temporal_attn=False):
#     (B, T, 768) → temporal_decay → projector(768→256→64) → (B, T, 64)

#   Architecture (use_temporal_attn=True):
#     (B, T, 768) → temporal_decay → projector → temporal_attn → residual → (B, T, 64)
# """

# import torch
# import torch.nn as nn
# from typing import Optional


# class NewsEncoder(nn.Module):

#     def __init__(
#         self,
#         input_dim:          int,
#         dim:                int,
#         dropout:            float = 0.1,
#         use_temporal_attn:  bool  = False,   # [FIX-P3a] default OFF
#     ):
#         super().__init__()
#         self.use_temporal_attn = use_temporal_attn

#         # 768 → 256 → dim (2-step compression, tránh info collapse 1-step)
#         mid = max(dim * 4, 256)

#         self.projector = nn.Sequential(
#             nn.Linear(input_dim, mid),   # 768 → 256
#             nn.GELU(),
#             nn.LayerNorm(mid),
#             nn.Linear(mid, dim),          # 256 → 64
#             nn.LayerNorm(dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#         )

#         # Temporal self-attention (optional)
#         if use_temporal_attn:
#             # num_heads=2: dim=64 → 32D/head, đủ cho window 20 bước
#             self.temporal_attn = nn.MultiheadAttention(
#                 embed_dim=dim,
#                 num_heads=2,
#                 batch_first=True,
#                 dropout=dropout,
#             )
#             self.attn_norm    = nn.LayerNorm(dim)
#             self.attn_dropout = nn.Dropout(dropout)

#         self._init_weights()

#     def _init_weights(self):
#         for layer in self.projector:
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#                 if layer.bias is not None:
#                     nn.init.zeros_(layer.bias)

#     def forward(
#         self,
#         s_n:       torch.Tensor,                        # (B, T, input_dim)
#         news_mask: Optional[torch.Tensor] = None,       # (B, T) bool: True = no news
#     ) -> torch.Tensor:                                   # (B, T, dim)
#         """
#         Args:
#             s_n       : News embedding sequence. Zero vectors for no-news days.
#             news_mask : True = no news at that timestep (excluded from attn keys).
#         """
#         # ── Step 1: Temporal recency decay ───────────────────────────────────
#         # decay[-1]=1.0 (most recent), decay[0]≈exp(-2.0) (oldest in T=20 window)
#         # Fixed prior: recent news có impact lớn hơn → không cần học
#         T = s_n.shape[1]
#         decay = torch.exp(
#             -0.1 * torch.arange(T, 0, -1, dtype=torch.float32)
#         ).to(s_n.device)
#         s_n = s_n * decay.unsqueeze(0).unsqueeze(-1)   # (B, T, D)

#         # ── Step 2: Project to shared latent space ────────────────────────────
#         v_n = self.projector(s_n)   # (B, T, dim)

#         # ── Step 3: Optional temporal self-attention ──────────────────────────
#         if self.use_temporal_attn:
#             # Safe masking: nếu ALL timesteps masked → PyTorch MHA → NaN
#             attn_mask = news_mask
#             if attn_mask is not None:
#                 fully_masked = attn_mask.all(dim=1, keepdim=True)   # (B, 1)
#                 if fully_masked.any():
#                     attn_mask = attn_mask & ~fully_masked

#             attn_out, _ = self.temporal_attn(
#                 query=v_n,
#                 key=v_n,
#                 value=v_n,
#                 key_padding_mask=attn_mask,
#                 need_weights=False,
#             )
#             # Residual: nếu attn saturate sớm, decay-weighted features vẫn chảy qua
#             v_n = self.attn_norm(v_n + self.attn_dropout(attn_out))

#         return v_n   # (B, T, dim)

# SỬA LẦN 1
# encoders/news_encoder.py
"""
V2 — Quality-aware NewsEncoder, Phase 2: external quality from pipeline.

Phase 2 vs Phase 1 changes:
  - quality_dim parameter: controls input size of quality_mlp
      Phase 1 (norm proxy): quality_dim=1 (default, backward-compat)
      Phase 2 (pipeline quality): quality_dim=4 [log_n, avg_conf, avg_rel, avg_impact]
  - forward() accepts optional news_quality: (B, T, quality_dim) tensor
      When provided  → use external quality (Phase 2 behavior)
      When None      → fallback to embedding-norm proxy (Phase 1 behavior)
  - Both modes output (v_n, g_n) tuple — interface unchanged from Phase 1

Architecture:
  (B, T, input_dim)
    → compute q_signal (B, T, quality_dim)  ← external or norm proxy
    → temporal decay
    → projector (input_dim → mid → dim)
    → explicit mask zero-out (no-news days)
    → dual gate: q_gate (quality_mlp) × f_gate (feature_gate)
    → optional temporal self-attention
    → (v_n: B,T,dim), (g_n: B,T,1)

NOTE on parameter mismatch when switching Phase 1 → Phase 2:
  quality_mlp input size changes (1 → 4), so checkpoints from Phase 1
  cannot be loaded into Phase 2 model. Start fresh when upgrading.
"""

# encoders/news_encoder.py
"""
V2 — Quality-aware NewsEncoder, Phase 2: external quality from pipeline.

Phase 2 vs Phase 1 changes:
  - quality_dim parameter: controls input size of quality_mlp
      Phase 1 (norm proxy): quality_dim=1 (default, backward-compat)
      Phase 2 (pipeline quality): quality_dim=4 [log_n, avg_conf, avg_rel, avg_impact]
  - forward() accepts optional news_quality: (B, T, quality_dim) tensor
      When provided  → use external quality (Phase 2 behavior)
      When None      → fallback to embedding-norm proxy (Phase 1 behavior)
  - Both modes output (v_n, g_n) tuple — interface unchanged from Phase 1

Architecture:
  (B, T, input_dim)
    → compute q_signal (B, T, quality_dim)  ← external or norm proxy
    → temporal decay
    → projector (input_dim → mid → dim)
    → explicit mask zero-out (no-news days)
    → dual gate: q_gate (quality_mlp) × f_gate (feature_gate)
    → optional temporal self-attention
    → (v_n: B,T,dim), (g_n: B,T,1)

NOTE on parameter mismatch when switching Phase 1 → Phase 2:
  quality_mlp input size changes (1 → 4), so checkpoints from Phase 1
  cannot be loaded into Phase 2 model. Start fresh when upgrading.
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
        quality_dim:       int   = 1,    # 1 = Phase 1 norm proxy; 4 = Phase 2 pipeline
    ):
        super().__init__()
        self.use_temporal_attn = use_temporal_attn
        self.learnable_decay   = learnable_decay
        self.quality_dim       = quality_dim

        mid = max(dim * 4, 256)

        # Projector: input_dim → mid → dim (2-step to avoid info collapse)
        self.projector = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Quality gate from quality signal (norm proxy OR external pipeline stats)
        # quality_dim=1 → Phase 1 (1 scalar per timestep)
        # quality_dim=4 → Phase 2 (4 stats per timestep from KG triples)
        self.quality_mlp = nn.Sequential(
            nn.Linear(quality_dim, max(dim // 4, 16)),
            nn.GELU(),
            nn.Linear(max(dim // 4, 16), 1),
        )

        # Feature gate from projected embedding (does the projected vector carry signal?)
        self.feature_gate = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

        if use_temporal_attn:
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=2,
                batch_first=True,
                dropout=dropout,
            )
            self.attn_norm    = nn.LayerNorm(dim)
            self.attn_dropout = nn.Dropout(dropout)

        if learnable_decay:
            # softplus(-2.3) ≈ 0.1 — init near the fixed rate for smooth warmup
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
        decay    = torch.exp(-alpha * time_idx)
        return s_n * decay.view(1, T, 1)

    def forward(
        self,
        s_n:          torch.Tensor,                   # (B, T, input_dim)
        news_mask:    Optional[torch.Tensor] = None,  # (B, T) bool: True = no news
        news_quality: Optional[torch.Tensor] = None,  # (B, T, quality_dim) or None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s_n          : News embedding sequence (zero vectors for no-news days).
            news_mask    : True = no news at that timestep.
            news_quality : External quality stats from pipeline (Phase 2).
                           Shape (B, T, quality_dim). When None → norm proxy fallback.
        Returns:
            v_n : (B, T, dim)  — encoded + gated news features
            g_n : (B, T, 1)    — gate values in [0,1] for monitoring
        """
        B, T, _ = s_n.shape

        # ── Step 1: Quality signal ─────────────────────────────────────────────
        # Phase 2: use external pipeline quality (triple count, confidence, etc.)
        # Phase 1 fallback: derive quality from embedding norm (no extra data needed)
        if news_quality is not None:
            # Validate shape matches declared quality_dim
            if news_quality.shape[-1] != self.quality_dim:
                raise ValueError(
                    f"news_quality dim mismatch: got {news_quality.shape[-1]}, "
                    f"expected {self.quality_dim}. "
                    f"Check QUALITY_DIM in config.py and data_loader/main.py defaults "
                    f"(all must use the same value)."
                )
            # news_quality: (B, T, quality_dim) — already znormed in data_loader
            q_signal = news_quality
        else:
            # Norm proxy fallback — used when news_quality=None:
            #   • During modality dropout (news zeroed, mask=all-True)
            #   • Phase 1 mode (quality_dim=1, no external quality)
            input_norm = s_n.norm(dim=-1, keepdim=True)            # (B, T, 1)
            max_norm   = input_norm.detach().max().clamp(min=1e-8)
            norm_proxy = input_norm / max_norm                      # (B, T, 1) ∈ [0,1]
            if self.quality_dim == 1:
                # Phase 1: norm proxy is the only quality signal
                q_signal = norm_proxy
            else:
                # Phase 2 architecture (quality_dim>1) but news_quality=None.
                # Put norm in slot [0], zeros elsewhere.
                # During modality dropout, v_n is masked to 0 anyway, so
                # the exact gate value here is inconsequential.
                q_signal = torch.cat([
                    norm_proxy,
                    torch.zeros(B, T, self.quality_dim - 1,
                                device=s_n.device, dtype=s_n.dtype),
                ], dim=-1)   # (B, T, quality_dim)

        # ── Step 2: Temporal decay ─────────────────────────────────────────────
        s_n = self._apply_temporal_decay(s_n)

        # ── Step 3: Project to latent space ───────────────────────────────────
        v_n = self.projector(s_n)   # (B, T, dim)

        # ── Step 4: Explicit zero-out no-news days after projector ─────────────
        # Fix: LayerNorm + bias in projector can create pseudo-signal from zero input.
        # Masking here ensures no-news days = zero after projection regardless of bias.
        if news_mask is not None:
            v_n = v_n.masked_fill(news_mask.unsqueeze(-1), 0.0)

        # ── Step 5: Dual gate ──────────────────────────────────────────────────
        # q_gate: quality-aware (how much to trust this day's news signal?)
        # f_gate: feature-aware (does the projected feature carry useful info?)
        # g_n = q_gate × f_gate: both conditions must be satisfied
        q_gate = torch.sigmoid(self.quality_mlp(q_signal))   # (B, T, 1)
        f_gate = torch.sigmoid(self.feature_gate(v_n))        # (B, T, 1)
        g_n    = q_gate * f_gate                               # (B, T, 1)

        # Hard-zero gate for no-news days: ensures mask days contribute nothing
        if news_mask is not None:
            g_n = g_n.masked_fill(news_mask.unsqueeze(-1), 0.0)

        v_n = v_n * g_n   # (B, T, dim)

        # ── Step 6: Optional temporal self-attention ───────────────────────────
        if self.use_temporal_attn:
            attn_mask = news_mask
            if attn_mask is not None:
                # Prevent NaN: if ALL positions masked → softmax undefined
                fully_masked = attn_mask.all(dim=1, keepdim=True)   # (B, 1)
                if fully_masked.any():
                    attn_mask = attn_mask & ~fully_masked

            attn_out, _ = self.temporal_attn(
                query=v_n, key=v_n, value=v_n,
                key_padding_mask=attn_mask,
                need_weights=False,
            )
            v_n = self.attn_norm(v_n + self.attn_dropout(attn_out))

            if news_mask is not None:
                v_n = v_n.masked_fill(news_mask.unsqueeze(-1), 0.0)

        return v_n, g_n