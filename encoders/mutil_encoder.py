# encoders/mutil_encoder.py
"""
V3 — Pass news_mask into NewsEncoder for temporal self-attention masking.

V3 change vs V2:
  forward() now accepts news_mask and passes it to NewsEncoder.
  This allows temporal_attn in NewsEncoder to exclude no-news timesteps.
"""

import torch
import torch.nn as nn
from typing import Optional

from .indicator_encoder import IndicatorSequenceEncoder
from .macro_encoder import MacroIndicatorEncoder
from .news_encoder import NewsEncoder


class MultimodalSourceEncoding(nn.Module):
    def __init__(self, price_dim, macro_dim, news_dim, dim, dropout=0.1):
        super().__init__()
        self.indicator_encoder = IndicatorSequenceEncoder(dim)
        self.macro_encoder     = MacroIndicatorEncoder(macro_dim, dim)
        self.news_encoder      = NewsEncoder(news_dim, dim, dropout=dropout)

    def forward(
        self,
        s_o, s_h, s_c,
        s_m,
        s_n,
        news_mask: Optional[torch.Tensor] = None,  # (B, T) bool
    ):
        """
        Args:
            s_o, s_h, s_c : price sequences (B, T, 1)
            s_m            : macro sequence  (B, T, macro_dim)
            s_n            : news embeddings (B, T, news_dim) — 768D FinBERT
            news_mask      : (B, T) BoolTensor, True = no news at that timestep

        Returns:
            v_m : macro encoding   (B, T, dim)
            v_i : price encoding   (B, T, dim)
            v_n : news encoding    (B, T, dim) or None
        """
        # 1. Encode price
        v_i = self.indicator_encoder(s_o, s_h, s_c)

        # 2. Encode macro
        v_m = self.macro_encoder(s_m)

        # 3. Encode news — pass news_mask for temporal attention masking
        if s_n is not None:
            v_n = self.news_encoder(s_n, news_mask=news_mask)
        else:
            v_n = None

        return v_m, v_i, v_n