"""
MultimodalSourceEncoding — combines price, macro, news encoders.

Returns: (v_m, v_i, v_n, g_n)
  v_m : (B, T, dim) — macro encoding
  v_i : (B, T, dim) — price/indicator encoding
  v_n : (B, T, dim) — news encoding (quality-gated)  or None
  g_n : (B, T, 1)   — gate values for monitoring     or None
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .indicator_encoder import IndicatorSequenceEncoder
from .macro_encoder import MacroIndicatorEncoder
from .news_encoder import NewsEncoder


class MultimodalSourceEncoding(nn.Module):

    def __init__(
        self,
        price_dim:   int,
        macro_dim:   int,
        news_dim:    int,
        dim:         int,
        dropout:     float = 0.1,
        quality_dim: int   = 4,
    ):
        super().__init__()
        self.indicator_encoder = IndicatorSequenceEncoder(dim)
        self.macro_encoder     = MacroIndicatorEncoder(macro_dim, dim)
        self.news_encoder      = NewsEncoder(
            input_dim=news_dim,
            dim=dim,
            dropout=dropout,
            quality_dim=quality_dim,
        )

    def forward(
        self,
        s_o, s_h, s_c,
        s_m,
        s_n,
        news_mask:    Optional[torch.Tensor] = None,
        news_quality: Optional[torch.Tensor] = None,
    ) -> Tuple:
        v_i = self.indicator_encoder(s_o, s_h, s_c)
        v_m = self.macro_encoder(s_m)

        if s_n is not None:
            v_n, g_n = self.news_encoder(
                s_n,
                news_mask=news_mask,
                news_quality=news_quality,
            )
        else:
            v_n = None
            g_n = None

        return v_m, v_i, v_n, g_n
