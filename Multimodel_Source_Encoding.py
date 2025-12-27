import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm as spectral_norm
import torch.nn.functional as F
import pandas as pd
import numpy as np



class IndicatorSequenceEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.proj_c = spectral_norm(nn.Linear(1, dim))
        self.proj_o = spectral_norm(nn.Linear(1, dim))
        self.proj_h = spectral_norm(nn.Linear(1, dim))
        self.combine = spectral_norm(nn.Linear(3 * dim, dim))

    def forward(self, s_o, s_h, s_c):
        """
        s_o, s_h, s_c: (T, 1) hoặc (batch, T, 1)
        """
        v_o = self.proj_o(s_o)
        v_h = self.proj_h(s_h)
        v_c = self.proj_c(s_c)

        v_i = self.combine(torch.cat([v_o, v_h, v_c], dim=-1))
        return v_i



class MacroIndicatorEncoder(nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.projector = nn.Linear(in_dim, dim)

    def forward(self, s_m):
        """
        s_m: (T, num_macro) hoặc (batch, T, num_macro)
        """
        v_m = self.projector(s_m)
        return v_m



class MultimodalSourceEncoding(nn.Module):
    def __init__(self, price_dim, macro_dim, dim):
        super().__init__()

        self.macro_encoder = MacroIndicatorEncoder(
            in_dim=macro_dim,
            dim=dim
        )

        self.indicator_encoder = IndicatorSequenceEncoder(dim)

    def forward(self, s_o, s_h, s_c, s_m):
        """
        s_o, s_h, s_c: (T,1) hoặc (batch,T,1)
        s_m: (T,num_macro) hoặc (batch,T,num_macro)
        """
        v_i = self.indicator_encoder(s_o, s_h, s_c)
        v_m = self.macro_encoder(s_m)

        return v_m, v_i


