import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as spectral_norm


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