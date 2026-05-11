import torch
import torch.nn as nn


class IndicatorSequenceEncoder(nn.Module):
    """
    Price indicator encoder: BiGRU over (open, high, close) sequences.

    Architecture (MSGCA Eq. 1-2):
      1. Per-component linear projection: O, H, C → dim each
      2. Combine via Linear(3*dim → dim)
      3. Bidirectional GRU: captures temporal patterns
      4. Residual + LayerNorm
    """

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.dim = dim

        self.proj_o = nn.Linear(1, dim)
        self.proj_h = nn.Linear(1, dim)
        self.proj_c = nn.Linear(1, dim)
        self.combine = nn.Linear(3 * dim, dim)

        self.gru = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.out_norm = nn.LayerNorm(dim)
        self.dropout  = nn.Dropout(dropout) if dropout > 0 else None

        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4 : n // 2].fill_(1.0)
        for layer in [self.proj_o, self.proj_h, self.proj_c, self.combine]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, s_o, s_h, s_c):
        """
        Args:
            s_o, s_h, s_c: (B, T, 1) — open, high, close price sequences
        Returns:
            (B, T, dim)
        """
        v_o = self.proj_o(s_o)
        v_h = self.proj_h(s_h)
        v_c = self.proj_c(s_c)
        fused = self.combine(torch.cat([v_o, v_h, v_c], dim=-1))

        gru_out, _ = self.gru(fused)
        out = self.out_norm(gru_out + fused)

        if self.dropout is not None:
            out = self.dropout(out)
        return out
