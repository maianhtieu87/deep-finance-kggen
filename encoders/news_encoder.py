import torch
import torch.nn as nn


class NewsEncoder(nn.Module):
    def __init__(self, input_dim, dim, dropout=0.1):
        super().__init__()
        # Bỏ spectral_norm, thêm LayerNorm và Dropout để ổn định features 1024D
        self.projector = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )


    def forward(self, s_n):
        # Proposal E: News Temporal Recency Decay
        # Recent news (end of window) gets higher weight, old news decays
        # decay[-1] = 1.0 (most recent), decay[0] ≈ 0.14 (oldest in window)
        T = s_n.shape[1]
        decay = torch.exp(-0.1 * torch.arange(T, 0, -1, dtype=torch.float32)).to(s_n.device)
        s_n = s_n * decay.unsqueeze(0).unsqueeze(-1)  # broadcast: (1, T, 1)
        return self.projector(s_n)
