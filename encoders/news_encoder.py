import torch
import torch.nn as nn

class NewsEncoder(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        # Project từ 1024 (Voyage) về 128 (Model Dim)
        self.projector = nn.Linear(input_dim, dim)
        self.act = nn.GELU()

    def forward(self, s_n):
        """
        s_n: (B, T, 1024)
        output: (B, T, dim)
        """
        return self.act(self.projector(s_n))