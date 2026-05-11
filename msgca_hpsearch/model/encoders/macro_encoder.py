from torch import nn


class MacroIndicatorEncoder(nn.Module):
    """Linear projection of macro indicators into latent space."""
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.projector = nn.Linear(in_dim, dim)

    def forward(self, s_m):
        """s_m: (B, T, macro_dim) → (B, T, dim)"""
        return self.projector(s_m)
