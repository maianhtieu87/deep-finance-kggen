from torch import nn

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