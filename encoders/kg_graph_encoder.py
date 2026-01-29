import torch
import torch.nn as nn

class KGGraphEncoder(nn.Module):
    def __init__(self, in_dim=384, hidden=128):
        super().__init__()
        self.lin = nn.Linear(in_dim, hidden)

    def forward(self, node_x, edge_index, ticker_idx):
        h = self.lin(node_x)
        return h[ticker_idx]
