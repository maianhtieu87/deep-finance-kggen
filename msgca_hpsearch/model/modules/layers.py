import torch
import torch.nn as nn


class ThreeLayerMLP(nn.Module):
    """
    3-layer MLP for dimension aggregation.
    Used in FinegrainedMovementPrediction for time and feature aggregation.
    """
    def __init__(self, d_in, d_out, d_h1, d_h2, final_activation=True, dropout=0.0):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(d_in, d_h1), nn.GELU(), nn.Dropout(dropout))
        self.layer2 = nn.Sequential(nn.Linear(d_h1, d_h2), nn.GELU(), nn.Dropout(dropout))
        if final_activation:
            self.layer3 = nn.Sequential(nn.Linear(d_h2, d_out), nn.GELU(), nn.Dropout(dropout))
        else:
            self.layer3 = nn.Linear(d_h2, d_out)

    def forward(self, x):
        return self.layer3(self.layer2(self.layer1(x)))
