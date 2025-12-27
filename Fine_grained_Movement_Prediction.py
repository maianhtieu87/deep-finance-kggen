import torch
import torch.nn as nn


# ======================================================
# 1. Time Dimension Aggregation
# ======================================================
class TimeDimensionAggregation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.mlp_indicator = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, h_im, v_i):
        """
        h_im : (B, T, T, D) or (B, T, D)
        v_i  : (B, T, D)
        """

        # ===== h_im =====
        if h_im.dim() == 4:
            # (B, T, T, D) → lấy temporal summary
            h_im = h_im.mean(dim=2)  # → (B, T, D)

        # ===== MLP =====
        h_im_out = self.mlp_fusion(h_im)    # (B, T, D)
        h_i_out  = self.mlp_indicator(v_i) # (B, T, D)

        return h_im_out, h_i_out


# ======================================================
# 2. Feature Dimension Aggregation (Classifier)
# ======================================================
class FeatureDimensionAggregation(nn.Module):
    def __init__(self, dim, num_classes=3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # x : (B, dim*2)
        return self.classifier(x)   # (B, num_classes)


# ======================================================
# 3. Fine-grained Movement Prediction Head
# ======================================================
class FinegrainedMovementPrediction(nn.Module):
    def __init__(self, dim, num_classes=3):
        super().__init__()
        self.time_agg = TimeDimensionAggregation(dim)
        self.head = FeatureDimensionAggregation(dim, num_classes)

    def forward(self, h_im, v_i):
        """
        h_im : (B, T, T, D) or (B, T, D)
        v_i  : (B, T, D)
        """

        # ===== 1. Time aggregation =====
        h_im_out, h_i_out = self.time_agg(h_im, v_i)
        # (B, T, D)

        # ===== 2. Last timestep =====
        h_im_final = h_im_out[:, -1, :]   # (B, D)
        h_i_final  = h_i_out[:, -1, :]    # (B, D)

        # ===== 3. Feature fusion =====
        combined = torch.cat([h_im_final, h_i_final], dim=-1)
        # (B, 2D)

        # ===== 4. Classification =====
        return self.head(combined)  # (B, 3)
