import torch
import torch.nn as nn

class ThreeLayerMLP(nn.Module):
    """
    Implementation of the 3-layer MLP structure for dimension aggregation.
    Structure: Linear -> GELU -> Linear -> GELU -> Linear -> (Optional GELU)
    For Time Dimension Aggregation, final_activation=True,
    for Feature Dimension Aggregation, final_activation=False (Keep raw logits for classifier).
    """
    def __init__(self, d_in, d_out, d_h1, d_h2, final_activation=True, dropout=0.0): # [NEW] Thêm tham số dropout
        super().__init__()
        
        # Layer 1
        self.layer1 = nn.Sequential(
            nn.Linear(d_in, d_h1),
            nn.GELU(),
            nn.Dropout(dropout) # [NEW]
        )
        
        # Layer 2
        self.layer2 = nn.Sequential(
            nn.Linear(d_h1, d_h2),
            nn.GELU(),
            nn.Dropout(dropout) # [NEW]
        )
        
        # Layer 3
        if final_activation:
            # Used for Time Aggregation (Latent feature extraction)
            self.layer3 = nn.Sequential(
                nn.Linear(d_h2, d_out),
                nn.GELU(),
                nn.Dropout(dropout) # [NEW]
            )
        else:
            # Used for Feature Aggregation (Final Classifier -> Raw Logits)
            # Không dùng Dropout ở lớp cuối cùng trước khi ra Logits để tránh mất ổn định
            self.layer3 = nn.Linear(d_h2, d_out)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x