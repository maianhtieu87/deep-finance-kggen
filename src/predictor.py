import torch
import torch.nn as nn
from .modules.layers import ThreeLayerMLP

class FinegrainedMovementPrediction(nn.Module):
    """
    Decoder module for Fine-grained Movement Prediction (Proposal Section 3.5).
    Replaces RNNs with MLPs for dimension aggregation.
    """
    def __init__(self, dim, window_size, num_classes=3, dropout=0.0): # [NEW] Nhận tham số dropout
        super().__init__()
        
        # --- 1. Time Dimension Aggregation (Reduce T -> 1) ---
        self.time_agg_fused = ThreeLayerMLP(
            d_in=window_size, 
            d_out=1, 
            d_h1=window_size // 2, 
            d_h2=window_size // 4, 
            final_activation=True,
            dropout=dropout # [NEW]
        )
        
        self.time_agg_orig = ThreeLayerMLP(
            d_in=window_size, 
            d_out=1, 
            d_h1=window_size // 2, 
            d_h2=window_size // 4, 
            final_activation=True,
            dropout=dropout # [NEW]
        )
        
        # --- 2. Feature Dimension Aggregation (Reduce 2D -> Classes) ---
        self.feat_agg = ThreeLayerMLP(
            d_in=2 * dim, 
            d_out=num_classes, 
            d_h1=dim, 
            d_h2=dim // 2, 
            final_activation=False, # No activation -> Raw Logits
            dropout=dropout # [NEW]
        )

    def forward(self, fused_seq, orig_seq):
        """
        Args:
            fused_seq: Multimodal features after fusion (B, T, D)
            orig_seq: Original Indicator/Price features (B, T, D)
        """
        # Transpose to (B, D, T) so Linear layers operate on T dimension
        fused_trans = fused_seq.transpose(1, 2)
        orig_trans = orig_seq.transpose(1, 2)
        
        # --- Step 1: Time Aggregation ---
        h_fused = self.time_agg_fused(fused_trans).squeeze(-1)
        h_orig = self.time_agg_orig(orig_trans).squeeze(-1)
        
        # --- Step 2: Concatenation ---
        h_final = torch.cat([h_fused, h_orig], dim=-1) # Shape: (B, 2D)
        
        # --- Step 3: Feature Aggregation (Prediction) ---
        logits = self.feat_agg(h_final) # Shape: (B, 3)
        
        return logits