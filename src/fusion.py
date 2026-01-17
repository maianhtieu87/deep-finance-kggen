import torch
import torch.nn as nn

class StableGatedCrossAttention(nn.Module):
    """
    Stable Gated Cross-Attention Mechanism
    Uses Indicator/Price features (Primary) to guide the selection of 
    Auxiliary features via a gating mechanism.
    """
    def __init__(self, dim, num_head, dropout=0.1):
        super().__init__()
        
        # 1. Unstable Fusion (Standard Multi-Head Attention)
        # Note: In PyTorch MHA, batch_first=True expects (Batch, Seq, Feature)
        # Output: H^l_{i,d}
        self.mha = nn.MultiheadAttention(embed_dim=dim, num_heads=num_head, batch_first=True, dropout=dropout)
        
        # 2. Gating Mechanism 
        # Gate is generated from the Primary modality
        # Input: H_i (Primary/Indicator) -> Output: H_b (Gate)
        self.gate_linear = nn.Linear(dim, dim)
        self.sigmoid = nn.Sigmoid()
        
        #3. Feature transformation weights 
        # Input: H^l_{i,d} (Unstable) -> Output: H_a (Transformed for gating)
        self.transform_linear = nn.Linear(dim, dim)
        
        # 4. Residual & Norm
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, primary, aux):
        """
        Args:
            primary (Tensor): The guiding modality (Price/Indicator). Shape: (B, T, D)
            aux (Tensor): The auxiliary - unstable fusion modality (e.g., News, Macro). Shape: (B, T, D)
        """
        # --- Step 1: Unstable Fusion ---
        # Query = Primary, Key = Aux, Value = Aux
        # Get H^l_{i,d} from Cross-Attention
        # unstable_fused shape: (B, T, D)
        unstable_fused, _ = self.mha(query=primary, key=aux, value=aux)
        
        # --- Step 2: Calculate H_a (Eq. 13) ---
        # Ha = H^l_{i,d} * Wa + b
        H_a = self.transform_linear(unstable_fused)
        
        # --- Step 3: Calculate H_b (Eq. 14) ---
        # Hb = Sigmoid(Hi * Wb + b')
        H_b = self.sigmoid(self.gate_linear(primary))
        
        # --- Step 4: Stable Feature Selection (Eq. 12) ---
        # Hi,d = Ha ⊙ Hb
        H_id = H_a * H_b 
        
        # --- Step 5: Residual Connection & Norm ---
        # Note: The paper implies Hi,d is the stable feature. 
        # Usually, Transformer blocks add Residual connection with the Query (Primary).
        output = self.norm(primary + self.dropout(H_id))
        
        return output