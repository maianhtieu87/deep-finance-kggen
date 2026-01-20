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
        
        
        # # --- Step 2: Calculate H_a (Eq. 13) ---
        # # Ha = H^l_{i,d} * Wa + b
        # H_a = self.transform_linear(unstable_fused)
        
        # # --- Step 3: Calculate H_b (Eq. 14) ---
        # # Hb = Sigmoid(Hi * Wb + b')
        # H_b = self.sigmoid(self.gate_linear(primary))
        
        # # --- Step 4: Stable Feature Selection (Eq. 12) ---
        # # Output = Gate * New_Info + (1 - Gate) * Old_Info
        # # Ý nghĩa: Gate quyết định bao nhiêu % thông tin mới được nạp vào, 
        # # và bao nhiêu % thông tin cũ được giữ lại.
        # H_id = H_a * H_b 
        
        # # --- Step 5: Residual Connection & Norm ---
        # # Note: The paper implies Hi,d is the stable feature. 
        # # Usually, Transformer blocks add Residual connection with the Query (Primary).
        # output = self.norm(primary + self.dropout(H_id))
        
        # 2. GATING MECHANISM (SỬA LẠI ĐÚNG CHUẨN HIGHWAY)
        # ---------------------------------------------------------
        
        # A. Tính Gate (H_b): Quyết định dựa trên Primary (Giá)
        # Gate chạy từ 0 đến 1. 
        # Gate -> 1: Tin vào News/Macro (Aux)
        # Gate -> 0: Tin vào Price (Primary)
        gate = self.sigmoid(self.gate_linear(primary)) 

        # B. Transform New Info (H_a)
        H_a = self.transform_linear(unstable_fused)
        
        # C. Áp dụng Gating (Highway Connection)
        # Đây là dòng quan trọng nhất sửa lỗi "Leaky Residual"
        
        # Nhánh 1: Thông tin Mới (News) được Gate cho phép đi qua
        gated_new = H_a * gate
        
        # Nhánh 2: Thông tin Cũ (Price) bị Gate chặn lại (Complementary Gate)
        # Nếu Gate=1 (News tốt), thì (1-Gate)=0 -> Price nhiễu bị loại bỏ hoàn toàn.
        gated_primary = primary * (1.0 - gate)
        
        # 3. KẾT HỢP & NORM
        # ---------------------------------------------------------
        # Áp dụng Dropout lên phần New Info (nơi chứa nhiều tham số học được)
        output = self.norm(gated_primary + self.dropout(gated_new))
        
        return output