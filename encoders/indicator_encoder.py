# import torch
# from torch import nn
# from torch.nn.utils.parametrizations import spectral_norm as spectral_norm


# class IndicatorSequenceEncoder(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim

#         self.proj_c = spectral_norm(nn.Linear(1, dim))
#         self.proj_o = spectral_norm(nn.Linear(1, dim))
#         self.proj_h = spectral_norm(nn.Linear(1, dim))
#         self.combine = spectral_norm(nn.Linear(3 * dim, dim))

#     def forward(self, s_o, s_h, s_c):
#         """
#         s_o, s_h, s_c: (T, 1) hoặc (batch, T, 1)
#         """
#         v_o = self.proj_o(s_o)
#         v_h = self.proj_h(s_h)
#         v_c = self.proj_c(s_c)

#         v_i = self.combine(torch.cat([v_o, v_h, v_c], dim=-1))
#         return v_i
    
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as spectral_norm
from configs.config import TrainConfig

class IndicatorSequenceEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Lấy dropout từ config hoặc default là 0.1
        dropout_rate = getattr(TrainConfig, 'drop_out', 0.1)

        # 1. Projection cho Close Price
        self.proj_c = nn.Sequential(
            spectral_norm(nn.Linear(1, dim)),
            nn.GELU(),
            nn.Dropout(dropout_rate)       
        )

        # 2. Projection cho Open Price
        self.proj_o = nn.Sequential(
            spectral_norm(nn.Linear(1, dim)),
            nn.GELU(), 
            nn.Dropout(dropout_rate)       
        )

        # 3. Projection cho High Price
        self.proj_h = nn.Sequential(
            spectral_norm(nn.Linear(1, dim)),
            nn.GELU(), 
            nn.Dropout(dropout_rate)       
        )

        # 4. Combine Layer: Gộp thông tin từ O, H, C
        # Input: 3 * dim -> Output: dim
        # Thêm LayerNorm để ổn định feature trước khi đưa vào Transformer/LSTM
        self.combine = nn.Sequential(
            spectral_norm(nn.Linear(3 * dim, dim)),
            nn.LayerNorm(dim),
            nn.GELU()
        )

    def forward(self, s_o, s_h, s_c):
        """
        Input:
            s_o, s_h, s_c: Tensor shape (batch, T, 1) hoặc (T, 1)
        Output:
            v_i: Tensor shape (batch, T, dim)
        """
        # Feature Projection
        v_o = self.proj_o(s_o)
        v_h = self.proj_h(s_h)
        v_c = self.proj_c(s_c)

        # Concatenate & Combine
        # Nối 3 vector theo chiều feature (dim=-1)
        merged = torch.cat([v_o, v_h, v_c], dim=-1)
        
        v_i = self.combine(merged)
        
        return v_i
    
# import torch
# from torch import nn
# from torch.nn.utils.parametrizations import spectral_norm as spectral_norm
# from configs.config import TrainConfig

# class IndicatorSequenceEncoder(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.dim = dim
#         self.proj_c = nn.Sequential(
#             spectral_norm(nn.Linear(1, dim)),
#             nn.GELU(), 
#             nn.Dropout(TrainConfig.drop_out)       
#         )
#         self.proj_o = nn.Sequential(
#             spectral_norm(nn.Linear(1, dim)),
#             nn.GELU(), 
#             nn.Dropout(TrainConfig.drop_out)       
#         )
#         self.proj_h = nn.Sequential(
#             spectral_norm(nn.Linear(1, dim)),
#             nn.GELU(), 
#             nn.Dropout(TrainConfig.drop_out)       
#         )
#         self.combine = nn.Sequential(
#             spectral_norm(nn.Linear(3 * dim, dim)),
#             nn.LayerNorm(dim),
#             nn.GELU()
#         )

#     def forward(self, s_o, s_h, s_c):
#         """
#         s_o, s_h, s_c: (T, 1) hoặc (batch, T, 1)
#         """
#         v_o = self.proj_o(s_o)
#         v_h = self.proj_h(s_h)
#         v_c = self.proj_c(s_c)

#         v_i = self.combine(torch.cat([v_o, v_h, v_c], dim=-1))
#         return v_i