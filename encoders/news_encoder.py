# import torch
# import torch.nn as nn
# from torch.nn.utils import spectral_norm

# class NewsEncoder(nn.Module):
#     def __init__(self, input_dim, dim):
#         super().__init__()
#         # Project từ 1024 (Voyage) về 128 (Model Dim)
#         self.projector = nn.Linear(input_dim, dim)
#         self.act = nn.GELU()

#     def forward(self, s_n):
#         """
#         s_n: (B, T, 1024)
#         output: (B, T, dim)
#         """
#         return self.act(self.projector(s_n))
 


# class NewsEncoder(nn.Module):
#     def __init__(self, input_dim, dim, dropout=0.1):
#         super().__init__()
#         self.projector = spectral_norm(nn.Linear(input_dim, dim))
#         self.act = nn.GELU()
#         # SpectralNorm là điểm khác biệt duy nhất so với encoder gốc
#         # Kiểm soát Lipschitz constant, ổn định hơn khi dim lớn

#     def forward(self, s_n):
#         return self.act(self.projector(s_n)) 
 
 
 
 
    
# class NewsEncoder(nn.Module):
#     def __init__(self, input_dim, dim, dropout=0.1):
#         super().__init__()
#         mid_dim = input_dim  // 2  # 512
        
#         self.stage1 = nn.Sequential(
#             nn.LayerNorm(input_dim),          # normalize Voyage output
#             spectral_norm(nn.Linear(input_dim, mid_dim)),
#             nn.GELU(),
#             nn.Dropout(0.3),
#         )
#         self.stage2 = nn.Sequential(
#             nn.LayerNorm(mid_dim),
#             spectral_norm(nn.Linear(mid_dim, dim)),
#             nn.GELU(),
#             nn.Dropout(0.2),
#         )

#     def forward(self, s_n):
#         """
#         s_n: (B, T, input_dim)
#         output: (B, T, dim)
#         """
#         x = self.stage1(s_n)
#         x = self.stage2(x)
#         return x
#         # return self.stage2(self.stage1(s_n))

import torch
import torch.nn as nn

class NewsEncoder(nn.Module):
    def __init__(self, input_dim, dim, dropout=0.1):
        super().__init__()
        # Bỏ spectral_norm, thêm LayerNorm và Dropout để ổn định features 1024D
        self.projector = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, s_n):
        # Proposal E: News Temporal Recency Decay
        # Recent news (end of window) gets higher weight, old news decays
        # decay[-1] = 1.0 (most recent), decay[0] ≈ 0.14 (oldest in window)
        T = s_n.shape[1]
        decay = torch.exp(-0.1 * torch.arange(T, 0, -1, dtype=torch.float32)).to(s_n.device)
        s_n = s_n * decay.unsqueeze(0).unsqueeze(-1)  # broadcast: (1, T, 1)
        return self.projector(s_n)

# import torch
# import torch.nn as nn
# from torch.nn.utils.parametrizations import spectral_norm
 
 
# class NewsEncoder(nn.Module):
#     def __init__(self, input_dim: int, dim: int, dropout: float = 0.1):
#         super().__init__()
        
#         self.projector = spectral_norm(nn.Linear(input_dim, dim))
#         self.norm      = nn.LayerNorm(dim)
#         self.act       = nn.GELU()
#         self.dropout   = nn.Dropout(dropout)
 
#     def forward(self, s_n: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             s_n: (B, T, 1024)   — normalized Voyage-3-Large embeddings
#         Returns:
#             v_n: (B, T, dim)    — projected news representation
#         """
#         return self.dropout(self.act(self.norm(self.projector(s_n))))



