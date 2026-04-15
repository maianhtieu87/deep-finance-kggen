# import torch
# import torch.nn as nn


# class NewsEncoder(nn.Module):
#     def __init__(self, input_dim, dim, dropout=0.1):
#         super().__init__()
#         # Bỏ spectral_norm, thêm LayerNorm và Dropout để ổn định features 1024D
#         self.projector = nn.Sequential(
#             nn.Linear(input_dim, dim),
#             nn.LayerNorm(dim),
#             nn.GELU(),
#             nn.Dropout(dropout)
#         )


#     def forward(self, s_n):
#         # Proposal E: News Temporal Recency Decay
#         # Recent news (end of window) gets higher weight, old news decays
#         # decay[-1] = 1.0 (most recent), decay[0] ≈ 0.14 (oldest in window)
#         T = s_n.shape[1]
#         decay = torch.exp(-0.1 * torch.arange(T, 0, -1, dtype=torch.float32)).to(s_n.device)
#         s_n = s_n * decay.unsqueeze(0).unsqueeze(-1)  # broadcast: (1, T, 1)
#         return self.projector(s_n)

# encoders/news_encoder.py
import torch
import torch.nn as nn


class NewsEncoder(nn.Module):
    """
    2-step projection: 1024 → mid → dim.
    
    Lý do dùng 2 bước:
      - Ratio 1024→64 (16x) trong 1 Linear là quá lớn → thông tin bị
        nén đột ngột, gradient vanish ở lớp đầu.
      - Dùng intermediate layer mid=max(dim*4, 256): 1024→256→64
        (hoặc 1024→256→128 nếu dim=128) — nhất quán với NewsProjector
        đang dùng trong baselines/models.py.
    
    Temporal decay giữ nguyên: các ngày gần nhất (end of window)
    được weight cao hơn so với ngày cũ.
    """

    def __init__(self, input_dim: int, dim: int, dropout: float = 0.1):
        super().__init__()
        mid = max(dim * 4, 256)   # 1024 → 256 → 64 (hoặc → 128 nếu dim=128)

        self.projector = nn.Sequential(
            nn.Linear(input_dim, mid),   # 1024 → 256
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, dim),          # 256 → 64
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, s_n: torch.Tensor) -> torch.Tensor:
        # Temporal recency decay: vị trí cuối window = weight cao nhất
        T = s_n.shape[1]
        decay = torch.exp(
            -0.1 * torch.arange(T, 0, -1, dtype=torch.float32)
        ).to(s_n.device)
        s_n = s_n * decay.unsqueeze(0).unsqueeze(-1)   # (B, T, D)
        return self.projector(s_n)                       # (B, T, dim)