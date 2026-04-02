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

        # v_i = self.combine(torch.cat([v_o, v_h, v_c], dim=-1))
        # return v_i
    
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm as spectral_norm
from configs.config import TrainConfig

class IndicatorSequenceEncoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
        # Lấy dropout từ config hoặc default là 0.1
        # dropout_rate = getattr(TrainConfig, 'drop_out', 0.1)

        # 1. Projection cho Close Price
        self.proj_c = nn.Sequential(
            spectral_norm(nn.Linear(1, dim)),
            # nn.GELU(),
            # nn.Dropout(dropout_rate)       
        )

        # 2. Projection cho Open Price
        self.proj_o = nn.Sequential(
            spectral_norm(nn.Linear(1, dim)),
            # nn.GELU(), 
            # nn.Dropout(dropout_rate)       
        )

        # 3. Projection cho High Price
        self.proj_h = nn.Sequential(
            spectral_norm(nn.Linear(1, dim)),
            # nn.GELU(), 
            # nn.Dropout(dropout_rate)       
        )

        # 4. Combine Layer: Gộp thông tin từ O, H, C
        # Input: 3 * dim -> Output: dim
        # Thêm LayerNorm để ổn định feature trước khi đưa vào Transformer/LSTM
        self.combine = nn.Sequential(
            spectral_norm(nn.Linear(3 * dim, dim)),
            # nn.LayerNorm(dim),
            # nn.GELU()
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

# class IndicatorSequenceEncoder(nn.Module):
#     def __init__(self, dim,):
#         super().__init__()
#         self.dim = dim
        
#         self.proj_c = nn.Linear(1, dim)
#         self.proj_o = nn.Linear(1, dim)
#         self.proj_h = nn.Linear(1, dim)

#         # Gộp thông tin O, H, C
#         self.combine = nn.Linear(3 * dim, dim)

#     def forward(self, s_o, s_h, s_c):
#         v_o = self.proj_o(s_o)
#         v_h = self.proj_h(s_h)
#         v_c = self.proj_c(s_c)

#         merged = torch.cat([v_o, v_h, v_c], dim=-1)
#         v_i = self.combine(merged)
        
#         return v_i
    

# import torch
# from torch import nn
# from torch.nn.utils.parametrizations import spectral_norm

# from configs.config import TrainConfig


# class IndicatorSequenceEncoder(nn.Module):
#     def __init__(self, dim: int, dropout: float = None):
#         super().__init__()
#         self.dim = dim
#         dropout = dropout if dropout is not None else getattr(TrainConfig, "drop_out", 0.1)

#         # ── Per-series feature projection ─────────────────────────────────
#         # SpectralNorm constrains Lipschitz constant → stable gradients
#         # GELU: smooth non-linearity, better than ReLU for financial time series
#         self.proj_o = nn.Sequential(
#             spectral_norm(nn.Linear(1, dim)),
#             nn.GELU(),
#         )
#         self.proj_h = nn.Sequential(
#             spectral_norm(nn.Linear(1, dim)),
#             nn.GELU(),
#         )
#         self.proj_c = nn.Sequential(
#             spectral_norm(nn.Linear(1, dim)),
#             nn.GELU(),
#         )

#         # ── Cross-series fusion ───────────────────────────────────────────
#         # Combines O, H, C into a single representation per timestep
#         self.combine = nn.Sequential(
#             spectral_norm(nn.Linear(3 * dim, dim)),
#             nn.LayerNorm(dim),
#             nn.GELU(),
#         )

#         # ── Temporal encoder (BiLSTM) ─────────────────────────────────────
#         # CRITICAL: without this, each timestep t is encoded in isolation.
#         # The linear-only encoder is PERMUTATION-INVARIANT — day 1 and day 20
#         # produce identical representations given the same O/H/C values.
#         # BiLSTM makes encoding of day t depend on days 0..t (forward)
#         # and days t..T-1 (backward) → captures momentum, trend, reversal.
#         #
#         # dim//2 per direction, bidirectional → total hidden = dim
#         # num_layers=1: sufficient for 20-step window; deeper LSTM risks
#         #               overfitting on this dataset size (~850 trading days)
#         self.temporal = nn.LSTM(
#             input_size=dim,
#             hidden_size=dim // 2,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.0,  # single layer → LSTM dropout has no effect
#         )

#         # ── Output projection ─────────────────────────────────────────────
#         # LayerNorm stabilizes LSTM outputs before feeding into MSGCA
#         # Dropout applied here (single regularization point)
#         self.out_proj = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Dropout(dropout),
#         )

#         self._init_lstm()

#     def _init_lstm(self):
#         """Orthogonal init for LSTM weights → better gradient flow at start."""
#         for name, param in self.temporal.named_parameters():
#             if "weight_ih" in name:
#                 nn.init.xavier_uniform_(param)
#             elif "weight_hh" in name:
#                 nn.init.orthogonal_(param)
#             elif "bias" in name:
#                 nn.init.zeros_(param)
#                 # Forget gate bias = 1 → remember by default at start
#                 n = param.size(0)
#                 param.data[n // 4: n // 2].fill_(1.0)

#     def forward(self, s_o: torch.Tensor, s_h: torch.Tensor, s_c: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             s_o, s_h, s_c: (B, T, 1)  — normalized OHLC sequences
#         Returns:
#             v_i: (B, T, dim)           — temporally-enriched price representation
#         """
#         # Per-series non-linear projection
#         v_o = self.proj_o(s_o)   # (B, T, dim)
#         v_h = self.proj_h(s_h)
#         v_c = self.proj_c(s_c)

#         # Fuse O, H, C features at each timestep
#         fused = self.combine(torch.cat([v_o, v_h, v_c], dim=-1))  # (B, T, dim)

#         # Temporal encoding over 20-day window
#         # lstm_out[b, t, :] encodes context from all 20 days
#         lstm_out, _ = self.temporal(fused)   # (B, T, dim)

#         # Residual + output norm
#         # Residual: preserve pre-LSTM features in case LSTM learns something harmful
#         v_i = self.out_proj(lstm_out + fused)  # (B, T, dim)

#         return v_i

# import torch
# from torch import nn
# from configs.config import TrainConfig

# class IndicatorSequenceEncoder(nn.Module):
#     def __init__(self, dim: int, dropout: float = None):
#         super().__init__()
#         self.dim = dim
#         dropout = dropout if dropout is not None else getattr(TrainConfig, "drop_out", 0.1)

#         # ── Per-series feature projection ─────────────────────────────────
#         # BỎ spectral_norm để tránh lỗi NaN/Inf do chia cho 0 với ma trận 1xDim
#         self.proj_o = nn.Sequential(
#             nn.Linear(1, dim),
#             nn.GELU(),
#         )
#         self.proj_h = nn.Sequential(
#             nn.Linear(1, dim),
#             nn.GELU(),
#         )
#         self.proj_c = nn.Sequential(
#             nn.Linear(1, dim),
#             nn.GELU(),
#         )

#         # ── Cross-series fusion ───────────────────────────────────────────
#         self.combine = nn.Sequential(
#             nn.Linear(3 * dim, dim),
#             nn.LayerNorm(dim),
#             nn.GELU(),
#         )

#         # ── Temporal encoder (BiLSTM) ─────────────────────────────────────
#         self.temporal = nn.LSTM(
#             input_size=dim,
#             hidden_size=dim // 2,
#             num_layers=1,
#             batch_first=True,
#             bidirectional=True,
#             dropout=0.0,  
#         )

#         # ── Output projection ─────────────────────────────────────────────
#         self.out_proj = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Dropout(dropout),
#         )

#         self._init_lstm()

#     def _init_lstm(self):
#         """Orthogonal init for LSTM weights → better gradient flow at start."""
#         for name, param in self.temporal.named_parameters():
#             if "weight_ih" in name:
#                 nn.init.xavier_uniform_(param)
#             elif "weight_hh" in name:
#                 nn.init.orthogonal_(param)
#             elif "bias" in name:
#                 nn.init.zeros_(param)
#                 # Forget gate bias = 1 → remember by default at start
#                 n = param.size(0)
#                 param.data[n // 4: n // 2].fill_(1.0)

#     def forward(self, s_o: torch.Tensor, s_h: torch.Tensor, s_c: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             s_o, s_h, s_c: (B, T, 1)  — normalized OHLC sequences
#         Returns:
#             v_i: (B, T, dim)           — temporally-enriched price representation
#         """
#         v_o = self.proj_o(s_o)   # (B, T, dim)
#         v_h = self.proj_h(s_h)
#         v_c = self.proj_c(s_c)

#         # Fuse O, H, C features at each timestep
#         fused = self.combine(torch.cat([v_o, v_h, v_c], dim=-1))  # (B, T, dim)

#         # Temporal encoding over 20-day window
#         lstm_out, _ = self.temporal(fused)   # (B, T, dim)

#         # Residual + output norm
#         v_i = self.out_proj(lstm_out + fused)  # (B, T, dim)

#         return v_i

