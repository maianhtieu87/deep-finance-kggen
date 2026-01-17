import torch.nn as nn
from .indicator_encoder import IndicatorSequenceEncoder
from .macro_encoder import MacroIndicatorEncoder
from .news_encoder import NewsEncoder # Import mới

class MultimodalSourceEncoding(nn.Module):
    def __init__(self, price_dim, macro_dim, news_dim, dim): # Thêm news_dim
        super().__init__()

        self.indicator_encoder = IndicatorSequenceEncoder(dim)
        
        self.macro_encoder = MacroIndicatorEncoder(
            in_dim=macro_dim,
            dim=dim
        )
        
        self.news_encoder = NewsEncoder(
            input_dim=news_dim, 
            dim=dim
        )

    def forward(self, s_o, s_h, s_c, s_m, s_n): # Thêm s_n
        """
        Input: Các tensor (B, T, Features)
        Output: 3 vectors đặc trưng (B, T, Dim)
        """
        v_i = self.indicator_encoder(s_o, s_h, s_c)
        v_m = self.macro_encoder(s_m)
        v_n = self.news_encoder(s_n) # Xử lý tin tức

        return v_m, v_i, v_n