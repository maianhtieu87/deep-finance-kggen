# encoders/mutil_encoder.py
import torch
import torch.nn as nn

from .indicator_encoder import IndicatorSequenceEncoder
from .macro_encoder import MacroIndicatorEncoder
from .news_encoder import NewsEncoder

class MultimodalSourceEncoding(nn.Module):
    def __init__(self, price_dim, macro_dim, news_dim, dim, dropout=0.1): 
        super().__init__()
        self.indicator_encoder = IndicatorSequenceEncoder(dim)
        self.macro_encoder = MacroIndicatorEncoder(macro_dim, dim)  
        self.news_encoder = NewsEncoder(news_dim, dim, dropout=dropout)

# class MultimodalSourceEncoding(nn.Module):
#     def __init__(self, price_dim, macro_dim, news_dim, dim):
#         super().__init__()
#         self.indicator_encoder = IndicatorSequenceEncoder(dim)
#         self.macro_encoder = MacroIndicatorEncoder(macro_dim, dim)
#         self.news_encoder = NewsEncoder(news_dim, dim)

    def forward(self, s_o, s_h, s_c, s_m, s_n):
        """
        Args:
            s_o, s_h, s_c: price inputs
            s_m: macro inputs
            s_n: news embedding tensor hoặc None

        Returns:
            v_m, v_i, v_n
        """
        # 1. Encode Price (Indicator)
        v_i = self.indicator_encoder(s_o, s_h, s_c)

        # 2. Encode Macro
        v_m = self.macro_encoder(s_m)

        # 3. Encode News (Check if None)
        if s_n is not None:
            v_n = self.news_encoder(s_n)
        else:
            v_n = None 

        return v_m, v_i, v_n

# import torch
# import torch.nn as nn

# from .indicator_encoder import IndicatorSequenceEncoder
# from .macro_encoder import MacroIndicatorEncoder
# from .news_encoder import NewsEncoder

# class MultimodalSourceEncoding(nn.Module):
#     def __init__(self, price_dim, macro_dim, news_dim, dim): 
#         super().__init__()
#         self.indicator_encoder = IndicatorSequenceEncoder(dim)
#         self.macro_encoder = MacroIndicatorEncoder(macro_dim, dim)  
#         self.news_encoder = NewsEncoder(news_dim, dim)

#     def forward(self, s_o, s_h, s_c, s_m, s_n):
#         v_i = self.indicator_encoder(s_o, s_h, s_c)
#         v_m = self.macro_encoder(s_m)

#         if s_n is not None:
#             v_n = self.news_encoder(s_n)
#         else:
#             v_n = None 

#         return v_m, v_i, v_n

