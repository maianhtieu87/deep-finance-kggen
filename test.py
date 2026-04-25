# Chạy nhanh để kiểm tra
from encoders.mutil_encoder import MultimodalSourceEncoding
import torch
enc = MultimodalSourceEncoding(1, 5, 768, 64, 0.1)
s_o = torch.randn(2, 20, 1)
s_m = torch.randn(2, 20, 5)
s_n = torch.randn(2, 20, 768)
out = enc(s_o, s_o, s_o, s_m, s_n)
print(type(out), len(out))  # (v_i, v_m, v_n) hay (v_m, v_i, v_n)?