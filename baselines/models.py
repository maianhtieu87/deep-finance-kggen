# baselines/models.py
"""
Triển khai đầy đủ 8 models cho RQ1:

Indicator-only:
  1. LSTMBaseline          — LSTM thuần túy
  2. ALSTMBaseline         — LSTM + Temporal Attention

Indicator + Graph:
  3. ESTIMATEBaseline      — ALSTM + graph features (concatenation)
  4. DTMLBaseline          — Attentive LSTM + cross-market attention

Indicator + Document:
  5. ALSTMWithDoc          — ALSTM + averaged doc embeddings
  6. SLOTBaseline          — Unified price+doc ALSTM (simplified)

LLM-based:
  7. LLMStockBaseline      — LLM embedding + indicator LSTM

Ablation của MSGCA:
  8. MSGCANoAdaptiveFusion — MSGCA-NAF: cross-attention không có gating

Input shapes (từ BaselineDataPrepare):
  - indicators     : (B, W, 3)
  - s_news_per_day : (B, W, 128)
  - s_graph_emb    : (B, 128)
  - s_m            : (B, W, macro_dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from encoders.kg_graph_encoder import KGGraphEncoder
from encoders.mutil_encoder import MultimodalSourceEncoding
from src.predictor import FinegrainedMovementPrediction


# ══════════════════════════════════════════════════════════════════════
# Shared sub-modules
# ══════════════════════════════════════════════════════════════════════

class TemporalAttention(nn.Module):
    """Temporal self-attention cho LSTM output."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_out: (B, T, H)
        Returns:
            context: (B, H)
        """
        scores  = self.attn(lstm_out).squeeze(-1)          # (B, T)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # (B, T, 1)
        return (weights * lstm_out).sum(dim=1)              # (B, H)


# ══════════════════════════════════════════════════════════════════════
# 1. LSTM Baseline
# ══════════════════════════════════════════════════════════════════════

class LSTMBaseline(nn.Module):
    """
    Indicator-only LSTM.
    Input: indicators (B, W, 3)
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64,
                 num_layers: int = 2, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc   = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, **kwargs):
        _, (h_n, _) = self.lstm(indicators)
        return self.fc(self.drop(h_n[-1]))


# ══════════════════════════════════════════════════════════════════════
# 2. ALSTM Baseline
# ══════════════════════════════════════════════════════════════════════

class ALSTMBaseline(nn.Module):
    """
    Indicator-only ALSTM (LSTM + Temporal Attention).
    Feng et al., 2018 — Adversarial training variant.
    Input: indicators (B, W, 3)
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64,
                 num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.lstm      = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = TemporalAttention(hidden_dim)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)
        self.drop      = nn.Dropout(dropout)

    def forward(self, indicators, **kwargs):
        lstm_out, (h_n, _) = self.lstm(indicators)
        context  = self.attention(lstm_out)                    # (B, H)
        combined = torch.cat([h_n[-1], context], dim=-1)       # (B, 2H)
        return self.fc(self.drop(combined))


# ══════════════════════════════════════════════════════════════════════
# 3. ESTIMATE Baseline
# ══════════════════════════════════════════════════════════════════════

class ESTIMATEBaseline(nn.Module):
    """
    Indicator + Graph (ESTIMATE — Huynh et al., 2023).
    Concatenate price ALSTM features với graph embedding.
    Input: indicators (B, W, 3), s_graph_emb (B, 128)
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64,
                 graph_dim: int = 128, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.lstm      = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = TemporalAttention(hidden_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        # Concatenate ALSTM context + h_n + graph → predict
        self.fc   = nn.Linear(hidden_dim * 3, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, s_graph_emb, **kwargs):
        lstm_out, (h_n, _) = self.lstm(indicators)
        context    = self.attention(lstm_out)                       # (B, H)
        graph_feat = self.graph_proj(s_graph_emb)                   # (B, H)
        combined   = torch.cat([h_n[-1], context, graph_feat], dim=-1)  # (B, 3H)
        return self.fc(self.drop(combined))


# ══════════════════════════════════════════════════════════════════════
# 4. DTML Baseline
# ══════════════════════════════════════════════════════════════════════

class DTMLBaseline(nn.Module):
    """
    Indicator + Graph (DTML — Yoo et al., KDD 2021).
    Attentive LSTM + Cross-market context từ graph embedding.
    
    Adaptation: Dùng graph_emb làm "market context" qua attention
    (thay cho cross-stock attention vì data_loader theo từng ticker).
    Input: indicators (B, W, 3), s_graph_emb (B, 128)
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64,
                 graph_dim: int = 128, num_classes: int = 3,
                 num_heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.lstm      = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = TemporalAttention(hidden_dim)
        self.graph_proj = nn.Linear(graph_dim, hidden_dim)

        # Cross-attention: stock feature queries market context
        self.market_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, s_graph_emb, **kwargs):
        lstm_out, (h_n, _) = self.lstm(indicators)
        context = self.attention(lstm_out)              # (B, H)

        # Market context từ graph embedding
        market = self.graph_proj(s_graph_emb).unsqueeze(1)   # (B, 1, H)
        stock  = context.unsqueeze(1)                         # (B, 1, H)

        # Cross-attention: stock attends to market
        attn_out, _ = self.market_attn(stock, market, market)
        stock_feat  = self.norm(stock + attn_out).squeeze(1) # (B, H)

        combined = torch.cat([h_n[-1], stock_feat], dim=-1)  # (B, 2H)
        return self.fc(self.drop(combined))


# ══════════════════════════════════════════════════════════════════════
# 5. ALSTM-W (ALSTM + averaged document embeddings)
# ══════════════════════════════════════════════════════════════════════

class ALSTMWithDoc(nn.Module):
    """
    Indicator + Document (ALSTM-W — Kaeley et al., 2023).
    Average doc embeddings qua time, concat với ALSTM output.
    Input: indicators (B, W, 3), s_news_per_day (B, W, 128)
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64,
                 doc_dim: int = 128, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.doc_proj  = nn.Linear(doc_dim, hidden_dim)
        self.lstm      = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attention = TemporalAttention(hidden_dim)
        # h_n + context + avg_doc
        self.fc   = nn.Linear(hidden_dim * 3, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, s_news_per_day, **kwargs):
        # Average doc embedding qua time
        doc_avg  = self.doc_proj(s_news_per_day.mean(dim=1))  # (B, H)

        lstm_out, (h_n, _) = self.lstm(indicators)
        context  = self.attention(lstm_out)                     # (B, H)
        combined = torch.cat([h_n[-1], context, doc_avg], dim=-1)  # (B, 3H)
        return self.fc(self.drop(combined))


# ══════════════════════════════════════════════════════════════════════
# 6. SLOT Baseline (Simplified)
# ══════════════════════════════════════════════════════════════════════

class SLOTBaseline(nn.Module):
    """
    Indicator + Document — SLOT (Soun et al., BigData 2022).
    Simplified: unified price+doc ALSTM (không có self-supervised pretraining).
    
    Core idea: project price + doc vào cùng không gian, concat per-timestep,
    rồi ALSTM → temporal attention → predict.
    Input: indicators (B, W, 3), s_news_per_day (B, W, 128)
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64,
                 doc_dim: int = 128, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.price_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.GELU()
        )
        self.doc_proj = nn.Sequential(
            nn.Linear(doc_dim, hidden_dim), nn.GELU()
        )
        # Unified = cat(price, doc) → ALSTM
        self.lstm      = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attention = TemporalAttention(hidden_dim)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)
        self.drop      = nn.Dropout(dropout)

    def forward(self, indicators, s_news_per_day, **kwargs):
        price_emb = self.price_proj(indicators)       # (B, W, H)
        doc_emb   = self.doc_proj(s_news_per_day)      # (B, W, H)
        unified   = torch.cat([price_emb, doc_emb], dim=-1)  # (B, W, 2H)

        lstm_out, (h_n, _) = self.lstm(unified)
        context  = self.attention(lstm_out)
        combined = torch.cat([h_n[-1], context], dim=-1)
        return self.fc(self.drop(combined))


# ══════════════════════════════════════════════════════════════════════
# 7. LLM-Stock Baseline
# ══════════════════════════════════════════════════════════════════════

class LLMStockBaseline(nn.Module):
    """
    LLM embedding-based model (LLM-Stock — Xie et al., 2023 + Zou et al., 2022).
    Sử dụng graph/LLM embeddings từng ngày là primary signal,
    kết hợp với price qua LSTM.
    Input: indicators (B, W, 3), s_news_per_day (B, W, 128)
    """
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64,
                 doc_dim: int = 128, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        # LLM embedding projection
        self.llm_proj = nn.Sequential(
            nn.Linear(doc_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        # Price projection
        self.price_proj = nn.Linear(input_dim, hidden_dim)

        # Concat → LSTM
        self.lstm      = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attention = TemporalAttention(hidden_dim)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)
        self.drop      = nn.Dropout(dropout)

    def forward(self, indicators, s_news_per_day, **kwargs):
        price_feat = self.price_proj(indicators)       # (B, W, H)
        llm_feat   = self.llm_proj(s_news_per_day)     # (B, W, H)
        combined   = torch.cat([price_feat, llm_feat], dim=-1)  # (B, W, 2H)

        lstm_out, (h_n, _) = self.lstm(combined)
        context  = self.attention(lstm_out)
        out      = torch.cat([h_n[-1], context], dim=-1)
        return self.fc(self.drop(out))


# ══════════════════════════════════════════════════════════════════════
# 8. MSGCA-NAF (No Adaptive Fusion) — Ablation model
# ══════════════════════════════════════════════════════════════════════

class PlainCrossAttention(nn.Module):
    """
    Cross-attention không có gating — dùng cho MSGCA-NAF.
    So sánh với StableGatedCrossAttention trong src/fusion.py.
    """
    def __init__(self, dim: int, num_head: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_head,
            batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, primary: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        H_cross, _ = self.cross_attn(query=primary, key=aux, value=aux, need_weights=False)
        return self.norm(primary + self.drop(H_cross))


class MSGCANoAdaptiveFusion(nn.Module):
    """
    MSGCA-NAF: Giống MSGCA nhưng thay StableGatedCrossAttention
    bằng PlainCrossAttention (không có gating).

    Sử dụng cùng encoder và predictor với model chính.
    Input: s_o, s_h, s_c (B,W,1) + s_n_graphs (list PyG) + s_m (B,W,M)
    """
    def __init__(self, price_dim: int, macro_dim: int, news_dim: int,
                 dim: int, input_dim: int, output_dim: int, num_head: int,
                 device, dropout: float = 0.1,
                 gnn_hidden_dim: int = 256, gnn_num_layers: int = 2,
                 gnn_heads: int = 4):
        super().__init__()
        self.device   = device
        self.news_dim = news_dim

        # ── Encoders (giống StockMovementModel) ──────────────────────
        self.kg_encoder = KGGraphEncoder(
            node_dim=news_dim, hidden_dim=gnn_hidden_dim,
            output_dim=dim, num_sage_layers=gnn_num_layers,
            use_gat=False, dropout=dropout,
        ).to(device)

        self.multimodal_encoder = MultimodalSourceEncoding(
            price_dim=price_dim, macro_dim=macro_dim,
            news_dim=dim, dim=dim,
        )

        # ── Fusion: PlainCrossAttention (không gating) ───────────────
        self.fusion_stage1 = PlainCrossAttention(dim, num_head, dropout)
        self.fusion_stage2 = PlainCrossAttention(dim, num_head, dropout)

        # ── Predictor ────────────────────────────────────────────────
        self.movement_predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=input_dim,
            num_classes=output_dim, dropout=dropout,
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def _encode_graphs(self, graph_list):
        B = len(graph_list)
        valid = [g for g in graph_list if g is not None and g.x.size(0) > 0]
        if not valid:
            return torch.zeros(B, 1, self.kg_encoder.output_dim).to(self.device)

        try:
            batched = Batch.from_data_list(valid).to(self.device)
            embs    = self.kg_encoder(batched.x, batched.edge_index, batched.batch)
            embs    = F.normalize(embs, p=2, dim=-1)
        except Exception:
            embs = torch.zeros(B, self.kg_encoder.output_dim).to(self.device)

        return embs.unsqueeze(1)  # (B, 1, dim)

    def forward(self, s_o, s_h, s_c, s_m, s_n_graphs, label=None,
                mode="train", return_preds=False):
        from sklearn.metrics import accuracy_score, matthews_corrcoef

        v_m, v_i, _ = self.multimodal_encoder(s_o, s_h, s_c, s_m, None)
        v_n          = self._encode_graphs(s_n_graphs)

        H1     = self.fusion_stage1(primary=v_i, aux=v_n)
        H_final = self.fusion_stage2(primary=H1,  aux=v_m)
        logits  = self.movement_predictor(fused_seq=H_final, orig_seq=v_i)
        logits  = torch.clamp(logits, -15, 15)

        if mode == "train":
            return self.loss_fn(logits, label.long().to(self.device))

        preds = logits.argmax(dim=1)
        lbl   = label.long().to(self.device)
        acc   = accuracy_score(lbl.cpu().numpy(), preds.cpu().numpy())
        mcc   = 0.0
        try:
            from sklearn.metrics import matthews_corrcoef
            mcc = matthews_corrcoef(lbl.cpu().numpy(), preds.cpu().numpy())
        except Exception:
            pass

        if return_preds:
            return acc, mcc, preds
        return acc, mcc


# ══════════════════════════════════════════════════════════════════════
# Registry — dùng trong trainer
# ══════════════════════════════════════════════════════════════════════

def build_baseline(name: str, cfg: dict) -> nn.Module:
    """
    Factory function khởi tạo baseline theo tên.

    Args:
        name: Tên model (xem BASELINE_REGISTRY)
        cfg : Dict chứa hyperparameters

    Returns:
        nn.Module instance
    """
    hidden     = cfg.get("hidden_dim", 64)
    doc_dim    = cfg.get("doc_dim", 128)
    graph_dim  = cfg.get("graph_dim", 128)
    num_cls    = cfg.get("num_classes", 3)
    dropout    = cfg.get("dropout", 0.1)
    num_heads  = cfg.get("num_heads", 2)

    registry = {
        "LSTM":      lambda: LSTMBaseline(3, hidden, 2, num_cls, dropout),
        "ALSTM":     lambda: ALSTMBaseline(3, hidden, num_cls, dropout),
        "ESTIMATE":  lambda: ESTIMATEBaseline(3, hidden, graph_dim, num_cls, dropout),
        "DTML":      lambda: DTMLBaseline(3, hidden, graph_dim, num_cls, num_heads, dropout),
        "ALSTM-W":   lambda: ALSTMWithDoc(3, hidden, doc_dim, num_cls, dropout),
        "SLOT":      lambda: SLOTBaseline(3, hidden, doc_dim, num_cls, dropout),
        "LLM-Stock": lambda: LLMStockBaseline(3, hidden, doc_dim, num_cls, dropout),
    }

    if name not in registry:
        raise ValueError(f"Unknown baseline '{name}'. Có: {list(registry.keys())}")

    return registry[name]()