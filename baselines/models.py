# baselines/models.py
"""
Baseline models cho RQ1-RQ3 — Deep Finance V5.3

Mapping modalities (V5.3 adaptation của paper MSGCA):
  Paper modality      → V5.3 equivalent
  ─────────────────────────────────────────────
  indicator sequence  → indicators (B,W,3) = price OHLC
  dynamic document    → s_n        (B,W,1024) = Voyage structured-triple embeddings
  relational graph    → s_m        (B,W,M)    = macro indicators (market context)

Models implemented:
  1. LSTMBaseline        — price only (LSTM)
  2. ALSTMBaseline       — price + temporal attention
  3. ESTIMATEBaseline    — price + macro concatenation
  4. DTMLBaseline        — price + macro cross-attention
  5. ALSTMWithDoc        — price + averaged Voyage embeddings
  6. SLOTBaseline        — unified price+news LSTM
  7. LLMStockBaseline    — Voyage primary + price LSTM
  8. MSGCANoGate         — MSGCA ablation: cross-attention without gating (RQ2)

RQ3 ablation variants:
  9. MSGCAPriceOnly      — price only (no news, no macro)
 10. MSGCAPriceNews      — price + news (no macro)
 11. MSGCAPriceMacro     — price + macro (no news)
 Full MSGCA: from src.model import StockMovementModel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
# Shared sub-modules
# ──────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Temporal self-attention trên LSTM output."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """(B,T,H) → (B,H)"""
        w = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=-1)  # (B,T)
        return (w.unsqueeze(-1) * lstm_out).sum(dim=1)


class NewsProjector(nn.Module):
    """Project 1024D Voyage embedding xuống hidden_dim."""
    def __init__(self, news_dim: int = 1024, hidden_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(news_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.proj(x)


# ──────────────────────────────────────────────────────────────────────
# 1. LSTM Baseline — price only
# ──────────────────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """Indicator-only LSTM. Input: indicators (B,W,3)."""

    def __init__(self, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden_dim, num_layers=2,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, **kw):
        _, (h_n, _) = self.lstm(indicators)
        return self.fc(self.drop(h_n[-1]))


# ──────────────────────────────────────────────────────────────────────
# 2. ALSTM Baseline — price + temporal attention
# ──────────────────────────────────────────────────────────────────────

class ALSTMBaseline(nn.Module):
    """ALSTM: LSTM + temporal attention. Input: indicators (B,W,3)."""

    def __init__(self, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn = TemporalAttention(hidden_dim)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, **kw):
        out, (h_n, _) = self.lstm(indicators)
        ctx  = self.attn(out)
        feat = torch.cat([h_n[-1], ctx], dim=-1)
        return self.fc(self.drop(feat))


# ──────────────────────────────────────────────────────────────────────
# 3. ESTIMATE Baseline — price + macro concatenation
#    (Paper: price + graph hypergraph; V5.3: price + macro)
# ──────────────────────────────────────────────────────────────────────

class ESTIMATEBaseline(nn.Module):
    """
    ESTIMATE (Huynh et al., WSDM 2023).
    Price ALSTM + macro market context via concatenation.
    Input: indicators (B,W,3), s_m (B,W,M)
    """

    def __init__(self, macro_dim=6, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.lstm      = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn      = TemporalAttention(hidden_dim)
        self.macro_avg = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.ReLU(),
        )
        self.fc   = nn.Linear(hidden_dim * 3, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, s_m, **kw):
        out, (h_n, _) = self.lstm(indicators)
        ctx   = self.attn(out)
        macro = self.macro_avg(s_m.mean(dim=1))
        feat  = torch.cat([h_n[-1], ctx, macro], dim=-1)
        return self.fc(self.drop(feat))


# ──────────────────────────────────────────────────────────────────────
# 4. DTML Baseline — price + macro cross-attention
#    (Paper: attentive LSTM + cross-market attention; V5.3: price + macro)
# ──────────────────────────────────────────────────────────────────────

class DTMLBaseline(nn.Module):
    """
    DTML (Yoo et al., KDD 2021) — adapted.
    Price ALSTM queries macro market context via cross-attention.
    Input: indicators (B,W,3), s_m (B,W,M)
    """

    def __init__(self, macro_dim=6, hidden_dim=64, num_heads=2,
                 num_classes=3, dropout=0.1):
        super().__init__()
        self.lstm       = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn       = TemporalAttention(hidden_dim)
        self.macro_proj = nn.Linear(macro_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm       = nn.LayerNorm(hidden_dim)
        self.fc         = nn.Linear(hidden_dim * 2, num_classes)
        self.drop       = nn.Dropout(dropout)

    def forward(self, indicators, s_m, **kw):
        out, (h_n, _) = self.lstm(indicators)
        ctx    = self.attn(out).unsqueeze(1)                    # (B,1,H)
        market = self.macro_proj(s_m)                            # (B,W,H)
        att, _ = self.cross_attn(ctx, market, market)
        stock  = self.norm(ctx + att).squeeze(1)                 # (B,H)
        feat   = torch.cat([h_n[-1], stock], dim=-1)
        return self.fc(self.drop(feat))


# ──────────────────────────────────────────────────────────────────────
# 5. ALSTM-W — price + averaged document embeddings
#    (Paper: ALSTM + Word2Vec/BERT avg; V5.3: ALSTM + avg Voyage 1024D)
# ──────────────────────────────────────────────────────────────────────

class ALSTMWithDoc(nn.Module):
    """
    ALSTM-W (Kaeley et al., 2023).
    Price ALSTM + average Voyage document embeddings.
    Input: indicators (B,W,3), s_n (B,W,1024)
    """

    def __init__(self, news_dim=1024, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.doc_proj  = NewsProjector(news_dim, hidden_dim)
        self.lstm      = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn      = TemporalAttention(hidden_dim)
        self.fc        = nn.Linear(hidden_dim * 3, num_classes)
        self.drop      = nn.Dropout(dropout)

    def forward(self, indicators, s_n, **kw):
        doc = self.doc_proj(s_n.mean(dim=1))          # (B,H) — avg over window
        out, (h_n, _) = self.lstm(indicators)
        ctx  = self.attn(out)
        feat = torch.cat([h_n[-1], ctx, doc], dim=-1)
        return self.fc(self.drop(feat))


# ──────────────────────────────────────────────────────────────────────
# 6. SLOT Baseline — unified price+news ALSTM
#    (Paper: unified price+tweet ALSTM; V5.3: price+Voyage per timestep)
# ──────────────────────────────────────────────────────────────────────

class SLOTBaseline(nn.Module):
    """
    SLOT (Soun et al., BigData 2022) — simplified.
    Project price+news per-timestep, then ALSTM.
    Input: indicators (B,W,3), s_n (B,W,1024)
    """

    def __init__(self, news_dim=1024, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.price_proj = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU())
        self.doc_proj   = NewsProjector(news_dim, hidden_dim)
        self.lstm       = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attn       = TemporalAttention(hidden_dim)
        self.fc         = nn.Linear(hidden_dim * 2, num_classes)
        self.drop       = nn.Dropout(dropout)

    def forward(self, indicators, s_n, **kw):
        price_emb = self.price_proj(indicators)  # (B,W,H)
        # project each timestep's 1024D embedding
        B, W, D = s_n.shape
        doc_emb = self.doc_proj(s_n.view(B * W, D)).view(B, W, -1)  # (B,W,H)
        unified = torch.cat([price_emb, doc_emb], dim=-1)            # (B,W,2H)
        out, (h_n, _) = self.lstm(unified)
        ctx  = self.attn(out)
        feat = torch.cat([h_n[-1], ctx], dim=-1)
        return self.fc(self.drop(feat))


# ──────────────────────────────────────────────────────────────────────
# 7. LLM-Stock Baseline — Voyage-primary + price LSTM
#    (Paper: LLM embedding primary; V5.3: Voyage structured-triple primary)
# ──────────────────────────────────────────────────────────────────────

class LLMStockBaseline(nn.Module):
    """
    LLM-Stock (Xie et al., 2023; Zou et al., 2022) — adapted.
    Voyage structured-triple embeddings as primary signal + price LSTM.
    Input: indicators (B,W,3), s_n (B,W,1024)
    """

    def __init__(self, news_dim=1024, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.voyage_proj = NewsProjector(news_dim, hidden_dim)
        self.price_proj  = nn.Linear(3, hidden_dim)
        self.lstm        = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attn        = TemporalAttention(hidden_dim)
        self.fc          = nn.Linear(hidden_dim * 2, num_classes)
        self.drop        = nn.Dropout(dropout)

    def forward(self, indicators, s_n, **kw):
        B, W, D = s_n.shape
        llm  = self.voyage_proj(s_n.view(B * W, D)).view(B, W, -1)  # (B,W,H)
        price = self.price_proj(indicators)                           # (B,W,H)
        fused = torch.cat([price, llm], dim=-1)
        out, (h_n, _) = self.lstm(fused)
        ctx  = self.attn(out)
        feat = torch.cat([h_n[-1], ctx], dim=-1)
        return self.fc(self.drop(feat))


# ──────────────────────────────────────────────────────────────────────
# 8. MSGCA-NAF (No Adaptive/Gate Fusion) — Ablation RQ2
#    Dùng plain cross-attention thay StableGatedCrossAttention
# ──────────────────────────────────────────────────────────────────────

class _PlainCrossAttn(nn.Module):
    """Cross-attention không có gating."""
    def __init__(self, dim, num_head, dropout=0.1):
        super().__init__()
        self.ca   = nn.MultiheadAttention(dim, num_head, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, primary, aux):
        H, _ = self.ca(primary, aux, aux, need_weights=False)
        return self.norm(primary + self.drop(H))


class MSGCANoGate(nn.Module):
    """
    MSGCA-CA / MSGCA-NAF: kiến trúc giống MSGCA nhưng thay gating bằng
    plain cross-attention. Dùng trong RQ2 ablation.

    Input: s_o, s_h, s_c (B,W,1), s_n (B,W,1024), s_m (B,W,M)
    """

    def __init__(self, macro_dim=6, news_dim=1024, dim=256, window_size=20,
                 num_head=4, num_classes=3, dropout=0.1):
        super().__init__()
        from encoders.mutil_encoder import MultimodalSourceEncoding
        from src.predictor import FinegrainedMovementPrediction

        self.encoder = MultimodalSourceEncoding(
            price_dim=1, macro_dim=macro_dim, news_dim=news_dim, dim=dim
        )
        self.stage1  = _PlainCrossAttn(dim, num_head, dropout)
        self.stage2  = _PlainCrossAttn(dim, num_head, dropout)
        self.predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=window_size, num_classes=num_classes, dropout=dropout
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, s_o, s_h, s_c, s_m, s_n, label=None,
                mode="train", return_preds=False):
        v_m, v_i, v_n = self.encoder(s_o, s_h, s_c, s_m, s_n)
        if v_n is None:
            v_n = torch.zeros_like(v_i)
        H1     = self.stage1(primary=v_i, aux=v_n)
        H_out  = self.stage2(primary=H1,  aux=v_m)
        logits = self.predictor(fused_seq=H_out, orig_seq=v_i)
        logits = torch.clamp(logits, -15, 15)

        if mode == "train":
            return self.loss_fn(logits, label.long().to(logits.device))

        preds = logits.argmax(dim=1)
        if return_preds:
            return preds
        from sklearn.metrics import accuracy_score, matthews_corrcoef
        lbl = label.cpu().numpy() if isinstance(label, torch.Tensor) else label
        return (accuracy_score(lbl, preds.cpu().numpy()),
                matthews_corrcoef(lbl, preds.cpu().numpy()))


# ──────────────────────────────────────────────────────────────────────
# RQ2: GLU ablation
# ──────────────────────────────────────────────────────────────────────

class _GLUFusion(nn.Module):
    """Gated Linear Unit fusion (simple, no cross-attention)."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)
        self.val_proj  = nn.Linear(dim, dim)
        self.norm      = nn.LayerNorm(dim)
        self.drop      = nn.Dropout(dropout)

    def forward(self, primary, aux):
        g   = torch.sigmoid(self.gate_proj(primary))
        val = self.val_proj(aux)
        return self.norm(primary + self.drop(g * val))


class MSGCAWithGLU(nn.Module):
    """
    MSGCA-GLU: thay gated cross-attention bằng GLU (Dauphin et al., 2017).
    Dùng trong RQ2 ablation.
    """

    def __init__(self, macro_dim=6, news_dim=1024, dim=256, window_size=20,
                 num_classes=3, dropout=0.1):
        super().__init__()
        from encoders.mutil_encoder import MultimodalSourceEncoding
        from src.predictor import FinegrainedMovementPrediction

        self.encoder = MultimodalSourceEncoding(
            price_dim=1, macro_dim=macro_dim, news_dim=news_dim, dim=dim
        )
        self.stage1  = _GLUFusion(dim, dropout)
        self.stage2  = _GLUFusion(dim, dropout)
        self.predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=window_size, num_classes=num_classes, dropout=dropout
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, s_o, s_h, s_c, s_m, s_n, label=None,
                mode="train", return_preds=False):
        v_m, v_i, v_n = self.encoder(s_o, s_h, s_c, s_m, s_n)
        if v_n is None:
            v_n = torch.zeros_like(v_i)
        H1     = self.stage1(primary=v_i, aux=v_n)
        H_out  = self.stage2(primary=H1,  aux=v_m)
        logits = self.predictor(fused_seq=H_out, orig_seq=v_i)
        logits = torch.clamp(logits, -15, 15)

        if mode == "train":
            return self.loss_fn(logits, label.long().to(logits.device))

        preds = logits.argmax(dim=1)
        if return_preds:
            return preds
        from sklearn.metrics import accuracy_score, matthews_corrcoef
        lbl = label.cpu().numpy() if isinstance(label, torch.Tensor) else label
        return (accuracy_score(lbl, preds.cpu().numpy()),
                matthews_corrcoef(lbl, preds.cpu().numpy()))


# ──────────────────────────────────────────────────────────────────────
# RQ3 Modality ablation — wrappers around MSGCA with zeroed inputs
# ──────────────────────────────────────────────────────────────────────

class MSGCAModalityAblation(nn.Module):
    """
    Wrapper quanh StockMovementModel để zero ra một modality.

    use_news  : False → s_n = zeros (no document modality)
    use_macro : False → s_m = zeros (no macro/market context modality)
    """

    def __init__(self, base_model, use_news=True, use_macro=True):
        super().__init__()
        self.model     = base_model
        self.use_news  = use_news
        self.use_macro = use_macro

    def forward(self, s_o, s_h, s_c, s_m, s_n, label=None,
                mode="train", return_preds=False, **kw):
        if not self.use_news:
            s_n = torch.zeros_like(s_n)
        if not self.use_macro:
            s_m = torch.zeros_like(s_m)
        return self.model(s_o, s_h, s_c, s_m, s_n, label,
                          mode=mode, return_preds=return_preds)


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────

FLAT_BASELINE_NAMES = ["LSTM", "ALSTM", "ESTIMATE", "DTML", "ALSTM-W", "SLOT", "LLM-Stock"]


def build_flat_baseline(name: str, macro_dim: int,
                        hidden_dim: int = 64, num_classes: int = 3,
                        dropout: float = 0.1) -> nn.Module:
    """
    Factory cho flat baselines (LSTM, ALSTM, ESTIMATE, DTML, ALSTM-W, SLOT, LLM-Stock).

    Parameters
    ----------
    name      : str — tên model (xem FLAT_BASELINE_NAMES)
    macro_dim : int — số chiều macro features (từ s_m)
    """
    NEWS_DIM = 1024   # Voyage-3-large output dimension

    registry = {
        "LSTM":      lambda: LSTMBaseline(hidden_dim, num_classes, dropout),
        "ALSTM":     lambda: ALSTMBaseline(hidden_dim, num_classes, dropout),
        "ESTIMATE":  lambda: ESTIMATEBaseline(macro_dim, hidden_dim, num_classes, dropout),
        "DTML":      lambda: DTMLBaseline(macro_dim, hidden_dim, 2, num_classes, dropout),
        "ALSTM-W":   lambda: ALSTMWithDoc(NEWS_DIM, hidden_dim, num_classes, dropout),
        "SLOT":      lambda: SLOTBaseline(NEWS_DIM, hidden_dim, num_classes, dropout),
        "LLM-Stock": lambda: LLMStockBaseline(NEWS_DIM, hidden_dim, num_classes, dropout),
    }
    if name not in registry:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(registry)}")
    return registry[name]()