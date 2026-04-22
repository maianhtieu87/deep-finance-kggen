# baselines/models.py
"""
Baseline models — Deep Finance V5.5

V5.5 change: NewsProjector updated for 768D FinBERT input (was 1024D Voyage).
  - Compression: 768 → max(hidden*4, 256) → hidden_dim
  - Ratio: ~12x instead of ~16x → less information loss

All other logic unchanged from V5.4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import TrainConfig


class TemporalAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=-1)
        return (w.unsqueeze(-1) * lstm_out).sum(dim=1)


class NewsProjector(nn.Module):
    """
    Project news embedding → hidden_dim.

    V5.6: Automatically adapts to news_dim (FinBERT/Voyage) based on config.
    2-stage bottleneck: news_dim → max(hidden*4, 256) → hidden_dim.
    """
    def __init__(self, news_dim: int = None, hidden_dim: int = 64):
        super().__init__()
        news_dim = news_dim or TrainConfig.news_embed_dim()
        mid = max(hidden_dim * 4, 256)
        self.proj = nn.Sequential(
            nn.Linear(news_dim, mid),
            nn.GELU(),
            nn.LayerNorm(mid),
            nn.Linear(mid, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


# ── Presence masks ────────────────────────────────────────────────────────────

def _window_news_mask(s_n: torch.Tensor) -> torch.Tensor:
    return (s_n.abs().sum(dim=(1, 2)) > 0).float().unsqueeze(-1)

def _timestep_news_mask(s_n: torch.Tensor) -> torch.Tensor:
    return (s_n.abs().sum(dim=-1) > 0).float().unsqueeze(-1)


# ── Category 1 ────────────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, **kw):
        _, (h_n, _) = self.lstm(indicators)
        return self.fc(self.drop(h_n[-1]))


class ALSTMBaseline(nn.Module):
    def __init__(self, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn = TemporalAttention(hidden_dim)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, **kw):
        out, (h_n, _) = self.lstm(indicators)
        feat = torch.cat([h_n[-1], self.attn(out)], dim=-1)
        return self.fc(self.drop(feat))


# ── Category 2 ────────────────────────────────────────────────────────────────

class ALSTMWithDoc(nn.Module):
    """ALSTM-W with dynamic news embeddings."""
    def __init__(self, news_dim=None, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        news_dim = news_dim or TrainConfig.news_embed_dim()
        self.doc_proj = NewsProjector(news_dim, hidden_dim)
        self.lstm     = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn     = TemporalAttention(hidden_dim)
        self.fc       = nn.Linear(hidden_dim * 3, num_classes)
        self.drop     = nn.Dropout(dropout)

    def forward(self, indicators, s_n, **kw):
        doc_mask = _window_news_mask(s_n)
        doc      = self.doc_proj(s_n.mean(dim=1)) * doc_mask
        out, (h_n, _) = self.lstm(indicators)
        feat = torch.cat([h_n[-1], self.attn(out), doc], dim=-1)
        return self.fc(self.drop(feat))


class SLOTBaseline(nn.Module):
    """SLOT with dynamic news embeddings."""
    def __init__(self, news_dim=None, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        news_dim = news_dim or TrainConfig.news_embed_dim()
        self.price_proj = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU())
        self.doc_proj   = NewsProjector(news_dim, hidden_dim)
        self.lstm       = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attn       = TemporalAttention(hidden_dim)
        self.fc         = nn.Linear(hidden_dim * 2, num_classes)
        self.drop       = nn.Dropout(dropout)

    def forward(self, indicators, s_n, **kw):
        B, W, D = s_n.shape
        ts_mask   = _timestep_news_mask(s_n)
        price_emb = self.price_proj(indicators)
        doc_emb   = self.doc_proj(s_n.view(B * W, D)).view(B, W, -1) * ts_mask
        unified   = torch.cat([price_emb, doc_emb], dim=-1)
        out, (h_n, _) = self.lstm(unified)
        feat = torch.cat([h_n[-1], self.attn(out)], dim=-1)
        return self.fc(self.drop(feat))


class LLMStockBaseline(nn.Module):
    """LLM-Stock with dynamic news embeddings."""
    def __init__(self, news_dim=None, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        news_dim = news_dim or TrainConfig.news_embed_dim()
        self.voyage_proj = NewsProjector(news_dim, hidden_dim)
        self.price_proj  = nn.Linear(3, hidden_dim)
        self.lstm        = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attn        = TemporalAttention(hidden_dim)
        self.fc          = nn.Linear(hidden_dim * 2, num_classes)
        self.drop        = nn.Dropout(dropout)

    def forward(self, indicators, s_n, **kw):
        B, W, D = s_n.shape
        ts_mask = _timestep_news_mask(s_n)
        llm     = self.voyage_proj(s_n.view(B * W, D)).view(B, W, -1) * ts_mask
        price   = self.price_proj(indicators)
        fused   = torch.cat([price, llm], dim=-1)
        out, (h_n, _) = self.lstm(fused)
        feat = torch.cat([h_n[-1], self.attn(out)], dim=-1)
        return self.fc(self.drop(feat))


# ── Category 3 ────────────────────────────────────────────────────────────────
# (Unchanged from V5.4 — macro-only models not affected by news_dim change)

class DARNNBaseline(nn.Module):
    def __init__(self, macro_dim=5, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.input_dim  = 3 + macro_dim
        self.hidden_dim = hidden_dim
        self.input_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2 + self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.input_dim),
        )
        self.encoder_lstm  = nn.LSTMCell(self.input_dim, hidden_dim)
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, indicators, s_m, **kw):
        B, W, _ = indicators.shape
        x = torch.cat([indicators, s_m], dim=-1)
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        c = torch.zeros(B, self.hidden_dim, device=x.device)
        encoder_states = []
        for t in range(W):
            x_t     = x[:, t, :]
            attn_in = torch.cat([h, c, x_t], dim=-1)
            e_t     = self.input_attn(attn_in)
            alpha_t = torch.softmax(e_t, dim=-1)
            x_tilde = alpha_t * x_t
            h, c    = self.encoder_lstm(x_tilde, (h, c))
            encoder_states.append(h.unsqueeze(1))
        H_enc   = torch.cat(encoder_states, dim=1)
        h_final = H_enc[:, -1, :]
        query   = h_final.unsqueeze(1).expand(-1, W, -1)
        beta_in = torch.cat([query, H_enc], dim=-1)
        beta_t  = torch.softmax(self.temporal_attn(beta_in).squeeze(-1), dim=-1)
        context = (beta_t.unsqueeze(-1) * H_enc).sum(dim=1)
        context = self.norm(self.context_proj(context))
        feat    = torch.cat([h_final, context], dim=-1)
        return self.fc(self.drop(feat))


class ESTIMATEBaseline(nn.Module):
    def __init__(self, macro_dim=5, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.lstm      = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn      = TemporalAttention(hidden_dim)
        self.macro_avg = nn.Sequential(
            nn.Linear(macro_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.fc   = nn.Linear(hidden_dim * 3, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, s_m, **kw):
        out, (h_n, _) = self.lstm(indicators)
        ctx   = self.attn(out)
        macro = self.macro_avg(s_m.mean(dim=1))
        feat  = torch.cat([h_n[-1], ctx, macro], dim=-1)
        return self.fc(self.drop(feat))


class DTMLBaseline(nn.Module):
    def __init__(self, macro_dim=5, hidden_dim=64, num_heads=2, num_classes=3, dropout=0.1):
        super().__init__()
        self.lstm       = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn       = TemporalAttention(hidden_dim)
        self.macro_proj = nn.Linear(macro_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, s_m, **kw):
        out, (h_n, _) = self.lstm(indicators)
        ctx    = self.attn(out).unsqueeze(1)
        market = self.macro_proj(s_m)
        att, _ = self.cross_attn(ctx, market, market, need_weights=False)
        stock  = self.norm(ctx + att).squeeze(1)
        feat   = torch.cat([h_n[-1], stock], dim=-1)
        return self.fc(self.drop(feat))


# ── MSGCA Ablations ────────────────────────────────────────────────────────────

class _PlainCrossAttn(nn.Module):
    def __init__(self, dim, num_head, dropout=0.1):
        super().__init__()
        self.ca   = nn.MultiheadAttention(dim, num_head, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, primary, aux):
        H, _ = self.ca(primary, aux, aux, need_weights=False)
        return self.norm(primary + self.drop(H))


class MSGCANoGate(nn.Module):
    def __init__(self, macro_dim=5, news_dim=None, dim=64,
                 window_size=14, num_head=2, num_classes=3, dropout=0.1):
        super().__init__()
        news_dim = news_dim or TrainConfig.news_embed_dim()
        from encoders.mutil_encoder import MultimodalSourceEncoding
        from src.predictor import FinegrainedMovementPrediction
        self.encoder   = MultimodalSourceEncoding(1, macro_dim, news_dim, dim, dropout)
        self.stage1    = _PlainCrossAttn(dim, num_head, dropout)
        self.stage2    = _PlainCrossAttn(dim, num_head, dropout)
        self.predictor = FinegrainedMovementPrediction(dim, window_size, num_classes, dropout)
        self.loss_fn   = nn.CrossEntropyLoss()

    def forward(self, s_o, s_h, s_c, s_m, s_n, label=None, mode="train",
                return_preds=False, **kw):
        v_m, v_i, v_n = self.encoder(s_o, s_h, s_c, s_m, s_n)
        if v_n is None: v_n = torch.zeros_like(v_i)
        H1     = self.stage1(primary=v_i, aux=v_n)
        H_out  = self.stage2(primary=H1,  aux=v_m)
        logits = self.predictor(fused_seq=H_out, orig_seq=v_i)
        if mode == "train": return self.loss_fn(logits, label.long().to(logits.device))
        preds = logits.argmax(dim=1)
        if return_preds: return preds
        from sklearn.metrics import accuracy_score, matthews_corrcoef
        lbl = label.cpu().numpy() if isinstance(label, torch.Tensor) else label
        return accuracy_score(lbl, preds.cpu().numpy()), matthews_corrcoef(lbl, preds.cpu().numpy())


class _GLUFusion(nn.Module):
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
    def __init__(self, macro_dim=5, news_dim=None, dim=64,
                 window_size=14, num_classes=3, dropout=0.1):
        super().__init__()
        news_dim = news_dim or TrainConfig.news_embed_dim()
        from encoders.mutil_encoder import MultimodalSourceEncoding
        from src.predictor import FinegrainedMovementPrediction
        self.encoder   = MultimodalSourceEncoding(1, macro_dim, news_dim, dim, dropout)
        self.stage1    = _GLUFusion(dim, dropout)
        self.stage2    = _GLUFusion(dim, dropout)
        self.predictor = FinegrainedMovementPrediction(dim, window_size, num_classes, dropout)
        self.loss_fn   = nn.CrossEntropyLoss()

    def forward(self, s_o, s_h, s_c, s_m, s_n, label=None, mode="train",
                return_preds=False, **kw):
        v_m, v_i, v_n = self.encoder(s_o, s_h, s_c, s_m, s_n)
        if v_n is None: v_n = torch.zeros_like(v_i)
        H1     = self.stage1(primary=v_i, aux=v_n)
        H_out  = self.stage2(primary=H1,  aux=v_m)
        logits = self.predictor(fused_seq=H_out, orig_seq=v_i)
        if mode == "train": return self.loss_fn(logits, label.long().to(logits.device))
        preds = logits.argmax(dim=1)
        if return_preds: return preds
        from sklearn.metrics import accuracy_score, matthews_corrcoef
        lbl = label.cpu().numpy() if isinstance(label, torch.Tensor) else label
        return accuracy_score(lbl, preds.cpu().numpy()), matthews_corrcoef(lbl, preds.cpu().numpy())


# ── Registry ───────────────────────────────────────────────────────────────────

FLAT_BASELINE_REGISTRY = {
    "LSTM":      ("cat1", lambda macro_dim, hidden_dim, num_classes, dropout:
                  LSTMBaseline(hidden_dim, num_classes, dropout)),
    "ALSTM":     ("cat1", lambda macro_dim, hidden_dim, num_classes, dropout:
                  ALSTMBaseline(hidden_dim, num_classes, dropout)),
    "ALSTM-W":   ("cat2", lambda macro_dim, hidden_dim, num_classes, dropout:
                  ALSTMWithDoc(TrainConfig.news_embed_dim(), hidden_dim, num_classes, dropout)),
    "SLOT":      ("cat2", lambda macro_dim, hidden_dim, num_classes, dropout:
                  SLOTBaseline(TrainConfig.news_embed_dim(), hidden_dim, num_classes, dropout)),
    "LLM-Stock": ("cat2", lambda macro_dim, hidden_dim, num_classes, dropout:
                  LLMStockBaseline(TrainConfig.news_embed_dim(), hidden_dim, num_classes, dropout)),
    "DA-RNN":    ("cat3", lambda macro_dim, hidden_dim, num_classes, dropout:
                  DARNNBaseline(macro_dim, hidden_dim, num_classes, dropout)),
    "ESTIMATE":  ("cat3", lambda macro_dim, hidden_dim, num_classes, dropout:
                  ESTIMATEBaseline(macro_dim, hidden_dim, num_classes, dropout)),
    "DTML":      ("cat3", lambda macro_dim, hidden_dim, num_classes, dropout:
                  DTMLBaseline(macro_dim, hidden_dim, 2, num_classes, dropout)),
}

FLAT_BASELINE_ORDER = ["LSTM","ALSTM","ALSTM-W","SLOT","LLM-Stock","DA-RNN","ESTIMATE","DTML"]

CATEGORY_LABELS = {
    "cat1": "Category 1 — Indicator-only",
    "cat2": "Category 2 — Indicator + Document (News, Dynamic Dim)",
    "cat3": "Category 3 — Indicator + Market Context (Macro)",
}


def build_flat_baseline(name, macro_dim, hidden_dim=64, num_classes=3, dropout=0.1):
    if name not in FLAT_BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(FLAT_BASELINE_REGISTRY)}")
    _, factory = FLAT_BASELINE_REGISTRY[name]
    return factory(macro_dim, hidden_dim, num_classes, dropout)