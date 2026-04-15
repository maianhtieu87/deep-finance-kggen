# baselines/models.py
"""
Baseline models — Deep Finance V5.4

CHANGES vs V5.3:
  1. Cat 2 — PRESENCE MASK for missing news (BUG FIX):
     Zero input through LayerNorm+bias projector does NOT stay zero.
     Added explicit doc_mask / timestep_mask to zero out no-news output.

  2. Cat 3 — DA-RNN-Macro added as proper exogenous baseline:
     DA-RNN (Qin et al., IJCAI 2017) is designed exactly for
     "target series + exogenous/driving series" prediction.
     Input attention selects relevant macro features per timestep.
     Temporal attention selects relevant encoder hidden states.
     This is a more principled Cat 3 baseline than graph-adapted models.

MODALITY MAPPING (V5.3 → V5.4, unchanged):
  Indicator sequence → s_o, s_h, s_c  (B,W,3)
  Dynamic Document   → s_n             (B,W,1024) — Voyage embeddings
  Market Context     → s_m             (B,W,M)    — macro indicators

CATEGORIES:
  Cat 1 — Indicator-only:           LSTM, ALSTM
  Cat 2 — Indicator + Document:     ALSTM-W, SLOT, LLM-Stock
  Cat 3 — Indicator + Macro:        DA-RNN, ESTIMATE, DTML
  Ablations:                        MSGCA-CA, MSGCA-GLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Temporal self-attention on LSTM output (B,T,H) → (B,H)."""
    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=-1)
        return (w.unsqueeze(-1) * lstm_out).sum(dim=1)


class NewsProjector(nn.Module):
    """
    Project 1024D Voyage embedding → hidden_dim.
    2-stage bottleneck: 1024 → 256 → hidden_dim.
    """
    def __init__(self, news_dim: int = 1024, hidden_dim: int = 64):
        super().__init__()
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


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 1 — Indicator-only (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """LSTM (Hochreiter & Schmidhuber, 1997). 2-layer, indicator-only."""
    def __init__(self, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden_dim, num_layers=2, batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, **kw):
        _, (h_n, _) = self.lstm(indicators)
        return self.fc(self.drop(h_n[-1]))


class ALSTMBaseline(nn.Module):
    """ALSTM (Qin et al., 2017). LSTM + temporal attention."""
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


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 2 — Indicator + Dynamic Document (Indicator + News)
# BUG FIX V5.4: presence mask prevents zero-padding leaking through projector
# ─────────────────────────────────────────────────────────────────────────────

def _window_news_mask(s_n: torch.Tensor) -> torch.Tensor:
    """
    Window-level presence mask: 1.0 if any timestep in window has non-zero news.
    s_n: (B, W, 1024) → mask: (B, 1)
    Used by ALSTMWithDoc (pools over W before projecting).
    """
    return (s_n.abs().sum(dim=(1, 2)) > 0).float().unsqueeze(-1)  # (B, 1)


def _timestep_news_mask(s_n: torch.Tensor) -> torch.Tensor:
    """
    Per-timestep presence mask: 1.0 if that timestep has non-zero news.
    s_n: (B, W, 1024) → mask: (B, W, 1)
    Used by SLOT and LLMStock (project per timestep).
    """
    return (s_n.abs().sum(dim=-1) > 0).float().unsqueeze(-1)  # (B, W, 1)


class ALSTMWithDoc(nn.Module):
    """
    ALSTM-W. ALSTM + window-averaged Voyage embeddings.

    V5.4 fix: doc_mask zeros out the projected document embedding when
    the entire window has no news (all-zero s_n). Without this, the
    bias term in NewsProjector produces a non-zero output for zero input,
    giving the model a spurious "no-news" pseudo-signal.
    """
    def __init__(self, news_dim=1024, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        self.doc_proj = NewsProjector(news_dim, hidden_dim)
        self.lstm     = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn     = TemporalAttention(hidden_dim)
        self.fc       = nn.Linear(hidden_dim * 3, num_classes)
        self.drop     = nn.Dropout(dropout)

    def forward(self, indicators, s_n, **kw):
        # BUG FIX: mask out windows with no news
        doc_mask = _window_news_mask(s_n)                   # (B, 1)
        doc      = self.doc_proj(s_n.mean(dim=1)) * doc_mask  # (B, H) — zeros where no news

        out, (h_n, _) = self.lstm(indicators)
        feat = torch.cat([h_n[-1], self.attn(out), doc], dim=-1)
        return self.fc(self.drop(feat))


class SLOTBaseline(nn.Module):
    """
    SLOT (Soun et al., BigData22) — simplified.
    Unified price + Voyage embeddings per-timestep.

    V5.4 fix: timestep_mask zeros out projected news for no-news timesteps.
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
        B, W, D = s_n.shape
        ts_mask   = _timestep_news_mask(s_n)                              # (B, W, 1)
        price_emb = self.price_proj(indicators)                            # (B, W, H)
        doc_emb   = self.doc_proj(s_n.view(B * W, D)).view(B, W, -1) * ts_mask  # (B, W, H)
        unified   = torch.cat([price_emb, doc_emb], dim=-1)               # (B, W, 2H)
        out, (h_n, _) = self.lstm(unified)
        feat = torch.cat([h_n[-1], self.attn(out)], dim=-1)
        return self.fc(self.drop(feat))


class LLMStockBaseline(nn.Module):
    """
    LLM-Stock (Xie et al., 2023) — adapted.
    Voyage embedding as primary signal + price LSTM as secondary.

    V5.4 fix: timestep_mask zeros out no-news timesteps before LSTM.
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
        ts_mask = _timestep_news_mask(s_n)                                  # (B, W, 1)
        llm     = self.voyage_proj(s_n.view(B * W, D)).view(B, W, -1) * ts_mask  # (B, W, H)
        price   = self.price_proj(indicators)                                # (B, W, H)
        fused   = torch.cat([price, llm], dim=-1)                            # (B, W, 2H)
        out, (h_n, _) = self.lstm(fused)
        feat = torch.cat([h_n[-1], self.attn(out)], dim=-1)
        return self.fc(self.drop(feat))


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 3 — Indicator + Market Context (Indicator + Macro)
# ─────────────────────────────────────────────────────────────────────────────

class DARNNBaseline(nn.Module):
    """
    DA-RNN (Qin et al., IJCAI 2017) — adapted for classification.
    https://arxiv.org/abs/1704.02971

    DA-RNN is designed for NARX: predict target series from its own
    history + n exogenous/driving series. This is exactly our setting:
      - Target series:   price (O/H/C) = 3 dims
      - Exogenous:       macro (VIX, sp500, dxy, wti, yield_spread) = M dims
      - Combined input:  (3 + M) driving series, predict next price movement

    Architecture:
      Stage 1 — Input Attention:
        At each timestep t, compute attention over the (3+M) input series.
        Attention uses previous encoder hidden state h_{t-1} and cell c_{t-1}.
        Selected input = weighted sum over series dimensions.

      Stage 2 — Temporal Attention (Encoder LSTM → Decoder):
        At the final step, compute attention over all T encoder hidden states.
        Context = weighted sum of encoder states.
        Decoder: context vector → FC → 3-class logits.

    Differences from paper (classification vs regression):
      - No separate decoder LSTM; use temporal-attention context directly.
      - Combined price+macro as input series (paper treats target separately;
        we merge because our "target" is a 3-class label, not a continuous value).
    """
    def __init__(self, macro_dim=5, hidden_dim=64, num_classes=3, dropout=0.1):
        super().__init__()
        # Input series: price (3) + macro (M)
        self.input_dim  = 3 + macro_dim
        self.hidden_dim = hidden_dim

        # Stage 1: Input attention
        # v_e (Eq.8): projects [h_{t-1}, c_{t-1}, x_k] → scalar score per series k
        self.input_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2 + self.input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, self.input_dim),   # score for each of the n input series
        )

        # Encoder LSTM (processes attended input)
        self.encoder_lstm = nn.LSTMCell(self.input_dim, hidden_dim)

        # Stage 2: Temporal attention
        # attention over T encoder hidden states
        self.temporal_attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # Context projection
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

        self.fc   = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, indicators, s_m, **kw):
        """
        indicators : (B, W, 3)  — price O/H/C
        s_m        : (B, W, M)  — macro series

        The two are treated as JOINT input series (W timesteps, 3+M features).
        Input attention selects which of the 3+M series to trust at each step.
        """
        B, W, _ = indicators.shape
        # Concatenate into joint driving series: (B, W, 3+M)
        x = torch.cat([indicators, s_m], dim=-1)

        # Initialize encoder hidden/cell states
        h = torch.zeros(B, self.hidden_dim, device=x.device)
        c = torch.zeros(B, self.hidden_dim, device=x.device)

        encoder_states = []

        # Stage 1: Input attention encoder
        for t in range(W):
            x_t = x[:, t, :]   # (B, 3+M) — all series at time t

            # Attention score for each input series k
            # Input to attention: [h_{t-1}, c_{t-1}, x_t]
            attn_in = torch.cat([h, c, x_t], dim=-1)   # (B, 2H + n_series)
            e_t = self.input_attn(attn_in)               # (B, n_series)
            alpha_t = torch.softmax(e_t, dim=-1)         # (B, n_series)

            # Attended input: element-wise weight each series
            x_tilde = alpha_t * x_t                     # (B, n_series)

            # Encoder step
            h, c = self.encoder_lstm(x_tilde, (h, c))
            encoder_states.append(h.unsqueeze(1))        # (B, 1, H)

        # Stack all encoder hidden states: (B, W, H)
        H_enc = torch.cat(encoder_states, dim=1)

        # Stage 2: Temporal attention over encoder states
        # Query: final hidden state h_T
        h_final = H_enc[:, -1, :]                        # (B, H)
        query   = h_final.unsqueeze(1).expand(-1, W, -1) # (B, W, H)
        beta_in = torch.cat([query, H_enc], dim=-1)      # (B, W, 2H)
        beta_t  = torch.softmax(
            self.temporal_attn(beta_in).squeeze(-1), dim=-1  # (B, W)
        )
        context = (beta_t.unsqueeze(-1) * H_enc).sum(dim=1)  # (B, H)

        # Combine context + final hidden state
        context = self.norm(self.context_proj(context))   # (B, H)
        feat    = torch.cat([h_final, context], dim=-1)   # (B, 2H)
        return self.fc(self.drop(feat))


class ESTIMATEBaseline(nn.Module):
    """
    ESTIMATE (Huynh et al., WSDM 2023) — adapted.
    Price ALSTM + macro via late-fusion concatenation.
    """
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
    """
    DTML (Yoo et al., KDD 2021) — adapted.
    Price ALSTM + macro via cross-attention (price queries macro).
    """
    def __init__(self, macro_dim=5, hidden_dim=64, num_heads=2, num_classes=3, dropout=0.1):
        super().__init__()
        self.lstm       = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn       = TemporalAttention(hidden_dim)
        self.macro_proj = nn.Linear(macro_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators, s_m, **kw):
        out, (h_n, _) = self.lstm(indicators)
        ctx    = self.attn(out).unsqueeze(1)     # (B,1,H) — query
        market = self.macro_proj(s_m)             # (B,W,H) — key/value
        att, _ = self.cross_attn(ctx, market, market, need_weights=False)
        stock  = self.norm(ctx + att).squeeze(1)
        feat   = torch.cat([h_n[-1], stock], dim=-1)
        return self.fc(self.drop(feat))


# ─────────────────────────────────────────────────────────────────────────────
# MSGCA ABLATIONS — RQ2 Fusion Strategy (unchanged from V5.3)
# ─────────────────────────────────────────────────────────────────────────────

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
    """MSGCA-CA: plain cross-attention, no gating (RQ2 ablation)."""
    def __init__(self, macro_dim=5, news_dim=1024, dim=64,
                 window_size=14, num_head=2, num_classes=3, dropout=0.1):
        super().__init__()
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
    """MSGCA-GLU: replace gated cross-attention with GLU (RQ2 ablation)."""
    def __init__(self, macro_dim=5, news_dim=1024, dim=64,
                 window_size=14, num_classes=3, dropout=0.1):
        super().__init__()
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


# ─────────────────────────────────────────────────────────────────────────────
# Registry & Factory
# ─────────────────────────────────────────────────────────────────────────────

FLAT_BASELINE_REGISTRY = {
    "LSTM":      ("cat1", lambda macro_dim, hidden_dim, num_classes, dropout:
                  LSTMBaseline(hidden_dim, num_classes, dropout)),
    "ALSTM":     ("cat1", lambda macro_dim, hidden_dim, num_classes, dropout:
                  ALSTMBaseline(hidden_dim, num_classes, dropout)),
    "ALSTM-W":   ("cat2", lambda macro_dim, hidden_dim, num_classes, dropout:
                  ALSTMWithDoc(1024, hidden_dim, num_classes, dropout)),
    "SLOT":      ("cat2", lambda macro_dim, hidden_dim, num_classes, dropout:
                  SLOTBaseline(1024, hidden_dim, num_classes, dropout)),
    "LLM-Stock": ("cat2", lambda macro_dim, hidden_dim, num_classes, dropout:
                  LLMStockBaseline(1024, hidden_dim, num_classes, dropout)),
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
    "cat2": "Category 2 — Indicator + Document (News)",
    "cat3": "Category 3 — Indicator + Market Context (Macro)",
}


def build_flat_baseline(name, macro_dim, hidden_dim=64, num_classes=3, dropout=0.1):
    if name not in FLAT_BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Available: {list(FLAT_BASELINE_REGISTRY)}")
    _, factory = FLAT_BASELINE_REGISTRY[name]
    return factory(macro_dim, hidden_dim, num_classes, dropout)