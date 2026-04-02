# baselines/models.py
"""
Baseline models — Deep Finance V5.3

━━━ MODALITY MAPPING (V5.3 adaptation của MSGCA paper) ━━━

  Paper original      │  V5.3 equivalent
  ──────────────────────────────────────────────────────
  Indicator sequence  │  s_o, s_h, s_c  (B,W,3) — price OHLC
  Dynamic Document    │  s_n             (B,W,1024) — Voyage-3-Large
                      │    structured triple embeddings (headline + content)
                      │    = document modality, higher quality than tweet Word2Vec
  Relational Graph    │  s_m             (B,W,M) — macro indicators
                      │    (VIX, yield spread, SP500) = market context signal

News (s_n) là Dynamic Document: pipeline extract structured financial event triples
từ full article, encode bằng Voyage-3-Large 1024D — richer hơn tweet embeddings.
Macro (s_m) thay relational graph: không có cross-stock graph data, dùng market-level
macro signals để capture broader market context tương tự graph hyperedges.

━━━ 3 CATEGORIES (khớp với paper MSGCA) ━━━

  Category 1 — Indicator-only:
    LSTM, ALSTM

  Category 2 — Indicator + Document (Indicator + News):
    ALSTM-W  (ALSTM + avg Voyage embeddings)
    SLOT     (unified price+news ALSTM per-timestep)
    LLM-Stock (Voyage-primary + price LSTM) ← extra, not in original paper

  Category 3 — Indicator + Market Context (Indicator + Macro):
    ESTIMATE (price ALSTM + macro via concatenation)
    DTML     (price ALSTM + macro via cross-attention)

  MSGCA ablations (RQ2):
    MSGCA-CA  (plain cross-attention, no gating)
    MSGCA-GLU (Gated Linear Unit instead of gated cross-attention)

  Full MSGCA — from src.model import StockMovementModel

━━━ HYPERPARAMETERS (baseline tier, from MSGCA paper Section 5.1) ━━━
  hidden_dim = 64, lr = 1e-4, epochs = 200, batch_size = 512, dropout = 0.1
  All with standard CrossEntropyLoss for fair comparison in RQ1.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Shared sub-modules
# ─────────────────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Temporal self-attention trên LSTM output (B,T,H) → (B,H)."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.attn(lstm_out).squeeze(-1), dim=-1)  # (B,T)
        return (w.unsqueeze(-1) * lstm_out).sum(dim=1)              # (B,H)


class NewsProjector(nn.Module):
    """
    Project 1024D Voyage embedding → hidden_dim cho baselines.

    Dùng 2-lớp projection vì ratio 1024→64 quá lớn cho 1 bước.
    1024 → mid → hidden_dim với LayerNorm để ổn định.
    """

    def __init__(self, news_dim: int = 1024, hidden_dim: int = 64):
        super().__init__()
        mid = max(hidden_dim * 4, 256)  # bottleneck: 1024 → 256 → 64
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
# CATEGORY 1 — Indicator-only
# ─────────────────────────────────────────────────────────────────────────────

class LSTMBaseline(nn.Module):
    """
    LSTM (Hochreiter & Schmidhuber, 1997).
    Baseline indicator-only, 2-layer LSTM.
    Input: indicators (B,W,3)
    """

    def __init__(self, hidden_dim: int = 64, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            3, hidden_dim, num_layers=2,
            batch_first=True, dropout=dropout,
        )
        self.fc   = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators: torch.Tensor, **kw) -> torch.Tensor:
        _, (h_n, _) = self.lstm(indicators)
        return self.fc(self.drop(h_n[-1]))


class ALSTMBaseline(nn.Module):
    """
    ALSTM (Qin et al., 2017).
    LSTM + temporal attention. Dùng cả hidden state lẫn attention context.
    Input: indicators (B,W,3)
    """

    def __init__(self, hidden_dim: int = 64, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn = TemporalAttention(hidden_dim)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, indicators: torch.Tensor, **kw) -> torch.Tensor:
        out, (h_n, _) = self.lstm(indicators)
        ctx  = self.attn(out)
        feat = torch.cat([h_n[-1], ctx], dim=-1)
        return self.fc(self.drop(feat))


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 2 — Indicator + Dynamic Document (= Indicator + News)
# ─────────────────────────────────────────────────────────────────────────────

class ALSTMWithDoc(nn.Module):
    """
    ALSTM-W (adapted).
    Paper: ALSTM + average Word2Vec embeddings.
    V5.3: ALSTM + average Voyage-3-Large structured-triple embeddings.

    Document modality: window-average của Voyage 1024D → projected → concat.
    Input: indicators (B,W,3), s_n (B,W,1024)
    """

    def __init__(
        self,
        news_dim: int = 1024,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.doc_proj = NewsProjector(news_dim, hidden_dim)
        self.lstm     = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn     = TemporalAttention(hidden_dim)
        self.fc       = nn.Linear(hidden_dim * 3, num_classes)
        self.drop     = nn.Dropout(dropout)

    def forward(
        self,
        indicators: torch.Tensor,
        s_n: torch.Tensor,
        **kw,
    ) -> torch.Tensor:
        # Average Voyage embeddings over window dimension
        doc  = self.doc_proj(s_n.mean(dim=1))       # (B, H) — avg over W
        out, (h_n, _) = self.lstm(indicators)
        ctx  = self.attn(out)                         # (B, H)
        feat = torch.cat([h_n[-1], ctx, doc], dim=-1) # (B, 3H)
        return self.fc(self.drop(feat))


class SLOTBaseline(nn.Module):
    """
    SLOT (Soun et al., BigData22) — simplified.
    Paper: unified price + tweet ALSTM (without SSL pretraining stage).
    V5.3: unified price + Voyage triple embeddings per-timestep.

    Document modality: per-timestep projection, concatenated với price.
    Input: indicators (B,W,3), s_n (B,W,1024)
    """

    def __init__(
        self,
        news_dim: int = 1024,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.price_proj = nn.Sequential(nn.Linear(3, hidden_dim), nn.GELU())
        self.doc_proj   = NewsProjector(news_dim, hidden_dim)
        self.lstm       = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attn       = TemporalAttention(hidden_dim)
        self.fc         = nn.Linear(hidden_dim * 2, num_classes)
        self.drop       = nn.Dropout(dropout)

    def forward(
        self,
        indicators: torch.Tensor,
        s_n: torch.Tensor,
        **kw,
    ) -> torch.Tensor:
        B, W, D = s_n.shape
        price_emb = self.price_proj(indicators)                         # (B,W,H)
        doc_emb   = self.doc_proj(s_n.view(B * W, D)).view(B, W, -1)  # (B,W,H)
        unified   = torch.cat([price_emb, doc_emb], dim=-1)             # (B,W,2H)
        out, (h_n, _) = self.lstm(unified)
        ctx  = self.attn(out)
        feat = torch.cat([h_n[-1], ctx], dim=-1)
        return self.fc(self.drop(feat))


class LLMStockBaseline(nn.Module):
    """
    LLM-Stock (Xie et al., 2023) — adapted. ← Extra baseline, not in original paper.
    Paper: LLM embedding as primary signal + price LSTM as secondary.
    V5.3: Voyage structured-triple embedding as primary + price LSTM.

    Document modality: PRIMARY signal, price LSTM as secondary context.
    Input: indicators (B,W,3), s_n (B,W,1024)
    """

    def __init__(
        self,
        news_dim: int = 1024,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.voyage_proj = NewsProjector(news_dim, hidden_dim)
        self.price_proj  = nn.Linear(3, hidden_dim)
        self.lstm        = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True)
        self.attn        = TemporalAttention(hidden_dim)
        self.fc          = nn.Linear(hidden_dim * 2, num_classes)
        self.drop        = nn.Dropout(dropout)

    def forward(
        self,
        indicators: torch.Tensor,
        s_n: torch.Tensor,
        **kw,
    ) -> torch.Tensor:
        B, W, D = s_n.shape
        llm   = self.voyage_proj(s_n.view(B * W, D)).view(B, W, -1)  # (B,W,H)
        price = self.price_proj(indicators)                             # (B,W,H)
        fused = torch.cat([price, llm], dim=-1)                        # (B,W,2H)
        out, (h_n, _) = self.lstm(fused)
        ctx  = self.attn(out)
        feat = torch.cat([h_n[-1], ctx], dim=-1)
        return self.fc(self.drop(feat))


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY 3 — Indicator + Market Context (= Indicator + Macro)
# Paper original: Indicator + Relational Graph
# V5.3 adaptation: Macro indicators thay thế cross-stock graph
# ─────────────────────────────────────────────────────────────────────────────

class ESTIMATEBaseline(nn.Module):
    """
    ESTIMATE (Huynh et al., WSDM 2023).
    Paper: price ALSTM + stock hypergraph features via concatenation.
    V5.3: price ALSTM + macro market context via concatenation.

    Market context modality: time-averaged macro → projected → concat.
    Input: indicators (B,W,3), s_m (B,W,M)
    """

    def __init__(
        self,
        macro_dim: int = 4,
        hidden_dim: int = 64,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
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

    def forward(
        self,
        indicators: torch.Tensor,
        s_m: torch.Tensor,
        **kw,
    ) -> torch.Tensor:
        out, (h_n, _) = self.lstm(indicators)
        ctx   = self.attn(out)                        # (B, H)
        macro = self.macro_avg(s_m.mean(dim=1))       # (B, H) — avg over W
        feat  = torch.cat([h_n[-1], ctx, macro], dim=-1)
        return self.fc(self.drop(feat))


class DTMLBaseline(nn.Module):
    """
    DTML (Yoo et al., KDD 2021) — adapted.
    Paper: attentive LSTM + cross-market attention (LSTM queries market graph).
    V5.3: price ALSTM queries macro market context via cross-attention.

    Market context modality: price LSTM output queries macro sequence.
    Input: indicators (B,W,3), s_m (B,W,M)
    """

    def __init__(
        self,
        macro_dim: int = 4,
        hidden_dim: int = 64,
        num_heads: int = 2,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm       = nn.LSTM(3, hidden_dim, batch_first=True)
        self.attn       = TemporalAttention(hidden_dim)
        self.macro_proj = nn.Linear(macro_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc   = nn.Linear(hidden_dim * 2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        indicators: torch.Tensor,
        s_m: torch.Tensor,
        **kw,
    ) -> torch.Tensor:
        out, (h_n, _) = self.lstm(indicators)
        ctx    = self.attn(out).unsqueeze(1)          # (B,1,H) — query
        market = self.macro_proj(s_m)                  # (B,W,H) — key/value
        att, _ = self.cross_attn(ctx, market, market, need_weights=False)
        stock  = self.norm(ctx + att).squeeze(1)       # (B,H)
        feat   = torch.cat([h_n[-1], stock], dim=-1)
        return self.fc(self.drop(feat))


# ─────────────────────────────────────────────────────────────────────────────
# MSGCA ABLATIONS — RQ2 Fusion Strategy
# ─────────────────────────────────────────────────────────────────────────────

class _PlainCrossAttn(nn.Module):
    """Cross-attention không có gating (MSGCA-CA ablation)."""

    def __init__(self, dim: int, num_head: int, dropout: float = 0.1):
        super().__init__()
        self.ca   = nn.MultiheadAttention(
            dim, num_head, batch_first=True, dropout=dropout,
        )
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, primary: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        H, _ = self.ca(primary, aux, aux, need_weights=False)
        return self.norm(primary + self.drop(H))


class MSGCANoGate(nn.Module):
    """
    MSGCA-CA (RQ2 ablation): kiến trúc giống MSGCA nhưng thay
    gated cross-attention bằng plain cross-attention (no gating).
    Dùng để kiểm tra necessity của gating mechanism.

    Input: s_o, s_h, s_c (B,W,1), s_n (B,W,1024), s_m (B,W,M)
    """

    def __init__(
        self,
        macro_dim: int = 4,
        news_dim: int = 1024,
        dim: int = 256,
        window_size: int = 20,
        num_head: int = 4,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        from encoders.mutil_encoder import MultimodalSourceEncoding
        from src.predictor import FinegrainedMovementPrediction

        self.encoder = MultimodalSourceEncoding(
            price_dim=1, macro_dim=macro_dim, news_dim=news_dim, dim=dim,
        )
        self.stage1    = _PlainCrossAttn(dim, num_head, dropout)
        self.stage2    = _PlainCrossAttn(dim, num_head, dropout)
        self.predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=window_size,
            num_classes=num_classes, dropout=dropout,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        s_o, s_h, s_c, s_m, s_n,
        label=None,
        mode: str = "train",
        return_preds: bool = False,
    ):
        v_m, v_i, v_n = self.encoder(s_o, s_h, s_c, s_m, s_n)
        if v_n is None:
            v_n = torch.zeros_like(v_i)
        H1     = self.stage1(primary=v_i, aux=v_n)
        H_out  = self.stage2(primary=H1,  aux=v_m)
        logits = self.predictor(fused_seq=H_out, orig_seq=v_i)

        if mode == "train":
            return self.loss_fn(logits, label.long().to(logits.device))

        preds = logits.argmax(dim=1)
        if return_preds:
            return preds
        from sklearn.metrics import accuracy_score, matthews_corrcoef
        lbl = label.cpu().numpy() if isinstance(label, torch.Tensor) else label
        return (
            accuracy_score(lbl, preds.cpu().numpy()),
            matthews_corrcoef(lbl, preds.cpu().numpy()),
        )


class _GLUFusion(nn.Module):
    """
    Gated Linear Unit fusion (Dauphin et al., 2017).
    Gate từ primary, value từ aux — không dùng cross-attention.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)
        self.val_proj  = nn.Linear(dim, dim)
        self.norm      = nn.LayerNorm(dim)
        self.drop      = nn.Dropout(dropout)

    def forward(self, primary: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        g   = torch.sigmoid(self.gate_proj(primary))
        val = self.val_proj(aux)
        return self.norm(primary + self.drop(g * val))


class MSGCAWithGLU(nn.Module):
    """
    MSGCA-GLU (RQ2 ablation): thay gated cross-attention bằng GLU.
    GLU không có cross-attention — gate từ primary, value từ aux directly.
    So sánh để xem cross-attention có cần thiết không.

    Input: s_o, s_h, s_c (B,W,1), s_n (B,W,1024), s_m (B,W,M)
    """

    def __init__(
        self,
        macro_dim: int = 4,
        news_dim: int = 1024,
        dim: int = 256,
        window_size: int = 20,
        num_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        from encoders.mutil_encoder import MultimodalSourceEncoding
        from src.predictor import FinegrainedMovementPrediction

        self.encoder = MultimodalSourceEncoding(
            price_dim=1, macro_dim=macro_dim, news_dim=news_dim, dim=dim,
        )
        self.stage1    = _GLUFusion(dim, dropout)
        self.stage2    = _GLUFusion(dim, dropout)
        self.predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=window_size,
            num_classes=num_classes, dropout=dropout,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        s_o, s_h, s_c, s_m, s_n,
        label=None,
        mode: str = "train",
        return_preds: bool = False,
    ):
        v_m, v_i, v_n = self.encoder(s_o, s_h, s_c, s_m, s_n)
        if v_n is None:
            v_n = torch.zeros_like(v_i)
        H1     = self.stage1(primary=v_i, aux=v_n)
        H_out  = self.stage2(primary=H1,  aux=v_m)
        logits = self.predictor(fused_seq=H_out, orig_seq=v_i)

        if mode == "train":
            return self.loss_fn(logits, label.long().to(logits.device))

        preds = logits.argmax(dim=1)
        if return_preds:
            return preds
        from sklearn.metrics import accuracy_score, matthews_corrcoef
        lbl = label.cpu().numpy() if isinstance(label, torch.Tensor) else label
        return (
            accuracy_score(lbl, preds.cpu().numpy()),
            matthews_corrcoef(lbl, preds.cpu().numpy()),
        )


# ─────────────────────────────────────────────────────────────────────────────
# RQ3 — Modality Ablation Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MSGCAModalityAblation(nn.Module):
    """
    Wrapper quanh StockMovementModel để zero ra một modality.
    Dùng trong RQ3 để đo đóng góp từng modality.

    use_news  = False → s_n = zeros (remove Document modality)
    use_macro = False → s_m = zeros (remove Market Context modality)
    """

    def __init__(self, base_model, use_news: bool = True, use_macro: bool = True):
        super().__init__()
        self.model     = base_model
        self.use_news  = use_news
        self.use_macro = use_macro

    def forward(
        self,
        s_o, s_h, s_c, s_m, s_n,
        label=None,
        mode: str = "train",
        return_preds: bool = False,
        **kw,
    ):
        if not self.use_news:
            s_n = torch.zeros_like(s_n)
        if not self.use_macro:
            s_m = torch.zeros_like(s_m)
        return self.model(
            s_o, s_h, s_c, s_m, s_n, label,
            mode=mode, return_preds=return_preds,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Registry & Factory
# ─────────────────────────────────────────────────────────────────────────────

# Flat baselines (Category 1, 2, 3) — forward nhận **kw nên unused args ok
FLAT_BASELINE_REGISTRY = {
    # Category 1: Indicator-only
    "LSTM":      ("cat1", lambda macro_dim, hidden_dim, num_classes, dropout:
                  LSTMBaseline(hidden_dim, num_classes, dropout)),
    "ALSTM":     ("cat1", lambda macro_dim, hidden_dim, num_classes, dropout:
                  ALSTMBaseline(hidden_dim, num_classes, dropout)),
    # Category 2: Indicator + Document (News)
    "ALSTM-W":   ("cat2", lambda macro_dim, hidden_dim, num_classes, dropout:
                  ALSTMWithDoc(1024, hidden_dim, num_classes, dropout)),
    "SLOT":      ("cat2", lambda macro_dim, hidden_dim, num_classes, dropout:
                  SLOTBaseline(1024, hidden_dim, num_classes, dropout)),
    "LLM-Stock": ("cat2", lambda macro_dim, hidden_dim, num_classes, dropout:
                  LLMStockBaseline(1024, hidden_dim, num_classes, dropout)),
    # Category 3: Indicator + Market Context (Macro)
    "ESTIMATE":  ("cat3", lambda macro_dim, hidden_dim, num_classes, dropout:
                  ESTIMATEBaseline(macro_dim, hidden_dim, num_classes, dropout)),
    "DTML":      ("cat3", lambda macro_dim, hidden_dim, num_classes, dropout:
                  DTMLBaseline(macro_dim, hidden_dim, 2, num_classes, dropout)),
}

# Ordered list for experiment output
FLAT_BASELINE_ORDER = ["LSTM", "ALSTM", "ALSTM-W", "SLOT", "LLM-Stock", "ESTIMATE", "DTML"]

CATEGORY_LABELS = {
    "cat1": "Category 1 — Indicator-only",
    "cat2": "Category 2 — Indicator + Document (News)",
    "cat3": "Category 3 — Indicator + Market Context (Macro)",
}


def build_flat_baseline(
    name: str,
    macro_dim: int,
    hidden_dim: int = 64,
    num_classes: int = 3,
    dropout: float = 0.1,
) -> nn.Module:
    """
    Factory cho flat baselines.

    Parameters
    ----------
    name      : str  — key trong FLAT_BASELINE_REGISTRY
    macro_dim : int  — số chiều macro features (từ s_m.shape[-1])

    Returns
    -------
    nn.Module với forward(**kw) nhận indicators, s_n, s_m
    """
    if name not in FLAT_BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{name}'. "
            f"Available: {list(FLAT_BASELINE_REGISTRY)}"
        )
    _, factory = FLAT_BASELINE_REGISTRY[name]
    return factory(macro_dim, hidden_dim, num_classes, dropout)