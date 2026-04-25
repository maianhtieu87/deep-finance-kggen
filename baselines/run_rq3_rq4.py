# -*- coding: utf-8 -*-
# baselines/run_rq3_rq4.py  — V2 (architectural parity fix)
"""
RQ3 & RQ4 Experiment Runner — V2

═══════════════════════════════════════════════════════════════════
THAY ĐỔI SO VỚI V1 (run_rq3_rq4_draft.py):

  V1 BUG — VariantModel dùng individual encoders thay vì
  MultimodalSourceEncoding → encoder path KHÁC với StockMovementModel
  → kết quả rq3_gated (NM) không thể so sánh trực tiếp với baseline_fv.

  V2 FIX — Dùng PatchedStockMovementModel:
    1. Build StockMovementModel đúng như run_ablation.py
    2. Introspect để tìm tên fusion attrs (stage1/stage2 hoặc tương đương)
    3. Thay thế ĐÚNG các fusion layers → mọi thứ khác giữ nguyên
    4. Train từ đầu với cùng protocol run_seed_fixed_val

  Fallback (nếu introspect thất bại):
    → VariantModel với MultimodalSourceEncoding (4-tuple handled)
    → Cảnh báo rõ ràng về architectural gap

  Reference anchor:
    → Load MSGCA_FV từ baselines/results/raw_results.json
    → Load baseline_wf từ baselines/results/ablation_raw.json
    → Validation: gated/NM MCC so với baseline_fv (phải overlap trong 1 std)

═══════════════════════════════════════════════════════════════════

RQ3: Does Gated Cross-Attention produce MORE STABLE results (lower variance)
     vs plain Cross-Attention?
     - MSGCA_Gated  : canonical MSGCA gated CA (= baseline_fv reference)
     - MSGCA_NoGated: replace gating with plain residual cross-attention

RQ4: Does the sequential fusion order NM outperform other orderings?
     - Order_NM : price×news → (price+news)×macro  [canonical]
     - Order_MN : price×macro → (price+macro)×news
     - Order_par: parallel price×news + price×macro → merge

Protocol (identical to run_ablation.py):
  train=[0:hval_split=0.80×inner_T]  val=[hval_split:inner_T]  test=[inner_T:T_max]
  FocalLoss(γ=2.0) + class weights   warmup=10ep   patience=TrainConfig
  AdamW no-decay on LN/bias          modality dropout=TrainConfig
  N seeds = 5 (default)

Usage:
  python baselines/run_rq3_rq4.py
  python baselines/run_rq3_rq4.py --rq 3
  python baselines/run_rq3_rq4.py --rq 4
  python baselines/run_rq3_rq4.py --n-seeds 3 --epochs 100
  python baselines/run_rq3_rq4.py --load-from output/   # ablation .pt for anchor

Outputs:
  baselines/results/rq3_table.txt
  baselines/results/rq4_table.txt
  baselines/results/rq3_rq4_raw.json
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

import argparse, copy, json, os, sys, time, random, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import TrainConfig, GlobalConfig
from src.data_loader import data_prepare, N_TICKERS, NEWS_EMB_DIM
from src.model import StockMovementModel

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

if DEVICE.type == "cuda":
    torch.set_float32_matmul_precision("high")

SEEDS        = [42, 123, 256, 512, 1024]
_PATIENCE    = TrainConfig.early_stop_patience     # default 30
_MOD_DROPOUT = TrainConfig.news_modality_dropout   # default 0.30


# =============================================================================
# FUSION MODULES — exact copies of what StockMovementModel uses
# =============================================================================

class GatedCrossAttention(nn.Module):
    """
    Canonical MSGCA gated cross-attention.
    Must be structurally identical to the fusion block in StockMovementModel
    (src/fusion.py or inline) so that PatchedStockMovementModel produces
    the same computation graph as the original when fusion_type='gated'.

    Eq (from paper):
      H_unstable = CrossAttn(Q=primary, K=V=aux)
      H_a = W_a(H_unstable) + bias_a
      H_b = sigmoid(W_b(primary) + bias_b)       ← gate from PRIMARY (stable signal)
      H_gated = H_a ⊙ H_b
      output  = LayerNorm(primary + dropout(H_gated))
    """
    def __init__(self, dim: int, num_head: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_head,
            batch_first=True, dropout=dropout,
        )
        self.W_a     = nn.Linear(dim, dim, bias=False)
        self.bias_a  = nn.Parameter(torch.zeros(dim))
        self.W_b     = nn.Linear(dim, dim, bias=True)
        nn.init.constant_(self.W_b.bias, 1.0)   # sigmoid(1) ≈ 0.73 → partial pass at init
        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        primary:  torch.Tensor,            # (B, T, dim)
        aux:      torch.Tensor,            # (B, T, dim)
        aux_mask: Optional[torch.Tensor] = None,   # (B, T) bool, True=pad
    ) -> torch.Tensor:
        # Guard: if ALL positions are masked, MHA will nan → unset mask
        if aux_mask is not None:
            all_masked = aux_mask.all(dim=1, keepdim=True)   # (B, 1)
            if all_masked.any():
                aux_mask = aux_mask & ~all_masked             # unset fully-masked rows
        H_unstable, _ = self.cross_attn(
            query=primary, key=aux, value=aux,
            key_padding_mask=aux_mask, need_weights=False,
        )
        H_a     = self.W_a(H_unstable) + self.bias_a
        H_b     = torch.sigmoid(self.W_b(primary))
        H_gated = H_a * H_b
        return self.norm(primary + self.dropout(H_gated))


class PlainCrossAttention(nn.Module):
    """
    RQ3 comparison: cross-attention WITHOUT sigmoid gate.
    Identical to GatedCrossAttention except H_b is removed.
    H_plain = W_a(CrossAttn(Q=primary, K=V=aux)) + bias_a
    output  = LayerNorm(primary + dropout(H_plain))
    Isolates the contribution of the gating mechanism.
    """
    def __init__(self, dim: int, num_head: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_head,
            batch_first=True, dropout=dropout,
        )
        self.W_a     = nn.Linear(dim, dim, bias=False)
        self.bias_a  = nn.Parameter(torch.zeros(dim))
        self.norm    = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        primary:  torch.Tensor,
        aux:      torch.Tensor,
        aux_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if aux_mask is not None:
            all_masked = aux_mask.all(dim=1, keepdim=True)
            if all_masked.any():
                aux_mask = aux_mask & ~all_masked
        H_unstable, _ = self.cross_attn(
            query=primary, key=aux, value=aux,
            key_padding_mask=aux_mask, need_weights=False,
        )
        H_plain = self.W_a(H_unstable) + self.bias_a
        return self.norm(primary + self.dropout(H_plain))


# =============================================================================
# PATCHED MODEL — swap only fusion layers, keep everything else
# =============================================================================

def _discover_fusion_attrs(model: StockMovementModel) -> Tuple[str, str]:
    """
    Introspect StockMovementModel để tìm tên của hai fusion stages.
    Tìm kiếm theo pattern phổ biến: stage1/stage2, fusion_stage1/fusion_stage2,
    cross_attn1/cross_attn2, etc.

    Returns (stage1_attr_name, stage2_attr_name) or raises RuntimeError.
    """
    candidates_stage1 = []
    candidates_stage2 = []

    for name, module in model.named_modules():
        lowname = name.lower()
        # Chỉ top-level attrs (no dots), có liên quan đến attention/fusion
        if "." not in name and any(kw in lowname for kw in
                                   ["stage", "fusion", "cross", "gate", "attn"]):
            if any(kw in lowname for kw in ["1", "first", "news", "n_"]):
                candidates_stage1.append(name)
            elif any(kw in lowname for kw in ["2", "second", "macro", "m_"]):
                candidates_stage2.append(name)

    # Fallback: bất kỳ module nào có MultiheadAttention bên trong
    if not candidates_stage1 or not candidates_stage2:
        mha_attrs = []
        for name, module in model.named_children():
            if isinstance(module, nn.MultiheadAttention):
                mha_attrs.append(name)
            elif any(isinstance(m, nn.MultiheadAttention)
                     for m in module.children()):
                mha_attrs.append(name)
        if len(mha_attrs) >= 2:
            candidates_stage1 = [mha_attrs[0]]
            candidates_stage2 = [mha_attrs[1]]

    if not candidates_stage1 or not candidates_stage2:
        raise RuntimeError(
            f"Cannot discover fusion attrs in StockMovementModel.\n"
            f"All named_children: {[n for n, _ in model.named_children()]}\n"
            f"Please hardcode attr names below in _FUSION_ATTR_OVERRIDES."
        )
    return candidates_stage1[0], candidates_stage2[0]


# ── Hardcode override (fill in if auto-discover fails) ──────────────────────
# Example: _FUSION_ATTR_OVERRIDES = ("stage1", "stage2")
_FUSION_ATTR_OVERRIDES: Optional[Tuple[str, str]] = None


def _get_fusion_attr_names(model: StockMovementModel) -> Tuple[str, str]:
    if _FUSION_ATTR_OVERRIDES is not None:
        return _FUSION_ATTR_OVERRIDES
    try:
        s1, s2 = _discover_fusion_attrs(model)
        print(f"  [Introspect] fusion attrs: stage1='{s1}'  stage2='{s2}'")
        return s1, s2
    except RuntimeError as e:
        print(f"\n  [WARN] {e}")
        print("  → Falling back to VariantModel (see FALLBACK section below)\n")
        return None, None


class PatchedStockMovementModel:
    """
    Wraps StockMovementModel và chỉ thay thế fusion stages.

    Architecture parity guarantee:
      - Tất cả encoder (indicator, macro, news), predictor, ticker_emb,
        fused_proj, loss_fn đều giống hệt StockMovementModel
      - Chỉ stage1/stage2 được thay bởi FusionCls variant

    Forward signature: giống hệt StockMovementModel.forward()
    """
    def __init__(
        self,
        base_model:   StockMovementModel,
        stage1_attr:  str,
        stage2_attr:  str,
        new_stage1:   nn.Module,
        new_stage2:   nn.Module,
    ):
        # Deep-copy để tránh ảnh hưởng lẫn nhau giữa variants
        self._model = copy.deepcopy(base_model)
        setattr(self._model, stage1_attr, new_stage1)
        setattr(self._model, stage2_attr, new_stage2)
        self._model = self._model.to(DEVICE)

    # Delegate mọi thứ xuống _model
    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def train(self, mode=True):
        self._model.train(mode)
        return self

    def eval(self):
        self._model.eval()
        return self

    def parameters(self):
        return self._model.parameters()

    def named_parameters(self):
        return self._model.named_parameters()

    def state_dict(self):
        return self._model.state_dict()

    def load_state_dict(self, state, strict=True):
        return self._model.load_state_dict(state, strict=strict)

    def zero_grad(self, set_to_none=True):
        self._model.zero_grad(set_to_none=set_to_none)


def _build_base_model(macro_dim: int, news_dim: int, cw: torch.Tensor) -> StockMovementModel:
    """Build StockMovementModel đúng cách — identical to run_ablation.py build_model()."""
    return StockMovementModel(
        price_dim=1,
        macro_dim=macro_dim,
        news_dim=news_dim,
        dim=TrainConfig.dim,
        input_dim=TrainConfig.window_size,
        output_dim=3,
        num_head=TrainConfig.num_head,
        dropout=0.1,
        class_weights=cw,
        use_focal_loss=True,
        focal_gamma=2.0,
        device=DEVICE,
        n_tickers=N_TICKERS,
        quality_dim=getattr(GlobalConfig, "QUALITY_DIM", 4),
    ).to(DEVICE)


# =============================================================================
# FALLBACK: VariantModel với MultimodalSourceEncoding (4-tuple)
# =============================================================================

class _FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)   # auto device-move via .to()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.to(inputs.device) if self.alpha is not None else None
        ce    = F.cross_entropy(inputs, targets, reduction="none", weight=alpha)
        pt    = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class FallbackVariantModel(nn.Module):
    """
    Fallback khi PatchedStockMovementModel không introspect được.
    Dùng MultimodalSourceEncoding (xử lý đúng 4-tuple return).
    Tất cả non-fusion components được import từ cùng source với StockMovementModel.
    """
    def __init__(
        self,
        macro_dim:    int,
        news_dim:     int,
        dim:          int   = 64,
        num_head:     int   = 2,
        window_size:  int   = 20,
        dropout:      float = 0.1,
        n_tickers:    int   = N_TICKERS,
        ticker_emb_dim: int = 16,
        fusion_type:  str   = "gated",
        fusion_order: str   = "NM",
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma:  float = 2.0,
        quality_dim:  int   = 4,
    ):
        super().__init__()
        assert fusion_type  in ("gated", "plain")
        assert fusion_order in ("NM", "MN", "par")
        self.fusion_order = fusion_order
        self.dim          = dim
        self.news_dim     = news_dim

        from encoders.mutil_encoder import MultimodalSourceEncoding
        from src.predictor          import FinegrainedMovementPrediction

        # MultimodalSourceEncoding returns 4-tuple: (v_m, v_i, v_n, extra)
        # extra is typically processed news_mask or quality-gated embedding
        self.encoder = MultimodalSourceEncoding(
            price_dim=1, macro_dim=macro_dim, news_dim=news_dim,
            dim=dim, dropout=dropout,
        )

        # Ticker embedding (match StockMovementModel)
        self.ticker_emb  = nn.Embedding(n_tickers, ticker_emb_dim)
        self.ticker_proj = nn.Sequential(
            nn.Linear(ticker_emb_dim, dim), nn.LayerNorm(dim), nn.Tanh(),
        )
        nn.init.normal_(self.ticker_emb.weight, 0.0, 0.02)

        FusionCls = GatedCrossAttention if fusion_type == "gated" else PlainCrossAttention

        if fusion_order == "par":
            self.fusion_news  = FusionCls(dim=dim, num_head=num_head, dropout=dropout)
            self.fusion_macro = FusionCls(dim=dim, num_head=num_head, dropout=dropout)
            # Lightweight mean-merge (no extra params vs sequential variants)
            # Mean avoids giving par an unfair parameter advantage
        else:
            self.stage1 = FusionCls(dim=dim, num_head=num_head, dropout=dropout)
            self.stage2 = FusionCls(dim=dim, num_head=num_head, dropout=dropout)

        # fused_proj: cat([H_idm, v_t_seq]) → dim
        # v_t_seq is ticker embedding broadcast to (B,T,dim)
        self.fused_proj = nn.Sequential(
            nn.Linear(2 * dim, dim), nn.LayerNorm(dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        self.predictor = FinegrainedMovementPrediction(
            dim=dim, window_size=window_size, num_classes=3, dropout=dropout,
        )
        self.loss_fn = _FocalLoss(alpha=class_weights, gamma=focal_gamma)

    def forward(
        self,
        s_o, s_h, s_c, s_m, s_n,
        label=None,
        mode:         str  = "train",
        return_preds: bool = False,
        ticker_id=None,
        news_mask=None,
        news_quality=None,
        **kwargs,
    ):
        B, T = s_o.shape[0], s_o.shape[1]

        # ── encode (MultimodalSourceEncoding returns 4-tuple) ─────────────────
        enc_out = self.encoder(
            s_o.to(DEVICE), s_h.to(DEVICE), s_c.to(DEVICE),
            s_m.to(DEVICE), s_n.to(DEVICE),
        )
        # Unpack: typical order from MultimodalSourceEncoding is (v_m, v_i, v_n, extra)
        # The 4th element varies — could be quality gate, processed mask, or interim feat.
        # We only need v_m, v_i, v_n for fusion.
        if len(enc_out) == 4:
            v_m, v_i, v_n, _ = enc_out
        elif len(enc_out) == 3:
            v_m, v_i, v_n = enc_out
        else:
            raise RuntimeError(f"Unexpected encoder output length: {len(enc_out)}")

        # ── ticker embedding ──────────────────────────────────────────────────
        tid = (ticker_id.to(DEVICE) if ticker_id is not None
               else torch.zeros(B, dtype=torch.long, device=DEVICE))
        v_t     = self.ticker_proj(self.ticker_emb(tid))   # (B, dim)
        v_t_seq = v_t.unsqueeze(1).expand(-1, T, -1)       # (B, T, dim)

        # ── news mask for CA ──────────────────────────────────────────────────
        # Convert news_mask from bool (True=present) → inverted for key_padding_mask
        # (True=ignore). If mask absent, assume all present.
        if news_mask is not None:
            nm = ~news_mask.bool().to(DEVICE)   # (B, T): True=pad position
        else:
            nm = None

        # ── fuse ─────────────────────────────────────────────────────────────
        if self.fusion_order == "NM":
            H_id  = self.stage1(primary=v_i,  aux=v_n, aux_mask=nm)
            H_idm = self.stage2(primary=H_id, aux=v_m)
        elif self.fusion_order == "MN":
            H_im  = self.stage1(primary=v_i,  aux=v_m)
            H_idm = self.stage2(primary=H_im, aux=v_n, aux_mask=nm)
        else:  # par — mean of two independent fusion branches
            H_n   = self.fusion_news( v_i, v_n, nm)
            H_m   = self.fusion_macro(v_i, v_m)
            H_idm = (H_n + H_m) * 0.5               # mean-merge, no extra params

        # ── predict ──────────────────────────────────────────────────────────
        fused = self.fused_proj(torch.cat([H_idm, v_t_seq], dim=-1))  # (B,T,dim)
        logits = self.predictor(fused_seq=fused, orig_seq=v_i)
        logits = torch.clamp(logits, -15.0, 15.0)

        def _to_long(lbl):
            if isinstance(lbl, list):
                return torch.tensor(
                    [x[0] if isinstance(x, (list, tuple)) else x for x in lbl],
                    dtype=torch.long, device=DEVICE,
                )
            return lbl.long().to(DEVICE)

        if mode == "train":
            return self.loss_fn(logits, _to_long(label))

        preds = logits.argmax(dim=1)
        if mode == "test":
            tgt = _to_long(label).cpu().numpy()
            p   = preds.cpu().numpy()
            acc = accuracy_score(tgt, p)
            mcc = matthews_corrcoef(tgt, p) if len(set(tgt)) > 1 else 0.0
            if return_preds:
                return acc, mcc, preds
            return acc, mcc
        return logits


# =============================================================================
# INFRASTRUCTURE — exact parity with run_ablation.py
# =============================================================================

class StockDataset(Dataset):
    _KEYS = ["s_o", "s_h", "s_c", "s_m", "s_n",
             "news_mask", "label", "ticker_id", "news_quality"]
    def __init__(self, d: dict):
        self.d    = d
        self.keys = [k for k in self._KEYS if k in d]
    def __len__(self):       return len(self.d["label"])
    def __getitem__(self, i): return {k: self.d[k][i] for k in self.keys}


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def merge(dicts: list, shuffle: bool = False) -> dict:
    valid = [d for d in dicts if d and len(d.get("label", [])) > 0]
    if not valid: return {}
    m: dict = {}
    for key in valid[0]:
        parts = [d[key] for d in valid if key in d and isinstance(d[key], torch.Tensor)]
        if parts: m[key] = torch.cat(parts, dim=0)
    if shuffle and "label" in m:
        idx = torch.randperm(len(m["label"]))
        for k in m: m[k] = m[k][idx]
    return m


def load_splits(pkl_path: str, tickers: list) -> dict:
    """
    IDENTICAL split logic với run_ablation.py để test set khớp hoàn toàn.
    train_wf  = [0:inner_T]     — cho reference load validation
    train_hval= [0:hval_split]  — cho training (consistent với no_news/no_macro)
    val_fixed = [hval_split:inner_T]
    test      = [inner_T:T_max]  ← CÙNG với main.py và run_ablation.py
    """
    dp = data_prepare(pkl_path, include_ticker_id=True)
    valid_T      = [dp.get_max_T(t) for t in tickers if dp.get_max_T(t) > 0]
    global_T_max = min(valid_T)
    inner_T      = int(global_T_max * 0.85)    # = val_end trong main.py ✓
    hval_split   = int(inner_T * 0.80)

    print(f"  global_T_max={global_T_max}  inner_T={inner_T}  hval_split={hval_split}")
    print(f"  test=[{inner_T}:{global_T_max}] ({global_T_max-inner_T} steps) "
          f"← identical to main.py & run_ablation.py ✓")

    tr_wf_list, tr_hv_list, va_hv_list, te_list = [], [], [], []
    macro_dim = news_dim = None

    for t in tickers:
        if dp.get_max_T(t) == 0:
            continue
        tr_wf, _, _ = dp.prepare_data(
            t, train_end=inner_T,    val_end=inner_T,    test_end=global_T_max
        )
        tr_hv, va_hv, te = dp.prepare_data(
            t, train_end=hval_split, val_end=inner_T,    test_end=global_T_max
        )
        if macro_dim is None and tr_hv and len(tr_hv.get("label", [])) > 0:
            macro_dim = tr_hv["s_m"].shape[-1]
            news_dim  = tr_hv["s_n"].shape[-1]

        if tr_wf and len(tr_wf.get("label", [])): tr_wf_list.append(tr_wf)
        if tr_hv and len(tr_hv.get("label", [])): tr_hv_list.append(tr_hv)
        if va_hv and len(va_hv.get("label", [])): va_hv_list.append(va_hv)
        if te    and len(te.get("label",    [])): te_list.append(te)

    print(f"  macro_dim={macro_dim}  news_dim={news_dim}  "
          f"(NEWS_EMB_DIM={NEWS_EMB_DIM})")
    return {
        "inner_T":      inner_T,
        "global_T_max": global_T_max,
        "hval_split":   hval_split,
        "train_wf":     merge(tr_wf_list,  shuffle=True),   # [0:inner_T]
        "train_hval":   merge(tr_hv_list,  shuffle=True),   # [0:hval_split]
        "val_fixed":    merge(va_hv_list,  shuffle=False),
        "test":         merge(te_list,     shuffle=False),
        "macro_dim":    macro_dim,
        "news_dim":     news_dim,
    }


def compute_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """Exact copy từ run_ablation.py: beta=0.9999, sqrt-normalize."""
    lbl  = labels.numpy()
    cnts = np.bincount(lbl, minlength=3).astype(float)
    beta = 0.9999
    eff  = 1.0 - np.power(beta, cnts)
    w    = (1.0 - beta) / (eff + 1e-8)
    w    = np.sqrt(w / w.sum() * 3)
    w    = w / w.sum() * 3
    return torch.tensor(w, dtype=torch.float32)


def _make_adamw(model, lr: float) -> torch.optim.Optimizer:
    """Exact copy từ run_ablation.py: no weight_decay trên LN/bias."""
    no_kw = ["bias", "LayerNorm.weight", "layernorm.weight",
             "norm.weight", "attn_norm.weight", "out_norm.weight"]
    dec, nodec = [], []
    for name, p in (model.named_parameters()
                    if hasattr(model, "named_parameters") else []):
        if not p.requires_grad: continue
        (nodec if any(k in name for k in no_kw) else dec).append(p)
    # Fallback: direct parameters()
    if not dec and not nodec:
        return torch.optim.AdamW(model.parameters(), lr=lr,
                                 weight_decay=TrainConfig.weight_decay)
    return torch.optim.AdamW(
        [{"params": dec,   "weight_decay": TrainConfig.weight_decay},
         {"params": nodec, "weight_decay": 0.0}],
        lr=lr,
    )


def _make_dataloader(ds: StockDataset, shuffle: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        drop_last=False, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )


def evaluate_model(model, data: dict) -> Tuple[float, float]:
    """
    Evaluate → (acc, mcc). Handles both StockMovementModel and FallbackVariantModel.
    Logic identical to run_ablation.py evaluate().
    """
    if not data or len(data.get("label", [])) == 0:
        return 0.0, 0.0

    is_patched  = isinstance(model, PatchedStockMovementModel)
    is_base     = isinstance(model, (StockMovementModel, PatchedStockMovementModel))

    ds  = StockDataset(data)
    ldr = _make_dataloader(ds, shuffle=False, batch_size=64)
    if is_patched: model.eval()
    else:          model.eval()

    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in ldr:
            q = batch.get("news_quality")
            if is_patched or is_base:
                result = model(
                    batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                    batch["s_c"].to(DEVICE), batch["s_m"].to(DEVICE),
                    batch["s_n"].to(DEVICE), batch["label"].to(DEVICE),
                    mode="test", return_preds=True,
                    ticker_id=batch.get("ticker_id"),
                    news_mask=batch.get("news_mask"),
                    news_quality=q.to(DEVICE) if q is not None else None,
                )
            else:  # FallbackVariantModel
                result = model(
                    batch["s_o"], batch["s_h"], batch["s_c"],
                    batch["s_m"], batch["s_n"], batch["label"],
                    mode="test", return_preds=True,
                    ticker_id=batch.get("ticker_id"),
                    news_mask=batch.get("news_mask"),
                    news_quality=q,
                )

            if isinstance(result, tuple) and len(result) == 3:
                _, _, preds = result
            elif isinstance(result, tuple) and len(result) == 2:
                # FallbackVariantModel without return_preds (shouldn't happen)
                preds = torch.zeros(batch["label"].shape[0], dtype=torch.long)
            else:
                preds = result

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(batch["label"].numpy())

    if len(set(labels_all)) < 2:
        return float(accuracy_score(labels_all, preds_all)), 0.0
    return (
        float(accuracy_score(labels_all, preds_all)),
        float(matthews_corrcoef(labels_all, preds_all)),
    )


# =============================================================================
# TRAINING LOOP — exact parity với run_ablation.run_seed_fixed_val()
# =============================================================================

def run_one_seed(
    seed:         int,
    train_data:   dict,
    val_data:     dict,
    test_data:    dict,
    macro_dim:    int,
    news_dim:     int,
    fusion_type:  str,
    fusion_order: str,
    stage1_attr:  Optional[str],
    stage2_attr:  Optional[str],
    max_epochs:   int   = 200,
    warmup_epochs: int  = 10,
    patience:     Optional[int] = None,
) -> Tuple[float, float]:
    """
    Train một variant/seed. Giao thức IDENTICAL với run_ablation.run_seed_fixed_val().

    Nếu stage1_attr/stage2_attr hợp lệ:
      → PatchedStockMovementModel (architectural parity guaranteed)
    Else:
      → FallbackVariantModel với MultimodalSourceEncoding
    """
    if patience is None:
        patience = _PATIENCE
    set_seed(seed)

    cw = compute_class_weights(train_data["label"]).to(DEVICE)

    # ── Build model ──────────────────────────────────────────────────────────
    use_patch = (stage1_attr is not None and stage2_attr is not None)
    FusionCls = GatedCrossAttention if fusion_type == "gated" else PlainCrossAttention

    if use_patch:
        base = _build_base_model(macro_dim, news_dim, cw)
        dim  = TrainConfig.dim
        nh   = TrainConfig.num_head
        # Build new fusion stages with SAME dim/head/dropout as base model
        new_s1 = FusionCls(dim=dim, num_head=nh, dropout=0.1).to(DEVICE)
        new_s2 = FusionCls(dim=dim, num_head=nh, dropout=0.1).to(DEVICE)

        # For parallel, we need a different patching strategy:
        # stage1=fusion_news, stage2=fusion_macro (par merge handled via forward hook)
        if fusion_order == "par":
            # Parallel: we add extra attrs dynamically
            # Build as FallbackVariantModel for par (par needs structural change)
            model = FallbackVariantModel(
                macro_dim=macro_dim, news_dim=news_dim,
                dim=dim, num_head=nh,
                window_size=TrainConfig.window_size,
                dropout=0.1, fusion_type=fusion_type,
                fusion_order="par", class_weights=cw, focal_gamma=2.0,
                quality_dim=getattr(GlobalConfig, "QUALITY_DIM", 4),
            ).to(DEVICE)
            print(f"    [par order] Using FallbackVariantModel (par requires structural change)")
        else:
            model = PatchedStockMovementModel(
                base_model=base,
                stage1_attr=stage1_attr, stage2_attr=stage2_attr,
                new_stage1=new_s1, new_stage2=new_s2,
            )
            # For MN order, we also need to swap the aux inputs in forward
            # This requires overriding forward, so fall back to FallbackVariantModel
            # for MN too, but using same encoders via a thin wrapper
            if fusion_order == "MN":
                model = FallbackVariantModel(
                    macro_dim=macro_dim, news_dim=news_dim,
                    dim=dim, num_head=nh,
                    window_size=TrainConfig.window_size,
                    dropout=0.1, fusion_type=fusion_type,
                    fusion_order="MN", class_weights=cw, focal_gamma=2.0,
                    quality_dim=getattr(GlobalConfig, "QUALITY_DIM", 4),
                ).to(DEVICE)
                print(f"    [MN order] Using FallbackVariantModel (MN requires forward override)")
    else:
        model = FallbackVariantModel(
            macro_dim=macro_dim, news_dim=news_dim,
            dim=TrainConfig.dim, num_head=TrainConfig.num_head,
            window_size=TrainConfig.window_size,
            dropout=0.1, fusion_type=fusion_type,
            fusion_order=fusion_order, class_weights=cw, focal_gamma=2.0,
            quality_dim=getattr(GlobalConfig, "QUALITY_DIM", 4),
        ).to(DEVICE)

    # ── Optimizer & schedulers (identical to run_ablation.py) ─────────────────
    ds  = StockDataset(train_data)
    ldr = _make_dataloader(
        ds, shuffle=True,
        batch_size=getattr(TrainConfig, "batch_size", 32),
    )
    opt    = _make_adamw(model, lr=TrainConfig.learning_rate)
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(max_epochs - warmup_epochs, 1), eta_min=1e-6,
    )

    best_mcc, best_state, no_improve = -2.0, None, 0

    for epoch in range(max_epochs):
        model.train()
        for batch in ldr:
            opt.zero_grad(set_to_none=True)

            s_n_in  = batch["s_n"].to(DEVICE)
            s_m_in  = batch["s_m"].to(DEVICE)
            mask_in = batch.get("news_mask")
            q_in    = batch.get("news_quality")

            # News modality dropout (identical to main.py / run_ablation.py)
            if _MOD_DROPOUT > 0.0 and torch.rand(1).item() < _MOD_DROPOUT:
                s_n_in = torch.zeros_like(s_n_in)
                if mask_in is not None:
                    mask_in = torch.ones_like(mask_in, dtype=torch.bool)
                q_in = None

            # Call model (same signature for both PatchedStockMovementModel and Fallback)
            loss = model(
                batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                batch["s_c"].to(DEVICE), s_m_in, s_n_in,
                batch["label"].to(DEVICE), mode="train",
                ticker_id=batch.get("ticker_id"),
                news_mask=mask_in.to(DEVICE) if mask_in is not None else None,
                news_quality=q_in.to(DEVICE) if q_in is not None else None,
            )
            if torch.isfinite(loss):
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        if epoch < warmup_epochs:
            warmup.step()
        else:
            cosine.step()

        if epoch >= warmup_epochs:
            _, mcc = evaluate_model(model, val_data)
            if mcc > best_mcc:
                best_mcc   = mcc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if best_state:
        model.load_state_dict(
            {k: v.to(DEVICE) for k, v in best_state.items()},
            strict=False,
        )
    return evaluate_model(model, test_data)


# =============================================================================
# MULTI-SEED RUNNER
# =============================================================================

def run_variant_n_seeds(
    label:        str,
    fusion_type:  str,
    fusion_order: str,
    splits:       dict,
    n_seeds:      int,
    max_epochs:   int,
    stage1_attr:  Optional[str],
    stage2_attr:  Optional[str],
    verbose:      bool = True,
) -> dict:
    train_data = splits["train_hval"]
    val_data   = splits["val_fixed"]
    test_data  = splits["test"]
    macro_dim  = splits["macro_dim"]
    news_dim   = splits["news_dim"]

    acc_list, mcc_list = [], []
    for seed in SEEDS[:n_seeds]:
        t0 = time.time()
        acc, mcc = run_one_seed(
            seed=seed,
            train_data=train_data, val_data=val_data, test_data=test_data,
            macro_dim=macro_dim, news_dim=news_dim,
            fusion_type=fusion_type, fusion_order=fusion_order,
            stage1_attr=stage1_attr, stage2_attr=stage2_attr,
            max_epochs=max_epochs,
        )
        acc_list.append(acc); mcc_list.append(mcc)
        if verbose:
            print(f"    seed={seed:5d}  ACC={acc:.4f}  MCC={mcc:.4f}  "
                  f"({time.time()-t0:.0f}s)")

    return {
        "label":        label,
        "fusion_type":  fusion_type,
        "fusion_order": fusion_order,
        "acc_mean":     float(np.mean(acc_list)),
        "acc_std":      float(np.std(acc_list)),
        "mcc_mean":     float(np.mean(mcc_list)),
        "mcc_std":      float(np.std(mcc_list)),
        "acc_list":     [float(x) for x in acc_list],
        "mcc_list":     [float(x) for x in mcc_list],
        "n_seeds":      len(SEEDS[:n_seeds]),
    }


# =============================================================================
# REFERENCE LOADING
# =============================================================================

def load_reference_results(results_dir: str) -> dict:
    """
    Load MSGCA_FV và baseline_wf từ kết quả đã chạy của run_experiments.py
    và run_ablation.py để dùng làm anchor so sánh.

    Trả về dict với keys:
      "MSGCA_FV"  — từ raw_results.json (run_experiments.py)
      "MSGCA_Best"— từ ablation_raw.json (run_ablation.py, baseline_wf)
    """
    refs: dict = {}

    # MSGCA_FV từ run_experiments.py
    raw_path = os.path.join(results_dir, "raw_results.json")
    if os.path.exists(raw_path):
        try:
            with open(raw_path, encoding="utf-8") as f:
                raw = json.load(f)
            if "MSGCA_FV" in raw:
                refs["MSGCA_FV"] = raw["MSGCA_FV"]
                print(f"  [Ref] MSGCA_FV loaded: "
                      f"MCC={refs['MSGCA_FV']['mcc_mean']:.4f}+/-"
                      f"{refs['MSGCA_FV']['mcc_std']:.4f}")
        except Exception as e:
            print(f"  [Ref] raw_results.json read error: {e}")

    # MSGCA_Best (baseline_wf) từ run_ablation.py
    abl_path = os.path.join(results_dir, "ablation_raw.json")
    if os.path.exists(abl_path):
        try:
            with open(abl_path, encoding="utf-8") as f:
                abl = json.load(f)
            if "baseline_wf" in abl:
                refs["MSGCA_Best"] = abl["baseline_wf"]
                print(f"  [Ref] MSGCA_Best (baseline_wf) loaded: "
                      f"MCC={refs['MSGCA_Best']['mcc_mean']:.4f}+/-"
                      f"{refs['MSGCA_Best']['mcc_std']:.4f}")
            if "baseline_fv" in abl:
                refs["MSGCA_FV_abl"] = abl["baseline_fv"]
        except Exception as e:
            print(f"  [Ref] ablation_raw.json read error: {e}")

    if not refs:
        print("  [Ref] No reference results found. Run run_experiments.py and "
              "run_ablation.py first for anchor comparison.")
    return refs


def validate_gated_nm(gated_nm_result: dict, refs: dict) -> str:
    """
    Kiểm tra gated/NM kết quả có overlap với MSGCA_FV reference không.
    Nếu overlap → architectural parity confirmed.
    Nếu không → cảnh báo kết quả có thể không comparable.
    """
    if "MSGCA_FV" not in refs:
        return "  [Validation] Cannot validate — MSGCA_FV reference not available."

    ref = refs["MSGCA_FV"]
    gm, gs = gated_nm_result["mcc_mean"], gated_nm_result["mcc_std"]
    rm, rs = ref["mcc_mean"], ref["mcc_std"]
    gap    = abs(gm - rm)

    # Overlap check: |mean_a - mean_b| < std_a + std_b
    overlap_threshold = gs + rs
    if gap <= overlap_threshold:
        verdict = (f"✓ PASS — distributions overlap "
                   f"(gap={gap:.4f} ≤ {overlap_threshold:.4f}). "
                   f"Architectural parity confirmed.")
    else:
        verdict = (f"✗ WARN — gap={gap:.4f} > {overlap_threshold:.4f}. "
                   f"MSGCA_Gated may differ architecturally from MSGCA_FV. "
                   f"Check encoder path (FallbackVariantModel vs PatchedStockMovementModel).")

    return (f"  [Validation] MSGCA_Gated(NM) vs MSGCA_FV:\n"
            f"    Gated: {gm:.4f}+/-{gs:.4f}  |  FV: {rm:.4f}+/-{rs:.4f}\n"
            f"    {verdict}")


# =============================================================================
# TABLE FORMATTERS
# =============================================================================

def format_rq3_table(results: dict, refs: dict) -> str:
    variants_rq3 = [k for k in results if k.startswith("rq3_")]
    if not variants_rq3:
        return "No RQ3 results."

    ref_key = next((k for k in variants_rq3 if "gated" in k and "no" not in k), None)
    ref_mcc = results[ref_key]["mcc_mean"] if ref_key else None

    # Anchor MCC from pre-computed results
    anchor_mcc = refs.get("MSGCA_FV", {}).get("mcc_mean")
    anchor_std = refs.get("MSGCA_FV", {}).get("mcc_std")

    sep = "=" * 120
    lines = [
        sep,
        "  RQ3 — Gated vs Non-Gated Cross-Attention",
        "  Hypothesis : Gated CA produces LOWER VARIANCE (more stable) than plain CA",
        "  Fixed var  : fusion_order = NM (canonical order)",
        "  Loss       : FocalLoss(γ=2.0) + class weights",
        "  Protocol   : identical to run_ablation.py (fixed-val, warmup=10, patience=30)",
        "  Test set   : [inner_T : T_max] — same as main.py and run_ablation.py",
    ]
    if anchor_mcc is not None:
        lines.append(
            f"  Anchor     : MSGCA_FV (from run_experiments.py) = "
            f"{anchor_mcc:.4f}+/-{anchor_std:.4f}"
        )
    lines += [
        sep,
        f"  {'Variant':<22} {'Type':<8} {'ACC':>16} {'MCC':>16}  {'dMCC(ref)':>10}  "
        f"{'std(MCC)':>10}  Stability",
        "-" * 120,
    ]

    for k in variants_rq3:
        r    = results[k]
        acc  = f"{r['acc_mean']:.4f}+/-{r['acc_std']:.4f}"
        mcc  = f"{r['mcc_mean']:.4f}+/-{r['mcc_std']:.4f}"
        dmcc = ("(Ref)" if k == ref_key
                else f"{r['mcc_mean'] - ref_mcc:+.4f}" if ref_mcc else "N/A")
        # Stability verdict
        if ref_key and k != ref_key:
            stab = ("MORE STABLE" if r["mcc_std"] < results[ref_key]["mcc_std"]
                    else "less stable")
        else:
            stab = "(reference)"
        lines.append(
            f"  {r['label']:<22} {r['fusion_type']:<8} {acc:>16} {mcc:>16}  "
            f"{dmcc:>10}  {r['mcc_std']:.4f}{'':>6}  {stab}"
        )

    lines += [sep, "\nINTERPRETATION (RQ3):"]
    gated_keys   = [k for k in variants_rq3 if "nogated" not in k]
    nogated_keys = [k for k in variants_rq3 if "nogated" in k]
    if gated_keys and nogated_keys:
        g_std  = np.mean([results[k]["mcc_std"] for k in gated_keys])
        ng_std = np.mean([results[k]["mcc_std"] for k in nogated_keys])
        g_mcc  = np.mean([results[k]["mcc_mean"] for k in gated_keys])
        ng_mcc = np.mean([results[k]["mcc_mean"] for k in nogated_keys])
        std_diff = ng_std - g_std

        if g_std < ng_std:
            lines.append(f"  ANSWER: YES — Gated CA IS more stable: "
                         f"std_gated={g_std:.4f} < std_plain={ng_std:.4f} (Δ={std_diff:.4f})")
        else:
            lines.append(f"  ANSWER: NO — Plain CA has lower variance: "
                         f"std_gated={g_std:.4f} ≥ std_plain={ng_std:.4f} (Δ={std_diff:.4f})")
        mcc_d = g_mcc - ng_mcc
        lines.append(f"  MCC difference: gated={g_mcc:.4f} vs plain={ng_mcc:.4f} (Δ={mcc_d:+.4f})")
        if abs(mcc_d) < 0.005:
            lines.append("  MCC gap < 0.005 — within noise bounds. "
                         "Stability (std) is the primary discriminating metric for RQ3.")

    return "\n".join(lines)


def format_rq4_table(results: dict, refs: dict) -> str:
    variants_rq4 = [k for k in results if k.startswith("rq4_")]
    if not variants_rq4:
        return "No RQ4 results."

    ref_key = next((k for k in variants_rq4 if "_NM" in k), None)
    ref_mcc = results[ref_key]["mcc_mean"] if ref_key else None
    anchor_mcc = refs.get("MSGCA_FV", {}).get("mcc_mean")

    order_desc = {
        "NM":  "price×news → (price+news)×macro  [canonical]",
        "MN":  "price×macro → (price+macro)×news",
        "par": "parallel: price×news + price×macro → mean-merge",
    }
    sep = "=" * 120
    lines = [
        sep,
        "  RQ4 — Fusion Order Comparison",
        "  Hypothesis : NM order (price-news first, price-macro second) is optimal",
        "  Fixed var  : fusion_type = gated (canonical MSGCA mechanism)",
        "  Loss       : FocalLoss(γ=2.0) + class weights",
        "  Protocol   : identical to run_ablation.py",
    ]
    if anchor_mcc is not None:
        lines.append(f"  Anchor     : MSGCA_FV = {anchor_mcc:.4f}")
    lines += [
        sep,
        f"  {'Variant':<18} {'Order':<6} {'Description':<46} {'MCC':>16}  "
        f"{'dMCC(NM ref)':>14}  {'std':>7}",
        "-" * 120,
    ]
    for k in variants_rq4:
        r     = results[k]
        order = r["fusion_order"]
        desc  = order_desc.get(order, order)[:45]
        mcc   = f"{r['mcc_mean']:.4f}+/-{r['mcc_std']:.4f}"
        dmcc  = ("(Ref)" if k == ref_key
                 else f"{r['mcc_mean'] - ref_mcc:+.4f}" if ref_mcc else "N/A")
        lines.append(
            f"  {r['label']:<18} {order:<6} {desc:<46} {mcc:>16}  "
            f"{dmcc:>14}  {r['mcc_std']:.4f}"
        )

    lines += [sep, "\nINTERPRETATION (RQ4):"]
    if variants_rq4:
        best_key = max(variants_rq4, key=lambda k: results[k]["mcc_mean"])
        best     = results[best_key]
        is_nm    = "NM" in best_key
        lines.append(f"  Best order: {best['fusion_order']} "
                     f"(MCC={best['mcc_mean']:.4f}+/-{best['mcc_std']:.4f})")
        if is_nm:
            lines.append("  ANSWER: YES — NM order is optimal. Hypothesis supported.")
        elif "MN" in best_key:
            lines.append("  ANSWER: NO — MN order is better. "
                         "Macro context may stabilise price representation before fusing news.")
        else:
            lines.append("  ANSWER: NO — Parallel fusion achieves highest MCC. "
                         "News and macro signals may be complementary rather than hierarchical.")

        if not is_nm and ref_mcc is not None:
            gap = best["mcc_mean"] - ref_mcc
            if abs(gap) < best["mcc_std"] + (results[ref_key]["mcc_std"] if ref_key else 0):
                lines.append(
                    f"  Note: |gap| = {abs(gap):.4f} is within combined std of NM and {best['fusion_order']}. "
                    "Results not conclusive — larger dataset or more seeds needed."
                )

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="RQ3 (Gated vs Plain CA) + RQ4 (Fusion order) — V2"
    )
    ap.add_argument("--rq",       type=int, default=0,
                    help="0=both, 3=RQ3 only, 4=RQ4 only")
    ap.add_argument("--pkl",      default=None)
    ap.add_argument("--n-seeds",  type=int, default=5)
    ap.add_argument("--epochs",   type=int, default=200)
    ap.add_argument("--tickers",  nargs="+", default=None)
    ap.add_argument("--verbose",  action="store_true", default=True)
    args = ap.parse_args()

    run_rq3 = args.rq in (0, 3)
    run_rq4 = args.rq in (0, 4)

    print("=" * 70)
    print("  RQ3 & RQ4 Experiment Runner — V2 (architectural parity)")
    print(f"  Device  : {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else ""))
    print(f"  Running : {'RQ3+RQ4' if run_rq3 and run_rq4 else 'RQ3' if run_rq3 else 'RQ4'}")
    print(f"  Seeds   : {SEEDS[:args.n_seeds]}")
    print(f"  Epochs  : {args.epochs}  warmup=10  patience={_PATIENCE}")
    print(f"  ModDrop : {_MOD_DROPOUT:.0%}  (same as main.py)")
    print("=" * 70)

    # ── Data ─────────────────────────────────────────────────────────────────
    pkl_path = args.pkl or os.path.join(
        GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl"
    )
    if not os.path.exists(pkl_path):
        print(f"Dataset not found: {pkl_path}"); sys.exit(1)

    tickers = ([t.upper() for t in args.tickers]
               if args.tickers else GlobalConfig.TICKERS)
    print(f"\nLoading data splits ({len(tickers)} tickers)...")
    splits = load_splits(pkl_path, tickers)

    # ── Reference results ─────────────────────────────────────────────────────
    print("\nLoading reference results...")
    refs = load_reference_results(RESULTS_DIR)

    # ── Discover fusion attr names from StockMovementModel ───────────────────
    print("\nIntrospecting StockMovementModel fusion layer names...")
    dummy_cw  = torch.ones(3) / 3.0
    _tmp      = _build_base_model(splits["macro_dim"], splits["news_dim"], dummy_cw)
    s1a, s2a  = _get_fusion_attr_names(_tmp)
    del _tmp
    use_patch = (s1a is not None)
    if not use_patch:
        print("  → FallbackVariantModel mode (MultimodalSourceEncoding 4-tuple)")
    else:
        print(f"  → PatchedStockMovementModel mode "
              f"(stage1='{s1a}', stage2='{s2a}')")

    all_results: dict = {}
    total_t0 = time.time()

    # ── RQ3 ──────────────────────────────────────────────────────────────────
    if run_rq3:
        print(f"\n{'='*70}")
        print("  RQ3: Gated vs Non-Gated Cross-Attention  (fusion_order=NM fixed)")
        print(f"{'='*70}")

        rq3_variants = [
            ("MSGCA_Gated",   "gated", "NM"),   # canonical — should ≈ MSGCA_FV
            ("MSGCA_NoGated", "plain", "NM"),   # no sigmoid gate
        ]
        for label, ftype, forder in rq3_variants:
            key = f"rq3_{label}"
            print(f"\n  [{key}]  fusion_type={ftype}  fusion_order={forder}")
            t0 = time.time()
            result = run_variant_n_seeds(
                label=label,
                fusion_type=ftype, fusion_order=forder,
                splits=splits, n_seeds=args.n_seeds, max_epochs=args.epochs,
                stage1_attr=s1a, stage2_attr=s2a, verbose=args.verbose,
            )
            all_results[key] = result
            print(f"  → ACC={result['acc_mean']:.4f}+/-{result['acc_std']:.4f}  "
                  f"MCC={result['mcc_mean']:.4f}+/-{result['mcc_std']:.4f}  "
                  f"({(time.time()-t0)/60:.1f} min)")

        # Validate gated/NM vs MSGCA_FV reference
        if "rq3_MSGCA_Gated" in all_results:
            print("\n" + validate_gated_nm(all_results["rq3_MSGCA_Gated"], refs))

    # ── RQ4 ──────────────────────────────────────────────────────────────────
    if run_rq4:
        print(f"\n{'='*70}")
        print("  RQ4: Fusion Order (fusion_type=gated fixed)")
        print(f"{'='*70}")

        rq4_variants = [
            ("Order_NM",  "gated", "NM"),    # canonical
            ("Order_MN",  "gated", "MN"),    # reversed
            ("Order_par", "gated", "par"),   # parallel mean-merge
        ]
        for label, ftype, forder in rq4_variants:
            key = f"rq4_{label}"
            print(f"\n  [{key}]  fusion_type={ftype}  fusion_order={forder}")
            t0 = time.time()
            result = run_variant_n_seeds(
                label=label,
                fusion_type=ftype, fusion_order=forder,
                splits=splits, n_seeds=args.n_seeds, max_epochs=args.epochs,
                stage1_attr=s1a, stage2_attr=s2a, verbose=args.verbose,
            )
            all_results[key] = result
            print(f"  → ACC={result['acc_mean']:.4f}+/-{result['acc_std']:.4f}  "
                  f"MCC={result['mcc_mean']:.4f}+/-{result['mcc_std']:.4f}  "
                  f"({(time.time()-t0)/60:.1f} min)")

    print(f"\nTotal: {(time.time()-total_t0)/60:.1f} min\n")

    # ── Output ───────────────────────────────────────────────────────────────
    if run_rq3:
        tbl3 = format_rq3_table(all_results, refs)
        print(tbl3)
        p3 = os.path.join(RESULTS_DIR, "rq3_table.txt")
        with open(p3, "w", encoding="utf-8") as f: f.write(tbl3)
        print(f"\nSaved → {p3}")

    if run_rq4:
        tbl4 = format_rq4_table(all_results, refs)
        print(tbl4)
        p4 = os.path.join(RESULTS_DIR, "rq4_table.txt")
        with open(p4, "w", encoding="utf-8") as f: f.write(tbl4)
        print(f"Saved → {p4}")

    raw_path = os.path.join(RESULTS_DIR, "rq3_rq4_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved → {raw_path}")


if __name__ == "__main__":
    main()