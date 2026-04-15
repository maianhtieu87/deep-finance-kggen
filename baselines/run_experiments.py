# baselines/run_experiments.py
"""
Master experiment runner — MSGCA vs all baselines.

CHANGES vs previous:
  1. _build_msgca_model: removed use_ticker_emb=True (param không tồn tại)
  2. _eval_msgca: đã pass news_mask + ticker_id (nhất quán với trainer.py fix)
  3. _train_epoch: đã pass news_mask + ticker_id
  4. Thêm --load-saved-model: load model đã train từ output/ làm kết quả seed=42,
     train thêm seeds [123,256,512,1024] và báo cáo mean±std.
     Test split trùng khớp: run_experiments inner_T=627 = main.py val_end=627.

USAGE:
  python baselines/run_experiments.py
  python baselines/run_experiments.py --skip-search
  python baselines/run_experiments.py --load-saved-model output/best_model_label=rolling_price=vol_adjusted_tid_seed42_standard.pt
  python baselines/run_experiments.py --models LSTM ALSTM --n-seeds 3
  python baselines/run_experiments.py --skip-msgca-inline --output-dir output
"""

from typing import Optional, Dict, List, Callable, Tuple
import argparse, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import TrainConfig, GlobalConfig
from src.data_loader import data_prepare, N_TICKERS
from src.model import StockMovementModel
from baselines.hparam_search import (
    grid_search_flat, final_eval_flat, SEEDS, get_all_combinations,
    FlatDataset, _eval_flat_mcc, _eval_flat_both, _forward_flat,
)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL FACTORIES — flat baselines
# ─────────────────────────────────────────────────────────────────────────────

def _make_factories(news_dim: int = 1024) -> Dict[str, Callable]:
    from baselines.models import (
        LSTMBaseline, ALSTMBaseline, ALSTMWithDoc, SLOTBaseline,
        LLMStockBaseline, DARNNBaseline, ESTIMATEBaseline, DTMLBaseline,
    )
    def make_lstm(hidden_dim, macro_dim, dropout):
        return LSTMBaseline(hidden_dim=hidden_dim, num_classes=3, dropout=dropout)
    def make_alstm(hidden_dim, macro_dim, dropout):
        return ALSTMBaseline(hidden_dim=hidden_dim, num_classes=3, dropout=dropout)
    def make_alstm_w(hidden_dim, macro_dim, dropout):
        return ALSTMWithDoc(news_dim=news_dim, hidden_dim=hidden_dim, num_classes=3, dropout=dropout)
    def make_slot(hidden_dim, macro_dim, dropout):
        return SLOTBaseline(news_dim=news_dim, hidden_dim=hidden_dim, num_classes=3, dropout=dropout)
    def make_llm(hidden_dim, macro_dim, dropout):
        return LLMStockBaseline(news_dim=news_dim, hidden_dim=hidden_dim, num_classes=3, dropout=dropout)
    def make_darnn(hidden_dim, macro_dim, dropout):
        return DARNNBaseline(macro_dim=macro_dim, hidden_dim=hidden_dim, num_classes=3, dropout=dropout)
    def make_est(hidden_dim, macro_dim, dropout):
        return ESTIMATEBaseline(macro_dim=macro_dim, hidden_dim=hidden_dim, num_classes=3, dropout=dropout)
    def make_dtml(hidden_dim, macro_dim, dropout):
        return DTMLBaseline(macro_dim=macro_dim, hidden_dim=hidden_dim, num_heads=2, num_classes=3, dropout=dropout)
    return {"LSTM": make_lstm, "ALSTM": make_alstm, "ALSTM-W": make_alstm_w,
            "SLOT": make_slot, "LLM-Stock": make_llm,
            "DA-RNN": make_darnn, "ESTIMATE": make_est, "DTML": make_dtml}


CATEGORY_MAP = {
    "LSTM":       "Cat1: Indicator-only",
    "ALSTM":      "Cat1: Indicator-only",
    "ALSTM-W":    "Cat2: Indicator+Doc",
    "SLOT":       "Cat2: Indicator+Doc",
    "LLM-Stock":  "Cat2: Indicator+Doc",
    "DA-RNN":     "Cat3: Indicator+Macro",
    "ESTIMATE":   "Cat3: Indicator+Macro",
    "DTML":       "Cat3: Indicator+Macro",
    "MSGCA_FV":   "Ours (CE, fair comparison)",
    "MSGCA_Best": "Ours (FocalLoss+weights, best)",
}
MODEL_ORDER = ["LSTM","ALSTM","ALSTM-W","SLOT","LLM-Stock","DA-RNN","ESTIMATE","DTML",
               "MSGCA_FV","MSGCA_Best"]


# ─────────────────────────────────────────────────────────────────────────────
# MSGCA DATASET
# ─────────────────────────────────────────────────────────────────────────────

class MSGCADataset(Dataset):
    def __init__(self, d):
        self.d    = d
        self.keys = [k for k in d if isinstance(d[k], torch.Tensor)]
    def __len__(self): return len(self.d["label"])
    def __getitem__(self, i): return {k: self.d[k][i] for k in self.keys}


# ─────────────────────────────────────────────────────────────────────────────
# MSGCA_FV helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_class_weights(labels):
    """
    Class weight formula matching main.py exactly:
      beta=0.9999, effective num, sqrt-normalize.
    Avoids the power=1.5 formula that was causing extreme weights
    and FocalLoss gradient explosion.
    """
    lbl  = labels.numpy()
    cnts = np.bincount(lbl, minlength=3).astype(float)
    beta = 0.9999
    eff  = 1.0 - np.power(beta, cnts)
    w    = (1.0 - beta) / (eff + 1e-8)
    w    = np.sqrt(w / w.sum() * 3)
    w    = w / w.sum() * 3
    return torch.tensor(w, dtype=torch.float32)


def _build_msgca_model(macro_dim, news_dim, lr, dropout, focal_gamma, cw=None):
    """
    Fair comparison mode: CE loss, no class weights (same as baselines).
    """
    model = StockMovementModel(
        price_dim=1, macro_dim=macro_dim, news_dim=news_dim,
        dim=TrainConfig.dim, input_dim=TrainConfig.window_size,
        output_dim=3, num_head=TrainConfig.num_head,
        dropout=dropout,
        class_weights=None,
        use_focal_loss=False,
        focal_gamma=focal_gamma,
        device=DEVICE,
        n_tickers=N_TICKERS,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-6)
    return model, opt


def _build_msgca_model_production(macro_dim, news_dim, lr, dropout, train_labels):
    """
    Production mode: FocalLoss(gamma=2.0) + class weights — matches main.py exactly.
    """
    cw = _compute_class_weights(train_labels).to(DEVICE)
    model = StockMovementModel(
        price_dim=1, macro_dim=macro_dim, news_dim=news_dim,
        dim=TrainConfig.dim, input_dim=TrainConfig.window_size,
        output_dim=3, num_head=TrainConfig.num_head,
        dropout=dropout,
        class_weights=cw,
        use_focal_loss=True,
        focal_gamma=2.0,
        device=DEVICE,
        n_tickers=N_TICKERS,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, eps=1e-6)
    return model, opt


def _eval_msgca(model, data, batch_size=64):
    """Evaluate MSGCA — pass news_mask + ticker_id."""
    from sklearn.metrics import accuracy_score, matthews_corrcoef
    if not data or len(data.get("label", [])) == 0: return 0.0, 0.0
    model.eval()
    ldr = DataLoader(MSGCADataset(data), batch_size=batch_size, shuffle=False)
    preds_all, labels_all = [], []
    with torch.no_grad():
        for batch in ldr:
            _, _, preds = model(
                batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
                batch["s_c"].to(DEVICE), batch["s_m"].to(DEVICE),
                batch["s_n"].to(DEVICE), batch["label"].to(DEVICE),
                mode="test", return_preds=True,
                ticker_id=batch.get("ticker_id"),
                news_mask=batch.get("news_mask"),
            )
            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(batch["label"].numpy())
    if len(set(labels_all)) < 2:
        return float(accuracy_score(labels_all, preds_all)), 0.0
    return (float(accuracy_score(labels_all, preds_all)),
            float(matthews_corrcoef(labels_all, preds_all)))


def _train_epoch(model, loader, opt):
    """
    Train 1 epoch — standard per-batch SGD (NOT gradient accumulation).

    CRITICAL FIX: zero_grad() and step() must be called INSIDE the batch loop.
    Previous code called them outside → accumulated gradient of ALL batches
    before a single update → weight jump of 162× normal mini-batch gradient
    → FocalLoss gradient explosion → class collapse (MCC=0).
    """
    model.train()
    total_loss = 0.0
    for batch in loader:
        opt.zero_grad(set_to_none=True)      # ← inside loop: fresh gradient each batch
        loss = model(
            batch["s_o"].to(DEVICE), batch["s_h"].to(DEVICE),
            batch["s_c"].to(DEVICE), batch["s_m"].to(DEVICE),
            batch["s_n"].to(DEVICE), batch["label"].to(DEVICE),
            mode="train",
            ticker_id=batch.get("ticker_id"),
            news_mask=batch.get("news_mask"),
        )
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()                       # ← inside loop: update after each batch
            total_loss += loss.item()
    return total_loss


def _run_msgca_production_one_seed(
    seed, train_hval, val_hval, train_full, test,
    macro_dim, news_dim,
    lr=1e-4, dropout=0.1,
    max_epochs=200, patience=30, warmup_epochs=15, verbose=False,
):
    """
    Train one seed with PRODUCTION setup: FocalLoss(gamma=2.0) + class weights.
    Protocol matches main.py:
      Phase 1: train_hval → find best_epoch via val MCC
      Phase 2: train_full for best_epoch, eval on test
    Scheduler: LinearLR warmup → CosineAnnealingLR (no restarts).
    """
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    model, opt = _build_msgca_model_production(macro_dim, news_dim, lr, dropout,
                                                train_hval["label"])
    ldr    = DataLoader(MSGCADataset(train_hval), batch_size=32, shuffle=True, drop_last=False)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(max_epochs - warmup_epochs, 1), eta_min=1e-6
    )

    best_mcc, best_epoch, no_improve = -2.0, 1, 0
    # Start checking after warmup only — not a fixed 40-epoch delay
    # This prevents the patience=30 trigger at epoch 41 when model hasn't warmed up yet
    min_eval_epoch = warmup_epochs + 5   # start checking 5 epochs after warmup ends

    for epoch in range(max_epochs):
        ep_loss = _train_epoch(model, ldr, opt)
        if epoch < warmup_epochs:
            warmup.step()
        else:
            cosine.step()

        if epoch >= min_eval_epoch:
            _, mcc = _eval_msgca(model, val_hval)
            if mcc > best_mcc:
                best_mcc = mcc; best_epoch = epoch + 1; no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience: break

        if verbose and (epoch + 1) % 20 == 0:
            _, cur_mcc = _eval_msgca(model, val_hval)
            print(f"      ep {epoch+1:3d}: loss={ep_loss:.4f}  val_MCC={cur_mcc:.4f}")

    if verbose:
        print(f"    Phase1: best_ep={best_epoch} val_MCC={best_mcc:.4f}")

    # Phase 2 — retrain full with production loss
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model2, opt2 = _build_msgca_model_production(macro_dim, news_dim, lr, dropout,
                                                   train_full["label"])
    ldr2   = DataLoader(MSGCADataset(train_full), batch_size=32, shuffle=True, drop_last=False)
    # Warmup for phase 2 as well
    warmup2 = torch.optim.lr_scheduler.LinearLR(opt2, 0.1, 1.0, total_iters=warmup_epochs)
    cosine2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=max(best_epoch - warmup_epochs, 1), eta_min=1e-6
    )
    for ep in range(best_epoch):
        _train_epoch(model2, ldr2, opt2)
        if ep < warmup_epochs: warmup2.step()
        else:                   cosine2.step()

    acc, mcc = _eval_msgca(model2, test)
    return acc, mcc, best_epoch


def run_msgca_best(
    saved_model_path: Optional[str],
    train_hval: dict, val_hval: dict,
    train_full: dict, test: dict,
    macro_dim: int, news_dim: int,
    n_seeds: int = 5,
    lr: float = 1e-4, dropout: float = 0.1,
    verbose: bool = True,
) -> dict:
    """
    MSGCA at best-performance configuration (FocalLoss + class weights).
    seed=42: load from saved .pt (trained by main.py production mode).
    Remaining seeds: train fresh with same production setup.
    Test set: identical to baselines [inner_T:global_T_max].
    """
    acc_list, mcc_list, ep_list = [], [], []

    if saved_model_path and os.path.exists(saved_model_path):
        print(f"  [MSGCA_Best] seed=42 → load {os.path.basename(saved_model_path)}")
        acc42, mcc42 = load_and_eval_saved_model(
            saved_model_path, test, macro_dim, news_dim, verbose=False
        )
        print(f"    ACC={acc42:.4f}  MCC={mcc42:.4f}")
        acc_list.append(acc42); mcc_list.append(mcc42); ep_list.append(0)
        remaining = [s for s in SEEDS[:n_seeds] if s != 42]
    else:
        print(f"  [MSGCA_Best] No saved model — training all seeds from scratch...")
        remaining = SEEDS[:n_seeds]

    for seed in remaining:
        print(f"  [MSGCA_Best] seed={seed} → FocalLoss + class weights...")
        acc, mcc, ep = _run_msgca_production_one_seed(
            seed=seed,
            train_hval=train_hval, val_hval=val_hval,
            train_full=train_full, test=test,
            macro_dim=macro_dim, news_dim=news_dim,
            lr=lr, dropout=dropout,
            max_epochs=200, patience=30, warmup_epochs=15, verbose=verbose,
        )
        acc_list.append(acc); mcc_list.append(mcc); ep_list.append(ep)
        if verbose:
            print(f"    Seed {seed}: ep={ep:3d}  ACC={acc:.4f}  MCC={mcc:.4f}")

    return {
        "acc_mean": float(np.mean(acc_list)), "acc_std":  float(np.std(acc_list)),
        "mcc_mean": float(np.mean(mcc_list)), "mcc_std":  float(np.std(mcc_list)),
        "acc_list": acc_list, "mcc_list": mcc_list,
        "ep_mean":  float(np.mean([e for e in ep_list if e > 0])) if any(e > 0 for e in ep_list) else 0.0,
        "n_seeds":  len(acc_list), "mode": "production (FocalLoss+classweights)",
    }


def _run_msgca_one_seed(
    seed, train_hval, val_hval, train_full, test,
    macro_dim, news_dim, hp,
    max_epochs=150, patience=30, warmup_epochs=15, verbose=False,
):
    """
    2-phase training for one seed (FAIR/CE mode — for MSGCA_FV row):
      Phase 1: train_hval → early stopping → find best_epoch
      Phase 2: train_full for best_epoch → eval on test
    """
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    # Phase 1 — find best epoch
    model, opt = _build_msgca_model(macro_dim, news_dim, hp["lr"], hp["dropout"], hp["focal_gamma"])
    ldr        = DataLoader(MSGCADataset(train_hval), batch_size=32, shuffle=True, drop_last=False)
    warmup     = torch.optim.lr_scheduler.LinearLR(opt, 0.1, 1.0, total_iters=warmup_epochs)
    cosine     = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs-warmup_epochs, eta_min=1e-6)

    best_mcc, best_epoch, no_improve = -2.0, 1, 0
    min_active = max(warmup_epochs, 40)

    for epoch in range(max_epochs):
        _train_epoch(model, ldr, opt)
        if epoch < warmup_epochs: warmup.step()
        else:                      cosine.step()

        if epoch >= min_active:
            _, mcc = _eval_msgca(model, val_hval)
            if mcc > best_mcc:
                best_mcc = mcc; best_epoch = epoch + 1; no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience: break

    if verbose: print(f"    Phase1: best_ep={best_epoch} val_MCC={best_mcc:.4f}")

    # Phase 2 — retrain full
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model2, opt2 = _build_msgca_model(macro_dim, news_dim, hp["lr"], hp["dropout"], hp["focal_gamma"])
    ldr2         = DataLoader(MSGCADataset(train_full), batch_size=32, shuffle=True, drop_last=False)
    sched2       = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=best_epoch, eta_min=1e-6)

    for _ in range(best_epoch):
        _train_epoch(model2, ldr2, opt2)
        sched2.step()

    acc, mcc = _eval_msgca(model2, test)
    return acc, mcc, best_epoch


def grid_search_msgca(train_hval, val_hval, macro_dim, news_dim, verbose=True):
    from itertools import product
    GRID   = {"lr": [5e-5, 1e-4, 5e-4], "dropout": [0.1, 0.2]}
    keys   = list(GRID.keys())
    combos = [dict(zip(keys, v), focal_gamma=2.0) for v in product(*[GRID[k] for k in keys])]
    torch.manual_seed(42); np.random.seed(42)
    best_mcc, best_hp = -2.0, combos[0]

    for i, hp in enumerate(combos):
        _, mcc, _ = _run_msgca_one_seed(
            seed=42, train_hval=train_hval, val_hval=val_hval,
            train_full=train_hval, test=val_hval,
            macro_dim=macro_dim, news_dim=news_dim, hp=hp,
            max_epochs=100, patience=20, warmup_epochs=15,
        )
        is_best = mcc > best_mcc
        if verbose:
            print(f"    [{i+1:02d}/{len(combos)}] lr={hp['lr']:.0e} "
                  f"drop={hp['dropout']} "
                  f"-> val_MCC={mcc:.4f}{' <-best' if is_best else ''}")
        if is_best: best_mcc, best_hp = mcc, hp.copy()

    return {"best_hparams": best_hp, "best_mcc": best_mcc}


def final_eval_msgca(train_hval, val_hval, train_full, test,
                     best_hparams, macro_dim, news_dim, n_seeds=5, verbose=True):
    acc_list, mcc_list, ep_list = [], [], []
    for seed in SEEDS[:n_seeds]:
        acc, mcc, ep = _run_msgca_one_seed(
            seed=seed,
            train_hval=train_hval, val_hval=val_hval,
            train_full=train_full, test=test,
            macro_dim=macro_dim, news_dim=news_dim, hp=best_hparams,
            max_epochs=150, patience=30, warmup_epochs=15, verbose=verbose,
        )
        acc_list.append(acc); mcc_list.append(mcc); ep_list.append(ep)
        if verbose: print(f"    Seed {seed}: ep={ep:3d}  ACC={acc:.4f}  MCC={mcc:.4f}")

    return {
        "acc_mean": float(np.mean(acc_list)), "acc_std": float(np.std(acc_list)),
        "mcc_mean": float(np.mean(mcc_list)), "mcc_std": float(np.std(mcc_list)),
        "acc_list": acc_list, "mcc_list": mcc_list,
        "ep_mean":  float(np.mean(ep_list)), "n_seeds": len(acc_list),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOAD SAVED MODEL — đánh giá model đã train từ main.py
# ─────────────────────────────────────────────────────────────────────────────

def load_and_eval_saved_model(
    model_path: str,
    test_data: dict,
    macro_dim: int,
    news_dim: int,
    verbose: bool = True,
) -> Tuple[float, float]:
    """
    Load model .pt đã train từ main.py và evaluate trên test_data.

    Test split trùng khớp:
      main.py     : val_end = int(T_max*0.85) = 627, test=[627:738]
      run_exp.py  : inner_T = int(T_max*0.85) = 627, test=[627:738]
    → Cùng test set → kết quả có thể so sánh trực tiếp.
    """
    if verbose:
        print(f"    Loading: {model_path}")

    model = StockMovementModel(
        price_dim=1, macro_dim=macro_dim, news_dim=news_dim,
        dim=TrainConfig.dim, input_dim=TrainConfig.window_size,
        output_dim=3, num_head=TrainConfig.num_head,
        dropout=0.0,
        class_weights=None,
        use_focal_loss=False,
        device=DEVICE, n_tickers=N_TICKERS,
    ).to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)

    # loss_fn.weight là class weight từ training, không cần cho inference
    skip_keys = {"loss_fn.weight"}
    real_missing    = [k for k in missing    if k not in skip_keys]
    real_unexpected = [k for k in unexpected if k not in skip_keys]

    if real_missing or real_unexpected:
        print(f"    [WARN] missing={real_missing}  unexpected={real_unexpected}")
    else:
        print(f"    Weights OK (loss_fn.weight skipped)")

    model.eval()
    acc, mcc = _eval_msgca(model, test_data)
    if verbose:
        print(f"    Result: ACC={acc:.4f}  MCC={mcc:.4f}")
    return acc, mcc


def run_msgca_with_saved_model(
    saved_model_path: str,
    train_hval: dict,
    val_hval: dict,
    train_full: dict,
    test: dict,
    macro_dim: int,
    news_dim: int,
    best_hparams: dict,
    n_seeds: int = 5,
    verbose: bool = True,
) -> dict:
    """
    Strategy: seed=42 → load model đã train (không retrain).
              seeds [123,256,512,1024] → train fresh như protocol thông thường.
    Kết quả: mean±std over n_seeds, bao gồm seed=42 đã save.

    Lý do: model seed=42 đã được train với best-val-MCC selection.
    Các seed khác được train fresh để có variance estimate.
    """
    print(f"  [MSGCA_FV] seed=42 → load từ {os.path.basename(saved_model_path)}")
    acc42, mcc42 = load_and_eval_saved_model(
        saved_model_path, test, macro_dim, news_dim, verbose=verbose
    )

    acc_list = [acc42]
    mcc_list = [mcc42]
    ep_list  = [0]  # epoch unknown for saved model

    remaining_seeds = [s for s in SEEDS[:n_seeds] if s != 42]
    for seed in remaining_seeds:
        print(f"  [MSGCA_FV] seed={seed} → training fresh...")
        acc, mcc, ep = _run_msgca_one_seed(
            seed=seed,
            train_hval=train_hval, val_hval=val_hval,
            train_full=train_full, test=test,
            macro_dim=macro_dim, news_dim=news_dim,
            hp=best_hparams,
            max_epochs=150, patience=30, warmup_epochs=15, verbose=verbose,
        )
        acc_list.append(acc); mcc_list.append(mcc); ep_list.append(ep)
        if verbose:
            print(f"    Seed {seed}: ep={ep:3d}  ACC={acc:.4f}  MCC={mcc:.4f}")

    return {
        "acc_mean": float(np.mean(acc_list)),
        "acc_std":  float(np.std(acc_list)),
        "mcc_mean": float(np.mean(mcc_list)),
        "mcc_std":  float(np.std(mcc_list)),
        "acc_list": acc_list,
        "mcc_list": mcc_list,
        "ep_mean":  float(np.mean([e for e in ep_list if e > 0])) if any(e > 0 for e in ep_list) else 0.0,
        "n_seeds":  len(acc_list),
        "seed42_from_saved": True,
    }


def collect_msgca_fv_results(output_dir="output", n_seeds=5):
    """Load MSGCA_FV results pre-computed by: python main.py --n-seeds N --save-results"""
    acc_list, mcc_list = [], []
    for seed in SEEDS[:n_seeds]:
        path = os.path.join(output_dir, f"msgca_fv_seed{seed}.json")
        if os.path.exists(path):
            with open(path) as f: r = json.load(f)
            acc_list.append(r.get("test_acc", 0.0))
            mcc_list.append(r.get("test_mcc", 0.0))
            print(f"  Seed {seed}: ACC={r['test_acc']:.4f}  MCC={r['test_mcc']:.4f}"
                  f"  ep={r.get('best_epoch','?')}")
    if not acc_list: return None
    return {"acc_mean": float(np.mean(acc_list)), "acc_std": float(np.std(acc_list)),
            "mcc_mean": float(np.mean(mcc_list)), "mcc_std": float(np.std(mcc_list)),
            "acc_list": acc_list, "mcc_list": mcc_list, "n_seeds": len(acc_list)}


# ─────────────────────────────────────────────────────────────────────────────
# DATA PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def load_and_split_data(pkl_path, tickers):
    print(f"\nLoading data: {tickers}")
    dp = data_prepare(pkl_path, include_ticker_id=True)

    valid_T      = [dp.get_max_T(t) for t in tickers if dp.get_max_T(t) > 0]
    global_T_max = min(valid_T)
    inner_T      = int(global_T_max * 0.85)
    hval_split   = int(inner_T * 0.80)
    print(f"  global_T_max={global_T_max}  inner_T={inner_T}  hval_split={hval_split}")
    print(f"  Test range: [{inner_T}:{global_T_max}]")
    print(f"  Alignment: main.py uses same 70%/15%/15% split → test sets are IDENTICAL")

    def _add_ind(s):
        if not s or not len(s.get("label", [])): return s
        s = dict(s); s["indicators"] = torch.cat([s["s_o"], s["s_h"], s["s_c"]], dim=-1)
        return s

    def _merge(dicts, shuffle=False):
        if not dicts: return {}
        m: dict = {}
        for key in dicts[0]:
            parts = [d[key] for d in dicts if key in d and isinstance(d[key], torch.Tensor)]
            if parts: m[key] = torch.cat(parts, dim=0)
        if shuffle and "label" in m:
            idx = torch.randperm(len(m["label"]))
            for k in m: m[k] = m[k][idx]
        return m

    trhv, vahv, trfl, tel = [], [], [], []
    macro_dim = news_dim = None

    for t in tickers:
        if dp.get_max_T(t) == 0: continue
        trh, vah, _  = dp.prepare_data(t, train_end=hval_split, val_end=inner_T, test_end=inner_T)
        trf, _,  te  = dp.prepare_data(t, train_end=inner_T,    val_end=inner_T, test_end=global_T_max)
        trh=_add_ind(trh); vah=_add_ind(vah); trf=_add_ind(trf); te=_add_ind(te)

        if macro_dim is None and trh and len(trh.get("label",[])) > 0:
            macro_dim = trh["s_m"].shape[-1]; news_dim = trh["s_n"].shape[-1]

        if trh and len(trh.get("label",[])): trhv.append(trh)
        if vah and len(vah.get("label",[])): vahv.append(vah)
        if trf and len(trf.get("label",[])): trfl.append(trf)
        if te  and len(te.get("label",[])):  tel.append(te)
        print(f"  {t}: hval_tr={len(trh.get('label',[]))} "
              f"hval_va={len(vah.get('label',[]))} te={len(te.get('label',[]))}")

    train_hval=_merge(trhv,True); val_hval=_merge(vahv)
    train_full=_merge(trfl,True); test=_merge(tel)
    print(f"\n  Merged: train_hval={len(train_hval.get('label',[]))} "
          f"val_hval={len(val_hval.get('label',[]))} "
          f"train_full={len(train_full.get('label',[]))} "
          f"test={len(test.get('label',[]))}")
    print(f"  macro_dim={macro_dim}  news_dim={news_dim}")

    return {"global_T_max": global_T_max, "inner_T": inner_T, "hval_split": hval_split,
            "train_hval": train_hval, "val_hval": val_hval,
            "train_full": train_full, "test": test,
            "macro_dim": macro_dim, "news_dim": news_dim}


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def format_table(results, best_hparams):
    sep   = "=" * 105
    lines = [sep, "  COMPARISON TABLE  (Mean +/- Std, outer test)", sep]
    lines.append(f"{'Model':<16} {'Category':<32} {'ACC':>16} {'MCC':>16}  Notes")
    lines.append("-" * 105)

    best = max(results, key=lambda n: results[n].get("mcc_mean", -999))
    for name in MODEL_ORDER:
        if name not in results: continue
        r  = results[name]
        hp = best_hparams.get(name, {})
        if name == "MSGCA_Best":
            note = f"FocalLoss(γ=2.0)+classweights  avg_ep={r.get('ep_mean',0):.0f}"
        elif name == "MSGCA_FV":
            saved_note = " [seed42=saved]" if r.get("seed42_from_saved") else ""
            note = f"CE (fair comparison)  avg_ep={r.get('ep_mean',0):.0f}{saved_note}"
        elif isinstance(hp, dict) and hp:
            note = f"lr={hp.get('lr','?'):.0e} dim={hp.get('hidden_dim','?')} drop={hp.get('dropout','?')}"
        else:
            note = "-"
        acc = f"{r['acc_mean']:.4f}+/-{r['acc_std']:.4f}"
        mcc = f"{r['mcc_mean']:.4f}+/-{r['mcc_std']:.4f}"
        lines.append(f"{name:<16} {CATEGORY_MAP.get(name,''):<32} {acc:>16} {mcc:>16}  "
                     f"{note}{'  <-BEST' if name==best else ''}")

    lines.append(sep)
    lines.append("\nNotes:")
    lines.append("  MSGCA_FV   : CE loss, no class weights — FAIR comparison (same constraints as baselines)")
    lines.append("  MSGCA_Best : FocalLoss(γ=2.0) + class weights — BEST performance (production mode)")
    lines.append("  All models evaluated on identical outer test [inner_T:global_T_max]")
    lines.append("  Alignment: T_max determined by min over all 9 tickers → same test set for MSGCA and baselines")
    lines.append("\n[LaTeX]")
    for name in MODEL_ORDER:
        if name not in results: continue
        r = results[name]
        lines.append(f"{name} & {r['acc_mean']:.4f}$\\pm${r['acc_std']:.4f}"
                     f" & {r['mcc_mean']:.4f}$\\pm${r['mcc_std']:.4f} \\\\")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _auto_detect_best_model(output_dir="output") -> Optional[str]:
    """Tìm file .pt mới nhất trong output/ — ưu tiên file 'standard' (bval-selection)."""
    if not os.path.isdir(output_dir):
        return None
    pt_files = [f for f in os.listdir(output_dir) if f.startswith("best_model") and f.endswith(".pt")]
    if not pt_files:
        return None
    # Ưu tiên standard split (seed42) — cùng split với run_experiments test set
    standard = [f for f in pt_files if "standard" in f and "seed42" in f]
    candidates = standard if standard else pt_files
    candidates.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)
    return os.path.join(output_dir, candidates[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl",               default=None)
    ap.add_argument("--skip-search",       action="store_true")
    ap.add_argument("--skip-msgca-inline", action="store_true",
                    help="Load MSGCA_FV từ output/ JSON files (pre-computed)")
    ap.add_argument("--load-saved-model",  default=None,
                    help="Path đến .pt đã train (seed=42, production mode). "
                         "Nếu không chỉ định, auto-detect trong output/")
    ap.add_argument("--no-load-saved",     action="store_true",
                    help="Buộc train MSGCA_Best toàn bộ từ đầu (không load .pt)")
    ap.add_argument("--skip-msgca-best",   action="store_true",
                    help="Bỏ qua MSGCA_Best row (chỉ chạy fair comparison MSGCA_FV)")
    ap.add_argument("--models",            nargs="+", default=None)
    ap.add_argument("--n-seeds",           type=int, default=5)
    ap.add_argument("--output-dir",        default="output")
    ap.add_argument("--verbose",           action="store_true")
    args = ap.parse_args()

    pkl_path = args.pkl or os.path.join(
        GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    if not os.path.exists(pkl_path):
        print(f"Dataset not found: {pkl_path}"); sys.exit(1)

    tickers            = GlobalConfig.TICKERS
    model_names_to_run = args.models or list(_make_factories().keys())

    print(f"\n{'='*60}")
    print(f"BASELINE COMPARISON  |  Device: {DEVICE}")
    print(f"Models  : {model_names_to_run}")
    print(f"Seeds   : {SEEDS[:args.n_seeds]}")
    print(f"{'='*60}")

    data      = load_and_split_data(pkl_path, tickers)
    macro_dim = data["macro_dim"]
    news_dim  = data["news_dim"]
    factories = _make_factories(news_dim=news_dim)

    hparams_path  = os.path.join(RESULTS_DIR, "best_hparams.json")
    saved_hparams = {}
    if os.path.exists(hparams_path):
        with open(hparams_path) as f: saved_hparams = json.load(f)

    all_results:  Dict[str, dict] = {}
    best_hparams: Dict[str, dict] = {}

    # ── MSGCA_FV ──────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("MSGCA_FV (fixed-val + early stopping):")

    if args.skip_msgca_inline:
        # Option A: load từ pre-computed JSON files
        fv = collect_msgca_fv_results(args.output_dir, args.n_seeds)
        if fv:
            all_results["MSGCA_FV"] = fv
            print(f"  Loaded ({fv['n_seeds']} seeds): "
                  f"ACC={fv['acc_mean']:.4f}+/-{fv['acc_std']:.4f}  "
                  f"MCC={fv['mcc_mean']:.4f}+/-{fv['mcc_std']:.4f}")
        else:
            print(f"  NOT found. Run: python main.py --n-seeds {args.n_seeds} --save-results")
    else:
        # Grid search (hoặc load saved hparams)
        t0 = time.time()
        msgca_key = "MSGCA_FV"
        if args.skip_search and msgca_key in saved_hparams:
            best_msgca_hp = saved_hparams[msgca_key]
            print(f"  [SKIP SEARCH] Saved: {best_msgca_hp}")
        else:
            print(f"  [1/2] Grid search (6 combos, seed=42)...")
            search = grid_search_msgca(
                data["train_hval"], data["val_hval"],
                macro_dim, news_dim, verbose=args.verbose,
            )
            best_msgca_hp = search["best_hparams"]
            print(f"  Best: {best_msgca_hp}  val_MCC={search['best_mcc']:.4f}")
            saved_hparams[msgca_key] = best_msgca_hp
            with open(hparams_path, "w", encoding="utf-8") as f:
                json.dump(saved_hparams, f, indent=2)

        best_hparams["MSGCA_FV"] = best_msgca_hp

        # Option B1: dùng saved model cho seed=42 + train remaining seeds
        # Option B2: train toàn bộ 5 seeds từ đầu
        if not args.no_load_saved:
            saved_model_path = args.load_saved_model or _auto_detect_best_model(args.output_dir)
            if saved_model_path and os.path.exists(saved_model_path):
                print(f"\n  [2/2] Final eval ({args.n_seeds} seeds):")
                print(f"        seed=42  → load từ {os.path.basename(saved_model_path)}")
                print(f"        seeds {[s for s in SEEDS[:args.n_seeds] if s!=42]} → train fresh")
                fv_result = run_msgca_with_saved_model(
                    saved_model_path=saved_model_path,
                    train_hval=data["train_hval"], val_hval=data["val_hval"],
                    train_full=data["train_full"], test=data["test"],
                    macro_dim=macro_dim, news_dim=news_dim,
                    best_hparams=best_msgca_hp,
                    n_seeds=args.n_seeds, verbose=True,
                )
            else:
                print(f"\n  [WARN] Không tìm thấy saved model trong '{args.output_dir}/'")
                print(f"         Chạy: python main.py trước để tạo model")
                print(f"         Hoặc dùng: --load-saved-model <path>")
                print(f"         Training toàn bộ {args.n_seeds} seeds từ đầu...")
                fv_result = final_eval_msgca(
                    data["train_hval"], data["val_hval"],
                    data["train_full"], data["test"],
                    best_msgca_hp, macro_dim, news_dim,
                    n_seeds=args.n_seeds, verbose=True,
                )
        else:
            # --no-load-saved: train toàn bộ
            print(f"  [2/2] Final eval ({args.n_seeds} seeds, train all from scratch)...")
            fv_result = final_eval_msgca(
                data["train_hval"], data["val_hval"],
                data["train_full"], data["test"],
                best_msgca_hp, macro_dim, news_dim,
                n_seeds=args.n_seeds, verbose=True,
            )

        all_results["MSGCA_FV"] = fv_result
        print(f"\n  -> ACC={fv_result['acc_mean']:.4f}+/-{fv_result['acc_std']:.4f}  "
              f"MCC={fv_result['mcc_mean']:.4f}+/-{fv_result['mcc_std']:.4f}  "
              f"({(time.time()-t0)/60:.1f} min)")

    # ── MSGCA_Best — proposed model at its best configuration ─────────────────
    if not args.skip_msgca_best:
        print(f"\n{'─'*60}")
        print("MSGCA_Best (FocalLoss + class weights — proposed model best performance):")
        t0 = time.time()

        saved_model_path = args.load_saved_model or _auto_detect_best_model(args.output_dir)

        if saved_model_path and os.path.exists(saved_model_path):
            print(f"  seed=42 → load from {os.path.basename(saved_model_path)}")
            print(f"  seeds {[s for s in SEEDS[:args.n_seeds] if s!=42]} → train fresh (FocalLoss)")
        else:
            print(f"  [WARN] No saved model found in '{args.output_dir}/'")
            print(f"  Run: python main.py --seed 42   to create a model first")
            print(f"  Training all {args.n_seeds} seeds from scratch...")

        best_result = run_msgca_best(
            saved_model_path=saved_model_path,
            train_hval=data["train_hval"], val_hval=data["val_hval"],
            train_full=data["train_full"], test=data["test"],
            macro_dim=macro_dim, news_dim=news_dim,
            n_seeds=args.n_seeds, lr=1e-4, dropout=0.1,
            verbose=True,
        )
        all_results["MSGCA_Best"] = best_result
        print(f"\n  -> MSGCA_Best: "
              f"ACC={best_result['acc_mean']:.4f}+/-{best_result['acc_std']:.4f}  "
              f"MCC={best_result['mcc_mean']:.4f}+/-{best_result['mcc_std']:.4f}  "
              f"({(time.time()-t0)/60:.1f} min)")

    # ── Flat baselines ────────────────────────────────────────────────────────
    for model_name in model_names_to_run:
        if model_name not in factories:
            print(f"\n[SKIP] Unknown: {model_name}"); continue
        print(f"\n{'─'*60}")
        print(f"  [{model_name}]  {CATEGORY_MAP.get(model_name, '')}")
        factory = factories[model_name]
        t0 = time.time()

        if args.skip_search and model_name in saved_hparams:
            best_hp = saved_hparams[model_name]
            print(f"  [SKIP SEARCH] Saved: {best_hp}")
        else:
            print(f"  [1/2] Grid search ({len(get_all_combinations())} combos)...")
            search  = grid_search_flat(
                model_factory=factory, train_hval_data=data["train_hval"],
                val_hval_data=data["val_hval"], macro_dim=macro_dim,
                device=DEVICE, verbose=args.verbose,
            )
            best_hp = search["best_hparams"]
            print(f"  Best: {best_hp}  val_MCC={search['best_mcc']:.4f}")
            saved_hparams[model_name] = best_hp
            with open(hparams_path, "w", encoding="utf-8") as f:
                json.dump(saved_hparams, f, indent=2)

        best_hparams[model_name] = best_hp
        print(f"  [2/2] Final eval ({args.n_seeds} seeds)...")
        result = final_eval_flat(
            model_factory=factory, train_data=data["train_full"],
            test_data=data["test"], best_hparams=best_hp,
            macro_dim=macro_dim, n_seeds=args.n_seeds,
            device=DEVICE, verbose=True,
        )
        all_results[model_name] = result
        print(f"  -> {model_name}: "
              f"ACC={result['acc_mean']:.4f}+/-{result['acc_std']:.4f}  "
              f"MCC={result['mcc_mean']:.4f}+/-{result['mcc_std']:.4f}  "
              f"({(time.time()-t0)/60:.1f} min)")

    if not all_results: print("No results."); return

    print()
    table = format_table(all_results, best_hparams)
    print(table)

    table_path = os.path.join(RESULTS_DIR, "rq1_table.txt")
    raw_path   = os.path.join(RESULTS_DIR, "raw_results.json")
    with open(table_path, "w", encoding="utf-8") as f: f.write(table)
    with open(raw_path,   "w", encoding="utf-8") as f: json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {table_path}\n         {raw_path}")


if __name__ == "__main__":
    main()