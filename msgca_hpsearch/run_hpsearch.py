#!/usr/bin/env python
"""
run_hpsearch.py — Standalone MSGCA_FV Hyperparameter Search
============================================================

Tìm hyperparameters tốt nhất cho MSGCA_FV (CE loss, fair comparison mode)
mà không cần chạy toàn bộ baseline pipeline.

USAGE
-----
# Full flow: grid search → final 5-seed evaluation
python run_hpsearch.py --data path/to/unified_dataset_test.pkl

# Chỉ search (nhanh, seed=42)
python run_hpsearch.py --data data.pkl --mode search

# Chỉ eval với HP đã tìm (load từ results/msgca_best_hparams.json)
python run_hpsearch.py --data data.pkl --mode eval

# Custom search space
python run_hpsearch.py --data data.pkl --lr 5e-5 1e-4 3e-4 --dropout 0.1 0.2

# Tùy chỉnh model architecture
python run_hpsearch.py --data data.pkl --dim 64 --num-head 2 --window-size 20

# Resume từ lần chạy bị crash
python run_hpsearch.py --data data.pkl --mode search --resume

# Quick smoke test (2 combos, 1 seed)
python run_hpsearch.py --data data.pkl --lr 1e-4 --dropout 0.1 --n-seeds 1 --max-epochs 20

NOTES
-----
- Dataset: unified_dataset_test.pkl (produced by deep-finance-kggen pipeline)
- Loss: Cross-Entropy (no focal loss, no class weights) — fair comparison mode
- Scheduler: LinearLR warmup → CosineAnnealingLR
- Early stopping: on val MCC, starts after max(warmup_epochs, 40) epochs
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# ── Make sure package is on path when running from any directory ──────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from data.loader import load_and_split, DEFAULT_TICKERS
from trainer.msgca_trainer import grid_search, final_eval, SEEDS
from model.stock_model import N_TICKERS

RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

HPARAMS_FILE = os.path.join(RESULTS_DIR, "msgca_best_hparams.json")
ALLRES_FILE  = os.path.join(RESULTS_DIR, "msgca_all_results.json")
FINAL_FILE   = os.path.join(RESULTS_DIR, "msgca_final_eval.json")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="MSGCA_FV Hyperparameter Search",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    ap.add_argument(
        "--data", required=True,
        help="Path to unified_dataset_test.pkl",
    )
    ap.add_argument(
        "--tickers", nargs="+", default=None,
        help=f"Tickers to use (default: {DEFAULT_TICKERS})",
    )

    # ── Mode ──────────────────────────────────────────────────────────────────
    ap.add_argument(
        "--mode", choices=["search", "eval", "full"], default="full",
        help=(
            "search: grid search only (seed=42)\n"
            "eval  : final n-seed eval with best HP (loads from results/)\n"
            "full  : search → eval (default)"
        ),
    )
    ap.add_argument(
        "--resume", action="store_true",
        help="Resume search from existing results/msgca_all_results.json",
    )

    # ── Search space ──────────────────────────────────────────────────────────
    ap.add_argument(
        "--lr", nargs="+", type=float, default=None,
        help="Learning rates to search (default: [5e-5, 1e-4, 3e-4, 5e-4])",
    )
    ap.add_argument(
        "--dropout", nargs="+", type=float, default=None,
        help="Dropout values to search (default: [0.1, 0.2, 0.3])",
    )

    # ── Model architecture ────────────────────────────────────────────────────
    ap.add_argument("--dim",         type=int,   default=64,   help="Model hidden dim (default: 64)")
    ap.add_argument("--num-head",    type=int,   default=2,    help="Attention heads (default: 2)")
    ap.add_argument("--window-size", type=int,   default=20,   help="Rolling window size (default: 20)")
    ap.add_argument("--news-dim",    type=int,   default=None,
                    help="News embedding dim. Auto-detected from data if not set.")
    ap.add_argument("--quality-dim", type=int,   default=4,    help="Quality stats dim (default: 4)")

    # ── Training control ──────────────────────────────────────────────────────
    ap.add_argument("--n-seeds",      type=int,   default=5,    help="Seeds for final eval (default: 5)")
    ap.add_argument("--max-epochs",   type=int,   default=150,  help="Max training epochs (default: 150)")
    ap.add_argument("--patience",     type=int,   default=30,   help="Early stopping patience (default: 30)")
    ap.add_argument("--warmup",       type=int,   default=15,   help="Warmup epochs (default: 15)")
    ap.add_argument("--mod-dropout",  type=float, default=0.30, help="News modality dropout prob (default: 0.30)")

    # ── Search-specific ───────────────────────────────────────────────────────
    ap.add_argument("--search-max-epochs", type=int, default=None,
                    help="Max epochs during search phase (default: min(max_epochs, 100))")
    ap.add_argument("--search-patience",   type=int, default=None,
                    help="Patience during search phase (default: min(patience, 20))")

    # ── Data split ────────────────────────────────────────────────────────────
    ap.add_argument("--train-ratio", type=float, default=0.70, help="Train ratio (default: 0.70)")
    ap.add_argument("--valid-ratio", type=float, default=0.15, help="Valid ratio (default: 0.15)")
    ap.add_argument("--price-mode",  default="vol_adjusted",
                    choices=["vol_adjusted", "pct_first", "absolute"])
    ap.add_argument("--label-mode",  default="rolling",
                    choices=["rolling", "fixed", "volatility"])

    ap.add_argument("--verbose", action="store_true", help="Print per-epoch progress")

    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _print_section(title: str):
    w = 60
    print(f"\n{'─'*w}")
    print(f"  {title}")
    print(f"{'─'*w}")


def _save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  Saved: {path}")


def _load_json(path: str):
    with open(path) as f:
        return json.load(f)


def _print_summary(result: dict, label: str = ""):
    print(f"\n  {'─'*40}")
    if label:
        print(f"  {label}")
    print(f"  ACC  : {result['acc_mean']:.4f} ± {result['acc_std']:.4f}")
    print(f"  MCC  : {result['mcc_mean']:.4f} ± {result['mcc_std']:.4f}")
    if "ep_mean" in result:
        print(f"  avg_ep: {result['ep_mean']:.0f}")
    print(f"  Seeds : {result.get('n_seeds', '?')}")
    print(f"  HP    : {result.get('hparams', {})}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"  MSGCA_FV Hyperparameter Search")
    print(f"{'='*60}")
    print(f"  Device     : {device}")
    print(f"  Mode       : {args.mode}")
    print(f"  Data       : {args.data}")
    print(f"  dim={args.dim}  num_head={args.num_head}  window={args.window_size}")
    print(f"  n_seeds={args.n_seeds}  max_epochs={args.max_epochs}  patience={args.patience}")
    print(f"  mod_dropout={args.mod_dropout}  warmup={args.warmup}")

    # ── Load and split data ───────────────────────────────────────────────────
    _print_section("Loading data")
    tickers = args.tickers or DEFAULT_TICKERS

    news_dim_override = args.news_dim
    data = load_and_split(
        pkl_path=args.data,
        tickers=tickers,
        news_dim=news_dim_override or 768,   # will be overridden by auto-detect
        quality_dim=args.quality_dim,
        window_size=args.window_size,
        price_mode=args.price_mode,
        label_mode=args.label_mode,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
    )

    macro_dim = data["macro_dim"]
    news_dim  = data["news_dim"]   # auto-detected from actual data

    if news_dim is None:
        news_dim = news_dim_override or 768
        print(f"  [WARN] Could not detect news_dim — using {news_dim}")

    print(f"\n  macro_dim={macro_dim}  news_dim={news_dim}")

    # Shared kwargs for all trainer calls
    model_kwargs = dict(
        dim=args.dim,
        num_head=args.num_head,
        window_size=args.window_size,
        n_tickers=N_TICKERS,
        quality_dim=args.quality_dim,
    )
    train_kwargs = dict(
        max_epochs=args.max_epochs,
        patience=args.patience,
        warmup_epochs=args.warmup,
        mod_dropout=args.mod_dropout,
        verbose=args.verbose,
    )

    # ── Build search grid ─────────────────────────────────────────────────────
    grid = {}
    if args.lr:
        grid["lr"] = args.lr
    if args.dropout:
        grid["dropout"] = args.dropout

    # ── MODE: search ─────────────────────────────────────────────────────────
    if args.mode in ("search", "full"):
        _print_section("Grid Search")

        resume_results = []
        if args.resume and os.path.exists(ALLRES_FILE):
            resume_results = _load_json(ALLRES_FILE)
            print(f"  Resuming from {len(resume_results)} previous results")

        search_max_epochs = args.search_max_epochs or min(args.max_epochs, 100)
        search_patience   = args.search_patience   or min(args.patience, 20)

        t0 = time.time()
        search_result = grid_search(
            train_hval=data["train_hval"],
            val_hval=data["val_hval"],
            macro_dim=macro_dim,
            news_dim=news_dim,
            device=device,
            grid=grid or None,
            max_epochs=search_max_epochs,
            patience=search_patience,
            warmup_epochs=args.warmup,
            mod_dropout=args.mod_dropout,
            verbose=True,
            resume_results=resume_results,
            **model_kwargs,
        )
        elapsed = time.time() - t0

        best_hp  = search_result["best_hparams"]
        best_mcc = search_result["best_mcc"]

        print(f"\n  Search done in {elapsed/60:.1f} min")
        print(f"  Best HP  : {best_hp}")
        print(f"  Best MCC : {best_mcc:.4f}")

        _save_json(best_hp, HPARAMS_FILE)
        _save_json(search_result["all_results"], ALLRES_FILE)

        if args.mode == "search":
            print("\nSearch complete. Run with --mode eval to evaluate.")
            return

    # ── MODE: eval ────────────────────────────────────────────────────────────
    if args.mode in ("eval", "full"):
        _print_section("Final Evaluation")

        if args.mode == "eval":
            if not os.path.exists(HPARAMS_FILE):
                print(f"ERROR: {HPARAMS_FILE} not found. Run --mode search first.")
                sys.exit(1)
            best_hp = _load_json(HPARAMS_FILE)
            print(f"  Loaded HP from {HPARAMS_FILE}: {best_hp}")

        print(f"  Seeds: {SEEDS[:args.n_seeds]}")
        print(f"  HP   : {best_hp}")

        t0 = time.time()
        final_result = final_eval(
            train_hval=data["train_hval"],
            val_hval=data["val_hval"],
            train_full=data["train_full"],
            test=data["test"],
            best_hparams=best_hp,
            macro_dim=macro_dim,
            news_dim=news_dim,
            device=device,
            n_seeds=args.n_seeds,
            **model_kwargs,
            **train_kwargs,
        )
        elapsed = time.time() - t0

        _print_summary(final_result, label="MSGCA_FV Final Results")
        print(f"\n  Eval done in {elapsed/60:.1f} min")

        _save_json(final_result, FINAL_FILE)

    print(f"\n{'='*60}")
    print("  Done. Results saved to results/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
