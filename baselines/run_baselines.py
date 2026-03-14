# baselines/run_baselines.py
"""
Run RQ1: So sánh toàn bộ baseline models + MSGCA (model chính) + MSGCA-NAF.

Sử dụng:
    python -m baselines.run_baselines

Output:
    baselines/results/rq1_results.json   — kết quả thô (mean±std mỗi model)
    baselines/results/rq1_table.txt      — bảng so sánh dạng text (giống Table 4 trong paper)

Hyperparameters tuân theo paper MSGCA:
    hidden_dim = 64, window_size = 20, lr = 1e-4, batch = 1024, epochs = 200
"""

import os
import sys
import json
import torch

# Đảm bảo import từ root project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.config import GlobalConfig, TrainConfig
from baselines.data_adapter import BaselineDataPrepare
from baselines.models import build_baseline, MSGCANoAdaptiveFusion
from baselines.trainer import run_baseline_multi_seed

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if TrainConfig.use_cuda and torch.cuda.is_available() else "cpu")

# Hyperparameters khớp với paper MSGCA (Section 5.1)
BASELINE_CFG = {
    "hidden_dim":  64,
    "doc_dim":     128,   # GNN graph embedding dim
    "graph_dim":   128,
    "num_classes": 3,
    "dropout":     0.1,
    "num_heads":   2,
}

TRAIN_HPS = {
    "epochs":     200,
    "lr":         1e-4,
    "batch_size": 1024,
    "n_runs":     5,
}

# Danh sách baselines cần chạy
FLAT_BASELINES = [
    "LSTM",
    "ALSTM",
    "ESTIMATE",
    "DTML",
    "ALSTM-W",
    "SLOT",
    "LLM-Stock",
]


# ──────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────

def load_baseline_data(pkl_path: str):
    """
    Load dữ liệu từ unified_dataset.pkl và chuẩn bị cho tất cả baselines.

    Returns
    -------
    flat_data : dict  — {ticker: (train, valid, test)} dạng flat tensor
    full_data : dict  — {ticker: (train, valid, test)} dạng PyG graph (cho MSGCA-NAF)
    """
    from src.data_loader import data_prepare as FullDataPrepare

    print(f"\n📦 Loading dataset: {pkl_path}")
    adapter   = BaselineDataPrepare(pkl_path)
    full_prep = FullDataPrepare(pkl_path)

    tickers = getattr(GlobalConfig, "TICKERS", ["TSLA", "AMZN", "MSFT", "NFLX"])

    flat_data = {}
    full_data = {}

    for ticker in tickers:
        print(f"\n── Ticker: {ticker}")

        # Flat embeddings cho baselines
        try:
            tr, va, te = adapter.prepare_baseline_data(ticker)
            if tr and len(tr.get("label", [])) >= 10:
                flat_data[ticker] = (tr, va, te)
        except Exception as e:
            print(f"   ⚠️  flat prep failed: {e}")

        # PyG graphs cho MSGCA-NAF
        try:
            tr_f, va_f, te_f = full_prep.prepare_data(ticker)
            if tr_f and len(tr_f.get("label", [])) >= 10:
                full_data[ticker] = (tr_f, va_f, te_f)
        except Exception as e:
            print(f"   ⚠️  full prep failed: {e}")

    print(f"\n✅ Loaded {len(flat_data)} tickers for flat baselines")
    print(f"✅ Loaded {len(full_data)} tickers for MSGCA-NAF")
    return flat_data, full_data


def merge_cross_ticker(data_dict: dict, mode: str):
    """
    Gộp dữ liệu nhiều tickers thành một dataset.
    mode: 'train' | 'valid' | 'test'
    """
    idx_map = {"train": 0, "valid": 1, "test": 2}
    idx = idx_map[mode]
    splits = [v[idx] for v in data_dict.values() if v[idx] and len(v[idx].get("label", [])) > 0]

    if not splits:
        return {}

    merged = {}
    for key in splits[0].keys():
        parts = [s[key] for s in splits]
        if isinstance(parts[0], torch.Tensor):
            merged[key] = torch.cat(parts, dim=0)
        else:
            # list (s_n_graphs)
            merged[key] = [g for sublist in parts for g in sublist]
    return merged


# ──────────────────────────────────────────────────────────────────────
# Results formatting
# ──────────────────────────────────────────────────────────────────────

def format_table(results: dict) -> str:
    """Tạo bảng so sánh dạng text giống Table 4 trong paper."""
    header = f"{'Model':<20} {'ACC':>12} {'MCC':>12}"
    sep    = "-" * 46
    lines  = [sep, header, sep]

    for name, res in results.items():
        acc_str = f"{res['acc_mean']:.4f}±{res['acc_std']:.4f}"
        mcc_str = f"{res['mcc_mean']:.4f}±{res['mcc_std']:.4f}"
        lines.append(f"{name:<20} {acc_str:>12} {mcc_str:>12}")

    lines.append(sep)
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    pkl_path = os.path.join(GlobalConfig.PROCESSED_PATH, "unified_dataset_test.pkl")
    if not os.path.exists(pkl_path):
        print(f"❌ Không tìm thấy dataset: {pkl_path}")
        print("   Hãy chạy main_test.py trước để tạo dữ liệu.")
        return

    # ── Load data ──────────────────────────────────────────────────
    flat_data, full_data = load_baseline_data(pkl_path)

    if not flat_data:
        print("❌ Không có dữ liệu cho baselines.")
        return

    # Gộp tất cả tickers
    train_flat = merge_cross_ticker(flat_data, "train")
    valid_flat = merge_cross_ticker(flat_data, "valid")
    test_flat  = merge_cross_ticker(flat_data, "test")

    train_full = merge_cross_ticker(full_data, "train")
    valid_full = merge_cross_ticker(full_data, "valid")
    test_full  = merge_cross_ticker(full_data, "test")

    print(f"\n📊 Flat data: Train={len(train_flat.get('label',[]))} | "
          f"Valid={len(valid_flat.get('label',[]))} | Test={len(test_flat.get('label',[]))}")

    all_results = {}

    # ── 1. Flat baselines ──────────────────────────────────────────
    for name in FLAT_BASELINES:
        print(f"\n{'='*50}")
        print(f"▶ Running: {name}")
        print(f"{'='*50}")

        def factory(n=name):
            return build_baseline(n, BASELINE_CFG)

        result = run_baseline_multi_seed(
            model_factory = factory,
            train_data    = train_flat,
            valid_data    = valid_flat,
            test_data     = test_flat,
            model_name    = name,
            n_runs        = TRAIN_HPS["n_runs"],
            epochs        = TRAIN_HPS["epochs"],
            lr            = TRAIN_HPS["lr"],
            batch_size    = TRAIN_HPS["batch_size"],
            device        = DEVICE,
            is_naf        = False,
        )
        all_results[name] = result

    # ── 2. MSGCA-NAF ──────────────────────────────────────────────
    if train_full and len(train_full.get("label", [])) > 0:
        print(f"\n{'='*50}")
        print(f"▶ Running: MSGCA-NAF")
        print(f"{'='*50}")

        macro_dim = train_full["s_m"].shape[-1]

        def naf_factory():
            return MSGCANoAdaptiveFusion(
                price_dim  = 1,
                macro_dim  = macro_dim,
                news_dim   = TrainConfig.news_embed_dim,
                dim        = TrainConfig.dim,
                input_dim  = TrainConfig.window_size,
                output_dim = TrainConfig.output_dim,
                num_head   = TrainConfig.num_head,
                device     = DEVICE,
                dropout    = 0.1,
                gnn_hidden_dim  = getattr(TrainConfig, "gnn_hidden_dim", 256),
                gnn_num_layers  = getattr(TrainConfig, "gnn_num_layers", 2),
                gnn_heads       = getattr(TrainConfig, "gnn_heads", 4),
            )

        result = run_baseline_multi_seed(
            model_factory = naf_factory,
            train_data    = train_full,
            valid_data    = valid_full,
            test_data     = test_full,
            model_name    = "MSGCA-NAF",
            n_runs        = TRAIN_HPS["n_runs"],
            epochs        = TRAIN_HPS["epochs"],
            lr            = TRAIN_HPS["lr"],
            batch_size    = getattr(TrainConfig, "batch_size", 128),
            device        = DEVICE,
            is_naf        = True,
        )
        all_results["MSGCA-NAF"] = result
    else:
        print("⚠️  Bỏ qua MSGCA-NAF (không đủ dữ liệu full PyG).")

    # ── 3. Tải kết quả MSGCA (model chính) nếu đã train ──────────
    msgca_model_path = os.path.join("output", "best_model.pt")
    if os.path.exists(msgca_model_path) and train_full:
        print(f"\n{'='*50}")
        print(f"▶ Evaluating MSGCA (best_model.pt)")
        print(f"{'='*50}")
        _eval_msgca_model(msgca_model_path, test_full, all_results, macro_dim=train_full["s_m"].shape[-1])

    # ── 4. Lưu kết quả ─────────────────────────────────────────────
    json_path = os.path.join(RESULTS_DIR, "rq1_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Raw results saved: {json_path}")

    table_str = format_table(all_results)
    table_path = os.path.join(RESULTS_DIR, "rq1_table.txt")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("RQ1 — Stock Movement Prediction Comparison\n\n")
        f.write(table_str)

    print(f"\n{'='*50}")
    print("📋 RQ1 RESULTS TABLE")
    print(f"{'='*50}")
    print(table_str)
    print(f"\n✅ Table saved: {table_path}")


def _eval_msgca_model(model_path: str, test_data: dict, all_results: dict, macro_dim: int):
    """Load và evaluate model MSGCA đã train từ main.py."""
    try:
        from src.model import StockMovementModel
        from main import evaluate as eval_main, StockGraphDataset

        model = StockMovementModel(
            price_dim  = 1,
            macro_dim  = macro_dim,
            news_dim   = TrainConfig.news_embed_dim,
            dim        = TrainConfig.dim,
            input_dim  = TrainConfig.window_size,
            output_dim = TrainConfig.output_dim,
            num_head   = TrainConfig.num_head,
            device     = DEVICE,
            dropout    = 0.1,
            use_focal_loss = TrainConfig.use_focal_loss,
            focal_gamma    = TrainConfig.focal_gamma,
            use_gnn        = getattr(TrainConfig, "use_gnn", True),
            gnn_type       = getattr(TrainConfig, "gnn_type", "sage"),
            gnn_hidden_dim = getattr(TrainConfig, "gnn_hidden_dim", 256),
            gnn_num_layers = getattr(TrainConfig, "gnn_num_layers", 2),
            gnn_heads      = getattr(TrainConfig, "gnn_heads", 4),
            gnn_pool       = getattr(TrainConfig, "gnn_pool", "attention"),
        ).to(DEVICE)

        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        acc, mcc = eval_main(model, test_data)

        all_results["MSGCA"] = {
            "acc_mean": acc, "acc_std": 0.0,
            "mcc_mean": mcc, "mcc_std": 0.0,
            "acc_list": [acc], "mcc_list": [mcc],
            "note": "single run (best_model.pt)",
        }
        print(f"  ✅ [MSGCA] ACC={acc:.4f} | MCC={mcc:.4f}")
    except Exception as e:
        print(f"  ⚠️  Không thể evaluate MSGCA: {e}")


if __name__ == "__main__":
    main()