# configs/config.py
"""
Configuration V4 — 9-ticker universe.

Thay đổi so với V3:
  - news_embed_dim = 1024  (Voyage-3-large output, dùng trực tiếp cho NewsEncoder)
  - kg_node_dim, kg_edge_attr_dim: kept for reference but NOT used in V4 model
  - use_gnn = False  (GATv2 pipeline removed)
"""

import os
from datetime import datetime


class GlobalConfig:

    BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR         = os.path.join(BASE_DIR, "data")
    RAW_PATH         = os.path.join(DATA_DIR, "raw")
    RAW_PRICE_PATH   = os.path.join(RAW_PATH, "price")
    RAW_MACRO_PATH   = os.path.join(RAW_PATH, "macro")
    RAW_NEWS_PATH    = os.path.join(RAW_PATH, "news")
    RAW_FILINGS_PATH = os.path.join(RAW_PATH, "filings")
    INTERIM_PATH     = os.path.join(DATA_DIR, "interim")
    PROCESSED_PATH   = os.path.join(DATA_DIR, "processed")

    START_DATE = "2022-01-01"
    END_DATE   = "2025-06-23"

    TICKERS = ["TSLA", "AAPL", "AMZN", "MSFT", "GOOGL", "META", "BA", "JPM", "WMT"]

    TICKER_SECTOR = {
        "TSLA":  "Consumer Discretionary",
        "AAPL":  "Technology",
        "AMZN":  "Consumer Discretionary",
        "MSFT":  "Technology",
        "GOOGL": "Technology",
        "META":  "Technology",
        "BA":    "Industrials",
        "JPM":   "Financials",
        "WMT":   "Consumer Staples",
    }

    TICKER_MAPPING = {t: t for t in TICKERS}

    MACRO_SYMBOLS = [
        "^GSPC",   # S&P 500
        "^DJI",    # Dow Jones
        "^IXIC",   # NASDAQ
        "^VIX",    # Volatility Index
        "^TNX",    # 10-Year Treasury Yield
    ]

    # Voyage embedding
    EMBED_MODEL       = "voyage-3-large"
    MAX_RETRIES       = 6
    BACKOFF_BASE      = 30
    MAX_TEXTS_PER_REQ = 40
    PAYMENT_ADDED     = True
    VOYAGE_RATE_LIMITS = {
        True:  {"RPM": 50,  "TPM": 400_000, "SLEEP": 1.0},
        False: {"RPM": 3,   "TPM": 10_000,  "SLEEP": 20.0},
    }
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

    KG_CACHE_DIRNAME        = "kg_article_cache"
    KG_VOYAGE_CACHE_DIRNAME = "kg_voyage_emb_cache"

    @classmethod
    def kg_cache_dir(cls) -> str:
        return os.path.join(cls.INTERIM_PATH, cls.KG_CACHE_DIRNAME)

    @classmethod
    def kg_voyage_cache_dir(cls) -> str:
        return os.path.join(cls.INTERIM_PATH, cls.KG_VOYAGE_CACHE_DIRNAME)


class TrainConfig:

    seed          = 42
    use_cuda      = True
    batch_size    = 32
    epoch_num     = 200
    learning_rate = 1e-4
    weight_decay  = 1e-4

    train_ratio = 0.7
    valid_ratio = 0.15

    window_size = 20
    dim         = 256
    output_dim  = 3
    num_head    = 4

    # V4: news_embed_dim = Voyage-3-large output dimension
    # NewsEncoder(1024, dim) projects 1024 → 256 during training
    news_embed_dim  = 1024   # was 128 in V3

    # GNN config (kept for reference but NOT used in V4)
    use_gnn        = False   # V4: no GATv2
    gnn_type       = "gat"
    gnn_hidden_dim = 128
    gnn_num_layers = 2
    gnn_heads      = 4
    gnn_pool       = "mean"

    kg_node_dim      = 1033  # reference only
    kg_edge_attr_dim = 17    # reference only

    use_focal_loss      = True
    focal_gamma         = 2.0
    use_label_smoothing = False
    label_smoothing     = 0.1

    kg_window_days = 3      # rolling window for embed_news.py (was 20 for graph)
    kg_top_triples = 5
    kg_use_voyage  = True
    kg_allow_llm   = False

    use_improved_resolver = False  # KMeans removed in V4
    resolver_kmeans_k     = 64
    resolver_min_cluster  = 3


class ModelConfig:

    price_lstm_hidden = 64
    price_lstm_layers = 2

    macro_lstm_hidden = 64
    macro_lstm_layers = 2

    # V4: no GATv2, but keep fields for reference
    kg_node_dim      = 1033
    kg_edge_attr_dim = 17
    kg_use_gat       = False
    kg_use_sage      = False

    fusion_num_heads = 4
    fusion_dropout   = 0.1

    predictor_hidden_dim = 256
    predictor_dropout    = 0.1


class PathConfig:

    @staticmethod
    def get_model_save_path(experiment_name: str = None) -> str:
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(GlobalConfig.BASE_DIR, "output", "models")
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"{experiment_name}.pt")

    @staticmethod
    def get_log_path(experiment_name: str = None) -> str:
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(GlobalConfig.BASE_DIR, "output", "logs")
        os.makedirs(log_dir, exist_ok=True)
        return os.path.join(log_dir, f"{experiment_name}.log")


def validate_config() -> bool:
    errors = []

    for path in [GlobalConfig.DATA_DIR, GlobalConfig.RAW_PATH, GlobalConfig.INTERIM_PATH]:
        if not os.path.exists(path):
            errors.append(f"Path does not exist: {path}")

    expected = {"TSLA", "AAPL", "AMZN", "MSFT", "GOOGL", "META", "BA", "JPM", "WMT"}
    actual   = set(GlobalConfig.TICKERS)
    if actual != expected:
        errors.append(f"TICKERS mismatch. Expected {sorted(expected)}, got {sorted(actual)}")

    if TrainConfig.news_embed_dim != 1024:
        errors.append(
            f"news_embed_dim should be 1024 (Voyage-3-large) but is {TrainConfig.news_embed_dim}"
        )

    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("Configuration validated (V4 — Voyage direct embedding)")
    return True


if __name__ == "__main__":
    print("=== Configuration V4 ===")
    print(f"Tickers  : {GlobalConfig.TICKERS}")
    print(f"Date     : {GlobalConfig.START_DATE} → {GlobalConfig.END_DATE}")
    print(f"News dim : {TrainConfig.news_embed_dim} (Voyage-3-large)")
    print(f"use_gnn  : {TrainConfig.use_gnn}")
    print()
    validate_config()