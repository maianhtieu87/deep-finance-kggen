# configs/config.py
"""
Configuration — 9-ticker universe + GNN/KG pipeline settings.

Tickers: TSLA  AAPL  AMZN  MSFT  GOOGL  META  BA  JPM  WMT
"""

import os
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL
# ─────────────────────────────────────────────────────────────────────────────

class GlobalConfig:

    # Paths
    BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR         = os.path.join(BASE_DIR, "data")
    RAW_PATH         = os.path.join(DATA_DIR, "raw")
    RAW_PRICE_PATH   = os.path.join(RAW_PATH, "price")
    RAW_MACRO_PATH   = os.path.join(RAW_PATH, "macro")
    RAW_NEWS_PATH    = os.path.join(RAW_PATH, "news")
    RAW_FILINGS_PATH = os.path.join(RAW_PATH, "filings")
    INTERIM_PATH     = os.path.join(DATA_DIR, "interim")
    PROCESSED_PATH   = os.path.join(DATA_DIR, "processed")

    # Date range
    START_DATE = "2022-01-01"
    END_DATE   = "2025-06-23"

    # ── Target universe (9 tickers) ──────────────────────────────────────────
    TICKERS = ["TSLA", "AAPL", "AMZN", "MSFT", "GOOGL", "META", "BA", "JPM", "WMT"]

    # Sectors for reference (used by any downstream analysis, not by extractor)
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

    # Raw → canonical ticker mapping (identity for clean symbols)
    TICKER_MAPPING = {t: t for t in TICKERS}

    # Macro indicators
    MACRO_SYMBOLS = [
        "^GSPC",   # S&P 500
        "^DJI",    # Dow Jones
        "^IXIC",   # NASDAQ
        "^VIX",    # Volatility Index
        "^TNX",    # 10-Year Treasury Yield
    ]

    # ── Voyage Embedding ─────────────────────────────────────────────────────
    EMBED_MODEL      = "voyage-3-large"
    MAX_RETRIES      = 6
    BACKOFF_BASE     = 30
    MAX_TEXTS_PER_REQ = 40
    PAYMENT_ADDED    = True
    VOYAGE_RATE_LIMITS = {
        True:  {"RPM": 50,  "TPM": 400_000, "SLEEP": 1.0},
        False: {"RPM": 3,   "TPM": 10_000,  "SLEEP": 20.0},
    }
    VOYAGE_API_KEY   = os.getenv("VOYAGE_API_KEY", "")

    # ── KG / Cache paths ─────────────────────────────────────────────────────
    # These match the defaults in KGGenNewsEmbedder so test scripts and
    # production runs share the same disk cache automatically.
    KG_CACHE_DIRNAME        = "kg_article_cache"
    KG_VOYAGE_CACHE_DIRNAME = "kg_voyage_emb_cache"

    @classmethod
    def kg_cache_dir(cls) -> str:
        return os.path.join(cls.INTERIM_PATH, cls.KG_CACHE_DIRNAME)

    @classmethod
    def kg_voyage_cache_dir(cls) -> str:
        return os.path.join(cls.INTERIM_PATH, cls.KG_VOYAGE_CACHE_DIRNAME)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

class TrainConfig:

    # Basic
    seed          = 42
    use_cuda      = True
    batch_size    = 32
    epoch_num     = 200
    learning_rate = 1e-4
    weight_decay  = 1e-4

    # Data splits
    train_ratio = 0.7
    valid_ratio = 0.15
    # test_ratio = 0.15 (implicit)

    # Architecture
    window_size = 20    # Temporal window (trading days)
    dim         = 256   # Hidden dimension throughout fusion layers
    output_dim  = 3     # Classes: DOWN(0) FLAT(1) UP(2)
    num_head    = 4     # Cross-attention heads

    # Feature dimensions
    # news_embed_dim = GATv2 graph_out_dim — must match KGGraphEncoderGATv2.output_dim
    news_embed_dim  = 128

    # ── GNN ──────────────────────────────────────────────────────────────────
    use_gnn        = True
    gnn_type       = "gat"       # "gat" | "sage"
    gnn_hidden_dim = 128
    gnn_num_layers = 2
    gnn_heads      = 4           # 128 / 4 = 32 per head
    gnn_pool       = "mean"      # "mean" | "max" | "attention"

    # Node feature dim: 1024 (Voyage) + 8 (entity_type one-hot) + 1 (target_flag)
    kg_node_dim      = 1033
    # Edge attr dim: 14 (relation one-hot) + 1 (confidence) + 1 (price_impact) + 1 (relevance)
    kg_edge_attr_dim = 17

    # ── Loss ─────────────────────────────────────────────────────────────────
    use_focal_loss      = True
    focal_gamma         = 2.0
    use_label_smoothing = False
    label_smoothing     = 0.1

    # ── KG Processing ────────────────────────────────────────────────────────
    kg_window_days  = 20     # Rolling window for graph aggregation
    kg_top_triples  = 5      # Max triples per article (soft cap)
    kg_use_voyage   = True
    kg_allow_llm    = False  # Allow LLM calls for cache-missing articles during graph rebuild

    # ── Entity Resolution ────────────────────────────────────────────────────
    use_improved_resolver  = True
    resolver_kmeans_k      = 64
    resolver_min_cluster   = 3


# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────

class ModelConfig:

    # Price encoder
    price_lstm_hidden = 64
    price_lstm_layers = 2

    # Macro encoder
    macro_lstm_hidden = 64
    macro_lstm_layers = 2

    # KG / GNN encoder
    # Must match TrainConfig.kg_node_dim — node features fed into KGGraphEncoderGATv2
    kg_node_dim      = 1033   # 1024 (Voyage) + 8 (entity_type) + 1 (target_flag)
    kg_edge_attr_dim = 17     # 14 (relation) + 3 (conf + impact + relevance)

    kg_use_gat  = True
    kg_use_sage = False

    # Fusion
    fusion_num_heads = 4
    fusion_dropout   = 0.1

    # Predictor
    predictor_hidden_dim = 256
    predictor_dropout    = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_config() -> bool:
    errors = []

    # Paths
    for path in [GlobalConfig.DATA_DIR, GlobalConfig.RAW_PATH, GlobalConfig.INTERIM_PATH]:
        if not os.path.exists(path):
            errors.append(f"Path does not exist: {path}")

    # Ticker universe
    expected = {"TSLA", "AAPL", "AMZN", "MSFT", "GOOGL", "META", "BA", "JPM", "WMT"}
    actual   = set(GlobalConfig.TICKERS)
    if actual != expected:
        errors.append(f"TICKERS mismatch. Expected {sorted(expected)}, got {sorted(actual)}")
    if set(GlobalConfig.TICKER_MAPPING.keys()) != actual:
        errors.append("TICKER_MAPPING keys do not match TICKERS")

    # GNN config
    if TrainConfig.use_gnn:
        if TrainConfig.gnn_type not in ("sage", "gat"):
            errors.append(f"Invalid gnn_type: {TrainConfig.gnn_type}")
        if TrainConfig.gnn_pool not in ("mean", "max", "attention"):
            errors.append(f"Invalid gnn_pool: {TrainConfig.gnn_pool}")

    # Dimension consistency
    if TrainConfig.kg_node_dim != ModelConfig.kg_node_dim:
        errors.append(
            f"kg_node_dim mismatch: TrainConfig={TrainConfig.kg_node_dim}, "
            f"ModelConfig={ModelConfig.kg_node_dim}"
        )
    if TrainConfig.kg_edge_attr_dim != ModelConfig.kg_edge_attr_dim:
        errors.append(
            f"kg_edge_attr_dim mismatch: TrainConfig={TrainConfig.kg_edge_attr_dim}, "
            f"ModelConfig={ModelConfig.kg_edge_attr_dim}"
        )
    if TrainConfig.gnn_hidden_dim % TrainConfig.gnn_heads != 0:
        errors.append(
            f"gnn_hidden_dim ({TrainConfig.gnn_hidden_dim}) must be divisible "
            f"by gnn_heads ({TrainConfig.gnn_heads})"
        )

    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("✅ Configuration validated")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Configuration Summary ===\n")

    print("Tickers :", GlobalConfig.TICKERS)
    print("Sectors :")
    for t, s in GlobalConfig.TICKER_SECTOR.items():
        print(f"  {t:<6} {s}")

    print(f"\nDate range : {GlobalConfig.START_DATE} → {GlobalConfig.END_DATE}")
    print(f"Data dir   : {GlobalConfig.DATA_DIR}")

    print("\nGNN:")
    print(f"  type={TrainConfig.gnn_type}  layers={TrainConfig.gnn_num_layers}"
          f"  hidden={TrainConfig.gnn_hidden_dim}  heads={TrainConfig.gnn_heads}"
          f"  pool={TrainConfig.gnn_pool}")
    print(f"  node_dim={TrainConfig.kg_node_dim}  edge_dim={TrainConfig.kg_edge_attr_dim}")
    print(f"  graph_out → news_embed_dim={TrainConfig.news_embed_dim}")

    print("\nKG cache paths:")
    print(f"  article  : {GlobalConfig.kg_cache_dir()}")
    print(f"  voyage   : {GlobalConfig.kg_voyage_cache_dir()}")

    print()
    validate_config()