# configs/config.py - UPDATED WITH GNN PARAMETERS
"""
Configuration file with GNN support

CHANGES:
- Added GNN-specific parameters to TrainConfig
- Updated documentation
"""

import os
from datetime import datetime


class GlobalConfig:
    """Global configuration for data paths and processing."""
    
    # === Project Paths ===
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    
    RAW_PATH = os.path.join(DATA_DIR, "raw")
    RAW_PRICE_PATH = os.path.join(RAW_PATH, "price")
    RAW_MACRO_PATH = os.path.join(RAW_PATH, "macro")
    RAW_NEWS_PATH = os.path.join(RAW_PATH, "news")
    RAW_FILINGS_PATH = os.path.join(RAW_PATH, "filings")
    
    INTERIM_PATH = os.path.join(DATA_DIR, "interim")
    PROCESSED_PATH = os.path.join(DATA_DIR, "processed")
    
    # === Data Collection Config ===
    START_DATE = "2022-01-01"
    END_DATE = "2024-12-31"
    
    # Target stocks
    TICKERS = ["TSLA", "AMZN", "MSFT", "NFLX"]
    
    # Macro indicators
    MACRO_SYMBOLS = [
        "^GSPC",    # S&P 500
        "^DJI",     # Dow Jones
        "^IXIC",    # NASDAQ
        "^VIX",     # Volatility Index
        "^TNX",     # 10-Year Treasury Yield
    ]
    
    # Ticker mapping (raw -> clean)
    TICKER_MAPPING = {
        "TSLA": "TSLA",
        "AMZN": "AMZN",
        "MSFT": "MSFT",
        "NFLX": "NFLX",
    }
        # --- Voyage Embedding Settings (nếu dùng lại text-embedding) ---
    EMBED_MODEL = "voyage-3-large"
    MAX_RETRIES = 6
    BACKOFF_BASE = 30
    MAX_TEXTS_PER_REQ = 40

    PAYMENT_ADDED = True 

    VOYAGE_RATE_LIMITS = {
        True:  {"RPM": 50, "TPM": 400_000, "SLEEP": 1.0},
        False: {"RPM": 3,  "TPM": 10_000,  "SLEEP": 20.0}
    }
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

class TrainConfig:
    """Training configuration with GNN parameters."""
    
    # === Basic Training ===
    seed = 42
    use_cuda = True
    batch_size = 32
    epoch_num = 200
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    # === Data Splits ===
    train_ratio = 0.7
    valid_ratio = 0.15
    # test_ratio = 0.15 (implicit)
    
    # === Model Architecture ===
    window_size = 20        # Temporal window (days)
    dim = 256               # Hidden dimension
    output_dim = 3          # Classes: DOWN, FLAT, UP
    num_head = 4            # Attention heads
    
    # === Feature Dimensions ===
    news_embed_dim = 128    # Node feature dimension from KG
    
    # ===== GNN PARAMETERS (NEW) =====
    use_gnn = True                  # Enable Graph Neural Network
    gnn_type = "gat"               # Options: "sage", "gat"
    gnn_hidden_dim = 512            # GNN hidden layer dimension
    gnn_num_layers = 3              # Number of GNN layers
    gnn_heads = 4                   # GAT attention heads (if gnn_type="gat")
    gnn_pool = "attention"          # Pooling method: "mean", "max", "attention"
    gnn_dropout = 0.1               # Dropout in GNN layers
    
    # ===== LOSS FUNCTION =====
    use_focal_loss = True           # Use Focal Loss for imbalanced data
    focal_gamma = 2.0               # Focal loss gamma parameter
    use_label_smoothing = False     # Alternative: Label smoothing
    label_smoothing = 0.1           # Smoothing factor if enabled
    
    # ===== ENTITY RESOLUTION (KG) =====
    use_improved_resolver = True    # Use 2-step resolution
    resolver_kmeans_k = 64         # KMeans clusters for entity resolution
    resolver_min_cluster = 3        # Min entities per cluster
    
    # ===== KG PROCESSING =====
    kg_window_days = 20             # Days to aggregate for graph building
    kg_top_triples = 5              # Top triples per article
    kg_use_voyage = True            # Use Voyage AI for embeddings
    kg_allow_llm = False            # Allow LLM calls for missing data


class ModelConfig:
    """Model-specific hyperparameters."""
    
    # === Price Encoder ===
    price_lstm_hidden = 64
    price_lstm_layers = 2
    
    # === Macro Encoder ===
    macro_lstm_hidden = 64
    macro_lstm_layers = 2
    
    # === News/KG Encoder (GNN) ===
    # Node feature input dimension
    kg_node_dim = 128  # Should match TrainConfig.news_embed_dim
    
    # GNN architecture (overridden by TrainConfig if use_gnn=True)
    kg_use_sage = False
    kg_use_gat = True
    kg_sage_aggr = "mean"  # "mean", "max", "sum"
    
    # === Fusion ===
    fusion_num_heads = 4
    fusion_dropout = 0.1
    
    # === Predictor ===
    predictor_hidden_dim = 256
    predictor_dropout = 0.1


class PathConfig:
    """Specific file paths for outputs."""
    
    @staticmethod
    def get_model_save_path(experiment_name: str = None) -> str:
        """Generate model save path with timestamp."""
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_dir = os.path.join(GlobalConfig.BASE_DIR, "output", "models")
        os.makedirs(save_dir, exist_ok=True)
        
        return os.path.join(save_dir, f"{experiment_name}.pt")
    
    @staticmethod
    def get_log_path(experiment_name: str = None) -> str:
        """Generate log file path."""
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        log_dir = os.path.join(GlobalConfig.BASE_DIR, "output", "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        return os.path.join(log_dir, f"{experiment_name}.log")


# === VALIDATION ===
def validate_config():
    """Validate configuration parameters."""
    errors = []
    
    # Check paths exist
    for path in [GlobalConfig.DATA_DIR, GlobalConfig.RAW_PATH, GlobalConfig.INTERIM_PATH]:
        if not os.path.exists(path):
            errors.append(f"Path does not exist: {path}")
    
    # Check GNN config consistency
    if TrainConfig.use_gnn:
        if TrainConfig.gnn_type not in ["sage", "gat"]:
            errors.append(f"Invalid gnn_type: {TrainConfig.gnn_type}")
        
        if TrainConfig.gnn_pool not in ["mean", "max", "attention"]:
            errors.append(f"Invalid gnn_pool: {TrainConfig.gnn_pool}")
    
    # Check dimensions
    if TrainConfig.news_embed_dim != ModelConfig.kg_node_dim:
        errors.append(f"Dimension mismatch: news_embed_dim={TrainConfig.news_embed_dim}, kg_node_dim={ModelConfig.kg_node_dim}")
    
    if errors:
        print("⚠️ Configuration Errors:")
        for err in errors:
            print(f"   - {err}")
        return False
    
    print("✅ Configuration validated successfully")
    return True


if __name__ == "__main__":
    print("=== Configuration Summary ===\n")
    
    print("📁 Global Config:")
    print(f"   Data Dir: {GlobalConfig.DATA_DIR}")
    print(f"   Tickers: {GlobalConfig.TICKERS}")
    print(f"   Date Range: {GlobalConfig.START_DATE} to {GlobalConfig.END_DATE}")
    
    print("\n🎯 Training Config:")
    print(f"   Batch Size: {TrainConfig.batch_size}")
    print(f"   Epochs: {TrainConfig.epoch_num}")
    print(f"   Learning Rate: {TrainConfig.learning_rate}")
    print(f"   Window Size: {TrainConfig.window_size}")
    
    print("\n🧠 GNN Config:")
    print(f"   Use GNN: {TrainConfig.use_gnn}")
    if TrainConfig.use_gnn:
        print(f"   GNN Type: {TrainConfig.gnn_type.upper()}")
        print(f"   Hidden Dim: {TrainConfig.gnn_hidden_dim}")
        print(f"   Num Layers: {TrainConfig.gnn_num_layers}")
        print(f"   Pooling: {TrainConfig.gnn_pool}")
    
    print("\n🔍 Validating...")
    validate_config()