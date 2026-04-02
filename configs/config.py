# configs/config.py
"""
Configuration V5.2 — 9-ticker universe.

V5.2 changes vs V5.1:
  - Removed duplicate KG_MAX_CONCURRENT declaration
  - Removed kg_tensor from builder.py (no longer needed)
  - Added KG_BATCH_CHUNK_SIZE for Gemini Batch API optimization
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

    # ─────────────────────────────────────────────────────────────────────────
    # KG EXTRACTION THRESHOLDS  ← SINGLE SOURCE OF TRUTH
    # ─────────────────────────────────────────────────────────────────────────
    KG_MIN_RELEVANCE  = 0.50   # skip triple if relevance_to_ticker < this
    KG_MIN_CONFIDENCE = 0.65   # skip triple if confidence < this
    KG_MAX_CONCURRENT = 5      # async concurrent Gemini API calls

    # ── V5.1: Article handling (replaces chunk-based approach) ────────────
    KG_MAX_ARTICLE_CHARS  = 15000
    KG_ENABLE_CHUNKING    = False
    KG_CHUNK_SIZE         = 5000
    KG_CHUNK_OVERLAP      = 0

    # ── V5.1: Dedup & quality control ─────────────────────────────────────
    KG_NORMALIZE_SUBJECT      = True
    KG_MAX_PER_ANALYST_FIRM   = 1
    KG_MAX_PER_ANALYST_RATING = 1
    KG_MAX_ARTICLES_PER_CALL  = 10    # max articles concat per LLM call (cost optimization)

    # ── V5.2: Async retry config (tránh mất data khi 429) ────────────────
    KG_ASYNC_MAX_RETRIES   = 3
    KG_ASYNC_BACKOFF_BASE  = 10.0
    KG_ASYNC_REQUEST_DELAY = 1.0

    # ── V5.2: Gemini Batch API config (50% cost saving) ──────────────────
    KG_BATCH_CHUNK_SIZE    = 5000   # max articles per batch job
    KG_BATCH_POLL_INTERVAL = 30     # seconds between status checks
    KG_BATCH_MAX_WAIT      = 86400  # 24h max wait per batch job

    # ── Voyage embedding ──────────────────────────────────────────────────
    EMBED_MODEL       = "voyage-finance-2"
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
    epoch_num     = 150
    learning_rate = 1e-4
    weight_decay  = 1e-4

    train_ratio = 0.7
    valid_ratio = 0.15

    window_size = 20
    dim         = 128
    output_dim  = 3
    num_head    = 2
    news_embed_dim  = 1024

    use_focal_loss      = True
    focal_gamma         = 2.0
    use_label_smoothing = False
    label_smoothing     = 0.1

    drop_out = 0.1


class ModelConfig:

    price_lstm_hidden = 64
    price_lstm_layers = 2

    macro_lstm_hidden = 64
    macro_lstm_layers = 2

    fusion_num_heads = TrainConfig.num_head
    fusion_dropout   = 0.1

    predictor_hidden_dim = TrainConfig.dim
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
        errors.append(f"news_embed_dim should be 1024 but is {TrainConfig.news_embed_dim}")
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  - {e}")
        return False
    print("Configuration validated (V5.2)")
    print(f"  KG thresholds: min_relevance={GlobalConfig.KG_MIN_RELEVANCE}  "
          f"min_confidence={GlobalConfig.KG_MIN_CONFIDENCE}")
    print(f"  Article handling: max_chars={GlobalConfig.KG_MAX_ARTICLE_CHARS}  "
          f"chunking={'ON' if GlobalConfig.KG_ENABLE_CHUNKING else 'OFF'}")
    print(f"  Batch API: chunk_size={GlobalConfig.KG_BATCH_CHUNK_SIZE}")
    return True


if __name__ == "__main__":
    print("=== Configuration V5.2 ===")
    print(f"Tickers  : {GlobalConfig.TICKERS}")
    print(f"Date     : {GlobalConfig.START_DATE} -> {GlobalConfig.END_DATE}")
    print(f"News dim : {TrainConfig.news_embed_dim} (Voyage-finance-2)")
    print(f"KG thresholds: rel>={GlobalConfig.KG_MIN_RELEVANCE}  conf>={GlobalConfig.KG_MIN_CONFIDENCE}")
    print(f"Article  : max_chars={GlobalConfig.KG_MAX_ARTICLE_CHARS}  chunking={GlobalConfig.KG_ENABLE_CHUNKING}")
    print(f"Batch    : chunk_size={GlobalConfig.KG_BATCH_CHUNK_SIZE}")
    print()
    validate_config()