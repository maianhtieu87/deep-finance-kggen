# configs/config.py
"""
Configuration V5.3


MACRO_SYMBOLS: 6 symbols fetched from Yahoo, but only 5 become model features.
  ^TNX and ^IRX are intermediate — used only to compute yield_spread fallback.


Final macro_dim = 5: vix, sp500, dxy, wti, yield_spread_10y_2y
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


    # TICKERS = ["TSLA", "AAPL", "AMZN", "MSFT", "GOOGL"]
    TICKERS = ["TSLA", "AAPL", "AMZN", "MSFT", "GOOGL", "META", "BA", "JPM", "WMT"]


    TICKER_SECTOR = {
        "TSLA": "Consumer Discretionary", "AAPL": "Technology",
        "AMZN": "Consumer Discretionary", "MSFT": "Technology",
        "GOOGL": "Technology",            "META": "Technology",
        "BA": "Industrials",              "JPM": "Financials",
        "WMT": "Consumer Staples",
    }


    TICKER_MAPPING = {t: t for t in TICKERS}


    # ── Yahoo fetch targets ───────────────────────────────────────────────────
    # 6 symbols fetched, but only 5 become model features (see MacroProcessor).
    # ^TNX and ^IRX are fallback sources for yield_spread_10y_2y computation only.
    MACRO_SYMBOLS = [
        "^GSPC",      # → sp500            [MODEL FEATURE]
        "^VIX",       # → vix              [MODEL FEATURE]
        "CL=F",       # → wti              [MODEL FEATURE]
        "DX-Y.NYB",   # → dxy              [MODEL FEATURE]
        "^TNX",       # → yield_spread fallback (10Y proxy when FRED unavailable)
        "^IRX",       # → yield_spread fallback (2Y  proxy when FRED unavailable)
    ]
    # yield_spread_10y_2y comes from FRED (DGS10 - DGS2) or above fallback → [MODEL FEATURE]


    KG_MIN_RELEVANCE  = 0.50
    KG_MIN_CONFIDENCE = 0.65
    KG_MAX_CONCURRENT = 5
    KG_MAX_ARTICLE_CHARS  = 15000
    KG_ENABLE_CHUNKING    = False
    KG_CHUNK_SIZE         = 5000
    KG_CHUNK_OVERLAP      = 0
    KG_NORMALIZE_SUBJECT      = True
    KG_MAX_PER_ANALYST_FIRM   = 1
    KG_MAX_PER_ANALYST_RATING = 1
    KG_MAX_ARTICLES_PER_CALL  = 10
    KG_ASYNC_MAX_RETRIES   = 3
    KG_ASYNC_BACKOFF_BASE  = 10.0
    KG_ASYNC_REQUEST_DELAY = 1.0
    KG_BATCH_CHUNK_SIZE    = 5000
    KG_BATCH_POLL_INTERVAL = 30
    KG_BATCH_MAX_WAIT      = 86400


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
    def kg_cache_dir(cls):
        return os.path.join(cls.INTERIM_PATH, cls.KG_CACHE_DIRNAME)


    @classmethod
    def kg_voyage_cache_dir(cls):
        return os.path.join(cls.INTERIM_PATH, cls.KG_VOYAGE_CACHE_DIRNAME)

class TrainConfig:
    seed = 42;  use_cuda = True;  
    batch_size = 32
    epoch_num = 150;  
    learning_rate = 1e-4;  
    weight_decay = 1e-4
   
    train_ratio = 0.7;  
    valid_ratio = 0.15
   
    window_size = 20;  
    dim = 64;  
    output_dim = 3;  
    num_head = 2
    news_embed_dim = 1024
   
    use_focal_loss = True;  
    focal_gamma = 2.0
    use_label_smoothing = False;  
    label_smoothing = 0.1
    drop_out = 0.1




class ModelConfig:
    price_lstm_hidden = 64;  price_lstm_layers = 2
    macro_lstm_hidden = 64;  macro_lstm_layers = 2
    fusion_num_heads  = TrainConfig.num_head;  fusion_dropout = 0.1
    predictor_hidden_dim = TrainConfig.dim;    predictor_dropout = 0.1




class PathConfig:
    @staticmethod
    def get_model_save_path(name=None):
        name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        d = os.path.join(GlobalConfig.BASE_DIR, "output", "models")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{name}.pt")


    @staticmethod
    def get_log_path(name=None):
        name = name or datetime.now().strftime("%Y%m%d_%H%M%S")
        d = os.path.join(GlobalConfig.BASE_DIR, "output", "logs")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{name}.log")

