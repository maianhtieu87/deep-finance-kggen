import os 

class GlobalConfig:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')

    # Data level paths
    RAW_PATH = os.path.join(DATA_DIR, 'raw')
    INTERIM_PATH = os.path.join(DATA_DIR, 'interim')
    PROCESSED_PATH = os.path.join(DATA_DIR, 'processed')

    RAW_PRICE_PATH = os.path.join(RAW_PATH, 'market_price')
    RAW_MACRO_PATH = os.path.join(RAW_PATH, 'macro')
    RAW_NEWS_PATH = os.path.join(RAW_PATH, 'news')
    RAW_FILINGS_PATH = os.path.join(RAW_PATH, 'filings')

    # Cũ: text-embedding news (Voyage) – giữ lại nếu muốn so sánh
    NEWS_EMBEDDING_OUTPUT_PATH = os.path.join(
        INTERIM_PATH,
        'news_headline_embeddings'
    )

    # --- API Keys ---
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_SECRET_KEY")
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "")

    # --- Data Settings ---
    START_DATE = '2024-05-01'
    END_DATE = '2025-06-26'

    TICKERS = ["TSLA", "AMZN", "MSFT", "NFLX"]

    TICKER_MAPPING = {
        'AMZN': 'AMZN', 'Amazon': 'AMZN',
        'MSFT': 'MSFT', 'Microsoft': 'MSFT',
        'TSLA': 'TSLA', 'Tesla': 'TSLA',
        'NFLX': 'NFLX', 'Netflix': 'NFLX'
    }

    MACRO_SYMBOLS = {
        'vix': '^VIX',
        'sp500': '^GSPC',
        'dxy': 'DX-Y.NYB',
        'wti': 'CL=F'
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

class TrainConfig:
    # reproducibility
    seed = 42
    use_cuda = True

    # splits
    train_ratio = 0.70
    valid_ratio = 0.15

    # ⚠ batch_size: dùng trong main.py
    batch_size = 128

    # Window Size (T): số ngày quá khứ dùng để dự báo
    window_size = 20

    # ⚠ NHỚ CHỈNH CHO KHỚP VỚI news_dim TỪ KG/GNN
    news_embed_dim = 256    
    
    # Model Hyperparameters
    dim = 128
    output_dim = 3          # {DOWN, FLAT, UP}
    num_head = 4

    # training
    epoch_num = 200
    learning_rate = 1e-4
    weight_decay = 1e-5