# configs/config.py
import os 

class GlobalConfig:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    #Data level paths
    RAW_PATH = os.path.join(DATA_DIR, 'raw')
    INTERIM_PATH = os.path.join(DATA_DIR, 'interim')
    PROCESSED_PATH = os.path.join(DATA_DIR, 'processed')

    
    RAW_PRICE_PATH = os.path.join(RAW_PATH, 'market_price')
    RAW_MACRO_PATH = os.path.join(RAW_PATH, 'macro')
    RAW_NEWS_PATH = os.path.join(RAW_PATH, 'news')
    RAW_FILINGS_PATH = os.path.join(RAW_PATH, 'filings')


    NEWS_EMBEDDING_OUTPUT_PATH = os.path.join(INTERIM_PATH, 'news_headline_embeddings')

    
    
    # --- API Keys (Load from env or set here) ---
    ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "YOUR_API_KEY")
    ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "YOUR_SECRET_KEY")
    VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "pa-CcuVotIUxYiHLOYmsJBAMESqQQnro316zjhoWs8LGrW")
    
    # --- Data Settings ---
    START_DATE = '2024-05-01'
    END_DATE = '2025-06-26'
    
    # List cổ phiếu (Example)
    TICKERS = ["TSLA", "AMZN", "MSFT", "NFLX"] 
    
    # Mapping để sửa lỗi tên không nhất quán (Logic từ file gốc)
    TICKER_MAPPING = {
        'AMZN': 'AMZN', 'Amazon': 'AMZN',
        'MSFT': 'MSFT', 'Microsoft': 'MSFT',
        'TSLA': 'TSLA', 'Tesla': 'TSLA',
        'NFLX': 'NFLX', 'Netflix': 'NFLX'
    }
    # Macro Symbols
    MACRO_SYMBOLS = {
        'vix': '^VIX',
        'sp500': '^GSPC',
        'dxy': 'DX-Y.NYB',
        'wti': 'CL=F'
    }
    
    # --- Voyage Embedding Settings (New) ---
    EMBED_MODEL = "voyage-3-large"
    MAX_RETRIES = 6
    BACKOFF_BASE = 30
    MAX_TEXTS_PER_REQ = 40
    
    # Toggle Payment Mode (True/False)
    PAYMENT_ADDED = True 
    
    # Rate Limits config
    VOYAGE_RATE_LIMITS = {
        True:  {"RPM": 50, "TPM": 400_000, "SLEEP": 1.0},
        False: {"RPM": 3,  "TPM": 10_000,  "SLEEP": 20.0}
    }


class TrainConfig:
    # reproducibility
    seed = 42
    use_cuda = True

    # data
    train_ratio = 0.70  # 70% để học
    valid_ratio = 0.15  # 15% để tinh chỉnh (chọn best model)
    train_batch_size = 64

    # Window Size (T): Số ngày quá khứ dùng để dự báo.
    window_size = 20
    news_embed_dim = 1024 
    
    # --- MODEL HYPERPARAMETERS ---
    # Model Hidden Dimension (D): Kích thước vector sau khi encode và fusion
    dim = 128
    # Output classes (Down, Flat, Up)
    output_dim = 3 
    #Attention heads
    num_head = 4 
    
    # training
    epoch_num = 200
    # learning_rate = 1e-4
    learning_rate = 1e-4
    weight_decay = 1e-5