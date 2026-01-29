import os
import pandas as pd
from configs.config import GlobalConfig
from data_pipeline.fetchers.yahoo_fetcher import YahooFetcher
from data_pipeline.processors.price_processor import PriceProcessor
from data_pipeline.processors.macro_processor import MacroProcessor
from data_pipeline.processors.news_processor import NewsProcessor, KGGenNewsEmbedder
from data_pipeline.builder import DatasetBuilder

def run_test_pipeline_skipping_news_fetch():
    print("🚀 STARTING TEST PIPELINE (Skipping News Fetching)...")

    EXISTING_NEWS_PATH = os.path.join(GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet")
    if not os.path.exists(EXISTING_NEWS_PATH):
        print(f"❌ ERROR: Không tìm thấy file tại {EXISTING_NEWS_PATH}")
        return

    # === PHASE A: PRICE + MACRO ===
    print("\n--- Phase A: Fetching (Price & Macro only) ---")
    yahoo = YahooFetcher()
    os.makedirs(GlobalConfig.RAW_PRICE_PATH, exist_ok=True)
    os.makedirs(GlobalConfig.RAW_MACRO_PATH, exist_ok=True)
    os.makedirs(GlobalConfig.PROCESSED_PATH, exist_ok=True)

    print(f"   Downloading Price Data ({GlobalConfig.START_DATE} to {GlobalConfig.END_DATE})...")
    raw_price_list = yahoo.download_data(GlobalConfig.START_DATE, GlobalConfig.END_DATE, GlobalConfig.TICKERS)
    
    print("   Downloading Macro Indicators...")
    raw_macro = yahoo.fetch_macro_indicators(GlobalConfig.START_DATE, GlobalConfig.END_DATE, GlobalConfig.MACRO_SYMBOLS)

    # === PHASE B: PROCESS ===
    print("\n--- Phase B: Processing ---")
    price_proc = PriceProcessor()
    macro_proc = MacroProcessor()
    news_proc = NewsProcessor()

    print("   Processing Price & Macro...")
    price_dict = price_proc.combine_to_nested_dict(raw_price_list, GlobalConfig.TICKERS)
    processed_price_macro = macro_proc.process_and_enrich(price_dict, raw_macro)
    
    trading_dates = list(processed_price_macro.keys())
    print(f"   Detected {len(trading_dates)} trading days.")

    print(f"   📥 Loading existing news from: {EXISTING_NEWS_PATH}")
    try:
        processed_news = pd.read_parquet(EXISTING_NEWS_PATH)
        print(f"   Loaded {len(processed_news)} news records.")

        if 'headline' in processed_news.columns and 'title' not in processed_news.columns:
            processed_news = processed_news.rename(columns={'headline': 'title'})
            print("   ✅ Renamed 'headline' -> 'title'.")

        if not pd.api.types.is_datetime64_any_dtype(processed_news['date']):
            processed_news['date'] = pd.to_datetime(processed_news['date']).dt.date

    except Exception as e:
        print(f"❌ ERROR reading parquet file: {e}")
        return

    print("   Aligning news to current Trading Days...")
    aligned_news = news_proc.align_to_trading_days(processed_news, trading_dates)
    print(f"   News after alignment: {len(aligned_news)} records.")

    # === PHASE B′: KGGEN OFFLINE ===
    print("\n--- Phase B.1: Embedding News (KGGen Extraction + Top-3 Filtering) ---")
    embedder = KGGenNewsEmbedder()
    embedding_json_path = embedder.process_and_save(aligned_news)

    # === PHASE C: FINAL UNION ===
    print("\n--- Phase C: Building Union File ---")
    builder = DatasetBuilder()

    filing_path = os.path.join(GlobalConfig.RAW_FILINGS_PATH, "final_summary_filing_data.parquet")
    if not os.path.exists(filing_path):
        print(f"   ⚠️ Warning: Filing file not found at {filing_path}. Creating dataset without filings.")
        pd.DataFrame(columns=['filedAt', 'ticker', 'formType', 'content_summary']).to_parquet("dummy_filings.parquet")
        filing_path = "dummy_filings.parquet"

    dataset = builder.create_synchronized_data(
        processed_price_macro, 
        aligned_news, 
        filing_path,
        embedding_path=embedding_json_path
    )
    
    builder.save(dataset, filename='unified_dataset_test.pkl')
    if os.path.exists("dummy_filings.parquet"): os.remove("dummy_filings.parquet")

    print("\n✅ TEST PIPELINE COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_test_pipeline_skipping_news_fetch()