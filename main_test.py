# main_test.py — V4
"""
Pipeline orchestrator V4.

Luồng:
  Phase A: Price + Macro data (Yahoo + FRED)
  Phase B: News alignment
  Phase B.1: Stage A — LLM extraction (extract_corpus.py)  [nếu cần]
  Phase B.2: Voyage embedding (embed_news.py)              [nếu cần]
  Phase C: Build unified_dataset.pkl (builder.py V4)
"""

import os
import json
import pandas as pd

from configs.config import GlobalConfig
from data_pipeline.fetchers.yahoo_fetcher import YahooFetcher
from data_pipeline.processors.price_processor import PriceProcessor
from data_pipeline.processors.macro_processor import MacroProcessor
from data_pipeline.processors.news_processor import NewsProcessor
from data_pipeline.builder import DatasetBuilder


def _check_news_embeddings(emb_path: str, n_sample: int = 5) -> bool:
    """Sanity check news_embeddings.json format."""
    if not os.path.exists(emb_path):
        print(f"  news_embeddings.json not found: {emb_path}")
        return False
    try:
        with open(emb_path) as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or len(obj) == 0:
            print("  news_embeddings.json is empty.")
            return False
        # Check a sample
        sample_dates = list(obj.keys())[:n_sample]
        ok = 0
        for d in sample_dates:
            tickers = obj[d]
            if not isinstance(tickers, dict):
                continue
            for t, emb in tickers.items():
                if isinstance(emb, list) and len(emb) == 1024:
                    ok += 1
                    break
        print(f"  Sanity check: {ok}/{len(sample_dates)} sampled dates have 1024D embeddings")
        return ok > 0
    except Exception as e:
        print(f"  Error reading news_embeddings.json: {e}")
        return False


def run_pipeline():
    print("Pipeline V4 starting...")

    EXISTING_NEWS_PATH = os.path.join(
        GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet"
    )
    if not os.path.exists(EXISTING_NEWS_PATH):
        print(f"News file not found: {EXISTING_NEWS_PATH}")
        return

    # ── Phase A: Price + Macro ────────────────────────────────────────────────
    print("\n--- Phase A: Price + Macro ---")
    yahoo = YahooFetcher()
    os.makedirs(GlobalConfig.RAW_PRICE_PATH, exist_ok=True)
    os.makedirs(GlobalConfig.RAW_MACRO_PATH, exist_ok=True)
    os.makedirs(GlobalConfig.PROCESSED_PATH, exist_ok=True)

    raw_price_list = yahoo.download_data(
        GlobalConfig.START_DATE, GlobalConfig.END_DATE, GlobalConfig.TICKERS
    )
    raw_macro = yahoo.fetch_macro_indicators(
        GlobalConfig.START_DATE, GlobalConfig.END_DATE, GlobalConfig.MACRO_SYMBOLS
    )

    price_proc = PriceProcessor()
    macro_proc = MacroProcessor()
    price_dict = price_proc.combine_to_nested_dict(raw_price_list, GlobalConfig.TICKERS)
    processed_price_macro = macro_proc.process_and_enrich(price_dict, raw_macro)
    trading_dates = list(processed_price_macro.keys())
    print(f"  {len(trading_dates)} trading days")

    # ── Phase B: News alignment ───────────────────────────────────────────────
    print("\n--- Phase B: News alignment ---")
    news_proc = NewsProcessor()
    processed_news = pd.read_parquet(EXISTING_NEWS_PATH)
    print(f"  Loaded {len(processed_news):,} news records")

    if "headline" in processed_news.columns and "title" not in processed_news.columns:
        processed_news = processed_news.rename(columns={"headline": "title"})
    if not pd.api.types.is_datetime64_any_dtype(processed_news["date"]):
        processed_news["date"] = pd.to_datetime(processed_news["date"]).dt.date

    aligned_news = news_proc.align_to_trading_days(processed_news, trading_dates)
    print(f"  After alignment: {len(aligned_news)} records")

    # ── Phase B.1: Stage A — LLM extraction (optional) ───────────────────────
    print("\n--- Phase B.1: LLM extraction (Stage A) ---")
    cache_dir = GlobalConfig.kg_cache_dir()
    cache_files = []
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]

    if cache_files:
        print(f"  Found {len(cache_files)} cached articles")
        ans = input("  Run Stage A to extract more (may cost API)? (y/n): ").strip().lower()
        if ans == "y":
            from extract_corpus import run_stage_a
            run_stage_a(
                news_df=processed_news,
                cache_dir=cache_dir,
                use_gemini_batch=False,
                max_concurrent=5,
                min_relevance=GlobalConfig.KG_MIN_RELEVANCE,
                min_confidence=GlobalConfig.KG_MIN_CONFIDENCE,
            )
    else:
        print("  No cache found.")
        ans = input("  Run Stage A now (requires GEMINI_API_KEY)? (y/n): ").strip().lower()
        if ans == "y":
            from extract_corpus import run_stage_a
            run_stage_a(
                news_df=processed_news,
                cache_dir=cache_dir,
                use_gemini_batch=False,
                max_concurrent=5,
                min_relevance=GlobalConfig.KG_MIN_RELEVANCE,
                min_confidence=GlobalConfig.KG_MIN_CONFIDENCE,
            )
        else:
            print("  Skipping extraction. News embeddings will be zeros.")

    # ── Phase B.2: Voyage embedding (embed_news.py) ───────────────────────────
    print("\n--- Phase B.2: Voyage embedding ---")
    emb_path = os.path.join(
        GlobalConfig.INTERIM_PATH, "kg_embeddings", "news_embeddings.json"
    )

    if os.path.exists(emb_path):
        ok = _check_news_embeddings(emb_path)
        if ok:
            ans = input("  Reuse existing news_embeddings.json? (y/n): ").strip().lower()
            if ans == "y":
                print("  Reusing existing embeddings.")
            else:
                _run_embed(processed_news, cache_dir, emb_path)
        else:
            print("  Existing file has issues — rebuilding...")
            _run_embed(processed_news, cache_dir, emb_path)
    else:
        print("  news_embeddings.json not found.")
        voyage_key = os.getenv("VOYAGE_API_KEY", GlobalConfig.VOYAGE_API_KEY)
        if voyage_key and voyage_key not in ("", "---"):
            print("  Running embed_news.py...")
            _run_embed(processed_news, cache_dir, emb_path)
        else:
            print("  VOYAGE_API_KEY not set — embeddings will be zeros.")
            print("  Set key and run: python embed_news.py")

    # ── Phase C: Build unified dataset ───────────────────────────────────────
    print("\n--- Phase C: Building unified_dataset.pkl ---")
    builder = DatasetBuilder()

    filing_path = os.path.join(
        GlobalConfig.RAW_FILINGS_PATH, "final_summary_filing_data.parquet"
    )
    dummy_filing = False
    if not os.path.exists(filing_path):
        print(f"  Filing not found — using dummy")
        pd.DataFrame(
            columns=["filedAt", "ticker", "formType", "content_summary"]
        ).to_parquet("dummy_filings.parquet")
        filing_path   = "dummy_filings.parquet"
        dummy_filing  = True

    dataset = builder.create_synchronized_data(
        price_macro_dict=processed_price_macro,
        news_df=aligned_news,
        filing_path=filing_path,
        embedding_path=emb_path if os.path.exists(emb_path) else None,
    )
    builder.save(dataset, filename="unified_dataset_test.pkl")

    if dummy_filing and os.path.exists("dummy_filings.parquet"):
        os.remove("dummy_filings.parquet")

    print("\nPipeline V4 complete!")
    print("Next: python main.py")


def _run_embed(news_df, cache_dir, emb_path):
    from embed_news import run_embed_news
    voyage_cache = GlobalConfig.kg_voyage_cache_dir()
    run_embed_news(
        news_df=news_df,
        cache_dir=cache_dir,
        output_path=emb_path,
        voyage_cache=voyage_cache,
        window_days=GlobalConfig.KG_WINDOW_EMBED,
        min_relevance=GlobalConfig.KG_MIN_RELEVANCE,
        min_confidence=GlobalConfig.KG_MIN_CONFIDENCE,
    )


if __name__ == "__main__":
    run_pipeline()