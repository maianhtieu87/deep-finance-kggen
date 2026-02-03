# main_test.py

import os
import json
import pandas as pd

from configs.config import GlobalConfig
from data_pipeline.fetchers.yahoo_fetcher import YahooFetcher
from data_pipeline.processors.price_processor import PriceProcessor
from data_pipeline.processors.macro_processor import MacroProcessor
from data_pipeline.processors.news_processor import NewsProcessor, KGGenNewsEmbedder
from data_pipeline.builder import DatasetBuilder


def _quick_check_kg_index(kg_index_path: str, n_samples: int = 5) -> bool:
    """
    Sanity check format + existence of kg_tensor_path
    Expect:
      embedded_kg.json: { "YYYY-MM-DD": [ {"date":..., "equity":..., "kg_tensor_path":...}, ... ], ... }
    """
    if not os.path.exists(kg_index_path):
        print(f"❌ KG index not found: {kg_index_path}")
        return False

    try:
        with open(kg_index_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as e:
        print(f"❌ Cannot read KG index JSON: {e}")
        return False

    if not isinstance(obj, dict) or len(obj) == 0:
        print("❌ KG index JSON is empty or not a dict.")
        return False

    # pick some records
    checked = 0
    missing = 0

    for date_str, recs in obj.items():
        if not isinstance(recs, list):
            continue
        for rec in recs:
            if not isinstance(rec, dict):
                continue
            if "kg_tensor_path" not in rec:
                continue
            checked += 1
            p = rec["kg_tensor_path"]
            if not isinstance(p, str) or not os.path.exists(p):
                missing += 1
            if checked >= n_samples:
                break
        if checked >= n_samples:
            break

    if checked == 0:
        print("⚠️ KG index has no usable records with 'kg_tensor_path'.")
        return False

    if missing > 0:
        print(f"⚠️ KG index check: {missing}/{checked} sampled tensor paths are missing.")
        print("   → Có thể bạn đã move folder data/interim/kg hoặc rebuild chưa xong.")
        # vẫn return True vì có thể sample trúng missing; nhưng bạn nên cân nhắc
        return True

    print(f"✅ KG index sanity-check OK. Sampled {checked} tensor paths exist.")
    return True


def run_test_pipeline_skipping_news_fetch():
    print("🚀 STARTING TEST PIPELINE (Skipping News Fetching)...")

    EXISTING_NEWS_PATH = os.path.join(GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet")
    if not os.path.exists(EXISTING_NEWS_PATH):
        print(f"❌ ERROR: Không tìm thấy file tại {EXISTING_NEWS_PATH}")
        return

    # ===== PHASE A: PRICE + MACRO =====
    print("\n--- Phase A: Fetching (Price & Macro only) ---")
    yahoo = YahooFetcher()
    os.makedirs(GlobalConfig.RAW_PRICE_PATH, exist_ok=True)
    os.makedirs(GlobalConfig.RAW_MACRO_PATH, exist_ok=True)
    os.makedirs(GlobalConfig.PROCESSED_PATH, exist_ok=True)

    print(f"   Downloading Price Data ({GlobalConfig.START_DATE} to {GlobalConfig.END_DATE})...")
    raw_price_list = yahoo.download_data(
        GlobalConfig.START_DATE,
        GlobalConfig.END_DATE,
        GlobalConfig.TICKERS
    )

    print("   Downloading Macro Indicators...")
    raw_macro = yahoo.fetch_macro_indicators(
        GlobalConfig.START_DATE,
        GlobalConfig.END_DATE,
        GlobalConfig.MACRO_SYMBOLS
    )

    # ===== PHASE B: PROCESS =====
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
    processed_news = pd.read_parquet(EXISTING_NEWS_PATH)
    print(f"   Loaded {len(processed_news)} news records.")

    if "headline" in processed_news.columns and "title" not in processed_news.columns:
        processed_news = processed_news.rename(columns={"headline": "title"})
        print("   ✅ Renamed 'headline' -> 'title'.")

    if not pd.api.types.is_datetime64_any_dtype(processed_news["date"]):
        processed_news["date"] = pd.to_datetime(processed_news["date"]).dt.date

    print("   Aligning news to current Trading Days...")
    aligned_news = news_proc.align_to_trading_days(processed_news, trading_dates)
    print(f"   News after alignment: {len(aligned_news)} records.")

    # ===== PHASE B.1: KG OFFLINE (REUSE) =====
    print("\n--- Phase B.1: KG (reuse existing outputs by default) ---")

    kg_index_path = os.path.join(GlobalConfig.INTERIM_PATH, "kg_embeddings", "embedded_kg.json")

    if os.path.exists(kg_index_path):
        print(f"   ✅ Found KG index: {kg_index_path}")

        ok = _quick_check_kg_index(kg_index_path, n_samples=5)
        if not ok:
            print("   ⚠️ KG index format/path seems problematic.")
            ans = input("   → Bạn có muốn rebuild KG lại từ đầu (tốn LLM)? (y/n): ").strip().lower()
            if ans == "y":
                print("   🧨 Rebuilding KG (LLM extraction + graph build)...")
                embedder = KGGenNewsEmbedder(
                    interim_root=GlobalConfig.INTERIM_PATH,
                    top_triples_per_article=5,
                    top_triples_per_day=None,   # giữ hết per-day (no top-k/day)
                    # NOTE: voyage resolution nếu bạn đã implement trong news_processor
                    # và đã set VOYAGE_API_KEY env. Nếu chưa, hãy để module tự handle.
                )
                kg_index_path = embedder.process_and_save(aligned_news)
            else:
                print("   ❌ Không rebuild nhưng KG index hiện không ổn. Dừng để tránh builder lỗi.")
                return
        else:
            # ✅ default: reuse (NO rebuild graph-only, NO voyage, NO llm)
            ans = input("   → Reuse KG đã build sẵn (skip build KG)? (y/n): ").strip().lower()
            if ans == "y":
                print("   ✅ Reusing existing KG index. (NO LLM / NO Voyage / NO graph rebuild)")
            else:
                print("   🧨 You chose to rebuild KG (LLM extraction + graph build)...")
                embedder = KGGenNewsEmbedder(
                    interim_root=GlobalConfig.INTERIM_PATH,
                    top_triples_per_article=5,
                    top_triples_per_day=None,   # giữ hết per-day (no top-k/day)
                )
                kg_index_path = embedder.process_and_save(aligned_news)
    else:
        print("   ❌ No KG index found.")
        ans = input("   → Build KG now (tốn LLM)? (y/n): ").strip().lower()
        if ans != "y":
            print("   ❌ Không có KG index để dùng. Dừng.")
            return
        embedder = KGGenNewsEmbedder(
            interim_root=GlobalConfig.INTERIM_PATH,
            top_triples_per_article=5,
            top_triples_per_day=None,
        )
        kg_index_path = embedder.process_and_save(aligned_news)

    embedding_json_path = kg_index_path

    # ===== PHASE C: FINAL UNION =====
    print("\n--- Phase C: Building Union File ---")
    builder = DatasetBuilder()

    filing_path = os.path.join(GlobalConfig.RAW_FILINGS_PATH, "final_summary_filing_data.parquet")
    if not os.path.exists(filing_path):
        print(f"   ⚠️ Warning: Filing file not found at {filing_path}. Creating dataset without filings.")
        pd.DataFrame(columns=["filedAt", "ticker", "formType", "content_summary"]).to_parquet("dummy_filings.parquet")
        filing_path = "dummy_filings.parquet"

    dataset = builder.create_synchronized_data(
        processed_price_macro,
        aligned_news,
        filing_path,
        embedding_path=embedding_json_path
    )

    builder.save(dataset, filename="unified_dataset_test.pkl")

    if os.path.exists("dummy_filings.parquet"):
        os.remove("dummy_filings.parquet")

    print("\n✅ TEST PIPELINE COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_test_pipeline_skipping_news_fetch()
