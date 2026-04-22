# main_test.py — V5.6
"""
Pipeline orchestrator V5.6.

V5.6: Embedder switch via config — set TrainConfig.news_embedder in configs/config.py:
  "finbert" → 768D  (FinBERT per-triple, local) → news_embeddings_finbert.json
  "voyage"  → 1024D (Voyage-finance-2, API key) → news_embeddings_voyage.json
All paths and dims are resolved automatically. No other file needs to change.
"""

import os
import json
import pandas as pd

from configs.config import GlobalConfig, TrainConfig
from data_pipeline.fetchers.yahoo_fetcher import YahooFetcher
from data_pipeline.processors.price_processor import PriceProcessor
from data_pipeline.processors.macro_processor import MacroProcessor
from data_pipeline.processors.news_processor import NewsProcessor
from data_pipeline.builder import DatasetBuilder

# Read from config — the ONLY place you need to change is TrainConfig.news_embedder
_ACTIVE_EMBEDDER   = TrainConfig.news_embedder          # "finbert" or "voyage"
_EXPECTED_NEWS_DIM = GlobalConfig.news_emb_dim()        # 768 or 1024
_EMB_PATH          = GlobalConfig.news_emb_path()       # correct JSON for active embedder
_EMBED_CMD         = ("python embed_news.py --finbert"
                      if _ACTIVE_EMBEDDER == "finbert"
                      else "python embed_news.py")


def _check_news_embeddings(emb_path: str, n_sample: int = 5) -> tuple[bool, int]:
    """
    Sanity-check news_embeddings.json.

    Returns (ok: bool, detected_dim: int).
    detected_dim = 0 if file is empty or unreadable.
    """
    if not os.path.exists(emb_path):
        print(f"  news_embeddings.json not found: {emb_path}")
        return False, 0
    try:
        with open(emb_path) as f:
            obj = json.load(f)
        if not isinstance(obj, dict) or len(obj) == 0:
            print("  news_embeddings.json is empty.")
            return False, 0

        sample_dates = list(obj.keys())[:n_sample]
        detected_dim = 0
        ok_count = 0

        for d in sample_dates:
            tickers = obj[d]
            if not isinstance(tickers, dict):
                continue
            for t, emb in tickers.items():
                if isinstance(emb, list) and len(emb) > 0:
                    detected_dim = len(emb)
                    ok_count += 1
                    break

        print(f"  Sanity check: {ok_count}/{len(sample_dates)} sampled dates have "
              f"{detected_dim}D embeddings")
        return ok_count > 0, detected_dim
    except Exception as e:
        print(f"  Error reading news_embeddings.json: {e}")
        return False, 0


def _run_embed_finbert(news_df, cache_dir, emb_path):
    """
    Phase B.2 — FinBERT per-triple embedder (V5, primary path).

    Reads KG triples from cache_dir, encodes each triple via FinBERT [CLS],
    aggregates per (date, ticker) with impact-weighted mean, saves 768D vectors.

    Requires: pip install transformers accelerate
    """
    from embed_news import run_embed_news_finbert
    voyage_cache = GlobalConfig.kg_voyage_cache_dir()  # kept as fallback cache dir
    run_embed_news_finbert(
        news_df=news_df,
        cache_dir=cache_dir,
        output_path=emb_path,
        finbert_model=GlobalConfig.FINBERT_MODEL,
        finbert_cache_dir=GlobalConfig.finbert_cache_dir(),
        min_relevance=GlobalConfig.KG_MIN_RELEVANCE,
        min_confidence=GlobalConfig.KG_MIN_CONFIDENCE,
    )


def _run_embed_voyage(news_df, cache_dir, emb_path):
    """
    Phase B.2 — Legacy Voyage embedder (kept for backward compat).

    Only used if VOYAGE_API_KEY is set and user explicitly chooses
    not to use FinBERT. Produces 1024D vectors.
    """
    from embed_news import run_embed_news
    voyage_cache = GlobalConfig.kg_voyage_cache_dir()
    run_embed_news(
        news_df=news_df,
        cache_dir=cache_dir,
        output_path=emb_path,
        voyage_cache=voyage_cache,
        min_relevance=GlobalConfig.KG_MIN_RELEVANCE,
        min_confidence=GlobalConfig.KG_MIN_CONFIDENCE,
    )


def run_pipeline():
    print("Pipeline V5.6 starting...")
    print(f"  Active embedder : {_ACTIVE_EMBEDDER}")
    print(f"  Expected dim    : {_EXPECTED_NEWS_DIM}D")
    print(f"  Embedding path  : {_EMB_PATH}")

    EXISTING_NEWS_PATH = os.path.join(
        GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet"
    )
    if not os.path.exists(EXISTING_NEWS_PATH):
        print(f"News file not found: {EXISTING_NEWS_PATH}")
        return

    # ── Phase A: Price + Macro ─────────────────────────────────────────────────
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

    # ── Phase B: News alignment ────────────────────────────────────────────────
    print("\n--- Phase B: News alignment ---")
    news_proc = NewsProcessor()
    processed_news = pd.read_parquet(EXISTING_NEWS_PATH)
    print(f"  Loaded {len(processed_news):,} news records")

    # Normalise column names
    if "headline" in processed_news.columns and "title" not in processed_news.columns:
        processed_news = processed_news.rename(columns={"headline": "title"})

    if "date" not in processed_news.columns and processed_news.index.name and \
            "date" in processed_news.index.name.lower():
        processed_news = processed_news.reset_index()

    if "date" not in processed_news.columns:
        DATE_CANDS = [
            "created_at", "createdAt", "published_at", "publishedAt",
            "publish_date", "pub_date", "Date", "DATE", "timestamp", "time", "news_date",
        ]
        date_col = next((c for c in DATE_CANDS if c in processed_news.columns), None)
        if date_col:
            processed_news = processed_news.rename(columns={date_col: "date"})
        else:
            print("🚨 Columns in processed_news:", processed_news.columns.tolist())
            raise KeyError("Cannot find date column in news data!")

    if not pd.api.types.is_datetime64_any_dtype(processed_news["date"]):
        processed_news["date"] = pd.to_datetime(processed_news["date"]).dt.date

    aligned_news = news_proc.align_to_trading_days(processed_news, trading_dates)
    print(f"  After alignment: {len(aligned_news)} records")

    # ── Phase B.1: Stage A — LLM extraction ───────────────────────────────────
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
                max_concurrent=5,
                min_relevance=GlobalConfig.KG_MIN_RELEVANCE,
                min_confidence=GlobalConfig.KG_MIN_CONFIDENCE,
            )
        else:
            print("  Skipping extraction. News embeddings will be zeros.")

    # ── Phase B.2: News embedding ──────────────────────────────────────────────
    _label = {"finbert": "FinBERT per-triple", "voyage": "Voyage-finance-2"}.get(
        _ACTIVE_EMBEDDER, _ACTIVE_EMBEDDER)
    print(f"\n--- Phase B.2: News embedding ({_label}, {_EXPECTED_NEWS_DIM}D) ---")
    emb_path = _EMB_PATH

    if os.path.exists(emb_path):
        ok, detected_dim = _check_news_embeddings(emb_path)

        if ok and detected_dim == _EXPECTED_NEWS_DIM:
            ans = input(
                f"  Existing file has {detected_dim}D vectors ({_label} ✓). Reuse? (y/n): "
            ).strip().lower()
            if ans == "y":
                print(f"  Reusing existing {_label} embeddings.")
            else:
                print(f"  Rebuilding with {_label}...")
                if _ACTIVE_EMBEDDER == "finbert":
                    _run_embed_finbert(processed_news, cache_dir, emb_path)
                else:
                    _run_embed_voyage(processed_news, cache_dir, emb_path)

        elif ok and detected_dim != _EXPECTED_NEWS_DIM:
            print(f"  [WARN] Existing file has {detected_dim}D (expected {_EXPECTED_NEWS_DIM}D). "
                  f"Rebuilding with {_label}...")
            if _ACTIVE_EMBEDDER == "finbert":
                _run_embed_finbert(processed_news, cache_dir, emb_path)
            else:
                _run_embed_voyage(processed_news, cache_dir, emb_path)

        else:
            print(f"  Existing file has issues — rebuilding with {_label}...")
            if _ACTIVE_EMBEDDER == "finbert":
                _run_embed_finbert(processed_news, cache_dir, emb_path)
            else:
                _run_embed_voyage(processed_news, cache_dir, emb_path)

    else:
        print(f"  {_label} embedding file not found: {emb_path}")
        if cache_files:
            print(f"  Found {len(cache_files)} extracted articles in cache.")
            print(f"  Running {_label} embedding...")
            if _ACTIVE_EMBEDDER == "finbert":
                _run_embed_finbert(processed_news, cache_dir, emb_path)
            else:
                _run_embed_voyage(processed_news, cache_dir, emb_path)
        else:
            print("  No KG cache found either.")
            print(f"  Run Stage A first: python extract_corpus.py")
            print(f"  Then run: {_EMBED_CMD}")
            print("  Continuing without news embeddings (all zeros).")

    # ── Phase C: Build unified dataset ────────────────────────────────────────
    print("\n--- Phase C: Building unified_dataset.pkl ---")
    builder = DatasetBuilder()

    filing_path = os.path.join(
        GlobalConfig.RAW_FILINGS_PATH, "final_summary_filing_data.parquet"
    )
    dummy_filing = False
    if not os.path.exists(filing_path):
        print("  Filing not found — using dummy")
        pd.DataFrame(
            columns=["filedAt", "ticker", "formType", "content_summary"]
        ).to_parquet("dummy_filings.parquet")
        filing_path  = "dummy_filings.parquet"
        dummy_filing = True

    dataset = builder.create_synchronized_data(
        price_macro_dict=processed_price_macro,
        news_df=aligned_news,
        filing_path=filing_path,
        embedding_path=emb_path if os.path.exists(emb_path) else None,
    )
    builder.save(dataset, filename="unified_dataset_test.pkl")

    if dummy_filing and os.path.exists("dummy_filings.parquet"):
        os.remove("dummy_filings.parquet")

    print("\nPipeline V5.6 complete!")
    print(f"  Embedder: {_label} | Dim: {_EXPECTED_NEWS_DIM}D")
    print(f"  PKL saved: data/processed/unified_dataset_test.pkl")
    print("Next: python main.py  (or python baselines/run_ablation.py)")


if __name__ == "__main__":
    run_pipeline()