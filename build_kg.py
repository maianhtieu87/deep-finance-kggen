# build_kg.py

import os
import pandas as pd
from datetime import datetime

from configs.config import GlobalConfig
from data_pipeline.processors.news_processor import NewsProcessor, KGGenNewsEmbedder


def main():
    print("🚀 BUILD_KG ONLY")

    news_path = os.path.join(GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet")
    print(f"📥 Load news: {news_path}")
    if not os.path.exists(news_path):
        print(f"❌ Missing news file: {news_path}")
        return

    df = pd.read_parquet(news_path)
    print(f"   rows={len(df)} cols={list(df.columns)}")

    news_proc = NewsProcessor()
    aligned_news = news_proc.align_to_trading_days(df, trading_days=None)
    print(f"✅ aligned_news rows={len(aligned_news)}")

    kg_index_path = os.path.join(GlobalConfig.INTERIM_PATH, "kg_embeddings", "embedded_kg.json")
    extracted_dir = os.path.join(GlobalConfig.INTERIM_PATH, "kg", "extracted_triples")

    # init embedder (IMPORTANT: top_triples_per_day=None => keep all per day)
    embedder = KGGenNewsEmbedder(
        interim_root=GlobalConfig.INTERIM_PATH,
        top_triples_per_article=5,
        top_triples_per_day=None,          # ✅ NO top-k/day
        use_voyage_resolution=True,        # ✅ voyage resolution
    )

    if os.path.exists(kg_index_path):
        print(f"⚠️ Found existing KG index: {kg_index_path}")
        ans = input("→ Rebuild LLM extraction (costly)? (y/n): ").strip().lower()

        if ans == "y":
            print("🧨 Rebuilding extraction + graphs (LLM calls)...")
            embedder.process_and_save(aligned_news)
        else:
            # if extracted triples exist => graph-only rebuild
            if os.path.exists(extracted_dir) and len(os.listdir(extracted_dir)) > 0:
                print("✅ Reuse extracted_triples. Rebuild graphs ONLY (NO LLM).")
                embedder.rebuild_graph_only()
            else:
                print("❌ No extracted_triples found. Cannot rebuild graph-only. Choose y to extract.")
    else:
        print("No KG index found. Running extraction + graphs...")
        embedder.process_and_save(aligned_news)


if __name__ == "__main__":
    main()
