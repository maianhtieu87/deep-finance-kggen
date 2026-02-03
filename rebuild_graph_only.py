# rebuild_graph_only.py
import os
import shutil
import time

from configs.config import GlobalConfig
from data_pipeline.processors.news_processor import KGGenNewsEmbedder

def main():
    print("🚀 REBUILD GRAPH-ONLY (NO LLM)")

    # backup old index
    kg_index = os.path.join(GlobalConfig.INTERIM_PATH, "kg_embeddings", "embedded_kg.json")
    if os.path.exists(kg_index):
        ts = time.strftime("%Y%m%d_%H%M%S")
        bak = os.path.join(GlobalConfig.INTERIM_PATH, "kg_embeddings", f"embedded_kg.backup_{ts}.json")
        shutil.copy2(kg_index, bak)
        print(f"🧷 Backup old KG index -> {bak}")

    embedder = KGGenNewsEmbedder(
        interim_root=GlobalConfig.INTERIM_PATH,
        window_days=20,
        kmeans_k=128,
        top_triples_per_article=5,

        # QUAN TRỌNG: graph-only => không cho phép gọi LLM khi thiếu cache
        allow_llm_when_missing=False,

        # Voyage-only
        use_voyage_resolution=True,
        use_voyage_node_features=True,
    )

    embedder.rebuild_graph_only()

if __name__ == "__main__":
    main()