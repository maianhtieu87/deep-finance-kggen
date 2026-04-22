from configs.config import GlobalConfig, TrainConfig
from src.data_loader import NEWS_EMB_DIM

print("=== Switch test ===")
print(f"Active embedder : {TrainConfig.news_embedder}")
print(f"news_embed_dim  : {TrainConfig.news_embed_dim()}")
print(f"NEWS_EMB_DIM    : {NEWS_EMB_DIM}")
print(f"FinBERT path    : {GlobalConfig.finbert_emb_path()}")
print(f"Voyage path     : {GlobalConfig.voyage_emb_path()}")
print(f"Active path     : {GlobalConfig.news_emb_path()}")
print()
print("--- Simulating switch to voyage ---")
TrainConfig.news_embedder = "voyage"
print(f"Active embedder : {TrainConfig.news_embedder}")
print(f"news_emb_dim()  : {GlobalConfig.news_emb_dim()}")
print(f"news_emb_path() : {GlobalConfig.news_emb_path()}")
