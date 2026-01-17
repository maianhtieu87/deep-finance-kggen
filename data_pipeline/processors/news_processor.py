import polars as pl
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import os
import json
import time
import voyageai
from typing import List
from configs.config import GlobalConfig as Config

class NewsProcessor:
    def merge_reuters_and_fix_nulls(self, alpaca_path, reuters_path, output_path):
        """
        Logic from [finmem]_data_pipeline.py:
        1. Load Alpaca & Reuters
        2. Align Reuters schema
        3. Concat
        4. FIX NULL DATES caused by rollover logic
        """
        if not os.path.exists(alpaca_path): return None
        news_df = pl.read_parquet(alpaca_path)
        
        # Load Reuters
        pdf = pd.read_excel(reuters_path)
        reuters_df = pl.from_pandas(pdf)
        
        # Rename/Cast Reuters to match Alpaca
        reuters_df = reuters_df.rename({
            'Title': 'title', 'datetime_utc': 'datetime', 'Content': 'content', 'Stock_Type': 'equity'
        })
        reuters_df = reuters_df.with_columns([
            pl.lit(None, dtype=pl.String).alias("author"),
            pl.lit(None, dtype=pl.String).alias("source"),
            pl.lit(None, dtype=pl.String).alias("summary"),
            pl.lit(None, dtype=pl.String).alias("url")
        ])
        
        # Cast datetime
        reuters_df = reuters_df.with_columns(
            pl.col("datetime").cast(pl.Datetime(time_unit='us')).alias("datetime"),
            pl.col("datetime").cast(pl.Date).alias("date")
        )
        # Select matching columns
        reuters_df = reuters_df.select(news_df.columns)
        
        # Concat
        combined = pl.concat([news_df, reuters_df]).sort("datetime")
        
        # --- CRITICAL FIX FROM DATA PIPELINE ---
        # Recalculate 'date' to fix nulls for end-of-month rollovers
        combined = combined.with_columns(
            pl.when(
                (pl.col("datetime").dt.hour() >= 16) & ((pl.col("datetime").dt.minute() > 0) | (pl.col("datetime").dt.second() > 0))
            )
            .then(pl.col("datetime").dt.offset_by("1d").cast(pl.Date))
            .otherwise(pl.col("datetime").cast(pl.Date))
            .alias("date")
        )
        
        combined.write_parquet(output_path)
        return combined.to_pandas()

    def align_to_trading_days(self, news_pd_df, trading_dates):
        """
        Logic shared by both files: adjust_trading_days_fast
        """
        def build_map(tr_dates, start, end):
            tr_dates = pd.to_datetime(sorted(pd.to_datetime(tr_dates).normalize()))
            cal_dates = pd.date_range(start=start, end=end, freq='D')
            mapping = {}
            idx = 0
            for d in tqdm(cal_dates, desc="Building map"):
                while idx < len(tr_dates) and tr_dates[idx] < d:
                    idx += 1
                if idx < len(tr_dates): mapping[d] = tr_dates[idx]
                else: mapping[d] = tr_dates[-1]
            return mapping

        news_pd_df['date'] = pd.to_datetime(news_pd_df['date']).dt.normalize()
        tr_dates_dt = pd.to_datetime(trading_dates).normalize()
        
        processed = news_pd_df[news_pd_df['date'].isin(tr_dates_dt)]
        raw = news_pd_df[~news_pd_df['date'].isin(tr_dates_dt)]
        
        if not raw.empty:
            start_d = news_pd_df['date'].min()
            end_d = news_pd_df['date'].max()
            date_map = build_map(trading_dates, start_d, end_d)
            
            tqdm.pandas(desc="Adjusting dates")
            raw['date'] = raw['date'].progress_map(lambda x: date_map.get(x, x))
            
        final = pd.concat([processed, raw]).sort_values(by='date').reset_index(drop=True)
        return final
    

class NewsEmbedder:
    """
    Logic from news_embed.py: 
    - Aggregates headlines by date/ticker
    - Embeds using Voyage AI with retry/backoff
    - Saves streaming JSONL
    """
    def __init__(self):
        os.environ["VOYAGE_API_KEY"] = Config.VOYAGE_API_KEY
        self.client = voyageai.Client()
        self.model = Config.EMBED_MODEL
        self.output_dir = Config.NEWS_EMBEDDING_OUTPUT_PATH
        self.jsonl_path = os.path.join(self.output_dir, "embedded_news.jsonl")
        self.json_path = os.path.join(self.output_dir, "embedded_news.json")
        
        limits = Config.VOYAGE_RATE_LIMITS[Config.PAYMENT_ADDED]
        self.sleep_time = limits["SLEEP"]
        self.max_retries = Config.MAX_RETRIES
        self.backoff_base = Config.BACKOFF_BASE

    def _safe_embed(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(self.max_retries):
            try:
                res = self.client.embed(texts=texts, model=self.model)
                return res.embeddings
            except Exception as e:
                wait = self.backoff_base * (2 ** attempt)
                print(f"[Retry {attempt+1}] {e} → sleep {wait}s")
                time.sleep(wait)
        raise RuntimeError("Embedding failed after max retries")

    def _aggregate_headlines(self, news_df: pd.DataFrame) -> pd.DataFrame:
        news_df['date'] = pd.to_datetime(news_df['date']).dt.date
        
        # Logic: Group by equity+date -> join titles with " | "
        grouped = (
            news_df.dropna(subset=["title"])
            .groupby(["equity", "date"])["title"]
            .apply(lambda x: " | ".join(x.astype(str)))
            .reset_index()
            .rename(columns={"title": "merged_title"})
        )
        return grouped

    def process_and_save(self, news_df: pd.DataFrame) -> str:
        grouped_df = self._aggregate_headlines(news_df)
        all_results = {}
        
        print(f"Starting embedding for {len(grouped_df)} grouped records...")
        
        with open(self.jsonl_path, "w", encoding="utf-8") as jsonl_file:
            for day, day_df in grouped_df.groupby("date"):
                print(f"📅 {day} | {len(day_df)} companies")
                
                texts = day_df["merged_title"].tolist()
                metas = day_df[["equity"]].to_dict("records")
                
                if texts:
                    embeddings = self._safe_embed(texts)
                    
                    day_records = []
                    for emb, text, meta in zip(embeddings, texts, metas):
                        record = {
                            "date": str(day),
                            "equity": meta["equity"],
                            "merged_title": text,
                            "embedding": emb
                        }
                        day_records.append(record)
                        jsonl_file.write(json.dumps(record) + "\n")
                    
                    all_results[str(day)] = day_records
                
                time.sleep(self.sleep_time)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f)
           
        print(f"✅ Embedding Done. Saved to: {self.json_path}")
        return self.json_path  