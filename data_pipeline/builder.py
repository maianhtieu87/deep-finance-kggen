import pandas as pd
import pickle
from configs.config import GlobalConfig as Config
import os
import json

class DatasetBuilder:
    def create_synchronized_data(self, price_macro_dict, news_df, filing_path, embedding_path):
        """
        Final Union Logic.
        FIX:
          - Ensure news_df['date'] is datetime-like before using .dt
          - Handle missing filing_path gracefully
          - Keep full news object (title, content, summary, source, url)
        """
        news_df["date"] = pd.to_datetime(news_df["date"]) # fix .dt.normalize() bug

        # -------------------------
        # 0) Filings (optional)
        # -------------------------
        filing_df = None
        if filing_path and os.path.exists(filing_path):
            filing_df = pd.read_parquet(filing_path)
            filing_df["filedAt"] = pd.to_datetime(filing_df["filedAt"], errors="coerce").dt.normalize()
            filing_df = filing_df.dropna(subset=["filedAt"])
        else:
            # allow running without filings
            filing_df = pd.DataFrame(columns=["filedAt", "ticker", "formType", "content_summary"])

        # -------------------------
        # 0.1) News dtype fix (✅ this fixes your error)
        # -------------------------
        if news_df is None:
            news_df = pd.DataFrame(columns=["date", "equity"])
        else:
            news_df = news_df.copy()

        if "date" in news_df.columns:
            # make datetime-like so .dt works
            news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
            news_df = news_df.dropna(subset=["date"])
            news_df["date"] = news_df["date"].dt.normalize()

        # ensure expected cols exist (avoid KeyError when selecting)
        for c in ["title", "content", "summary", "source", "url"]:
            if c not in news_df.columns:
                news_df[c] = None

        # -------------------------
        # 1) Load embeddings index
        # -------------------------
        embedding_data = {}
        if embedding_path and os.path.exists(embedding_path):
            print(f"Loading embeddings from {embedding_path}...")
            with open(embedding_path, "r", encoding="utf-8") as f:
                raw_embed_data = json.load(f)

            # Standardize key to YYYY-MM-DD
            for k, v in raw_embed_data.items():
                clean_key = str(k)[:10]
                embedding_data[clean_key] = v

        synchronized_data = {}
        mapping = Config.TICKER_MAPPING

        # -------------------------
        # 2) Iterate trading dates
        # -------------------------
        for date_obj, data in price_macro_dict.items():
            date_dt = pd.to_datetime(date_obj).normalize()
            date_str = str(date_obj)[:10]   # ✅ ensure YYYY-MM-DD
            synchronized_data[date_obj] = {}

            # 2.1 Price
            synchronized_data[date_obj]["price"] = {}
            for t, v in data.items():
                if t != "macro" and t in mapping:
                    synchronized_data[date_obj]["price"][mapping[t]] = v

            # 2.2 Macro
            synchronized_data[date_obj]["macro"] = data.get("macro", {})

            # 2.3 News (full object)
            synchronized_data[date_obj]["news"] = {}

            if len(news_df) > 0:
                date_news = news_df[news_df["date"] == date_dt]

                for ticker in date_news["equity"].unique():
                    if ticker in mapping:
                        clean_ticker = mapping[ticker]

                        news_records = date_news[date_news["equity"] == ticker][
                            ["title", "content", "summary", "source", "url"]
                        ].to_dict(orient="records")

                        if clean_ticker not in synchronized_data[date_obj]["news"]:
                            synchronized_data[date_obj]["news"][clean_ticker] = []

                        synchronized_data[date_obj]["news"][clean_ticker].extend(news_records)

            # 2.4 News Embeddings (KG index)
            synchronized_data[date_obj]["kg_tensor"] = {}

            if date_str in embedding_data:
                for rec in embedding_data[date_str]:
                    raw_ticker = rec.get("equity")
                    if raw_ticker in mapping:
                        clean_ticker = mapping[raw_ticker]

                        # ✅ KG version: may have "kg_tensor_path" instead of "embedding"
                        if "kg_tensor_path" in rec:
                            synchronized_data[date_obj]["kg_tensor"][clean_ticker] = rec["kg_tensor_path"]
                        else:
                            synchronized_data[date_obj]["kg_tensor"][clean_ticker] = rec.get("embedding", [])

            # 2.5 Filings
            date_filings = filing_df[filing_df["filedAt"] == date_dt]
            synchronized_data[date_obj]["filing_q"] = {}
            synchronized_data[date_obj]["filing_k"] = {}

            if len(date_filings) > 0:
                for ticker in date_filings["ticker"].unique():
                    if ticker in mapping:
                        clean_ticker = mapping[ticker]
                        tf = date_filings[date_filings["ticker"] == ticker]

                        q_txt = tf[tf["formType"] == "10-Q"]["content_summary"].tolist()
                        k_txt = tf[tf["formType"] == "10-K"]["content_summary"].tolist()

                        if q_txt:
                            synchronized_data[date_obj]["filing_q"][clean_ticker] = " ".join(q_txt)
                        if k_txt:
                            synchronized_data[date_obj]["filing_k"][clean_ticker] = " ".join(k_txt)

        return synchronized_data

    def save(self, data, filename="unified_dataset.pkl"):
        os.makedirs(Config.PROCESSED_PATH, exist_ok=True)
        path = os.path.join(Config.PROCESSED_PATH, filename)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Data saved to {path}")
