import pandas as pd
import pickle
from configs.config import GlobalConfig as Config
import os 
import json

class DatasetBuilder:
    def create_synchronized_data(self, price_macro_dict, news_df, filing_path, embedding_path):
        """
        Final Union Logic.
        FIX: Stores full news object (title, content, summary) instead of just one string.
        """
        # Load Filings
        filing_df = pd.read_parquet(filing_path)
        filing_df['filedAt'] = pd.to_datetime(filing_df['filedAt']).dt.normalize()
        
        embedding_data = {}
        if embedding_path and os.path.exists(embedding_path):
            print(f"Loading embeddings from {embedding_path}...")
            with open(embedding_path, 'r') as f:
                raw_embed_data = json.load(f)
            #Standardize key to YYYY-MM-DD
            for k,v in raw_embed_data.items():
                clean_key = str(k)[:10]
                embedding_data[clean_key] = v
                
                 
        synchronized_data = {}
        mapping = Config.TICKER_MAPPING
        
        for date_obj, data in price_macro_dict.items():
            date_dt = pd.to_datetime(date_obj).normalize()
            date_str = str(date_obj)
            synchronized_data[date_obj] = {}
            
            # 1. Price (Align Ticker Names)
            synchronized_data[date_obj]['price'] = {}
            for t, v in data.items():
                if t != 'macro' and t in mapping:
                    synchronized_data[date_obj]['price'][mapping[t]] = v
                    
            # 2. Macro
            synchronized_data[date_obj]['macro'] = data.get('macro', {})
            
            # 3. News (FIX: Store Full Object)
            date_news = news_df[news_df['date'].dt.normalize() == date_dt]
            synchronized_data[date_obj]['news'] = {}
            
            for ticker in date_news['equity'].unique():
                if ticker in mapping:
                    clean_ticker = mapping[ticker]
                    
                    # Convert rows to list of dicts
                    # We preserve 'title' (headline), 'content', and 'summary'
                    news_records = date_news[date_news['equity'] == ticker][
                        ['title', 'content', 'summary', 'source', 'url']
                    ].to_dict(orient='records')
                    
                    if clean_ticker not in synchronized_data[date_obj]['news']:
                        synchronized_data[date_obj]['news'][clean_ticker] = []
                    
                    synchronized_data[date_obj]['news'][clean_ticker].extend(news_records)


            # 4. News Embeddings
            synchronized_data[date_obj]['news_embedding'] = {}
            
            if date_str in embedding_data:
                for rec in embedding_data[date_str]:
                    raw_ticker = rec['equity']
                    if raw_ticker in mapping:
                        clean_ticker = mapping[raw_ticker]
                        synchronized_data[date_obj]['news_embedding'][clean_ticker] = rec['embedding']
                        
            # 4. Filings
            date_filings = filing_df[filing_df['filedAt'] == date_dt]
            synchronized_data[date_obj]['filing_q'] = {}
            synchronized_data[date_obj]['filing_k'] = {}
            
            for ticker in date_filings['ticker'].unique():
                if ticker in mapping:
                    clean_ticker = mapping[ticker]
                    tf = date_filings[date_filings['ticker'] == ticker]
                    
                    q_txt = tf[tf['formType'] == '10-Q']['content_summary'].tolist()
                    k_txt = tf[tf['formType'] == '10-K']['content_summary'].tolist()
                    
                    if q_txt: synchronized_data[date_obj]['filing_q'][clean_ticker] = " ".join(q_txt)
                    if k_txt: synchronized_data[date_obj]['filing_k'][clean_ticker] = " ".join(k_txt)
                    
        return synchronized_data

    def save(self, data, filename='unified_dataset.pkl'):
    # Ensure PROCESSED_PATH exists
        os.makedirs(Config.PROCESSED_PATH, exist_ok=True)
        path = os.path.join(Config.PROCESSED_PATH, filename)
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {path}")