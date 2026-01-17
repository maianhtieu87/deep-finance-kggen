import os
import time
import shutil
import httpx
import polars as pl
from rich import print
from tqdm import tqdm
from uuid import uuid4
from datetime import date, timedelta, datetime
from typing import List, Dict, Tuple, Union
from tenacity import retry, stop_after_attempt, wait_fixed
from configs.config import GlobalConfig as Config

# Constants
END_POINT_TEMPLATE = "https://data.alpaca.markets/v1beta1/news?start={start_date}&end={end_date}&limit=50&symbols={symbol}"
END_POINT_TEMPLATE_LINK_PAGE = "https://data.alpaca.markets/v1beta1/news?limit=50&symbol={symbol}&page_token={page_token}"
NUM_NEWS_PER_RECORD = 200

class ScraperError(Exception): pass
class RecordContainerFull(Exception): pass

def round_to_next_day(date_col: pl.Expr) -> pl.Expr:
    """Logic from [finmem]_data_pipeline.py: Roll over if hour >= 16"""
    condition = (date_col.dt.hour() >= 16) & ((date_col.dt.minute() > 0) | (date_col.dt.second() > 0))
    return pl.when(condition).then(date_col.dt.offset_by("1d")).otherwise(date_col)

class ParseRecordContainer:
    """Strict logic from Data Pipeline file"""
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.record_counter = 0
        self.author_list = []
        self.content_list = []
        self.date_list = []
        self.source_list = []
        self.summary_list = []
        self.title_list = []
        self.url_list = []

    def add_records(self, records: List[Dict[str, str]]) -> None:
        for cur_record in records:
            self.author_list.append(cur_record.get("author"))
            self.content_list.append(cur_record.get("content"))
            # Logic: Parse ISO format and strip Z
            d_str = cur_record["created_at"].rstrip("Z")
            self.date_list.append(datetime.fromisoformat(d_str))
            
            self.source_list.append(cur_record.get("source"))
            self.summary_list.append(cur_record.get("summary"))
            self.title_list.append(cur_record.get("headline"))
            self.url_list.append(cur_record.get("url"))
            
            self.record_counter += 1
            if self.record_counter == NUM_NEWS_PER_RECORD:
                raise RecordContainerFull

    def pop(self, align_next_date: bool = True) -> Union[pl.DataFrame, None]:
        if self.record_counter == 0: return None
        return_df = pl.DataFrame({
            "author": self.author_list, "content": self.content_list,
            "datetime": self.date_list, "source": self.source_list,
            "summary": self.summary_list, "title": self.title_list,
            "url": self.url_list,
        })
        # Logic: Apply round_to_next_day within Polars
        if align_next_date:
            return_df = return_df.with_columns(round_to_next_day(return_df["datetime"]).alias("date"))
        else:
            return_df = return_df.with_columns(pl.col("datetime").date().alias("date"))
        return return_df.with_columns(pl.lit(self.symbol).alias("equity"))

class AlpacaNewsFetcher:
    def __init__(self):
        self.temp_dir = os.path.join(Config.RAW_NEWS_PATH, "temp")
        self.primary_dir = os.path.join(Config.RAW_NEWS_PATH, "03_primary")
        if os.path.exists(self.temp_dir): shutil.rmtree(self.temp_dir)
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.primary_dir, exist_ok=True)
        
        self.headers = {
            "APCA-API-KEY-ID": Config.ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": Config.ALPACA_SECRET_KEY,
        }

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(60))
    def _query_one_record(self, args: Tuple[date, str]) -> None:
        d, symbol = args
        next_d = d + timedelta(days=1)
        container = ParseRecordContainer(symbol)
        
        with httpx.Client() as client:
            url = END_POINT_TEMPLATE.format(
                start_date=d.strftime("%Y-%m-%d"),
                end_date=next_d.strftime("%Y-%m-%d"),
                symbol=symbol
            ) + "&include_content=True&exclude_contentless=True"
            
            resp = client.get(url, headers=self.headers)
            if resp.status_code != 200: raise ScraperError(resp.text)
            
            res_json = resp.json()
            next_token = res_json.get("next_page_token")
            container.add_records(res_json.get("news", []))
            
            while next_token:
                try:
                    url = END_POINT_TEMPLATE_LINK_PAGE.format(
                        symbol=symbol, page_token=next_token
                    ) + "&include_content=True&exclude_contentless=True"
                    resp = client.get(url, headers=self.headers)
                    if resp.status_code != 200: raise ScraperError(resp.text)
                    
                    res_json = resp.json()
                    next_token = res_json.get("next_page_token")
                    container.add_records(res_json.get("news", []))
                except RecordContainerFull:
                    break
        
        result = container.pop(align_next_date=True)
        if result is not None:
            result.write_parquet(os.path.join(self.temp_dir, f"{uuid4()}.parquet"))

    def fetch_all(self, price_dict: dict):
        """Orchestrate fetching based on existing price data dates"""
        # Logic: Extract unique (date, equity) from price dict
        args_list = []
        # Assuming price_dict structure: {date: {ticker: ...}}
        for d_obj, tickers_data in price_dict.items():
            if isinstance(d_obj, str): d_obj = datetime.strptime(d_obj, "%Y-%m-%d").date()
            if hasattr(d_obj, 'date'): d_obj = d_obj.date()
            
            # Logic: tickers_data keys are tickers
            for ticker in tickers_data.keys():
                if ticker != 'macro':
                    args_list.append((d_obj, ticker))
        
        # Deduplicate
        args_list = list(set(args_list))
        
        print(f"Fetching news for {len(args_list)} (date, ticker) pairs...")
        with tqdm(total=len(args_list)) as pbar:
            for i, arg in enumerate(args_list):
                try:
                    self._query_one_record(arg)
                except Exception as e:
                    print(f"Error: {e}")
                pbar.update(1)
                if (i + 1) % 3000 == 0: time.sleep(90)
        
        # Merge
        files = [os.path.join(self.temp_dir, f) for f in os.listdir(self.temp_dir) if f.endswith(".parquet")]
        if files:
            df = pl.concat([pl.read_parquet(f) for f in files])
            df.write_parquet(os.path.join(self.primary_dir, "news.parquet"))
            return df
        return None