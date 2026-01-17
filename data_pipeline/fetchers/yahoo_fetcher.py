import yfinance as yf
import pandas as pd
from typing import List, Dict
from pandas_datareader import data as pdr

class YahooFetcher:
    def download_data(self, start_day: str, end_day: str, tickers: List[str]) -> List[pd.DataFrame]:
        """
        RESOLVED CONFLICT: Follows logic from [finmem]_macro_indicators_retrieval.py
        Fetches 'Open', 'High', 'Close' (not just Close).
        Renames columns to {ticker}_open, {ticker}_high, {ticker}_close.
        """
        df_list = []
        for ticker in tickers:
            print(f'Downloading data for {ticker}')
            # Logic: Auto_adjust=False implies we want raw/adjusted appropriately, 
            # but usually 'Adj Close' is separate. 
            # Macro file uses: data = yf.download(...) -> reset_index -> rename
            data = yf.download(ticker, start=start_day, end=end_day)
            
            # Reset index to make Date a column
            data = data.reset_index()
            data['Date'] = data['Date'].dt.date
            
            # Select OHLC
            # Note: yfinance might return MultiIndex if multiple tickers, but here we loop one by one.
            if 'Open' in data.columns and 'High' in data.columns and 'Close' in data.columns:
                data = data[['Date', 'Open', 'High', 'Close']]
                data = data.rename(columns={
                    'Date': 'date',
                    'Open': f'{ticker}_open',
                    'High': f'{ticker}_high',
                    'Close': f'{ticker}_close'
                })
                df_list.append(data)
            else:
                 print(f"Warning: Missing OHLC data for {ticker}")
                 
        return df_list

    def fetch_macro_indicators(self, start_date: str, end_date: str, symbols: Dict[str, str]):
        """
        Fetches Macro data (Yahoo + FRED).
        Updated to accept 'symbols' argument.
        """
        # 1. Fetch from Yahoo
        print("Fetching Macro from Yahoo...")
        try:
            # yfinance download accepts list of tickers
            ticker_list = list(symbols.values())
            macro_data = yf.download(ticker_list, start=start_date, end=end_date, auto_adjust=True, progress=False)
            
            # Handle yfinance Output Structure (it varies by version/number of tickers)
            if 'Close' in macro_data.columns and isinstance(macro_data.columns, pd.MultiIndex):
                macro_data = macro_data['Close']
            elif 'Close' in macro_data.columns: # Single ticker case or flat index
                pass # Use as is if it's just the close prices
                
            # Rename columns to standardized keys (vix, sp500, etc.)
            # Map values back to keys: {'^VIX': 'vix', ...}
            inv_map = {v: k for k, v in symbols.items()}
            macro_data = macro_data.rename(columns=inv_map)
            
        except Exception as e:
            print(f"Yahoo Macro Error: {e}")
            macro_data = pd.DataFrame()

        # 2. Fetch from FRED
        try:
            print("Fetching Yields from FRED...")
            fred_2y = pdr.DataReader("DGS2", "fred", start_date, end_date)
            fred_10y = pdr.DataReader("DGS10", "fred", start_date, end_date)
            
            # Merge into macro_data
            # We use assignment which aligns on Index (Date) automatically
            macro_data['us2y'] = fred_2y['DGS2']
            macro_data['us10y'] = fred_10y['DGS10']
            
        except Exception as e:
            print(f"FRED Error: {e}")

        # Reset index to make 'date' a column
        macro_data = macro_data.reset_index()
        # Rename 'Date' index to 'date' column
        if 'Date' in macro_data.columns:
            macro_data = macro_data.rename(columns={'Date': 'date'})
            macro_data['date'] = pd.to_datetime(macro_data['date']).dt.date
            
        return macro_data