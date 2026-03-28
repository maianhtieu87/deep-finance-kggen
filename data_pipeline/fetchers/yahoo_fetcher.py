import yfinance as yf
import pandas as pd
from typing import List, Dict, Union
from pandas_datareader import data as pdr


class YahooFetcher:

    def download_data(self, start_day: str, end_day: str, tickers: List[str]) -> List[pd.DataFrame]:
        """
        Tải OHLC cho từng ticker. Trả về List[DataFrame] với cột
        date, {TICKER}_open, {TICKER}_high, {TICKER}_close.
        """
        df_list = []
        for ticker in tickers:
            print(f'Downloading data for {ticker}')
            data = yf.download(ticker, start=start_day, end=end_day, progress=False)
            data = data.reset_index()
            data['Date'] = pd.to_datetime(data['Date']).dt.date

            # Flatten MultiIndex nếu có (yfinance >= 0.2.x)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = ['_'.join(filter(None, col)).strip() for col in data.columns]

            # Tìm các cột Open/High/Close (có thể là "Open" hoặc "Open_TSLA")
            open_col  = next((c for c in data.columns if c.lower().startswith("open")),  None)
            high_col  = next((c for c in data.columns if c.lower().startswith("high")),  None)
            close_col = next((c for c in data.columns if c.lower().startswith("close")), None)
            date_col  = next((c for c in data.columns if c.lower() == "date"), None)

            if not all([open_col, high_col, close_col, date_col]):
                print(f"Warning: Missing OHLC columns for {ticker}. Cols: {list(data.columns)}")
                continue

            out = data[[date_col, open_col, high_col, close_col]].copy()
            out = out.rename(columns={
                date_col:  "date",
                open_col:  f"{ticker}_open",
                high_col:  f"{ticker}_high",
                close_col: f"{ticker}_close",
            })
            df_list.append(out)

        return df_list

    def fetch_macro_indicators(
        self,
        start_date: str,
        end_date:   str,
        symbols:    Union[List[str], Dict[str, str]],
    ) -> pd.DataFrame:
        """
        Tải macro indicators từ Yahoo Finance + FRED.

        symbols có thể là:
          - List[str]:         ["^GSPC", "^VIX", ...]  → dùng ticker làm key
          - Dict[str, str]:    {"sp500": "^GSPC", ...} → dùng key làm tên cột

        FIX: GlobalConfig.MACRO_SYMBOLS là List[str], không phải Dict.
        Code cũ gọi .values() trên list → TypeError.
        """
        # Chuẩn hóa symbols → dict {canonical_name: yahoo_symbol}
        if isinstance(symbols, dict):
            symbol_map = symbols  # {"vix": "^VIX", ...}
        else:
            # List → tạo canonical name từ ticker symbol
            symbol_map = {}
            for sym in symbols:
                name = (
                    sym.replace("^", "")
                       .replace(".", "_")
                       .lower()
                )
                # Override với tên chuẩn cho các symbol quen thuộc
                friendly = {
                    "gspc":  "sp500",
                    "dji":   "dji",
                    "ixic":  "nasdaq",
                    "vix":   "vix",
                    "tnx":   "us10y_yahoo",  # sẽ bị override bởi FRED
                    "tyx":   "us30y",
                    "irx":   "us3m",
                    "dxy":   "dxy",
                    "cl=f":  "wti",
                }
                symbol_map[friendly.get(name, name)] = sym

        ticker_list = list(symbol_map.values())

        # 1. Tải từ Yahoo
        print("Fetching Macro from Yahoo...")
        macro_data = pd.DataFrame()
        try:
            raw = yf.download(
                ticker_list,
                start=start_date, end=end_date,
                auto_adjust=True, progress=False,
            )

            # Lấy Close prices
            if isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.get_level_values(0):
                    raw = raw["Close"]
                else:
                    # Flatten
                    raw.columns = ["_".join(filter(None, c)) for c in raw.columns]

            # Rename: yahoo symbol → canonical name
            inv_map = {v: k for k, v in symbol_map.items()}
            raw = raw.rename(columns=inv_map)
            macro_data = raw.copy()

        except Exception as e:
            print(f"  Yahoo Macro Error: {e}")

        # 2. Tải yields từ FRED (override Yahoo TNX nếu có)
        try:
            print("  Fetching Yields from FRED...")
            fred_2y  = pdr.DataReader("DGS2",  "fred", start_date, end_date)
            fred_10y = pdr.DataReader("DGS10", "fred", start_date, end_date)
            macro_data["us2y"]  = fred_2y["DGS2"]
            macro_data["us10y"] = fred_10y["DGS10"]
        except Exception as e:
            print(f"  FRED Error: {e}")

        # 3. Chuẩn hóa index thành cột date
        macro_data = macro_data.reset_index()
        date_col = next(
            (c for c in macro_data.columns if c.lower() in ("date", "index")), None
        )
        if date_col and date_col != "date":
            macro_data = macro_data.rename(columns={date_col: "date"})
        if "date" in macro_data.columns:
            macro_data["date"] = pd.to_datetime(macro_data["date"]).dt.date

        print(f"  Macro columns: {[c for c in macro_data.columns if c != 'date']}")
        return macro_data