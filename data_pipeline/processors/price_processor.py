import pickle
import pandas as pd
from typing import List, Dict

class PriceProcessor:
    def combine_to_nested_dict(self, df_list: List[pd.DataFrame], tickers: List[str]) -> Dict:
        """
        RESOLVED CONFLICT: Follows [finmem]_macro_indicators_retrieval.py
        Structure: { 'YYYY-MM-DD': { 'TSLA': {'open':..., 'high':..., 'close':...} } }
        """
        combined_dict = {}

        for df, ticker in zip(df_list, tickers):
            # Logic MultiIndex handling from Macro file
            # The file checks for ('date', '') or just 'date'
            if isinstance(df.columns, pd.MultiIndex):
                # Try to find date col
                if ('date', '') in df.columns: date_series = df[('date', '')]
                else: date_series = df.index # Fallback
            else:
                date_series = df['date']

            # Target keys from Macro file
            columns_to_keep = {
                'open': f'{ticker}_open',
                'high': f'{ticker}_high',
                'close': f'{ticker}_close',
            }

            for idx, date in enumerate(date_series):
                if date not in combined_dict:
                    combined_dict[date] = {}
                
                combined_dict[date][ticker] = {}
                
                # Logic: Handle MultiIndex vs SingleIndex access
                for key, col_name in columns_to_keep.items():
                    val = None
                    try:
                        # Try direct access (Single Index)
                        if col_name in df.columns:
                            val = df[col_name].iloc[idx]
                        # Try MultiIndex tuple ('Price', 'TSLA_open') - pattern in some yf versions
                        # But our fetcher renamed cols to flat strings like 'TSLA_open'.
                        # So direct access should work if fetcher worked.
                    except KeyError:
                        pass
                    
                    if val is not None:
                        combined_dict[date][ticker][key] = val

        return combined_dict

    def save(self, data: Dict, path: str):
        with open(path, 'wb') as file:
            pickle.dump(data, file)