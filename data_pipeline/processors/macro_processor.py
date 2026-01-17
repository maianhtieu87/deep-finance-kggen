import pandas as pd
import pickle

class MacroProcessor:
    def process_and_enrich(self, price_dict: dict, macro_df: pd.DataFrame) -> dict:
        """
        Nhận vào price_dict và macro_df raw.
        Thực hiện logic: Calculate Spread -> Return -> FFill -> Shift(1) -> Merge vào Dict.
        """
        if 'date' in macro_df.columns: 
            macro_df['date'] = pd.to_datetime(macro_df['date'])
            macro_df = macro_df.set_index('date').sort_index()
        elif not isinstance(macro_df.index, pd.DatetimeIndex):
            macro_df.index = pd.to_datetime(macro_df.index)
            
        
        
        # --- Step 3: Compute spread + returns (Logic gốc) ---
        if 'us10y' in macro_df.columns and 'us2y' in macro_df.columns:
            macro_df['yield_spread_10y_2y'] = macro_df['us10y'] - macro_df['us2y']

        if 'sp500' in macro_df.columns:
            macro_df['sp500_return'] = macro_df['sp500'].pct_change(fill_method=None)

        # --- Step 4 & 5: Forward-fill & Lag by 1 day (Logic gốc) ---
        macro_df = macro_df.ffill()
        macro_df = macro_df.shift(1)

        # --- Step 6: Merge into price_data ---
        macro_cols = ['vix', 'yield_spread_10y_2y', 'sp500', 'sp500_return', 'dxy', 'wti']
        available_cols = [c for c in macro_cols if c in macro_df.columns]
        
        all_dates = sorted(price_dict.keys())
        
        # Logic Loop dates để map macro (tương tự file gốc)
        for date in all_dates:
            # Tìm ngày macro gần nhất trước đó
            ts_date = pd.to_datetime(date)
            prev_date = macro_df.index[macro_df.index <= ts_date].max()
            
            if pd.isna(prev_date):
                continue
                
            row = macro_df.loc[prev_date]
            
            if 'macro' not in price_dict[date]:
                price_dict[date]['macro'] = {}

            for col in available_cols:
                price_dict[date]['macro'][col] = row[col]
                
        return price_dict