# src/data_loader.py - FIXED VERSION (Series error resolved, No Leakage & Rolling Quantile Added)
"""
Fixed Data Loader với Graph Structure Support, 
Rolling Quantile Labeling & Leakage-Free Normalization

FIXES:
- ✅ Handle both dict and Series/DataFrame for price_data
- ✅ Proper scalar conversion from Series
- ✅ No "Series is ambiguous" error
- ✅ Tích hợp Rolling Quantile Labeling từ Project 1
- ✅ Sửa lỗi Data Leakage: Tính Mean/Std chỉ trên tập Train
"""

import torch
import pickle
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from configs.config import TrainConfig, GlobalConfig
import os


class data_prepare:
    """
    Prepare stock movement prediction data with KG graph support.
    """
    
    def __init__(self, dataset_path: str):
        with open(dataset_path, "rb") as f:
            self.raw_data = pickle.load(f)
        
        self.window_size = TrainConfig.window_size
        self.news_dim = TrainConfig.news_embed_dim 
        
        print(f"📦 Loaded dataset with {len(self.raw_data)} trading days")
    
    def _load_graph_from_path(self, kg_tensor_path):
        if not kg_tensor_path or not isinstance(kg_tensor_path, str):
            return None
            
        if not os.path.exists(kg_tensor_path):
            return None
        
        try:
            graph_data = torch.load(kg_tensor_path, map_location='cpu')
            
            if isinstance(graph_data, Data):
                return graph_data
            elif isinstance(graph_data, dict):
                x = graph_data.get('x', graph_data.get('node_features'))
                edge_index = graph_data.get('edge_index', torch.zeros(2, 0, dtype=torch.long))
                if x is None:
                    return None
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                return Data(x=x, edge_index=edge_index)
            else:
                x = graph_data
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                return Data(
                    x=x,
                    edge_index=torch.zeros(2, 0, dtype=torch.long)
                )
                
        except Exception as e:
            return None
    
    def _create_empty_graph(self):
        return Data(
            x=torch.zeros(1, self.news_dim),
            edge_index=torch.zeros(2, 0, dtype=torch.long)
        )
    
    def prepare_data(self, target_ticker: str):
        dates = sorted(self.raw_data.keys())
        
        rows = []
        for date_key in dates:
            day_data = self.raw_data[date_key]
            
            price_data = day_data.get("price", {}).get(target_ticker)
            if price_data is None:
                continue
            
            if isinstance(price_data, dict):
                s_o = price_data.get("Open") or price_data.get("open")
                s_h = price_data.get("High") or price_data.get("high")
                s_c = price_data.get("Close") or price_data.get("close")
            else:
                try:
                    if hasattr(price_data, '__getitem__'):
                        s_o = price_data.get("Open", price_data.get("open", None))
                        s_h = price_data.get("High", price_data.get("high", None))
                        s_c = price_data.get("Close", price_data.get("close", None))
                        
                        if isinstance(s_o, pd.Series):
                            s_o = s_o.iloc[0] if len(s_o) > 0 else None
                        if isinstance(s_h, pd.Series):
                            s_h = s_h.iloc[0] if len(s_h) > 0 else None
                        if isinstance(s_c, pd.Series):
                            s_c = s_c.iloc[0] if len(s_c) > 0 else None
                    else:
                        s_o = s_h = s_c = None
                except Exception:
                    s_o = s_h = s_c = None
            
            if s_o is None or s_h is None or s_c is None:
                continue
            
            try:
                if hasattr(s_o, 'iloc'): s_o = float(s_o.iloc[0])
                else: s_o = float(s_o)
                
                if hasattr(s_h, 'iloc'): s_h = float(s_h.iloc[0])
                else: s_h = float(s_h)
                
                if hasattr(s_c, 'iloc'): s_c = float(s_c.iloc[0])
                else: s_c = float(s_c)
            except (ValueError, TypeError):
                continue
            
            macro_data = day_data.get("macro", {})
            kg_tensor_path = day_data.get("kg_tensor", {}).get(target_ticker)
            
            rows.append({
                "date": date_key,
                "s_o": s_o,
                "s_h": s_h,
                "s_c": s_c,
                "macro": macro_data,
                "kg_path": kg_tensor_path
            })
        
        if len(rows) < self.window_size + 1:
            print(f"⚠️ {target_ticker}: Not enough data ({len(rows)} days)")
            return {}, {}, {}
        
        print(f"📊 {target_ticker}: Collected {len(rows)} days with valid price data")
        
        # ==============================================================================
        # [FEATURE 1] ROLLING QUANTILE LABELING (Dynamic & No Look-Ahead)
        # ==============================================================================
        close_prices = pd.Series([float(row["s_c"]) for row in rows])
        returns_series = close_prices.pct_change().fillna(0)
        
        rolling_window = 20
        # shift(1) để loại bỏ Look-Ahead Bias (Dữ liệu quá khứ không chứa tương lai)
        roll_low  = returns_series.rolling(window=rolling_window).quantile(0.33).shift(1)
        roll_high = returns_series.rolling(window=rolling_window).quantile(0.66).shift(1)
        
        T = len(rows) - self.window_size
        
        s_o_all = np.zeros((T, self.window_size, 1))
        s_h_all = np.zeros((T, self.window_size, 1))
        s_c_all = np.zeros((T, self.window_size, 1))
        
        first_macro = rows[0]["macro"]
        macro_keys = sorted(first_macro.keys())
        macro_dim = len(macro_keys)
        s_m_all = np.zeros((T, self.window_size, macro_dim))
        
        graph_paths_all = []
        labels = np.zeros(T, dtype=int)
        
        for t in range(T):
            window_rows = rows[t:t + self.window_size]
            target_row = rows[t + self.window_size]
            
            for w, row in enumerate(window_rows):
                s_o_all[t, w, 0] = float(row["s_o"])
                s_h_all[t, w, 0] = float(row["s_h"])
                s_c_all[t, w, 0] = float(row["s_c"])
                
                macro_vec = [row["macro"].get(k, 0) for k in macro_keys]
                s_m_all[t, w, :] = macro_vec
            
            graph_paths_all.append(target_row["kg_path"])
            
            # Gán nhãn Rolling Quantile
            target_idx = t + self.window_size
            current_return = returns_series.iloc[target_idx]
            current_roll_low = roll_low.iloc[target_idx]
            current_roll_high = roll_high.iloc[target_idx]
            
            if pd.isna(current_roll_low) or pd.isna(current_roll_high):
                labels[t] = 1 # Mặc định Flat cho những ngày đầu chuỗi rolling
            else:
                is_noise = abs(current_return) < 0.001
                if is_noise:
                    labels[t] = 1 # Force Flat
                elif current_return < current_roll_low:
                    labels[t] = 0 # DOWN
                elif current_return > current_roll_high:
                    labels[t] = 2 # UP
                else:
                    labels[t] = 1 # FLAT
                    
        # Log phân phối nhãn
        unique, counts = np.unique(labels, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f" ⚖️ Label Distribution (Rolling Quantile 33/66): {dist}")
        
        # ==============================================================================
        # [FEATURE 2] DATA LEAKAGE FIX (Split indices BEFORE Normalization)
        # ==============================================================================
        train_ratio = getattr(TrainConfig, "train_ratio", 0.7)
        valid_ratio = getattr(TrainConfig, "valid_ratio", 0.15)
        
        train_end = int(T * train_ratio)
        valid_end = int(T * (train_ratio + valid_ratio))

        # Tính Mean/Std CHỈ TRÊN TẬP TRAIN để chống Look-Ahead Bias
        s_o_mean = s_o_all[:train_end].mean()
        s_o_std  = s_o_all[:train_end].std() + 1e-8
        
        s_h_mean = s_h_all[:train_end].mean()
        s_h_std  = s_h_all[:train_end].std() + 1e-8
        
        s_c_mean = s_c_all[:train_end].mean()
        s_c_std  = s_c_all[:train_end].std() + 1e-8
        
        s_m_mean = s_m_all[:train_end].mean(axis=(0, 1), keepdims=True)
        s_m_std  = s_m_all[:train_end].std(axis=(0, 1), keepdims=True) + 1e-8
        
        # Áp dụng Mean/Std của Train lên TOÀN BỘ dữ liệu
        s_o_all = (s_o_all - s_o_mean) / s_o_std
        s_h_all = (s_h_all - s_h_mean) / s_h_std
        s_c_all = (s_c_all - s_c_mean) / s_c_std
        s_m_all = (s_m_all - s_m_mean) / s_m_std
        
        # Tensors
        s_o_tensor = torch.tensor(s_o_all, dtype=torch.float32)
        s_h_tensor = torch.tensor(s_h_all, dtype=torch.float32)
        s_c_tensor = torch.tensor(s_c_all, dtype=torch.float32)
        s_m_tensor = torch.tensor(s_m_all, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        print(f"📊 Loading {T} graphs for {target_ticker}...")
        graph_list = []
        for path in graph_paths_all:
            graph = self._load_graph_from_path(path)
            if graph is None:
                graph = self._create_empty_graph()
            graph_list.append(graph)
        
        train_data = {
            "s_o": s_o_tensor[:train_end],
            "s_h": s_h_tensor[:train_end],
            "s_c": s_c_tensor[:train_end],
            "s_m": s_m_tensor[:train_end],
            "s_n_graphs": graph_list[:train_end],
            "label": label_tensor[:train_end]
        }
        
        valid_data = {
            "s_o": s_o_tensor[train_end:valid_end],
            "s_h": s_h_tensor[train_end:valid_end],
            "s_c": s_c_tensor[train_end:valid_end],
            "s_m": s_m_tensor[train_end:valid_end],
            "s_n_graphs": graph_list[train_end:valid_end],
            "label": label_tensor[train_end:valid_end]
        }
        
        test_data = {
            "s_o": s_o_tensor[valid_end:],
            "s_h": s_h_tensor[valid_end:],
            "s_c": s_c_tensor[valid_end:],
            "s_m": s_m_tensor[valid_end:],
            "s_n_graphs": graph_list[valid_end:],
            "label": label_tensor[valid_end:]
        }
        
        print(f"✅ {target_ticker}: Train={len(train_data['label'])}, "
              f"Valid={len(valid_data['label'])}, Test={len(test_data['label'])}")
        
        return train_data, valid_data, test_data