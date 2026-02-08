# src/data_loader.py - FIXED VERSION (Series error resolved)
"""
Fixed Data Loader với Graph Structure Support

FIXES:
- ✅ Handle both dict and Series/DataFrame for price_data
- ✅ Proper scalar conversion from Series
- ✅ No FutureWarning
- ✅ No "Series is ambiguous" error
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
    
    Key Features:
    - Load pre-built KG graphs from tensor paths
    - Proper normalization (Z-score for price/macro, L2 for news)
    - Support for dynamic graph batching
    - Handle both dict and Series format for price data
    """
    
    def __init__(self, dataset_path: str):
        """
        Args:
            dataset_path: Path to unified_dataset.pkl
        """
        with open(dataset_path, "rb") as f:
            self.raw_data = pickle.load(f)
        
        self.window_size = TrainConfig.window_size
        self.news_dim = TrainConfig.news_embed_dim  # Expected node feature dim
        
        print(f"📦 Loaded dataset with {len(self.raw_data)} trading days")
    
    def _load_graph_from_path(self, kg_tensor_path):
        """
        Load pre-built graph tensor from disk.
        
        Args:
            kg_tensor_path: Path to saved .pt file containing graph data
        
        Returns:
            Data: PyTorch Geometric Data object with x and edge_index
                  or None if loading fails
        """
        if not kg_tensor_path or not isinstance(kg_tensor_path, str):
            return None
            
        if not os.path.exists(kg_tensor_path):
            # print(f"⚠️ Graph file not found: {kg_tensor_path}")
            return None
        
        try:
            graph_data = torch.load(kg_tensor_path, map_location='cpu')
            
            # Handle different save formats
            if isinstance(graph_data, Data):
                # Already a PyG Data object
                return graph_data
            
            elif isinstance(graph_data, dict):
                # Extract x and edge_index from dict
                x = graph_data.get('x', graph_data.get('node_features'))
                edge_index = graph_data.get('edge_index', torch.zeros(2, 0, dtype=torch.long))
                
                if x is None:
                    return None
                
                # Ensure correct shapes
                if x.dim() == 1:
                    x = x.unsqueeze(0)  # (D,) -> (1, D)
                
                return Data(x=x, edge_index=edge_index)
            
            else:
                # Assume it's raw node features tensor
                x = graph_data
                if x.dim() == 1:
                    x = x.unsqueeze(0)
                
                return Data(
                    x=x,
                    edge_index=torch.zeros(2, 0, dtype=torch.long)
                )
                
        except Exception as e:
            # print(f"❌ Error loading graph from {kg_tensor_path}: {e}")
            return None
    
    def _create_empty_graph(self):
        """Create placeholder graph for missing data."""
        return Data(
            x=torch.zeros(1, self.news_dim),
            edge_index=torch.zeros(2, 0, dtype=torch.long)
        )
    
    def prepare_data(self, target_ticker: str):
        """
        Prepare train/valid/test splits for a specific ticker.
        
        Args:
            target_ticker: Stock ticker (e.g., "TSLA")
        
        Returns:
            train_dict, valid_dict, test_dict: Dicts containing:
                - s_o, s_h, s_c: Price features (T, W, 1)
                - s_m: Macro features (T, W, macro_dim)
                - s_n_graphs: List of PyG Data objects (length T)
                - label: Labels (T,)
        """
        dates = sorted(self.raw_data.keys())
        
        # Collect raw data
        rows = []
        for date_key in dates:
            day_data = self.raw_data[date_key]
            
            # Check if ticker has data
            price_data = day_data.get("price", {}).get(target_ticker)
            if price_data is None:
                continue
            
            # ========================================
            # FIX: Handle both dict and Series/DataFrame
            # ========================================
            if isinstance(price_data, dict):
                # Dict format
                s_o = price_data.get("Open") or price_data.get("open")
                s_h = price_data.get("High") or price_data.get("high")
                s_c = price_data.get("Close") or price_data.get("close")
            else:
                # DataFrame/Series format
                try:
                    # Try getting values
                    if hasattr(price_data, '__getitem__'):
                        s_o = price_data.get("Open", price_data.get("open", None))
                        s_h = price_data.get("High", price_data.get("high", None))
                        s_c = price_data.get("Close", price_data.get("close", None))
                        
                        # Convert Series to scalar
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
            
            # Check validity (now safe - all scalars)
            if s_o is None or s_h is None or s_c is None:
                continue
            
            # Ensure float type
            try:
                # Handle Series properly (avoid FutureWarning)
                if hasattr(s_o, 'iloc'):
                    s_o = float(s_o.iloc[0])
                else:
                    s_o = float(s_o)
                
                if hasattr(s_h, 'iloc'):
                    s_h = float(s_h.iloc[0])
                else:
                    s_h = float(s_h)
                
                if hasattr(s_c, 'iloc'):
                    s_c = float(s_c.iloc[0])
                else:
                    s_c = float(s_c)
            except (ValueError, TypeError):
                continue
            
            # Extract macro
            macro_data = day_data.get("macro", {})
            
            # Extract KG graph path
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
        
        # Create sliding windows
        T = len(rows) - self.window_size
        
        # Initialize arrays
        s_o_all = np.zeros((T, self.window_size, 1))
        s_h_all = np.zeros((T, self.window_size, 1))
        s_c_all = np.zeros((T, self.window_size, 1))
        
        # Determine macro dimension
        first_macro = rows[0]["macro"]
        macro_keys = sorted(first_macro.keys())
        macro_dim = len(macro_keys)
        s_m_all = np.zeros((T, self.window_size, macro_dim))
        
        # Store graph paths (will load later)
        graph_paths_all = []
        
        labels = np.zeros(T, dtype=int)
        
        # Build windows
        for t in range(T):
            # Window data
            window_rows = rows[t:t + self.window_size]
            target_row = rows[t + self.window_size]
            
            # ========================================
            # FIX: Proper scalar assignment
            # ========================================
            for w, row in enumerate(window_rows):
                # Extract values (already scalars from above)
                s_o_val = row["s_o"]
                s_h_val = row["s_h"]
                s_c_val = row["s_c"]
                
                # Additional safety: ensure float
                s_o_all[t, w, 0] = float(s_o_val)
                s_h_all[t, w, 0] = float(s_h_val)
                s_c_all[t, w, 0] = float(s_c_val)
                
                # Fill macro
                macro_vec = [row["macro"].get(k, 0) for k in macro_keys]
                s_m_all[t, w, :] = macro_vec
            
            # Store graph path for target day
            graph_paths_all.append(target_row["kg_path"])
            
            # ========================================
            # FIX: Label calculation with scalar values
            # ========================================
            current_close = window_rows[-1]["s_c"]
            next_close = target_row["s_c"]
            
            # Ensure scalars (already are, but double-check)
            current_close = float(current_close)
            next_close = float(next_close)
            
            if next_close > current_close * 1.005:
                labels[t] = 2  # UP
            elif next_close < current_close * 0.995:
                labels[t] = 0  # DOWN
            else:
                labels[t] = 1  # FLAT
        
        # ===== NORMALIZATION =====
        # ✅ Z-score for Price & Macro (Proper for numeric features)
        s_o_all = (s_o_all - s_o_all.mean()) / (s_o_all.std() + 1e-8)
        s_h_all = (s_h_all - s_h_all.mean()) / (s_h_all.std() + 1e-8)
        s_c_all = (s_c_all - s_c_all.mean()) / (s_c_all.std() + 1e-8)
        s_m_all = (s_m_all - s_m_all.mean()) / (s_m_all.std() + 1e-8)
        
        # ✅ NO Z-score for News (preserve embedding geometry)
        # Graphs will be L2-normalized in the model if needed
        
        # Convert to tensors
        s_o_tensor = torch.tensor(s_o_all, dtype=torch.float32)
        s_h_tensor = torch.tensor(s_h_all, dtype=torch.float32)
        s_c_tensor = torch.tensor(s_c_all, dtype=torch.float32)
        s_m_tensor = torch.tensor(s_m_all, dtype=torch.float32)
        label_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Load graphs from paths
        print(f"📊 Loading {T} graphs for {target_ticker}...")
        graph_list = []
        for path in graph_paths_all:
            graph = self._load_graph_from_path(path)
            if graph is None:
                graph = self._create_empty_graph()
            graph_list.append(graph)
        
        # Create dataset dict
        dataset = {
            "s_o": s_o_tensor,
            "s_h": s_h_tensor,
            "s_c": s_c_tensor,
            "s_m": s_m_tensor,
            "s_n_graphs": graph_list,  # List of Data objects
            "label": label_tensor
        }
        
        # Split train/valid/test
        train_ratio = getattr(TrainConfig, "train_ratio", 0.7)
        valid_ratio = getattr(TrainConfig, "valid_ratio", 0.15)
        
        train_end = int(T * train_ratio)
        valid_end = int(T * (train_ratio + valid_ratio))
        
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