# data_pipeline/kg/tensorizer.py
import torch
import numpy as np
from torch_geometric.data import Data
from configs.config import GlobalConfig
from data_pipeline.kg.voyage_embedder import VoyageEmbedder

class GraphTensorizer:
    def __init__(self, use_voyage=True, node_dim=128):
        self.use_voyage = use_voyage
        self.node_dim = node_dim
        
        if use_voyage:
            self.embedder = VoyageEmbedder(model=GlobalConfig.EMBED_MODEL)
        else:
            self.embedder = None # Fallback logic if needed

    def build_graphs_by_ticker(self, triples, target_tickers):
        """
        Input: List of (s, p, o)
        Output: Dict { "TSLA": Data(...), "AMZN": Data(...) }
        """
        graphs = {}
        
        # 1. Filter triples relevant to each ticker
        # (Logic đơn giản: nếu entity chứa tên ticker hoặc alias)
        # Bạn có thể dùng logic mapping phức tạp hơn ở đây
        
        for ticker in target_tickers:
            relevant_triples = []
            keywords = [ticker.lower(), GlobalConfig.TICKER_MAPPING.get(ticker, "").lower()]
            
            for s, p, o in triples:
                # Check nếu s hoặc o có liên quan đến ticker
                s_lower, o_lower = s.lower(), o.lower()
                if any(k in s_lower or k in o_lower for k in keywords):
                    relevant_triples.append((s, p, o))
            
            if not relevant_triples:
                # Empty graph
                graphs[ticker] = Data(x=torch.zeros(1, self.node_dim), edge_index=torch.zeros(2, 0, dtype=torch.long))
                continue
                
            # 2. Build Graph for this ticker
            graphs[ticker] = self._triples_to_pyg_data(relevant_triples)
            
        return graphs

    def _triples_to_pyg_data(self, triples):
        # Get unique entities
        entities = sorted(list(set([t[0] for t in triples] + [t[2] for t in triples])))
        entity_to_idx = {e: i for i, e in enumerate(entities)}
        
        # Build Edge Index
        edge_indices = []
        for s, p, o in triples:
            src = entity_to_idx[s]
            dst = entity_to_idx[o]
            # Undirected or directed? Let's do directed + reverse later if needed
            edge_indices.append([src, dst])
            # Add self-loops? Optional
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0:
             edge_index = torch.zeros(2, 0, dtype=torch.long)

        # Build Node Features (Embeddings)
        if self.use_voyage:
            try:
                # Batch embed entities
                embeddings = self.embedder.embed_texts(entities)
                x = torch.tensor(embeddings, dtype=torch.float32)
                
                # Project nếu dim không khớp (Voyage-3-large là 1024, model là 128)
                # Lưu ý: Tốt nhất là để model GNN có input 1024 rồi project xuống.
                # Nhưng nếu muốn tiết kiệm bộ nhớ, ta có thể PCA hoặc Linear ở đây.
                # Tuy nhiên, để đơn giản, ta giả định model GNN input_dim khớp với Voyage
                # HOẶC trong NewsProcessor/Data Loader ta cho phép input khác nhau.
                
                # FIX NHANH: Nếu config đang set news_embed_dim=128 mà Voyage ra 1024
                # Ta sẽ giữ nguyên 1024 ở đây, và sửa 'node_dim' trong model config thành 1024
                pass 
                
            except Exception as e:
                print(f"⚠️ Embedding failed: {e}")
                x = torch.zeros(len(entities), self.node_dim)
        else:
            x = torch.randn(len(entities), self.node_dim) # Random fallback
            
        return Data(x=x, edge_index=edge_index)