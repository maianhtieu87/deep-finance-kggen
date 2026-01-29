import torch

def tensorize(triples, ticker):
    nodes = list({s for s,_,_ in triples} | {o for _,_,o in triples})
    node2id = {n:i for i,n in enumerate(nodes)}

    edges = [(node2id[s], node2id[o]) for s,_,o in triples]
    edge_index = torch.tensor(edges).t().contiguous()

    node_x = torch.randn(len(nodes), 384)   # SBERT embedding placeholder
    ticker_idx = node2id.get(ticker, 0)

    return {
        "node_x": node_x,
        "edge_index": edge_index,
        "ticker_idx": ticker_idx
    }
