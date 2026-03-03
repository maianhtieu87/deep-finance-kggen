# visualize_kg.py - Knowledge Graph Visualization & Quality Analysis
"""
Mục tiêu:
1. Visualize structure của KG (nodes, edges)
2. Kiểm tra embedding quality
3. So sánh graphs giữa các ngày/tickers
4. Export statistics cho analysis
"""

import os
import sys
import json
import pickle
import numpy as np
import torch
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"📊 {title}")
    print(f"{'='*70}")


# ============================================================
# 1. LOAD AND ANALYZE RAW TRIPLES
# ============================================================

def load_triples_from_cache(cache_dir: str, limit: int = 100) -> List[dict]:
    """Load cached article-level triples"""
    if not os.path.exists(cache_dir):
        print(f"❌ Cache dir not found: {cache_dir}")
        return []
    
    files = [f for f in os.listdir(cache_dir) if f.endswith('.json')][:limit]
    results = []
    
    for f in files:
        try:
            with open(os.path.join(cache_dir, f), 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                results.append(data)
        except:
            pass
    
    return results


def analyze_triple_distribution(triples_data: List[dict]) -> dict:
    """Analyze triple quality and distribution"""
    print_section("1. TRIPLE DISTRIBUTION ANALYSIS")
    
    all_subjects = []
    all_predicates = []
    all_objects = []
    triple_counts = []
    
    for item in triples_data:
        triples = item.get('triples', [])
        triple_counts.append(len(triples))
        
        for t in triples:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                all_subjects.append(str(t[0]).lower().strip())
                all_predicates.append(str(t[1]).lower().strip())
                all_objects.append(str(t[2]).lower().strip())
    
    print(f"\n📈 Basic Statistics:")
    print(f"   Total articles with triples: {len(triples_data)}")
    print(f"   Total triples: {len(all_subjects)}")
    print(f"   Avg triples per article: {np.mean(triple_counts):.2f}")
    print(f"   Min triples: {min(triple_counts) if triple_counts else 0}")
    print(f"   Max triples: {max(triple_counts) if triple_counts else 0}")
    
    print(f"\n📋 Top 10 Subjects:")
    for subj, count in Counter(all_subjects).most_common(10):
        print(f"   {count:4d} | {subj[:60]}")
    
    print(f"\n📋 Top 10 Predicates:")
    for pred, count in Counter(all_predicates).most_common(10):
        print(f"   {count:4d} | {pred[:60]}")
    
    print(f"\n📋 Top 10 Objects:")
    for obj, count in Counter(all_objects).most_common(10):
        print(f"   {count:4d} | {obj[:60]}")
    
    # Check for degenerate patterns
    print(f"\n⚠️ Quality Checks:")
    
    # Check for repeated predicates
    pred_counts = Counter(all_predicates)
    if pred_counts:
        top_pred = pred_counts.most_common(1)[0]
        if top_pred[1] / len(all_predicates) > 0.3:
            print(f"   ⚠️ WARNING: Predicate '{top_pred[0]}' appears in {top_pred[1]/len(all_predicates)*100:.1f}% of triples!")
            print(f"      → This may indicate low diversity in extracted relations")
    
    # Check for self-loops
    self_loops = sum(1 for s, _, o in zip(all_subjects, all_predicates, all_objects) if s == o)
    if self_loops > 0:
        print(f"   ⚠️ WARNING: {self_loops} self-loop triples (subject == object)")
    
    # Check for very short entities
    short_subjects = sum(1 for s in all_subjects if len(s) < 3)
    short_objects = sum(1 for o in all_objects if len(o) < 3)
    if short_subjects > len(all_subjects) * 0.1:
        print(f"   ⚠️ WARNING: {short_subjects} subjects are very short (<3 chars)")
    
    return {
        'total_articles': len(triples_data),
        'total_triples': len(all_subjects),
        'avg_triples': np.mean(triple_counts) if triple_counts else 0,
        'subject_distribution': Counter(all_subjects),
        'predicate_distribution': Counter(all_predicates),
    }


# ============================================================
# 2. VISUALIZE GRAPH STRUCTURE
# ============================================================

def visualize_single_graph(tensor_path: str, output_path: str = None):
    """Visualize a single graph from tensor file"""
    
    if not os.path.exists(tensor_path):
        print(f"❌ Tensor not found: {tensor_path}")
        return
    
    data = torch.load(tensor_path, map_location='cpu')
    
    # Extract components
    if isinstance(data, dict):
        nodes = data.get('nodes', [])
        edge_index = data.get('edge_index')
        node_x = data.get('node_x', data.get('x'))
        graph_emb = data.get('graph_emb')
    else:
        print("❌ Unknown tensor format")
        return
    
    print(f"\n📊 Graph Statistics:")
    print(f"   Nodes: {len(nodes)}")
    print(f"   Node features shape: {node_x.shape if node_x is not None else 'N/A'}")
    print(f"   Edges: {edge_index.shape[1] if edge_index is not None else 0}")
    print(f"   Graph embedding shape: {graph_emb.shape if isinstance(graph_emb, torch.Tensor) else len(graph_emb) if graph_emb else 'N/A'}")
    
    if len(nodes) == 0:
        print("   ⚠️ Empty graph - no nodes!")
        return
    
    # Build NetworkX graph
    G = nx.DiGraph()
    
    for i, node in enumerate(nodes):
        G.add_node(i, label=node[:30] + "..." if len(node) > 30 else node)
    
    if edge_index is not None and edge_index.shape[1] > 0:
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Graph structure
    ax1 = axes[0]
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', arrows=True, arrowsize=15)
    
    labels = {i: G.nodes[i]['label'] for i in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=8)
    
    ax1.set_title(f"Graph Structure ({len(nodes)} nodes, {G.number_of_edges()} edges)")
    ax1.axis('off')
    
    # Node feature heatmap
    ax2 = axes[1]
    if node_x is not None and node_x.numel() > 0:
        # Show first 50 dimensions
        feature_sample = node_x[:, :min(50, node_x.shape[1])].numpy()
        im = ax2.imshow(feature_sample, aspect='auto', cmap='RdBu_r')
        ax2.set_xlabel('Feature Dimension (first 50)')
        ax2.set_ylabel('Node Index')
        ax2.set_title('Node Feature Heatmap')
        plt.colorbar(im, ax=ax2)
    else:
        ax2.text(0.5, 0.5, 'No node features', ha='center', va='center')
        ax2.set_title('Node Features: N/A')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   📸 Saved to: {output_path}")
    
    plt.close()
    
    return G


def visualize_graph_statistics(tensor_dir: str, ticker: str, output_dir: str):
    """Generate statistics plots for all graphs of a ticker"""
    print_section(f"2. GRAPH STATISTICS FOR {ticker}")
    
    ticker_dir = os.path.join(tensor_dir, ticker)
    if not os.path.exists(ticker_dir):
        print(f"❌ Directory not found: {ticker_dir}")
        return
    
    files = sorted([f for f in os.listdir(ticker_dir) if f.endswith('.pt')])
    
    stats = {
        'dates': [],
        'num_nodes': [],
        'num_edges': [],
        'node_feat_mean': [],
        'node_feat_std': [],
        'graph_emb_norm': [],
        'is_empty': [],
    }
    
    for f in files:
        try:
            data = torch.load(os.path.join(ticker_dir, f), map_location='cpu')
            
            date = f.replace('.pt', '')
            stats['dates'].append(date)
            
            if isinstance(data, dict):
                node_x = data.get('node_x', data.get('x'))
                edge_index = data.get('edge_index')
                graph_emb = data.get('graph_emb')
                
                num_nodes = node_x.shape[0] if node_x is not None else 0
                num_edges = edge_index.shape[1] if edge_index is not None else 0
                
                stats['num_nodes'].append(num_nodes)
                stats['num_edges'].append(num_edges)
                stats['is_empty'].append(num_nodes == 0)
                
                if node_x is not None and node_x.numel() > 0:
                    stats['node_feat_mean'].append(node_x.mean().item())
                    stats['node_feat_std'].append(node_x.std().item())
                else:
                    stats['node_feat_mean'].append(0)
                    stats['node_feat_std'].append(0)
                
                if graph_emb is not None:
                    if isinstance(graph_emb, torch.Tensor):
                        stats['graph_emb_norm'].append(graph_emb.norm().item())
                    else:
                        stats['graph_emb_norm'].append(np.linalg.norm(graph_emb))
                else:
                    stats['graph_emb_norm'].append(0)
            else:
                stats['num_nodes'].append(0)
                stats['num_edges'].append(0)
                stats['is_empty'].append(True)
                stats['node_feat_mean'].append(0)
                stats['node_feat_std'].append(0)
                stats['graph_emb_norm'].append(0)
                
        except Exception as e:
            print(f"   ❌ Error loading {f}: {e}")
    
    # Print summary
    print(f"\n📈 Summary for {ticker}:")
    print(f"   Total graphs: {len(stats['num_nodes'])}")
    print(f"   Empty graphs: {sum(stats['is_empty'])} ({sum(stats['is_empty'])/len(stats['is_empty'])*100:.1f}%)")
    print(f"   Avg nodes: {np.mean(stats['num_nodes']):.1f}")
    print(f"   Avg edges: {np.mean(stats['num_edges']):.1f}")
    print(f"   Avg graph emb norm: {np.mean(stats['graph_emb_norm']):.4f}")
    
    # Create visualization
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Nodes over time
    ax1 = axes[0, 0]
    ax1.plot(stats['num_nodes'], alpha=0.7)
    ax1.set_xlabel('Time Index')
    ax1.set_ylabel('Number of Nodes')
    ax1.set_title(f'{ticker} - Nodes per Graph over Time')
    ax1.axhline(y=np.mean(stats['num_nodes']), color='r', linestyle='--', label=f'Mean: {np.mean(stats["num_nodes"]):.1f}')
    ax1.legend()
    
    # 2. Edges over time
    ax2 = axes[0, 1]
    ax2.plot(stats['num_edges'], alpha=0.7, color='green')
    ax2.set_xlabel('Time Index')
    ax2.set_ylabel('Number of Edges')
    ax2.set_title(f'{ticker} - Edges per Graph over Time')
    ax2.axhline(y=np.mean(stats['num_edges']), color='r', linestyle='--', label=f'Mean: {np.mean(stats["num_edges"]):.1f}')
    ax2.legend()
    
    # 3. Graph embedding norm
    ax3 = axes[1, 0]
    ax3.plot(stats['graph_emb_norm'], alpha=0.7, color='purple')
    ax3.set_xlabel('Time Index')
    ax3.set_ylabel('Graph Embedding L2 Norm')
    ax3.set_title(f'{ticker} - Graph Embedding Magnitude')
    ax3.axhline(y=np.mean(stats['graph_emb_norm']), color='r', linestyle='--')
    
    # 4. Distribution histogram
    ax4 = axes[1, 1]
    ax4.hist(stats['num_nodes'], bins=30, alpha=0.7, label='Nodes')
    ax4.hist(stats['num_edges'], bins=30, alpha=0.7, label='Edges')
    ax4.set_xlabel('Count')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'{ticker} - Distribution of Graph Sizes')
    ax4.legend()
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{ticker}_graph_stats.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   📸 Saved to: {output_path}")
    plt.close()
    
    return stats


# ============================================================
# 3. EMBEDDING QUALITY ANALYSIS
# ============================================================

def analyze_embedding_quality(tensor_dir: str, tickers: List[str]):
    """Analyze quality of graph embeddings"""
    print_section("3. EMBEDDING QUALITY ANALYSIS")
    
    all_embeddings = []
    all_labels = []  # ticker labels
    
    for ticker in tickers:
        ticker_dir = os.path.join(tensor_dir, ticker)
        if not os.path.exists(ticker_dir):
            continue
        
        files = [f for f in os.listdir(ticker_dir) if f.endswith('.pt')]
        
        for f in files[:50]:  # Sample 50 per ticker
            try:
                data = torch.load(os.path.join(ticker_dir, f), map_location='cpu')
                if isinstance(data, dict):
                    graph_emb = data.get('graph_emb')
                    if graph_emb is not None:
                        if isinstance(graph_emb, torch.Tensor):
                            emb = graph_emb.numpy()
                        else:
                            emb = np.array(graph_emb)
                        
                        if emb.ndim == 1 and len(emb) > 0:
                            all_embeddings.append(emb)
                            all_labels.append(ticker)
            except:
                pass
    
    if not all_embeddings:
        print("❌ No embeddings found")
        return
    
    embeddings = np.stack(all_embeddings)
    
    print(f"\n📊 Embedding Statistics:")
    print(f"   Total embeddings: {len(embeddings)}")
    print(f"   Embedding dimension: {embeddings.shape[1]}")
    print(f"   Mean L2 norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}")
    print(f"   Std L2 norm: {np.linalg.norm(embeddings, axis=1).std():.4f}")
    
    # Check for collapsed embeddings
    zero_embeddings = (np.abs(embeddings).sum(axis=1) < 1e-6).sum()
    print(f"   Zero embeddings: {zero_embeddings} ({zero_embeddings/len(embeddings)*100:.1f}%)")
    
    if zero_embeddings / len(embeddings) > 0.3:
        print(f"   ⚠️ WARNING: Over 30% of embeddings are zero!")
        print(f"      → GNN may not be propagating information properly")
    
    # Check variance per dimension
    dim_variance = np.var(embeddings, axis=0)
    low_var_dims = (dim_variance < 1e-6).sum()
    print(f"   Low-variance dimensions: {low_var_dims} / {embeddings.shape[1]}")
    
    if low_var_dims > embeddings.shape[1] * 0.5:
        print(f"   ⚠️ WARNING: Over 50% of dimensions have near-zero variance!")
        print(f"      → Embeddings may not be discriminative")
    
    # Check similarity within vs across tickers
    print(f"\n📏 Intra-class vs Inter-class Similarity:")
    
    for ticker in tickers:
        ticker_mask = np.array(all_labels) == ticker
        if ticker_mask.sum() > 1:
            ticker_embs = embeddings[ticker_mask]
            
            # Cosine similarity within ticker
            norms = np.linalg.norm(ticker_embs, axis=1, keepdims=True)
            normalized = ticker_embs / (norms + 1e-8)
            sim_matrix = normalized @ normalized.T
            
            # Upper triangle (exclude diagonal)
            triu_indices = np.triu_indices(len(sim_matrix), k=1)
            intra_sim = sim_matrix[triu_indices].mean()
            
            print(f"   {ticker} intra-class cosine similarity: {intra_sim:.4f}")
    
    return embeddings, all_labels


# ============================================================
# 4. COMPARE GRAPHS ACROSS TIME
# ============================================================

def compare_graphs_over_time(tensor_dir: str, ticker: str, num_samples: int = 5):
    """Compare how graphs evolve over time for a ticker"""
    print_section(f"4. GRAPH EVOLUTION FOR {ticker}")
    
    ticker_dir = os.path.join(tensor_dir, ticker)
    if not os.path.exists(ticker_dir):
        print(f"❌ Directory not found: {ticker_dir}")
        return
    
    files = sorted([f for f in os.listdir(ticker_dir) if f.endswith('.pt')])
    
    # Sample evenly
    indices = np.linspace(0, len(files)-1, num_samples, dtype=int)
    sampled_files = [files[i] for i in indices]
    
    print(f"\n📅 Sampled dates:")
    
    for f in sampled_files:
        date = f.replace('.pt', '')
        fpath = os.path.join(ticker_dir, f)
        
        try:
            data = torch.load(fpath, map_location='cpu')
            
            if isinstance(data, dict):
                nodes = data.get('nodes', [])
                node_x = data.get('node_x', data.get('x'))
                edge_index = data.get('edge_index')
                
                num_nodes = len(nodes) if nodes else (node_x.shape[0] if node_x is not None else 0)
                num_edges = edge_index.shape[1] if edge_index is not None else 0
                
                print(f"\n   📆 {date}:")
                print(f"      Nodes: {num_nodes}, Edges: {num_edges}")
                
                if nodes:
                    print(f"      Sample entities: {nodes[:3]}")
                
        except Exception as e:
            print(f"   ❌ {date}: Error - {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("📊 KNOWLEDGE GRAPH VISUALIZATION & QUALITY ANALYSIS")
    print("=" * 70)
    
    from configs.config import GlobalConfig
    
    interim_path = GlobalConfig.INTERIM_PATH
    tensor_dir = os.path.join(interim_path, "kg", "tensors")
    cache_dir = os.path.join(interim_path, "kg_article_cache")
    output_dir = os.path.join("output", "kg_analysis")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📁 Tensor directory: {tensor_dir}")
    print(f"📁 Cache directory: {cache_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    tickers = ['TSLA', 'AMZN', 'MSFT', 'NFLX']
    
    # 1. Analyze raw triples
    if os.path.exists(cache_dir):
        triples_data = load_triples_from_cache(cache_dir, limit=200)
        if triples_data:
            analyze_triple_distribution(triples_data)
    
    # 2. Visualize graph statistics per ticker
    if os.path.exists(tensor_dir):
        for ticker in tickers:
            visualize_graph_statistics(tensor_dir, ticker, output_dir)
    
    # 3. Embedding quality analysis
    if os.path.exists(tensor_dir):
        analyze_embedding_quality(tensor_dir, tickers)
    
    # 4. Compare graphs over time
    if os.path.exists(tensor_dir):
        for ticker in tickers[:1]:  # Just first ticker
            compare_graphs_over_time(tensor_dir, ticker, num_samples=5)
    
    # 5. Visualize sample graph
    sample_tensor = os.path.join(tensor_dir, "TSLA")
    if os.path.exists(sample_tensor):
        files = sorted([f for f in os.listdir(sample_tensor) if f.endswith('.pt')])
        if files:
            mid_file = files[len(files)//2]
            print_section("5. SAMPLE GRAPH VISUALIZATION")
            visualize_single_graph(
                os.path.join(sample_tensor, mid_file),
                os.path.join(output_dir, "sample_graph.png")
            )
    
    print("\n" + "=" * 70)
    print("✅ ANALYSIS COMPLETE")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()