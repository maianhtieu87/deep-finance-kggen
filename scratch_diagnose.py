"""Diagnostic script: analyze WHY news & macro modules hurt performance."""
import torch, sys, numpy as np
sys.path.insert(0, 'd:/deep-finance-kggen')

from src.model import StockMovementModel
from src.data_loader import data_prepare, N_TICKERS
from configs.config import TrainConfig
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = StockMovementModel(
    price_dim=1, macro_dim=5, news_dim=768,
    dim=64, input_dim=20, output_dim=3, num_head=2,
    device=DEVICE, dropout=0.1, class_weights=None,
    use_focal_loss=True, focal_gamma=2.0, n_tickers=N_TICKERS,
).to(DEVICE)

dp = data_prepare('d:/deep-finance-kggen/data/processed/unified_dataset_test.pkl', include_ticker_id=True)

tickers = ['TSLA','AAPL','AMZN','MSFT','GOOGL','META','BA','JPM','WMT']
tr_list = []
for t in tickers:
    tr, va, te = dp.prepare_data(t)
    if tr and len(tr.get('label',[])) > 0:
        tr_list.append(tr)

merged = {}
for key in tr_list[0]:
    parts = [d[key] for d in tr_list if key in d]
    if parts and isinstance(parts[0], torch.Tensor):
        merged[key] = torch.cat(parts, dim=0)

print(f'Training samples: {len(merged["label"])}')

class DS(Dataset):
    def __init__(self, d):
        self.d = d
        self.keys = [k for k in d if isinstance(d[k], torch.Tensor)]
    def __len__(self): return len(self.d['label'])
    def __getitem__(self, i): return {k: self.d[k][i] for k in self.keys}

ds = DS(merged)
ldr = DataLoader(ds, batch_size=32, shuffle=True)
opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

for ep in range(10):
    model.train()
    total_loss, n_batches = 0, 0
    for batch in ldr:
        opt.zero_grad()
        loss = model(
            batch['s_o'].to(DEVICE), batch['s_h'].to(DEVICE),
            batch['s_c'].to(DEVICE), batch['s_m'].to(DEVICE), batch['s_n'].to(DEVICE),
            label=batch['label'].to(DEVICE), mode='train',
            ticker_id=batch.get('ticker_id'), news_mask=batch.get('news_mask'),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        n_batches += 1
    print(f'  Epoch {ep}: loss={total_loss/n_batches:.4f}')

# ============================================================
# ANALYSIS: Direct probing (no hooks)
# ============================================================
model.eval()
print(f'\n{"="*60}')
print(f'DIRECT PROBING ANALYSIS')
print(f'{"="*60}')

with torch.no_grad():
    batch = next(iter(ldr))
    s_o = batch['s_o'].to(DEVICE)
    s_h = batch['s_h'].to(DEVICE)
    s_c = batch['s_c'].to(DEVICE)
    s_m = batch['s_m'].to(DEVICE)
    s_n = batch['s_n'].to(DEVICE)
    nm = batch.get('news_mask')
    if nm is not None: nm = nm.to(DEVICE)

    # Step 1: Encode
    v_m, v_i, v_n = model.multimodal_encoder(s_o, s_h, s_c, s_m, s_n, news_mask=nm)

    print(f'\n--- ENCODED REPRESENTATIONS ---')
    print(f'v_i (price): mean={v_i.mean():.4f}, std={v_i.std():.4f}, L2/sample={v_i.reshape(v_i.shape[0],-1).norm(dim=1).mean():.2f}')
    print(f'v_m (macro): mean={v_m.mean():.4f}, std={v_m.std():.4f}, L2/sample={v_m.reshape(v_m.shape[0],-1).norm(dim=1).mean():.2f}')
    print(f'v_n (news):  mean={v_n.mean():.4f}, std={v_n.std():.4f}, L2/sample={v_n.reshape(v_n.shape[0],-1).norm(dim=1).mean():.2f}')
    ratio_nv = v_n.reshape(v_n.shape[0],-1).norm(dim=1).mean() / v_i.reshape(v_i.shape[0],-1).norm(dim=1).mean()
    ratio_mv = v_m.reshape(v_m.shape[0],-1).norm(dim=1).mean() / v_i.reshape(v_i.shape[0],-1).norm(dim=1).mean()
    print(f'  L2 ratio news/price:  {ratio_nv:.2f}x')
    print(f'  L2 ratio macro/price: {ratio_mv:.2f}x')

    # Step 2: Fusion Stage 1 decomposition
    f1 = model.fusion_stage1
    # Handle safe masking same as forward()
    attn_mask = nm
    if attn_mask is not None:
        fully_masked = attn_mask.all(dim=1, keepdim=True)
        if fully_masked.any():
            attn_mask = attn_mask & ~fully_masked

    H_unstable1, attn_w1 = f1.cross_attn(query=v_i, key=v_n, value=v_n,
                                           key_padding_mask=attn_mask, need_weights=True)
    H_a1 = f1.W_a(H_unstable1) + f1.bias_a
    H_b1 = torch.sigmoid(f1.W_b(v_i))
    H_gated1 = H_a1 * H_b1

    print(f'\n--- STAGE 1: price × news ---')
    print(f'  H_unstable L2/sample: {H_unstable1.reshape(H_unstable1.shape[0],-1).norm(dim=1).mean():.4f}')
    print(f'  H_a (transformed) L2: {H_a1.reshape(H_a1.shape[0],-1).norm(dim=1).mean():.4f}')
    print(f'  Gate H_b: mean={H_b1.mean():.4f}, std={H_b1.std():.4f}')
    print(f'    >0.5: {(H_b1>0.5).float().mean():.3f}, >0.9: {(H_b1>0.9).float().mean():.3f}, <0.1: {(H_b1<0.1).float().mean():.3f}')
    print(f'  H_gated L2/sample:    {H_gated1.reshape(H_gated1.shape[0],-1).norm(dim=1).mean():.4f}')
    
    delta1 = H_gated1.norm(dim=-1)
    base1 = v_i.norm(dim=-1)
    rc1 = delta1 / (base1 + 1e-8)
    print(f'  Relative change |H_gated|/|v_i|: mean={rc1.mean():.4f}, max={rc1.max():.4f}')
    print(f'  Attention entropy: {(-attn_w1 * (attn_w1+1e-10).log()).sum(dim=-1).mean():.4f}')

    # Stage 1 output
    H_id = f1.norm(v_i + f1.dropout(H_gated1))

    # Step 3: Fusion Stage 2
    f2 = model.fusion_stage2
    H_unstable2, _ = f2.cross_attn(query=H_id, key=v_m, value=v_m, need_weights=False)
    H_a2 = f2.W_a(H_unstable2) + f2.bias_a
    H_b2 = torch.sigmoid(f2.W_b(H_id))
    H_gated2 = H_a2 * H_b2

    print(f'\n--- STAGE 2: fused × macro ---')
    print(f'  Gate H_b: mean={H_b2.mean():.4f}, std={H_b2.std():.4f}')
    print(f'    >0.5: {(H_b2>0.5).float().mean():.3f}, >0.9: {(H_b2>0.9).float().mean():.3f}, <0.1: {(H_b2<0.1).float().mean():.3f}')
    rc2 = H_gated2.norm(dim=-1) / (H_id.norm(dim=-1) + 1e-8)
    print(f'  Relative change |H_gated|/|H_id|: mean={rc2.mean():.4f}, max={rc2.max():.4f}')

# ============================================================
# NEWS SIGNAL VS LABELS
# ============================================================
print(f'\n{"="*60}')
print(f'NEWS SIGNAL-TO-NOISE: CORRELATION WITH LABELS')
print(f'{"="*60}')

labels = merged['label'].numpy()
s_n_all = merged['s_n']
mask_all = merged['news_mask']

last_day_news = s_n_all[:, -1, :]
last_day_mask = mask_all[:, -1]
has_news = ~last_day_mask
n_with = has_news.sum().item()
print(f'  Samples with news on last day: {n_with}/{len(labels)} ({100*n_with/len(labels):.1f}%)')

if n_with > 100:
    news_vecs = last_day_news[has_news].numpy()
    news_labels = labels[has_news.numpy()]
    means = {}
    for cls in [0, 1, 2]:
        idx = news_labels == cls
        if idx.sum() > 0:
            mean_vec = news_vecs[idx].mean(axis=0)
            means[cls] = mean_vec
            print(f'  Class {cls} ({["DOWN","HOLD","UP"][cls]}): n={idx.sum()}, mean_norm={np.linalg.norm(mean_vec):.4f}')

    if len(means) == 3:
        print(f'\n  Inter-class cosine similarities (higher = less discriminative):')
        for a in range(3):
            for b in range(a+1, 3):
                cos = np.dot(means[a], means[b]) / (np.linalg.norm(means[a]) * np.linalg.norm(means[b]) + 1e-8)
                print(f'    Cosine({["DOWN","HOLD","UP"][a]} vs {["DOWN","HOLD","UP"][b]}): {cos:.4f}')

# ============================================================
# MACRO SIGNAL VS LABELS
# ============================================================
print(f'\n{"="*60}')
print(f'MACRO SIGNAL-TO-NOISE: CORRELATION WITH LABELS')
print(f'{"="*60}')

from scipy import stats
s_m_all = merged['s_m'].numpy()
last_macro = s_m_all[:, -1, :]
macro_keys = ['dxy', 'sp500', 'vix', 'wti', 'yield_spread']

for cls in [0, 1, 2]:
    idx = labels == cls
    if idx.sum() > 0:
        mean_m = last_macro[idx].mean(axis=0)
        print(f'  Class {cls} ({["DOWN","HOLD","UP"][cls]}): n={idx.sum()}, macro_mean={mean_m.round(4)}')

print(f'\n  Spearman correlation (each macro feature vs label):')
for i, name in enumerate(macro_keys):
    r, p = stats.spearmanr(last_macro[:, i], labels)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    print(f'    {name:20s}: r={r:+.4f}, p={p:.4f} {sig}')

# ============================================================
# NEWS TEMPORAL REDUNDANCY IN WINDOW  
# ============================================================
print(f'\n{"="*60}')
print(f'NEWS TEMPORAL REDUNDANCY IN WINDOW')
print(f'{"="*60}')

# For TSLA, compute cosine sim between consecutive days within a window
s_n_tsla = merged['s_n'][:594]  # first ticker
mask_tsla = merged['news_mask'][:594]
intra_window_sims = []
for i in range(min(200, len(s_n_tsla))):
    window = s_n_tsla[i]  # (20, 768)
    mask = mask_tsla[i]   # (20,)
    valid_idx = (~mask).nonzero(as_tuple=True)[0]
    if len(valid_idx) >= 2:
        for j in range(len(valid_idx)-1):
            a = window[valid_idx[j]]
            b = window[valid_idx[j+1]]
            cos = torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
            intra_window_sims.append(cos)

if intra_window_sims:
    sims = np.array(intra_window_sims)
    print(f'  Intra-window consecutive cosine sim:')
    print(f'    mean={sims.mean():.4f}, std={sims.std():.4f}')
    print(f'    median={np.median(sims):.4f}')
    print(f'    >0.9: {(sims>0.9).mean()*100:.1f}%')
    print(f'    >0.7: {(sims>0.7).mean()*100:.1f}%')

print(f'\n{"="*60}')
print(f'DIAGNOSIS COMPLETE')
print(f'{"="*60}')

