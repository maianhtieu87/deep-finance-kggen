# baselines/data_adapter.py
"""
BaselineDataAdapter — V5.4

V5.4 change: news_dim updated to 768 (FinBERT) in docstring.
No structural change — news_dim is read from data dynamically.
"""

import torch
from src.data_loader import data_prepare, NEWS_EMB_DIM
from configs.config import GlobalConfig


def prepare_for_baselines(pkl_path: str, ticker: str) -> tuple[dict, dict, dict]:
    """
    Load và chuẩn bị dữ liệu cho tất cả baseline models.

    Returns dicts with:
        s_o, s_h, s_c  : (N, W, 1)
        s_m            : (N, W, M)
        s_n            : (N, W, 768)  — FinBERT news embeddings (V5.4)
        label          : (N,)
        indicators     : (N, W, 3)    — cat(s_o, s_h, s_c)
    """
    dp = data_prepare(pkl_path)
    train, valid, test = dp.prepare_data(ticker)

    for split in (train, valid, test):
        if split and len(split.get("label", [])) > 0:
            split["indicators"] = torch.cat(
                [split["s_o"], split["s_h"], split["s_c"]], dim=-1
            )

    return train, valid, test


def merge_tickers(
    splits_per_ticker: dict[str, tuple[dict, dict, dict]],
    shuffle_train: bool = True,
) -> tuple[dict, dict, dict]:
    def _merge(dicts):
        if not dicts:
            return {}
        merged = {}
        for key in dicts[0].keys():
            parts = [d[key] for d in dicts if d and key in d]
            if parts and isinstance(parts[0], torch.Tensor):
                merged[key] = torch.cat(parts, dim=0)
        return merged

    trains = [v[0] for v in splits_per_ticker.values() if v[0] and len(v[0].get("label", [])) > 0]
    valids = [v[1] for v in splits_per_ticker.values() if v[1] and len(v[1].get("label", [])) > 0]
    tests  = [v[2] for v in splits_per_ticker.values() if v[2] and len(v[2].get("label", [])) > 0]

    merged_train = _merge(trains)
    merged_valid = _merge(valids)
    merged_test  = _merge(tests)

    if shuffle_train and "label" in merged_train:
        idx = torch.randperm(len(merged_train["label"]))
        for k in merged_train:
            merged_train[k] = merged_train[k][idx]

    return merged_train, merged_valid, merged_test


def load_all_available_tickers(pkl_path: str) -> tuple[dict, dict, dict]:
    splits = {}
    for ticker in GlobalConfig.TICKERS:
        try:
            tr, va, te = prepare_for_baselines(pkl_path, ticker)
            if tr and len(tr.get("label", [])) >= 100:
                splits[ticker] = (tr, va, te)
                n_train = len(tr["label"])
                n_news  = int((tr["s_n"].abs().sum(dim=-1).sum(dim=-1) > 0).sum())
                print(f"  {ticker}: {n_train} train  news_coverage={n_news}/{n_train}  "
                      f"news_dim={tr['s_n'].shape[-1]}D")
            else:
                print(f"  {ticker}: skip (insufficient data)")
        except Exception as e:
            print(f"  {ticker}: skip ({e})")

    if not splits:
        raise RuntimeError("No ticker data available. Run main_test.py first.")

    return merge_tickers(splits, shuffle_train=True)