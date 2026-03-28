# baselines/data_adapter.py
"""
BaselineDataAdapter — V5.3

Wrapper mỏng quanh src.data_loader.data_prepare.
Thêm key 'indicators' = cat(s_o, s_h, s_c) để baselines dùng dạng (B, W, 3).

V5.3 mapping (khớp với pipeline hiện tại, không dùng PyG/GNN):
  indicators   (B, W, 3)    — giá OHLC ghép    ← "indicator sequence"
  s_n          (B, W, 1024) — Voyage embeddings ← "dynamic document"
  s_m          (B, W, M)    — macro indicators  ← "market context" (thay graph)
  label        (B,)         — 0=DOWN,1=FLAT,2=UP

Lý do không dùng lại data_adapter.py cũ:
  - Cũ dùng kg_tensor (PyG graphs) → không tồn tại trong V5.3
  - Cũ dùng 128D GNN embedding → V5.3 dùng 1024D Voyage
  - Cũ import KGGraphEncoder → bị xóa từ V4
"""

import torch
from src.data_loader import data_prepare
from configs.config import GlobalConfig


def prepare_for_baselines(pkl_path: str, ticker: str) -> tuple[dict, dict, dict]:
    """
    Load và chuẩn bị dữ liệu cho tất cả baseline models.

    Tái sử dụng hoàn toàn data_prepare.prepare_data() — không duplicate logic.
    Chỉ thêm key 'indicators' cho convenience.

    Parameters
    ----------
    pkl_path : str
        Path đến unified_dataset.pkl
    ticker : str
        Ticker symbol, e.g. "TSLA"

    Returns
    -------
    train_data, valid_data, test_data : dict
        Mỗi dict chứa:
            s_o, s_h, s_c  : (N, W, 1)     — giá riêng
            s_m            : (N, W, M)     — macro
            s_n            : (N, W, 1024)  — Voyage news embeddings
            label          : (N,)          — 0/1/2
            indicators     : (N, W, 3)     — cat(s_o, s_h, s_c) cho flat baselines
    """
    dp = data_prepare(pkl_path)
    train, valid, test = dp.prepare_data(ticker)

    for split in (train, valid, test):
        if split and len(split.get("label", [])) > 0:
            split["indicators"] = torch.cat(
                [split["s_o"], split["s_h"], split["s_c"]], dim=-1
            )  # (N, W, 3)

    return train, valid, test


def merge_tickers(
    splits_per_ticker: dict[str, tuple[dict, dict, dict]],
    shuffle_train: bool = True,
) -> tuple[dict, dict, dict]:
    """
    Gộp data nhiều tickers thành một dataset.

    Parameters
    ----------
    splits_per_ticker : {ticker: (train, valid, test)}
    shuffle_train     : bool — shuffle train sau khi gộp

    Returns
    -------
    (merged_train, merged_valid, merged_test)
    """
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
    """
    Load tất cả tickers có trong GlobalConfig.TICKERS.
    Skip ticker nào không có đủ data (ít nhất 100 training samples).

    Returns merged train/valid/test.
    """
    splits = {}
    for ticker in GlobalConfig.TICKERS:
        try:
            tr, va, te = prepare_for_baselines(pkl_path, ticker)
            if tr and len(tr.get("label", [])) >= 100:
                splits[ticker] = (tr, va, te)
                n_train = len(tr["label"])
                n_news  = int((tr["s_n"].abs().sum(dim=-1).sum(dim=-1) > 0).sum())
                print(f"  {ticker}: {n_train} train  news_coverage={n_news}/{n_train}")
            else:
                print(f"  {ticker}: skip (insufficient data)")
        except Exception as e:
            print(f"  {ticker}: skip ({e})")

    if not splits:
        raise RuntimeError("No ticker data available. Run main_test.py first.")

    return merge_tickers(splits, shuffle_train=True)