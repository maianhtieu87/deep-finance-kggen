#!/usr/bin/env python3
# embed_news.py — V8.0 (FinBERT per-triple + Voyage legacy, unified CLI)
"""
Stage A.5 — News Embedding

V8.0 changes vs V7.1:
  Added FinBERT per-triple embedding path (PRIMARY, recommended):
    --finbert          Use FinBERT [CLS] with impact-weighted aggregation
                       Output: 768D vectors (FinBERT CLS dimension)

  Kept Voyage path (LEGACY, requires API key):
    (no flag)          Use Voyage voyage-finance-2
                       Output: 1024D vectors

  FinBERT approach rationale:
    - Directional embedding space: "Apple cuts guidance" and
      "Apple raises guidance" land on opposite sides (FinBERT was
      fine-tuned on financial sentiment labels).
    - Per-triple encoding: each KG triple (~20 tokens) encoded
      individually, then aggregated with weights =
      |price_impact_score| × confidence × relevance_to_ticker
      (Gemini-scored fields directly weight the aggregation).
    - Local inference: no API cost, deterministic, auditable.
    - Paper 3 (Chronos-Fuse) confirms fine-tuning sentence encoders
      causes semantic collapse → FinBERT kept FROZEN.

  Backward compatibility:
    - run_embed_news() (Voyage) unchanged
    - run_embed_news_finbert() is the new function
    - Output JSON format identical: {"YYYY-MM-DD": {"TICKER": [Nd]}}
    - news_embeddings.json dimension changes 1024→768 when using FinBERT

V7.1 features retained:
  - Unified SHA1 + meta-based cache reading
  - Merge-on-write for per-ticker runs
  - --check-cache, --check-output, --force-sha1, --force-meta flags

Usage:
    python embed_news.py --finbert                     # FinBERT, all tickers (PRIMARY)
    python embed_news.py --finbert --ticker TSLA       # FinBERT, 1 ticker
    python embed_news.py --finbert --date 2023         # FinBERT, date filter
    python embed_news.py                               # Voyage legacy (needs API key)
    python embed_news.py --ticker TSLA                 # Voyage, 1 ticker
    python embed_news.py --check-cache                 # cache stats, no embedding
    python embed_news.py --check-output                # output JSON stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import GlobalConfig
from configs.ticker_aliases import TICKER_NAME_MAP


# ─────────────────────────────────────────────────────────────────────────────
# FINBERT PER-TRIPLE EMBEDDER  (V8.0 — PRIMARY PATH)
# ─────────────────────────────────────────────────────────────────────────────

class FinBERTPerTripleEmbedder:
    """
    FinBERT [CLS] embedding with impact-weighted aggregation per (date, ticker).

    Pipeline:
      For each (date, ticker):
        1. Get filtered KG triples from cache
        2. Format each triple as short text (~20–40 tokens)
        3. FinBERT [CLS] → 768D embedding per triple
        4. Weighted mean: weight = |price_impact_score| × confidence × relevance
        5. Output: 768D daily embedding (or zero vector if no triples)

    Key design decisions:
      - FROZEN encoder: Paper 3 (Chronos-Fuse) shows fine-tuning causes
        semantic collapse (MSE degrades 6× vs frozen).
      - Per-triple encoding: avoids the Voyage problem of serializing all
        triples into one long text that loses individual impact scores.
      - Relation → verb mapping: FinBERT understands natural language verbs
        better than uppercase codes like "POS_IMPACTS".
      - Directional space: FinBERT fine-tuned on financial sentiment →
        "CUTS guidance" and "RAISES guidance" are antipodal in embedding space.
    """

    EMBED_DIM = 768

    RELATION_VERBS: Dict[str, str] = {
        "ANNOUNCES":    "announces",
        "RAISES":       "raises",
        "CUTS":         "cuts",
        "INVESTS_IN":   "invests in",
        "DIVESTS":      "divests from",
        "APPOINTS":     "appoints",
        "POS_IMPACTS":  "positively impacts",
        "NEG_IMPACTS":  "negatively impacts",
        "COMPETES_WITH":"competes with",
        "REGULATES":    "regulates",
        "SUPPLIES_TO":  "supplies to",
        "CONTROLS":     "controls",
        "SIGNALS":      "signals",
        "RELATES_TO":   "relates to",
    }

    def __init__(
        self,
        model_name:  str = "ProsusAI/finbert",
        device:      str = None,
        batch_size:  int = 32,
        cache_dir:   str = None,
    ):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise RuntimeError(
                "FinBERT requires: pip install transformers accelerate\n"
                "Run: pip install transformers accelerate"
            )

        import torch as _torch
        if device is None:
            device = "cuda" if _torch.cuda.is_available() else "cpu"
        self.device     = device
        self.batch_size = batch_size
        self._torch     = _torch

        print(f"Loading FinBERT '{model_name}' → device: {device}")
        cache_dir_hf = cache_dir  # HuggingFace model cache (None = default ~/.cache)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir_hf)
        self.model     = AutoModel.from_pretrained(model_name, cache_dir=cache_dir_hf)
        self.model.eval()
        self.model.to(self.device)

        # Freeze ALL parameters — no fine-tuning (semantic collapse risk)
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"  FinBERT loaded. Parameters FROZEN. Output dim: {self.EMBED_DIM}D")

    def _format_triple(self, triple: Dict, ticker: str) -> str:
        """
        Format one KG triple as short natural-language text for FinBERT.

        Template: "{subject} {verb} {object}. {reasoning}. Impact on {ticker}."
        Target length: ~20–40 tokens (well within FinBERT 512-token limit).
        """
        subj   = triple.get("subject", {}).get("name", "")
        rel    = triple.get("relation", "")
        obj    = triple.get("object",  {}).get("name", "")
        reason = (triple.get("reasoning") or "").strip()

        verb = self.RELATION_VERBS.get(rel, rel.lower().replace("_", " "))
        text = f"{subj} {verb} {obj}"

        # Append reasoning if compact (Gemini writes ≤15-word reasons)
        if reason and len(reason) <= 80:
            text += f". {reason}"

        # Ticker context: FinBERT knows who this signal is for
        text += f". Impact on {ticker}."

        return text

    def _encode_batch(self, texts: List[str]) -> "np.ndarray":
        """Encode list of texts → [CLS] hidden states (N, 768)."""
        import torch

        if not texts:
            return np.zeros((0, self.EMBED_DIM), dtype=np.float32)

        all_cls = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                max_length=128,   # triple texts are short; 128 is generous
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            # [CLS] token = index 0 of last_hidden_state
            cls = outputs.last_hidden_state[:, 0, :]  # (B, 768)
            all_cls.append(cls.cpu().numpy())

        return np.vstack(all_cls).astype(np.float32)

    def embed_triples(self, triples: List[Dict], ticker: str) -> List[float]:
        """
        Convert KG triples for one (date, ticker) → 768D embedding.

        Aggregation: impact-weighted mean.
          weight_i = |price_impact_score_i| × confidence_i × relevance_i

        Returns zero vector (list of 768 floats) if triples is empty.
        news_mask in the model will mark this day as "no news".
        """
        if not triples:
            return [0.0] * self.EMBED_DIM

        # Step 1: format each triple as NL text
        texts = [self._format_triple(t, ticker) for t in triples]

        # Step 2: encode
        embeddings = self._encode_batch(texts)  # (N, 768)

        # Step 3: compute Gemini-scored weights
        weights = []
        for t in triples:
            impact = abs(float(t.get("price_impact_score", 0.0)))
            conf   = float(t.get("confidence", 0.65))
            rel    = float(t.get("relevance_to_ticker", 0.5))
            w      = impact * conf * rel
            weights.append(max(w, 1e-6))  # floor to avoid all-zero weight array

        weights_arr = np.array(weights, dtype=np.float32)
        weights_arr /= weights_arr.sum()  # normalize to sum=1

        # Step 4: weighted mean
        daily_emb = (embeddings * weights_arr[:, np.newaxis]).sum(axis=0)
        return daily_emb.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# VOYAGE EMBEDDER  (V7.1 — LEGACY PATH, kept for backward compat)
# ─────────────────────────────────────────────────────────────────────────────

class VoyageEmbedder:
    """Rate-limited Voyage embedder with disk cache (SHA1-keyed). Legacy path."""

    def __init__(self, cache_dir: str):
        self.api_key = os.getenv("VOYAGE_API_KEY", GlobalConfig.VOYAGE_API_KEY)
        if not self.api_key or self.api_key.strip() in ("", "---"):
            raise RuntimeError("VOYAGE_API_KEY not set.")
        self.model        = getattr(GlobalConfig, "EMBED_MODEL", "voyage-3-large")
        self.max_texts    = getattr(GlobalConfig, "MAX_TEXTS_PER_REQ", 40)
        self.max_retries  = getattr(GlobalConfig, "MAX_RETRIES", 6)
        self.backoff_base = getattr(GlobalConfig, "BACKOFF_BASE", 30)
        payment_added     = bool(getattr(GlobalConfig, "PAYMENT_ADDED", True))
        rl = GlobalConfig.VOYAGE_RATE_LIMITS[payment_added]
        self.base_sleep = float(rl.get("SLEEP", 1.0))
        self.rpm        = int(rl.get("RPM", 50))
        self.cache_dir  = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self._req_times: List[float] = []

    def _sha1(self, s: str) -> str:
        return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

    def _cache_path(self, text: str) -> str:
        return os.path.join(self.cache_dir, f"{self._sha1(text)}.json")

    def _load(self, text: str) -> Optional[List[float]]:
        p = self._cache_path(text)
        if not os.path.exists(p):
            return None
        try:
            with open(p) as f:
                emb = json.load(f).get("embedding")
            if isinstance(emb, list) and len(emb) > 0:
                return emb
        except Exception:
            pass
        return None

    def _save(self, text: str, emb: List[float]):
        with open(self._cache_path(text), "w") as f:
            json.dump({"embedding": emb}, f)

    def _rpm_guard(self):
        now = time.time()
        self._req_times = [t for t in self._req_times if now - t < 60.0]
        if len(self._req_times) >= self.rpm:
            wait = max(0.0, 60.0 - (now - min(self._req_times))) + 0.5
            print(f"  Voyage RPM limit — sleep {wait:.1f}s")
            time.sleep(wait)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        out: List[Optional[List[float]]] = [None] * len(texts)
        miss_i, miss_t = [], []
        for i, t in enumerate(texts):
            t = (t or "").strip()
            if not t:
                out[i] = [0.0] * 1024
                continue
            cached = self._load(t)
            if cached is not None:
                out[i] = cached
            else:
                miss_i.append(i)
                miss_t.append(t)

        if not miss_t:
            return [o if o is not None else [0.0] * 1024 for o in out]

        url     = "https://api.voyageai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}

        def _chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i: i + n]

        pos = 0
        for batch_texts in _chunks(miss_t, self.max_texts):
            batch_idx = miss_i[pos: pos + len(batch_texts)]
            pos += len(batch_texts)
            payload = {"model": self.model, "input": batch_texts}
            for attempt in range(self.max_retries):
                try:
                    self._rpm_guard()
                    if self.base_sleep > 0:
                        time.sleep(self.base_sleep)
                    r = requests.post(url, headers=headers, json=payload,
                                      timeout=(15, 120))
                    self._req_times.append(time.time())
                    if r.status_code == 429:
                        wait = self.backoff_base * (2 ** attempt) + 2.0
                        print(f"  429 rate limit — sleep {wait:.0f}s")
                        time.sleep(wait)
                        continue
                    r.raise_for_status()
                    for bi, item in enumerate(r.json().get("data", [])):
                        emb = item.get("embedding", [])
                        out[batch_idx[bi]] = emb
                        self._save(batch_texts[bi], emb)
                    break
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        print(f"  Voyage failed: {e}")
                        for idx in batch_idx:
                            out[idx] = [0.0] * 1024
                    else:
                        time.sleep(self.backoff_base * (2 ** attempt))

        return [o if o is not None else [0.0] * 1024 for o in out]


# ─────────────────────────────────────────────────────────────────────────────
# SHARED UTILS
# ─────────────────────────────────────────────────────────────────────────────

def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _parse_tickers(val: Any) -> List[str]:
    if isinstance(val, list):
        return [t.strip().upper() for t in val if isinstance(t, str) and t.strip()]
    if isinstance(val, str):
        return [t.strip().upper() for t in val.split(",") if t.strip()]
    return []

TITLE_WEIGHT = 3

def detect_primary_ticker(title: str, content: str, tickers: List[str]) -> str:
    if not tickers:
        return ""
    if len(tickers) == 1:
        return tickers[0]
    t_up = (title   or "").upper()
    c_up = (content or "").upper()
    counts = {}
    for t in tickers:
        score = 0
        for name in TICKER_NAME_MAP.get(t, [t]):
            n_up = name.upper()
            score += t_up.count(n_up) * TITLE_WEIGHT + c_up.count(n_up)
        counts[t] = score
    best = max(counts.values())
    if best == 0:
        return tickers[0]
    for t in tickers:
        if counts[t] == best:
            return t

def _ticker_mentioned_in_triple(ticker: str, triple: Dict) -> bool:
    tl = ticker.lower()
    sn = triple.get("subject", {}).get("name", "").lower()
    on = triple.get("object",  {}).get("name", "").lower()
    if tl in sn or tl in on:
        return True
    for name in TICKER_NAME_MAP.get(ticker.upper(), []):
        if name.lower() in sn or name.lower() in on:
            return True
    return False

def triples_to_text(triples: List[Dict], ticker: str) -> str:
    """Serialize triples to plain text for Voyage embedding (legacy path)."""
    if not triples:
        return ""
    sorted_triples = sorted(
        triples,
        key=lambda t: abs(float(t.get("price_impact_score", 0))),
        reverse=True,
    )
    parts = [f"TARGET: {ticker}"]
    for t in sorted_triples:
        subj   = t.get("subject", {}).get("name", "")
        rel    = t.get("relation", "")
        obj    = t.get("object",  {}).get("name", "")
        reason = t.get("reasoning", "")
        if not subj or not obj:
            continue
        line = f"{subj} {rel} {obj}"
        if reason:
            line += f". {reason}"
        parts.append(line)
    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# CACHE STORE (shared by both Voyage and FinBERT paths)
# ─────────────────────────────────────────────────────────────────────────────

class CacheStore:
    """
    Unified cache store: preload from BOTH new-format (_meta) and old-format (SHA1).

    meta_entries : List[Dict]  — files with _meta (new format V5.2+)
    sha1_no_meta : Set[str]    — SHA1s of files without _meta (old format)
    """

    def __init__(self, cache_dir: str):
        self.cache_dir    = cache_dir
        self.meta_entries: List[Dict] = []
        self.sha1_no_meta: Set[str]  = set()
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if not os.path.exists(self.cache_dir):
            self._loaded = True
            return

        files = [
            f for f in os.listdir(self.cache_dir)
            if f.endswith(".json") and not f.startswith("_")
        ]
        n_meta, n_no_meta = 0, 0

        for fname in files:
            path = os.path.join(self.cache_dir, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            triples  = data.get("triples", [])
            meta     = data.get("_meta", {})
            date_val = meta.get("date")
            primary  = str(meta.get("primary_ticker", "")).upper()

            if date_val and triples:
                self.meta_entries.append({
                    "date":           str(date_val),
                    "primary_ticker": primary,
                    "triples":        triples,
                })
                n_meta += 1
            elif triples:
                sha1 = fname[:-5]
                self.sha1_no_meta.add(sha1)
                n_no_meta += 1

        self._loaded = True
        print(f"  Cache scan: {n_meta} new-format (with _meta), "
              f"{n_no_meta} old-format (SHA1-only)")

    def get_triples_meta(
        self,
        date_str: str, ticker: str,
        min_relevance: float, min_confidence: float,
    ) -> List[Dict]:
        """Lookup via _meta entries (new format). In-memory, fast."""
        all_triples: List[Dict] = []
        for entry in self.meta_entries:
            if entry["date"] != date_str:
                continue
            primary = entry["primary_ticker"]
            raw     = entry["triples"]
            if primary == ticker.upper():
                filtered = [
                    t for t in raw
                    if float(t.get("confidence", 0))          >= min_confidence
                    and float(t.get("relevance_to_ticker", 0)) >= min_relevance
                ]
            else:
                filtered = [
                    t for t in raw
                    if _ticker_mentioned_in_triple(ticker, t)
                    and float(t.get("confidence", 0))          >= min_confidence
                    and float(t.get("relevance_to_ticker", 0)) >= min_relevance
                ]
            all_triples.extend(filtered)
        return all_triples

    def get_triples_sha1(
        self,
        day_df: pd.DataFrame, ticker: str,
        min_relevance: float, min_confidence: float,
    ) -> List[Dict]:
        """Fallback: SHA1-based lookup (old-format files, no _meta)."""
        if not self.sha1_no_meta:
            return []
        all_triples: List[Dict] = []
        for _, row in day_df.iterrows():
            title   = _norm(str(row.get("title",   "") or ""))
            content = _norm(str(row.get("content", "") or ""))
            parts   = []
            if title:
                parts.append(f"HEADLINES:\n- {title}")
            if content:
                parts.append(f"ARTICLES:\n{content}")
            full_text = "\n\n".join(parts)
            if not full_text:
                continue
            h = _sha1(full_text)
            if h not in self.sha1_no_meta:
                continue
            path = os.path.join(self.cache_dir, f"{h}.json")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw = data.get("triples", [])
            except Exception:
                continue
            primary = str(row.get("primary_ticker") or ticker)
            if primary.upper() == ticker.upper():
                filtered = [
                    t for t in raw
                    if float(t.get("confidence", 0))          >= min_confidence
                    and float(t.get("relevance_to_ticker", 0)) >= min_relevance
                ]
            else:
                filtered = [
                    t for t in raw
                    if _ticker_mentioned_in_triple(ticker, t)
                    and float(t.get("confidence", 0))          >= min_confidence
                    and float(t.get("relevance_to_ticker", 0)) >= min_relevance
                ]
            all_triples.extend(filtered)
        return all_triples


def _dedup_triples(triples: List[Dict]) -> List[Dict]:
    """Simple exact dedup by (subject, relation, object)."""
    seen: set = set()
    result: List[Dict] = []
    for t in triples:
        key = (
            t.get("subject", {}).get("name", ""),
            t.get("relation", ""),
            t.get("object",  {}).get("name", ""),
        )
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS
# ─────────────────────────────────────────────────────────────────────────────

def check_output_stats(output_path: str):
    """Print stats about existing news_embeddings.json."""
    print(f"\nOutput file: {output_path}")
    if not os.path.exists(output_path):
        print("  Not found.")
        return
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Read error: {e}")
        return

    print(f"  Total dates: {len(data)}")
    if not data:
        return

    all_tickers: Dict[str, int] = {}
    detected_dims: set = set()
    n_zero, n_total = 0, 0

    for date_str, ticker_dict in data.items():
        for t, vec in ticker_dict.items():
            all_tickers[t] = all_tickers.get(t, 0) + 1
            n_total += 1
            if isinstance(vec, list):
                if len(vec) > 0:
                    detected_dims.add(len(vec))
                if all(v == 0.0 for v in vec[:10]):
                    n_zero += 1

    dates_sorted = sorted(data.keys())
    print(f"  Date range: {dates_sorted[0]} → {dates_sorted[-1]}")
    print(f"  Embedding dim(s) detected: {sorted(detected_dims)}")
    if 768 in detected_dims:
        print(f"  → 768D = FinBERT CLS (current pipeline)")
    if 1024 in detected_dims:
        print(f"  → 1024D = Voyage legacy (rebuild with --finbert to upgrade)")
    print(f"  Tickers in file:")
    for t, cnt in sorted(all_tickers.items()):
        print(f"    {t}: {cnt} days")
    print(f"  Zero vectors (no news): {n_zero}/{n_total} "
          f"({100 * n_zero / max(n_total, 1):.1f}%)")


def check_cache_stats(cache_dir: str):
    """Print stats about cache directory."""
    print(f"\nCache directory: {cache_dir}")
    if not os.path.exists(cache_dir):
        print("  Not found.")
        return

    files = [f for f in os.listdir(cache_dir)
             if f.endswith(".json") and not f.startswith("_")]
    print(f"  Total files: {len(files)}")

    n_with_meta, n_no_meta, n_empty = 0, 0, 0
    date_set:   set = set()
    ticker_set: set = set()

    for fname in files:
        path = os.path.join(cache_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        triples  = data.get("triples", [])
        meta     = data.get("_meta", {})
        date_val = meta.get("date")
        primary  = meta.get("primary_ticker", "")

        if not triples:
            n_empty += 1
        elif date_val:
            n_with_meta += 1
            date_set.add(str(date_val)[:10])
            if primary:
                ticker_set.add(primary.upper())
        else:
            n_no_meta += 1

    print(f"  New-format (with _meta, has triples): {n_with_meta}")
    print(f"  Old-format (no _meta):                {n_no_meta}")
    print(f"  Empty (no triples):                   {n_empty}")
    if date_set:
        ds = sorted(date_set)
        print(f"  Date range (new-format): {ds[0]} → {ds[-1]}")
        print(f"  Tickers in _meta: {sorted(ticker_set)}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SHARED DATAFRAME NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_news_df(
    news_df:       pd.DataFrame,
    ticker_filter: Optional[str] = None,
    date_prefix:   Optional[str] = None,
) -> pd.DataFrame:
    """
    Normalise news DataFrame columns and apply optional filters.
    Shared by both Voyage and FinBERT paths.
    """
    df = news_df.copy()
    if "headline" in df.columns and "title" not in df.columns:
        df = df.rename(columns={"headline": "title"})
    if "ticker" in df.columns and "equity" not in df.columns:
        df = df.rename(columns={"ticker": "equity"})
    if "content" not in df.columns:
        for alt in ("body", "text"):
            if alt in df.columns:
                df = df.rename(columns={alt: "content"})
                break
    if "content" not in df.columns: df["content"] = ""
    if "title"   not in df.columns: df["title"]   = ""

    if "date" not in df.columns:
        DATE_CANDS = [
            "created_at", "createdAt", "published_at", "publishedAt",
            "publish_date", "pub_date", "Date", "DATE", "timestamp", "news_date",
        ]
        date_col = next((c for c in DATE_CANDS if c in df.columns), None)
        if date_col is None:
            date_col = next(
                (c for c in df.columns if any(
                    k in c.lower() for k in ("date", "time", "publish", "creat"))),
                None,
            )
        if date_col is not None:
            df = df.rename(columns={date_col: "date"})

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    ticker_col = next((c for c in ("symbols", "equity") if c in df.columns), None)
    if ticker_col is None:
        raise ValueError("No ticker column found (expected 'symbols' or 'equity').")

    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)
    df = df[df["_all_tickers"].map(len) > 0]

    df["primary_ticker"] = df.apply(
        lambda row: detect_primary_ticker(
            str(row.get("title",   "") or ""),
            str(row.get("content", "") or ""),
            row["_all_tickers"],
        ),
        axis=1,
    )

    df = df.explode("_all_tickers").rename(columns={"_all_tickers": "equity"})
    df = df[df["equity"].notna() & (df["equity"] != "")].reset_index(drop=True)
    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)

    if ticker_filter:
        df = df[df["equity"] == ticker_filter.upper()]
    if date_prefix:
        df = df[df["date"].astype(str).str.startswith(date_prefix)]

    return df


def _load_existing_output(output_path: str) -> Dict[str, Dict[str, List[float]]]:
    """Load existing news_embeddings.json. Return {} on error."""
    if not os.path.exists(output_path):
        return {}
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"  Warning: could not load existing output ({e}) — starting fresh")
    return {}


def _save_output(
    new_output:    Dict[str, Dict[str, List[float]]],
    output_path:   str,
    ticker_filter: Optional[str],
) -> str:
    """
    Merge-on-write + save.

    When ticker_filter is set, load existing output and merge so that
    other tickers already in the file are preserved.
    When no filter (full rebuild), write new_output directly.
    """
    if ticker_filter:
        existing = _load_existing_output(output_path)
        if existing:
            n_ex = len(set(t for d in existing.values() for t in d))
            print(f"\nMerging with existing output:")
            print(f"  Existing: {len(existing)} dates, ~{n_ex} tickers")
            print(f"  New data: {len(new_output)} dates for {ticker_filter.upper()}")
            merged: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
            for ds, td in existing.items():
                merged[ds].update(td)
            for ds, td in new_output.items():
                merged[ds].update(td)
            final = merged
            n_f = len(set(t for d in final.values() for t in d))
            print(f"  Final: {len(final)} dates, ~{n_f} tickers")
        else:
            print(f"\nNo existing output — creating new file for {ticker_filter.upper()}")
            final = new_output
    else:
        final = new_output

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dict(final), f, ensure_ascii=False)

    print(f"\nSaved: {output_path}")
    print(f"  Dates: {len(final)}")
    if final:
        sample_date = next(iter(final))
        print(f"  Sample ({sample_date}): tickers = {sorted(final[sample_date].keys())}")
    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# QUALITY OUTPUT HELPERS  (V8.1)
# ─────────────────────────────────────────────────────────────────────────────

QUALITY_DIM = 4
"""
Quality vector layout per (date, ticker):
  [0] log(1 + n_triples)      — signal volume, log-scaled để tránh outlier
  [1] avg_confidence           — extractor confidence trung bình
  [2] avg_relevance_to_ticker  — độ liên quan trực tiếp với ticker
  [3] avg_abs_impact           — độ mạnh tác động giá trung bình
"""


def _compute_quality_4d(triples: List[Dict]) -> List[float]:
    """
    Compute 4D quality vector từ danh sách triples đã filter.
    Returns [0.0, 0.0, 0.0, 0.0] nếu triples rỗng.
    """
    if not triples:
        return [0.0, 0.0, 0.0, 0.0]

    n   = len(triples)
    confs   = [float(t.get("confidence",          0.65)) for t in triples]
    rels    = [float(t.get("relevance_to_ticker",  0.50)) for t in triples]
    impacts = [abs(float(t.get("price_impact_score", 0.0))) for t in triples]

    return [
        float(np.log1p(n)),         # log(1 + n_triples)
        float(np.mean(confs)),      # avg_confidence
        float(np.mean(rels)),       # avg_relevance_to_ticker
        float(np.mean(impacts)),    # avg_abs_impact
    ]


def _save_quality_output(
    quality_dict:  Dict[str, Dict[str, List[float]]],
    quality_path:  str,
    ticker_filter: Optional[str],
) -> None:
    """
    Merge-on-write + save quality JSON.
    Cùng logic với _save_output() để đảm bảo per-ticker run không xóa ticker khác.

    Format: {"YYYY-MM-DD": {"TICKER": [4 floats]}}
    """
    if ticker_filter:
        existing: Dict[str, Dict[str, List[float]]] = {}
        if os.path.exists(quality_path):
            try:
                with open(quality_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
            except Exception as e:
                print(f"  [quality] Could not load existing ({e}) — starting fresh")
        if existing:
            merged: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
            for ds, td in existing.items():
                merged[ds].update(td)
            for ds, td in quality_dict.items():
                merged[ds].update(td)
            final = merged
        else:
            final = quality_dict
    else:
        final = quality_dict

    os.makedirs(os.path.dirname(os.path.abspath(quality_path)), exist_ok=True)
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(dict(final), f, ensure_ascii=False)

    total_pairs = sum(len(v) for v in final.values())
    print(f"  Quality saved : {quality_path}")
    print(f"  Quality pairs : {total_pairs} (dates={len(final)}, dim={QUALITY_DIM})")


# ─────────────────────────────────────────────────────────────────────────────
# FINBERT EMBED FUNCTION  (V8.0 — PRIMARY)
# ─────────────────────────────────────────────────────────────────────────────

def run_embed_news_finbert(
    news_df:           pd.DataFrame,
    cache_dir:         str,
    output_path:       str,
    finbert_model:     str          = "ProsusAI/finbert",
    finbert_cache_dir: Optional[str] = None,
    min_relevance:     float        = None,
    min_confidence:    float        = None,
    ticker_filter:     Optional[str] = None,
    date_prefix:       Optional[str] = None,
    device:            Optional[str] = None,
    quality_output_path: Optional[str] = None,   # V8.1: path for 4D quality JSON
) -> str:
    """
    FinBERT per-triple embedding pipeline.

    V8.1 adds quality output as a side-effect of the same triple loop:
      Embedding output  (768D): output_path            (e.g. news_embeddings_finbert.json)
      Quality output     (4D) : quality_output_path    (e.g. news_embeddings_finbert_quality.json)

    Quality vector per (date, ticker): [log(1+n_triples), avg_conf, avg_rel, avg_abs_impact]
    If quality_output_path is None → auto-derived from output_path (recommended).

    1. Normalise DataFrame
    2. Load KG triple cache (meta + SHA1 fallback)
    3. For each (ticker, date): embed triples → 768D weighted mean + 4D quality stats
    4. Merge-on-write + save both outputs

    Output format: {"YYYY-MM-DD": {"TICKER": [768D vector]}}
    Zero vector = no triples (news_mask handles this in model).
    """
    if min_relevance  is None: min_relevance  = GlobalConfig.KG_MIN_RELEVANCE
    if min_confidence is None: min_confidence = GlobalConfig.KG_MIN_CONFIDENCE

    # V8.1: Auto-derive quality path nếu không cung cấp
    if quality_output_path is None:
        quality_output_path = output_path.replace(".json", "_quality.json")

    # Lazy-load FinBERT (slow on first call, ~10s)
    embedder = FinBERTPerTripleEmbedder(
        model_name=finbert_model,
        device=device,
        cache_dir=finbert_cache_dir,
    )
    EMPTY_VEC     = [0.0] * FinBERTPerTripleEmbedder.EMBED_DIM
    EMPTY_QUALITY = [0.0] * QUALITY_DIM

    df = _normalize_news_df(news_df, ticker_filter, date_prefix)
    tickers = sorted(df["equity"].unique())

    print(f"\nEmbed news V8.1 (FinBERT per-triple, {FinBERTPerTripleEmbedder.EMBED_DIM}D + {QUALITY_DIM}D quality)")
    print(f"  Tickers      : {len(tickers)}")
    print(f"  Cache dir    : {cache_dir}")
    print(f"  Output (emb) : {output_path}")
    print(f"  Output (qty) : {quality_output_path}")
    if ticker_filter:
        print(f"  Filter       : {ticker_filter.upper()} — will MERGE into existing output")
    else:
        print(f"  Filter       : none — full REBUILD")
    print()

    cache_store = CacheStore(cache_dir)
    cache_store.load()

    new_output:  Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    quality_out: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    stats = {"n_with_triples": 0, "n_empty": 0, "total_triples": 0}

    for ticker in tickers:
        df_t  = df[df["equity"] == ticker].copy()
        dates = sorted(df_t["date"].unique())

        for d in dates:
            date_str = str(d)
            day_df   = df_t[df_t["date"] == d]

            # Get triples: meta first, SHA1 fallback, then dedup
            meta_triples = cache_store.get_triples_meta(
                date_str, ticker, min_relevance, min_confidence)
            sha1_triples = cache_store.get_triples_sha1(
                day_df, ticker, min_relevance, min_confidence)
            all_triples  = _dedup_triples(meta_triples + sha1_triples)

            if all_triples:
                stats["n_with_triples"] += 1
                stats["total_triples"]  += len(all_triples)
                daily_emb = embedder.embed_triples(all_triples, ticker)
                new_output[date_str][ticker]  = daily_emb
                # V8.1: quality stats từ cùng all_triples (zero extra cost)
                quality_out[date_str][ticker] = _compute_quality_4d(all_triples)
            else:
                stats["n_empty"] += 1
                new_output[date_str][ticker]  = EMPTY_VEC
                quality_out[date_str][ticker] = EMPTY_QUALITY

    avg_t = stats["total_triples"] / max(stats["n_with_triples"], 1)
    print(f"Embedding stats:")
    print(f"  Days with triples : {stats['n_with_triples']}")
    print(f"  Days empty (zeros): {stats['n_empty']}")
    print(f"  Avg triples/day   : {avg_t:.1f}")

    # V8.1: Save quality output TRƯỚC embedding output để nếu crash thì biết quality đã xong
    _save_quality_output(quality_out, quality_output_path, ticker_filter)
    return _save_output(new_output, output_path, ticker_filter)


# ─────────────────────────────────────────────────────────────────────────────
# VOYAGE EMBED FUNCTION  (V7.1 — LEGACY)
# ─────────────────────────────────────────────────────────────────────────────

def run_embed_news(
    news_df:        pd.DataFrame,
    cache_dir:      str,
    output_path:    str,
    voyage_cache:   str,
    min_relevance:  float = None,
    min_confidence: float = None,
    ticker_filter:  Optional[str] = None,
    date_prefix:    Optional[str] = None,
    force_sha1:     bool = False,
    force_meta:     bool = False,
) -> str:
    """
    Voyage embedding pipeline (legacy, 1024D).
    Kept for backward compatibility. Use run_embed_news_finbert() for new work.
    """
    if min_relevance  is None: min_relevance  = GlobalConfig.KG_MIN_RELEVANCE
    if min_confidence is None: min_confidence = GlobalConfig.KG_MIN_CONFIDENCE

    voyage = VoyageEmbedder(cache_dir=voyage_cache)

    df = _normalize_news_df(news_df, ticker_filter, date_prefix)
    tickers = sorted(df["equity"].unique())

    mode_str = ("sha1-only" if force_sha1 else
                "meta-only" if force_meta else
                "unified (meta + sha1 fallback)")
    print(f"\nEmbed news V7.1 (Voyage 1024D, legacy): {len(tickers)} tickers — mode: {mode_str}")
    print(f"Cache dir : {cache_dir}")
    print(f"Output    : {output_path}")
    if ticker_filter:
        print(f"Ticker filter: {ticker_filter.upper()} — will MERGE into existing output")
    else:
        print(f"No ticker filter — will REBUILD entire output")
    print()

    cache_store = CacheStore(cache_dir)
    cache_store.load()

    ticker_date_text: Dict[Tuple[str, str], str] = {}
    stats = {"meta_only": 0, "sha1_only": 0, "both": 0, "empty": 0}

    for ticker in tickers:
        df_t  = df[df["equity"] == ticker].copy()
        dates = sorted(df_t["date"].unique())

        for d in dates:
            date_str = str(d)
            day_df   = df_t[df_t["date"] == d]

            meta_triples: List[Dict] = []
            if not force_sha1:
                meta_triples = cache_store.get_triples_meta(
                    date_str, ticker, min_relevance, min_confidence)

            sha1_triples: List[Dict] = []
            if not force_meta:
                sha1_triples = cache_store.get_triples_sha1(
                    day_df, ticker, min_relevance, min_confidence)

            all_triples = _dedup_triples(meta_triples + sha1_triples)

            has_meta = bool(meta_triples)
            has_sha1 = bool(sha1_triples)
            if has_meta and has_sha1:   stats["both"] += 1
            elif has_meta:              stats["meta_only"] += 1
            elif has_sha1:              stats["sha1_only"] += 1
            else:                       stats["empty"] += 1

            text = triples_to_text(all_triples, ticker)
            ticker_date_text[(ticker, date_str)] = text

    print(f"Cache coverage per (ticker, date) pair:")
    print(f"  Meta-based only (new format):  {stats['meta_only']}")
    print(f"  SHA1-based only (old format):  {stats['sha1_only']}")
    print(f"  Both sources (merged):         {stats['both']}")
    print(f"  No cache found (zeros):        {stats['empty']}")
    print(f"  Total pairs: {len(ticker_date_text)}")

    unique_texts = list(set(ticker_date_text.values()))
    non_empty    = [t for t in unique_texts if t]
    empty_vec    = [0.0] * 1024

    if non_empty:
        print(f"\nEmbedding {len(non_empty)} unique texts via Voyage...")
        embeddings_list = voyage.embed_texts(non_empty)
        text_to_emb     = {t: e for t, e in zip(non_empty, embeddings_list)}
    else:
        text_to_emb = {}
        print("\nNo text to embed (all pairs empty).")

    new_output: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for (ticker, date_str), text in ticker_date_text.items():
        new_output[date_str][ticker] = text_to_emb.get(text, empty_vec)

    return _save_output(new_output, output_path, ticker_filter)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage A.5 — News Embedding\n"
            "  --finbert  : FinBERT per-triple (768D, PRIMARY, recommended)\n"
            "  (no flag)  : Voyage legacy (1024D, requires VOYAGE_API_KEY)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Shared args ──────────────────────────────────────────────────────────
    parser.add_argument("--news",      default=None,
                        help="Path to news parquet (default: interim/concatenated_news_filtered.parquet)")
    parser.add_argument("--cache-dir", default=None,
                        help="Article triple cache dir (default: from GlobalConfig)")
    parser.add_argument("--output",    default=None,
                        help="Output JSON path (default: interim/kg_embeddings/news_embeddings_finbert.json)")
    parser.add_argument("--quality-output", default=None,
                        help="[FinBERT] Quality stats JSON path. "
                             "Default: same dir as --output, with '_quality' suffix. "
                             "E.g. news_embeddings_finbert_quality.json (4D per ticker/day)")
    parser.add_argument("--ticker",    default=None,
                        help="Filter to 1 ticker, e.g. TSLA (merges into existing output)")
    parser.add_argument("--date",      default=None,
                        help="Date prefix filter, e.g. '2023-06'")
    parser.add_argument("--min-relevance",  type=float, default=None,
                        help="Min relevance_to_ticker threshold (default: GlobalConfig)")
    parser.add_argument("--min-confidence", type=float, default=None,
                        help="Min confidence threshold (default: GlobalConfig)")

    # ── FinBERT-specific args ────────────────────────────────────────────────
    parser.add_argument("--finbert", action="store_true",
                        help="Use FinBERT per-triple embedder (768D, PRIMARY). "
                             "Requires: pip install transformers accelerate")
    parser.add_argument("--finbert-model", default=None,
                        help="HuggingFace model name (default: ProsusAI/finbert)")
    parser.add_argument("--finbert-device", default=None,
                        help="Device for FinBERT inference, e.g. cuda or cpu")

    # ── Voyage-specific args ─────────────────────────────────────────────────
    parser.add_argument("--force-sha1",  action="store_true",
                        help="[Voyage] Only use SHA1 lookup (debug)")
    parser.add_argument("--force-meta",  action="store_true",
                        help="[Voyage] Only use _meta scan (debug)")

    # ── Diagnostic modes ────────────────────────────────────────────────────
    parser.add_argument("--check-cache",  action="store_true",
                        help="Print cache stats and exit (no embedding)")
    parser.add_argument("--check-output", action="store_true",
                        help="Print current news_embeddings.json stats and exit")

    args = parser.parse_args()

    # ── Resolve paths ────────────────────────────────────────────────────────
    cache_dir = args.cache_dir or GlobalConfig.kg_cache_dir()
    if args.output:
        output_path = args.output
    elif args.finbert:
        output_path = GlobalConfig.finbert_emb_path()  # news_embeddings_finbert.json (768D)
    else:
        output_path = GlobalConfig.voyage_emb_path()   # news_embeddings_voyage.json  (1024D)

    # ── Diagnostic modes ────────────────────────────────────────────────────
    if args.check_cache:
        check_cache_stats(cache_dir)
        return

    if args.check_output:
        check_output_stats(output_path)
        return

    # ── Validate common inputs ───────────────────────────────────────────────
    news_path = args.news or os.path.join(
        GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet"
    )
    if not os.path.exists(news_path):
        print(f"News file not found: {news_path}")
        sys.exit(1)

    if not os.path.exists(cache_dir):
        print(f"Cache dir not found: {cache_dir}")
        print("Run Stage A first: python extract_corpus.py")
        sys.exit(1)

    df = pd.read_parquet(news_path)
    print(f"Loaded {len(df):,} rows from {news_path}")

    # ── Route to embedder ────────────────────────────────────────────────────
    if args.finbert:
        # FinBERT path (PRIMARY)
        finbert_model = (
            args.finbert_model
            or getattr(GlobalConfig, "FINBERT_MODEL", "ProsusAI/finbert")
        )
        finbert_cache = getattr(GlobalConfig, "finbert_cache_dir", lambda: None)()

        run_embed_news_finbert(
            news_df=df,
            cache_dir=cache_dir,
            output_path=output_path,
            finbert_model=finbert_model,
            finbert_cache_dir=finbert_cache,
            min_relevance=args.min_relevance,
            min_confidence=args.min_confidence,
            ticker_filter=args.ticker,
            date_prefix=args.date,
            device=args.finbert_device,
            quality_output_path=args.quality_output,  # V8.1: None → auto-derive
        )

    else:
        # Voyage path (LEGACY)
        if args.force_sha1 and args.force_meta:
            print("Cannot use --force-sha1 and --force-meta together.")
            sys.exit(1)

        voyage_cache = GlobalConfig.kg_voyage_cache_dir()
        run_embed_news(
            news_df=df,
            cache_dir=cache_dir,
            output_path=output_path,
            voyage_cache=voyage_cache,
            min_relevance=args.min_relevance,
            min_confidence=args.min_confidence,
            ticker_filter=args.ticker,
            date_prefix=args.date,
            force_sha1=args.force_sha1,
            force_meta=args.force_meta,
        )


if __name__ == "__main__":
    main()