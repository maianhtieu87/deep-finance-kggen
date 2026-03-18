# embed_news.py
"""
Stage A.5 — News Embedding
==========================
Đọc SHA-1 cache (triples JSON từ Stage A) → tổng hợp reasoning text
của các triples theo (ticker, date) → Voyage embed → lưu news_embeddings.json.

Đây là bước nối giữa Stage A (LLM extraction) và main_test.py (dataset builder).
Không cần GATv2, không cần graph building.

Luồng:
  extract_corpus.py  →  embed_news.py  →  main_test.py  →  main.py

Output file: data/interim/kg_embeddings/news_embeddings.json
Format:
  {
    "YYYY-MM-DD": {
      "TSLA": [0.12, -0.34, ...],   # 1024D Voyage embedding
      "AAPL": [...]
    },
    ...
  }

Cách tổng hợp text từ triples:
  Với mỗi (ticker, date), lấy tất cả triples từ cache của ngày đó,
  ghép thành 1 chuỗi text bao gồm:
    - subject + relation + object (signal structure)
    - reasoning (price mechanism)
    - price_impact_score (signed magnitude)
  Sau đó embed toàn bộ chuỗi này → 1 vector 1024D.

Usage:
    python embed_news.py
    python embed_news.py --ticker TSLA
    python embed_news.py --date 2022-06
    python embed_news.py --window 3    # dùng rolling window 3 ngày khi aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import GlobalConfig


# ─────────────────────────────────────────────────────────────────────────────
# VOYAGE EMBEDDER (standalone, no dependency on news_processor)
# ─────────────────────────────────────────────────────────────────────────────

class VoyageEmbedder:
    """Simple rate-limited Voyage embedder with disk cache."""

    def __init__(self, cache_dir: str):
        self.api_key = os.getenv("VOYAGE_API_KEY", GlobalConfig.VOYAGE_API_KEY)
        if not self.api_key or self.api_key.strip() in ("", "---"):
            raise RuntimeError("VOYAGE_API_KEY not set.")

        self.model        = getattr(GlobalConfig, "EMBED_MODEL", "voyage-3-large")
        self.max_texts    = getattr(GlobalConfig, "MAX_TEXTS_PER_REQ", 40)
        self.max_retries  = getattr(GlobalConfig, "MAX_RETRIES", 6)
        self.backoff_base = getattr(GlobalConfig, "BACKOFF_BASE", 30)

        payment_added = bool(getattr(GlobalConfig, "PAYMENT_ADDED", True))
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
        if os.path.exists(p):
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
        """Embed list of texts. Returns list of 1024D vectors."""
        out        = [None] * len(texts)
        miss_i     = []
        miss_t     = []

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

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i: i + n]

        pos = 0
        for batch_texts in chunks(miss_t, self.max_texts):
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
                    embs = r.json().get("data", [])
                    for bi, item in enumerate(embs):
                        emb = item.get("embedding", [])
                        idx = batch_idx[bi]
                        out[idx] = emb
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
# CACHE READING
# ─────────────────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _parse_tickers(val: Any) -> List[str]:
    if isinstance(val, list):
        return [t.strip().upper() for t in val if isinstance(t, str) and t.strip()]
    if isinstance(val, str):
        return [t.strip().upper() for t in val.split(",") if t.strip()]
    return []


def _load_article_cache(cache_dir: str, sha1: str) -> Optional[List[Dict]]:
    p = os.path.join(cache_dir, f"{sha1}.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f).get("triples", [])
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# TRIPLE → TEXT CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def triples_to_text(triples: List[Dict], ticker: str) -> str:
    """
    Chuyển list of triples thành 1 chuỗi text để embed.

    Format mỗi triple:
      "{subject} {RELATION} {object}. {reasoning} (impact={price_impact_score:+.2f})"

    Triples được sort theo |price_impact_score| descending để signal mạnh nhất
    xuất hiện đầu tiên (quan trọng cho truncation nếu text quá dài).

    Nếu không có triple nào → trả về chuỗi rỗng "" → sẽ embed thành zero vector.
    """
    if not triples:
        return ""

    # Sort by abs(impact) descending
    sorted_triples = sorted(
        triples,
        key=lambda t: abs(float(t.get("price_impact_score", 0))),
        reverse=True,
    )

    parts = [f"TARGET: {ticker}"]
    for t in sorted_triples:
        subj    = t.get("subject", {}).get("name", "")
        rel     = t.get("relation", "")
        obj     = t.get("object",  {}).get("name", "")
        reason  = t.get("reasoning", "")
        impact  = float(t.get("price_impact_score", 0))
        conf    = float(t.get("confidence", 0))
        rel_s   = float(t.get("relevance_to_ticker", 0))

        if not subj or not obj:
            continue

        line = f"{subj} {rel} {obj}"
        if reason:
            line += f". {reason}"
        line += f" (impact={impact:+.2f}, conf={conf:.2f}, rel={rel_s:.2f})"
        parts.append(line)

    return " | ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# RESCORE (simplified version — same logic as extractor_batch.py)
# ─────────────────────────────────────────────────────────────────────────────

TICKER_NAME_MAP: Dict[str, List[str]] = {
    "TSLA":  ["Tesla", "TSLA"],
    "AAPL":  ["Apple", "AAPL"],
    "AMZN":  ["Amazon", "AMZN"],
    "MSFT":  ["Microsoft", "MSFT"],
    "GOOGL": ["Google", "Alphabet", "GOOGL"],
    "GOOG":  ["Google", "Alphabet", "GOOG"],
    "META":  ["Meta", "Facebook", "META"],
    "BA":    ["Boeing", "BA"],
    "JPM":   ["JPMorgan", "JP Morgan", "JPM"],
    "WMT":   ["Walmart", "WMT"],
    "NVDA":  ["Nvidia", "NVDA"],
    "NFLX":  ["Netflix", "NFLX"],
    "INTC":  ["Intel", "INTC"],
    "AMD":   ["AMD", "Advanced Micro"],
    "RIVN":  ["Rivian", "RIVN"],
    "LCID":  ["Lucid", "LCID"],
}

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
            score += t_up.count(n_up) * TITLE_WEIGHT
            score += c_up.count(n_up)
        counts[t] = score
    best = max(counts.values())
    if best == 0:
        return tickers[0]
    for t in tickers:
        if counts[t] == best:
            return t


def _ticker_mentioned_in_text(ticker: str, text_upper: str) -> bool:
    for name in TICKER_NAME_MAP.get(ticker, [ticker]):
        name_upper = name.upper()
        if len(name) <= 3:
            if re.search(r'\b' + re.escape(name_upper) + r'\b', text_upper):
                return True
        else:
            if name_upper in text_upper:
                return True
    return False


def rescore_for_ticker(
    triples: List[Dict],
    primary_ticker: str,
    target_ticker: str,
    article_text: str = "",
    all_tickers: Optional[List[str]] = None,
    min_relevance: float = 0.30,
    min_confidence: float = 0.35,
) -> List[Dict]:
    """Simplified rescore — same 3-tier logic as extractor_batch.py."""
    if primary_ticker.upper() == target_ticker.upper():
        return [t for t in triples
                if float(t.get("confidence", 0)) >= min_confidence
                and float(t.get("relevance_to_ticker", 0)) >= min_relevance]

    text_upper     = (article_text or "").upper()
    target_in_text = _ticker_mentioned_in_text(target_ticker, text_upper)
    others_in_text = any(
        _ticker_mentioned_in_text(t, text_upper)
        for t in (all_tickers or [])
        if t.upper() != target_ticker.upper()
    )

    if target_in_text:
        strict_mode   = False
        effective_min = min_relevance
    elif others_in_text:
        strict_mode   = True
        effective_min = 0.75
    else:
        strict_mode   = False
        effective_min = max(0.50, min_relevance)

    target_lower = target_ticker.lower()
    out = []
    for t in triples:
        if float(t.get("confidence", 0)) < min_confidence:
            continue
        t2       = dict(t)
        subj     = t.get("subject", {}).get("name", "").lower()
        obj      = t.get("object",  {}).get("name", "").lower()
        mentions = target_lower in subj or target_lower in obj

        # Also check company name variants
        if not mentions:
            for name in TICKER_NAME_MAP.get(target_ticker.upper(), []):
                if name.lower() in subj or name.lower() in obj:
                    mentions = True
                    break

        orig_rel = float(t.get("relevance_to_ticker", 0.0))

        if strict_mode:
            if not mentions:
                continue
            t2["relevance_to_ticker"] = min(1.0, orig_rel * 1.1)
        else:
            t2["relevance_to_ticker"] = min(1.0, orig_rel * 1.1) if mentions else orig_rel * 0.4

        if t2["relevance_to_ticker"] >= effective_min:
            out.append(t2)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ROLLING WINDOW AGGREGATE
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_triples_rolling(
    per_day_triples: List[List[Dict]],
) -> List[Dict]:
    """Gộp triples trong rolling window, dedup theo (subj, rel, obj)."""
    seen, result = set(), []
    for day_triples in per_day_triples:
        for t in day_triples:
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
# BUILD COMBINED TEXT FROM CACHE  (per ticker, per date)
# ─────────────────────────────────────────────────────────────────────────────

def build_combined_text_from_cache(
    day_df: pd.DataFrame,
    ticker: str,
    date_str: str,
    cache_dir: str,
    min_relevance: float,
    min_confidence: float,
) -> Tuple[List[Dict], str]:
    """
    Đọc cache articles của ngày này cho ticker này,
    áp dụng rescore, trả về (triples, text_for_embedding).
    """
    sha1_to_meta: Dict[str, Dict] = {}
    sha1_to_raw:  Dict[str, Optional[List]] = {}

    for _, row in day_df.iterrows():
        title   = _norm(str(row.get("title",   "") or ""))
        content = _norm(str(row.get("content", "") or ""))

        # Build combined text (titles first, content after)
        parts = []
        if title:
            parts.append(f"HEADLINES:\n- {title}")
        if content:
            parts.append(f"ARTICLES:\n{content}")
        full_text = "\n\n".join(parts)
        if not full_text:
            continue

        h = _sha1(full_text)
        if h in sha1_to_raw:
            # Merge tickers
            row_tickers = _parse_tickers(
                str(row.get("symbols", "") or row.get("equity", ticker) or ticker)
            )
            existing = set(sha1_to_meta[h]["all_tickers"])
            sha1_to_meta[h]["all_tickers"] = list(existing | set(row_tickers))
            continue

        primary = str(row.get("primary_ticker") or detect_primary_ticker(
            title, content,
            _parse_tickers(str(row.get("symbols", "") or row.get("equity", ticker) or ticker))
        ))
        all_t = _parse_tickers(
            str(row.get("symbols", "") or row.get("equity", ticker) or ticker)
        ) or [primary]

        sha1_to_meta[h] = {
            "full_text": full_text,
            "primary_ticker": primary,
            "all_tickers": all_t,
        }

        cached = _load_article_cache(cache_dir, h)
        sha1_to_raw[h] = cached  # None if not cached

    # Fan-out + rescore
    all_triples: List[Dict] = []
    for h, raw in sha1_to_raw.items():
        if not raw:
            continue
        meta     = sha1_to_meta[h]
        primary  = meta["primary_ticker"]
        all_t    = meta["all_tickers"]
        full_txt = meta["full_text"]

        rescored = rescore_for_ticker(
            raw, primary, ticker,
            article_text=full_txt,
            all_tickers=all_t,
            min_relevance=min_relevance,
            min_confidence=min_confidence,
        )
        all_triples.extend(rescored)

    # Dedup
    seen, deduped = set(), []
    for t in all_triples:
        key = (
            t.get("subject", {}).get("name", ""),
            t.get("relation", ""),
            t.get("object",  {}).get("name", ""),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(t)

    text = triples_to_text(deduped, ticker)
    return deduped, text


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Stage A.5
# ─────────────────────────────────────────────────────────────────────────────

def run_embed_news(
    news_df:        pd.DataFrame,
    cache_dir:      str,
    output_path:    str,
    voyage_cache:   str,
    window_days:    int   = 3,
    min_relevance:  float = 0.30,
    min_confidence: float = 0.35,
    ticker_filter:  Optional[str] = None,
    date_prefix:    Optional[str] = None,
) -> str:
    """
    Đọc cache triples → aggregate theo rolling window → Voyage embed.

    Args:
        news_df       : news DataFrame (để biết ngày nào có ticker nào)
        cache_dir     : thư mục SHA-1 cache từ Stage A
        output_path   : đường dẫn file output JSON
        voyage_cache  : thư mục cache embedding Voyage
        window_days   : rolling window để aggregate triples (default 3)
        ticker_filter : chỉ process 1 ticker
        date_prefix   : chỉ process ngày bắt đầu với prefix này

    Returns:
        output_path (path to news_embeddings.json)
    """
    voyage = VoyageEmbedder(cache_dir=voyage_cache)

    # Normalize DataFrame
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
    if "content" not in df.columns:
        df["content"] = ""
    if "title" not in df.columns:
        df["title"] = ""

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])

    ticker_col = next((c for c in ("symbols", "equity") if c in df.columns), None)
    if ticker_col is None:
        raise ValueError("No ticker column found.")

    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)
    df = df[df["_all_tickers"].map(len) > 0]

    # Detect primary ticker
    df["primary_ticker"] = df.apply(
        lambda row: detect_primary_ticker(
            str(row.get("title",   "") or ""),
            str(row.get("content", "") or ""),
            row["_all_tickers"],
        ),
        axis=1,
    )

    # Explode per ticker
    df = df.explode("_all_tickers").rename(columns={"_all_tickers": "equity"})
    df = df[df["equity"].notna() & (df["equity"] != "")].reset_index(drop=True)
    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)  # restore after explode

    # Apply filters
    if ticker_filter:
        df = df[df["equity"] == ticker_filter.upper()]
    if date_prefix:
        df = df[df["date"].astype(str).str.startswith(date_prefix)]

    tickers = sorted(df["equity"].unique())
    print(f"\nEmbed news: {len(tickers)} tickers, window={window_days}")
    print(f"Cache dir : {cache_dir}")
    print(f"Output    : {output_path}\n")

    # Collect all texts to embed in one pass (batch efficiency)
    # Key: (ticker, date_str) → text
    ticker_date_text: Dict[Tuple[str, str], str] = {}
    ticker_date_triples: Dict[Tuple[str, str], List[Dict]] = {}

    miss_count = 0
    for ticker in tickers:
        df_t  = df[df["equity"] == ticker].copy()
        dates = sorted(df_t["date"].unique())

        # Rolling window of per-day triples
        day_triple_list: List[Tuple[str, List[Dict]]] = []  # (date_str, triples)

        for d in dates:
            date_str = str(d)
            day_df   = df_t[df_t["date"] == d]

            triples, _ = build_combined_text_from_cache(
                day_df=day_df,
                ticker=ticker,
                date_str=date_str,
                cache_dir=cache_dir,
                min_relevance=min_relevance,
                min_confidence=min_confidence,
            )

            if not triples:
                miss_count += 1

            day_triple_list.append((date_str, triples))

        # Apply rolling window
        for i, (date_str, _) in enumerate(day_triple_list):
            start = max(0, i - window_days + 1)
            window = [t for _, t in day_triple_list[start: i + 1]]
            aggregated = aggregate_triples_rolling(window)
            text = triples_to_text(aggregated, ticker)
            ticker_date_text[(ticker, date_str)]    = text
            ticker_date_triples[(ticker, date_str)] = aggregated

    print(f"Days with no cache entries: {miss_count}")
    print(f"Total (ticker, date) pairs: {len(ticker_date_text)}")

    # Batch embed — collect unique texts first to avoid redundant API calls
    unique_texts = list(set(ticker_date_text.values()))
    non_empty    = [t for t in unique_texts if t]
    empty_vec    = [0.0] * 1024

    if non_empty:
        print(f"Embedding {len(non_empty)} unique texts via Voyage...")
        embeddings_list = voyage.embed_texts(non_empty)
        text_to_emb     = {t: e for t, e in zip(non_empty, embeddings_list)}
    else:
        text_to_emb = {}

    # Build output structure: {"YYYY-MM-DD": {"TSLA": [...]}}
    output: Dict[str, Dict[str, List[float]]] = defaultdict(dict)
    for (ticker, date_str), text in ticker_date_text.items():
        emb = text_to_emb.get(text, empty_vec)
        output[date_str][ticker] = emb

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dict(output), f, ensure_ascii=False)

    print(f"\nSaved news_embeddings.json: {len(output)} dates")
    print(f"Output path: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Stage A.5 — Embed news triples via Voyage")
    parser.add_argument("--news",       default=None,  help="Path to news parquet")
    parser.add_argument("--cache-dir",  default=None,  help="SHA-1 cache dir (default: GlobalConfig)")
    parser.add_argument("--output",     default=None,  help="Output JSON path (default: GlobalConfig)")
    parser.add_argument("--window",     type=int, default=3,
                        help="Rolling window days for triple aggregation (default 3)")
    parser.add_argument("--ticker",     default=None)
    parser.add_argument("--date",       default=None,  help="Date prefix filter, e.g. '2022-06'")
    parser.add_argument("--min-relevance",  type=float, default=GlobalConfig.KG_MIN_RELEVANCE)
    parser.add_argument("--min-confidence", type=float, default=GlobalConfig.KG_MIN_CONFIDENCE)
    args = parser.parse_args()

    news_path = args.news or os.path.join(
        GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet"
    )
    if not os.path.exists(news_path):
        print(f"News file not found: {news_path}")
        sys.exit(1)

    cache_dir = args.cache_dir or GlobalConfig.kg_cache_dir()
    if not os.path.exists(cache_dir):
        print(f"Cache dir not found: {cache_dir}")
        print("Run Stage A first: python extract_corpus.py")
        sys.exit(1)

    output_path = args.output or os.path.join(
        GlobalConfig.INTERIM_PATH, "kg_embeddings", "news_embeddings.json"
    )
    voyage_cache = GlobalConfig.kg_voyage_cache_dir()

    df = pd.read_parquet(news_path)
    print(f"Loaded {len(df):,} rows from {news_path}")

    run_embed_news(
        news_df=df,
        cache_dir=cache_dir,
        output_path=output_path,
        voyage_cache=voyage_cache,
        window_days=args.window,
        min_relevance=args.min_relevance,
        min_confidence=args.min_confidence,
        ticker_filter=args.ticker,
        date_prefix=args.date,
    )


if __name__ == "__main__":
    main()