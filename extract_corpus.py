# extract_corpus.py — V5.2
"""
Stage A: LLM Extraction.

V5.2 changes vs V4.1:
  - --batch mode REFACTORED: gom tất cả uncached articles → submit ít batch jobs lớn
    → 50% cost saving, interrupt-safe (per-article cache)
  - 2-phase batch: Phase 1 (collect uncached) → Phase 2 (submit + cache)
  - Resume logic: save batch job state to disk, resume polling on restart
  - Interactive mode (without --batch) unchanged — uses AsyncConcurrentExtractor

Modes:
  python extract_corpus.py                  # Interactive (full price, fast)
  python extract_corpus.py --batch          # Batch API (50% off, slower)
  python extract_corpus.py --batch --ticker WMT  # Batch, 1 ticker only
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, sys, time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from google import genai
from configs.config import GlobalConfig
from data_pipeline.kg.extractor_batch import (
    AsyncConcurrentExtractor, GeminiBatchAPIExtractor,
    detect_primary_ticker, build_combined_text, get_article_pieces,
    dedup_triples, apply_quality_filters, smart_dedup_triples, limit_signals_per_source,
    filter_triples_for_ticker, _sha1, _norm, _filter_and_clamp, _parse_tickers,
    tag_triples_source, build_user_prompt,
    FINDKG_LITE_SYSTEM_PROMPT, _GEN_CONFIG_DICT, MODEL_ID,
)

# ─────────────────────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(cache_dir, sha1): return os.path.join(cache_dir, f"{sha1}.json")

def _load_cache(cache_dir, sha1):
    p = _cache_path(cache_dir, sha1)
    if not os.path.exists(p): return None
    try:
        with open(p, "r", encoding="utf-8") as f: return json.load(f).get("triples", [])
    except Exception: return None

def _save_cache(cache_dir, sha1, triples, meta=None):
    os.makedirs(cache_dir, exist_ok=True)
    payload = {"triples": triples, "_v": "v5.2"}
    if meta: payload["_meta"] = meta
    with open(_cache_path(cache_dir, sha1), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

# ─────────────────────────────────────────────────────────────────────────────
# NEWS NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def normalize_news_df(df):
    df = df.copy()
    col_map = {}
    if "headline" in df.columns and "title" not in df.columns: col_map["headline"] = "title"
    if "ticker" in df.columns and "equity" not in df.columns: col_map["ticker"] = "equity"
    if col_map: df = df.rename(columns=col_map)
    if "content" not in df.columns:
        for alt in ("body", "text"):
            if alt in df.columns: df = df.rename(columns={alt: "content"}); break
    if "content" not in df.columns: df["content"] = ""
    if "title"   not in df.columns: df["title"] = ""
    # Auto-detect date column
    if "date" not in df.columns:
        DATE_CANDS = ["created_at", "createdAt", "published_at", "publishedAt",
                      "publish_date", "pub_date", "Date", "DATE", "timestamp", "time", "news_date"]
        date_col = next((c for c in DATE_CANDS if c in df.columns), None)
        if date_col is None:
            date_col = next((c for c in df.columns if any(k in c.lower() for k in ("date", "time", "publish", "creat"))), None)
        if date_col is None:
            raise ValueError(f"Missing date column. Has: {list(df.columns)}")
        df = df.rename(columns={date_col: "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    ticker_col = next((c for c in ("symbols", "equity") if c in df.columns), None)
    if ticker_col is None: raise ValueError("No ticker column found.")
    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)
    df = df[df["_all_tickers"].map(len) > 0]
    df["primary_ticker"] = df.apply(
        lambda row: detect_primary_ticker(str(row.get("title","") or ""), str(row.get("content","") or ""), row["_all_tickers"]), axis=1)
    df = df.explode("_all_tickers").rename(columns={"_all_tickers": "equity"})
    df = df[df["equity"].notna() & (df["equity"] != "")].reset_index(drop=True)
    df["_all_tickers"] = df[ticker_col].apply(_parse_tickers)
    return df

# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MODE (unchanged from V4.1)
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_cache_per_article(day_df, ticker, date_str, extractor, cache_dir, min_relevance, min_confidence):
    """
    Extract triples for (ticker, date) using interactive API.
    Per-article cache: each article saved immediately after extraction.
    """
    sha1_to_meta, sha1_to_raw, cache_hits = {}, {}, 0
    for _, row in day_df.iterrows():
        title   = _norm(str(row.get("title","")   or ""))
        content = _norm(str(row.get("content","") or ""))
        full_text = build_combined_text([title] if title else [], [content] if content else [])
        if not full_text: continue
        h = _sha1(full_text)
        if h in sha1_to_raw:
            existing = set(sha1_to_meta[h]["all_tickers"])
            sha1_to_meta[h]["all_tickers"] = list(existing | set(row.get("_all_tickers") or [ticker]))
            continue
        sha1_to_meta[h] = {"full_text": full_text, "primary_ticker": str(row.get("primary_ticker") or ticker),
                           "date": date_str, "all_tickers": list(row.get("_all_tickers") or [ticker])}
        cached = _load_cache(cache_dir, h)
        if cached is not None: sha1_to_raw[h] = cached; cache_hits += 1
        else: sha1_to_raw[h] = None

    uncached = [h for h, v in sha1_to_raw.items() if v is None]
    if uncached:
        print(f"  [{ticker} {date_str}] cache_hits={cache_hits} to_extract={len(uncached)}")
        piece_jobs = []
        for h in uncached:
            for piece in get_article_pieces(sha1_to_meta[h]["full_text"]):
                piece_jobs.append((h, {"text": piece, "ticker": sha1_to_meta[h]["primary_ticker"], "date": sha1_to_meta[h]["date"]}))

        n_pieces = len(piece_jobs)
        n_articles = len(uncached)
        if n_pieces == n_articles:
            print(f"  Extracting: {n_articles} articles (full-text, no chunking)")
        else:
            print(f"  Extracting: {n_articles} articles -> {n_pieces} pieces (chunking enabled)")

        results = extractor.extract_batch([j[1] for j in piece_jobs])
        sha1_pieces = defaultdict(list)
        for (sha1, _), triples in zip(piece_jobs, results): sha1_pieces[sha1].extend(triples or [])
        for h in uncached:
            merged  = _filter_and_clamp(sha1_pieces[h], min_relevance, min_confidence)
            deduped = apply_quality_filters(merged)
            deduped = tag_triples_source(deduped, h)
            sha1_to_raw[h] = deduped
            _save_cache(cache_dir, h, deduped, meta=sha1_to_meta[h])
    else:
        if sha1_to_meta: print(f"  [{ticker} {date_str}] all {cache_hits} article(s) from cache")

    all_triples = []
    for h, raw in sha1_to_raw.items():
        if not raw: continue
        meta = sha1_to_meta[h]
        filtered = filter_triples_for_ticker(raw, meta["primary_ticker"], ticker, min_relevance)
        filtered = [t for t in filtered if float(t.get("confidence",0)) >= min_confidence]
        all_triples.extend(filtered)

    all_triples = smart_dedup_triples(all_triples)
    all_triples = limit_signals_per_source(all_triples)
    return dedup_triples(all_triples)


def run_stage_a(news_df, cache_dir, max_concurrent=None, min_relevance=None,
                min_confidence=None, ticker_filter=None, date_prefix=None):
    """Interactive mode: AsyncConcurrentExtractor, per-article cache."""
    os.makedirs(cache_dir, exist_ok=True)
    _mr   = min_relevance  if min_relevance  is not None else GlobalConfig.KG_MIN_RELEVANCE
    _mc   = min_confidence if min_confidence is not None else GlobalConfig.KG_MIN_CONFIDENCE
    _conc = max_concurrent if max_concurrent is not None else GlobalConfig.KG_MAX_CONCURRENT
    df = normalize_news_df(news_df)
    if ticker_filter: df = df[df["equity"] == ticker_filter.upper()]
    if date_prefix:   df = df[df["date"].astype(str).str.startswith(date_prefix)]
    if len(df) == 0: print("No data after filter."); return {}
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: raise RuntimeError("GEMINI_API_KEY not set.")

    extractor = AsyncConcurrentExtractor(api_key=api_key, min_relevance=_mr, min_confidence=_mc, max_concurrent=_conc)
    print(f"Extractor: AsyncConcurrentExtractor (max_concurrent={_conc})")
    print(f"Thresholds: min_relevance={_mr}  min_confidence={_mc}")
    print(f"Mode: full-article  max_chars={GlobalConfig.KG_MAX_ARTICLE_CHARS}")

    tickers = sorted(df["equity"].unique())
    print(f"\nStage A: {len(tickers)} tickers x {df['date'].nunique()} dates\nCache: {cache_dir}\n")
    summary = {}
    for ticker in tickers:
        df_t = df[df["equity"] == ticker].copy(); summary[ticker] = {}
        for d in sorted(df_t["date"].unique()):
            date_str = str(d)
            triples  = extract_and_cache_per_article(df_t[df_t["date"]==d], ticker, date_str, extractor, cache_dir, _mr, _mc)
            summary[ticker][date_str] = len(triples)
    print("\nStage A complete.")
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# BATCH-OPTIMIZED MODE (50% cost saving via Gemini Batch API)
# ═════════════════════════════════════════════════════════════════════════════
#
# Luồng 2 pha:
#   Phase 1: Scan toàn bộ corpus → thu thập tất cả uncached articles (unique by SHA1)
#   Phase 2: Gom thành ít batch jobs lớn → submit → poll → save to cache per-article
#
# An toàn khi interrupt:
#   - Mỗi article được lưu vào cache NGAY SAU KHI batch job hoàn thành
#   - Batch job state lưu vào disk → resume polling nếu restart
#   - Trước mỗi chunk, kiểm tra lại cache → skip articles đã cached
#
# Sau khi hoàn thành: chạy `python embed_news.py` (đọc từ cache)
# ═════════════════════════════════════════════════════════════════════════════

def _collect_all_uncached(df: pd.DataFrame, cache_dir: str) -> Tuple[Dict, int]:
    """
    Phase 1: Scan toàn bộ corpus, thu thập articles chưa có cache.

    Deduplicate by SHA1: cùng 1 bài báo tag nhiều tickers chỉ extract 1 lần.
    Returns: (sha1_map, cached_count)
      sha1_map: {sha1: {"full_text": ..., "primary_ticker": ..., "date": ...}}
    """
    sha1_map = {}
    sha1_seen = set()
    cached_count = 0

    for _, row in df.iterrows():
        title   = _norm(str(row.get("title","")   or ""))
        content = _norm(str(row.get("content","") or ""))
        full_text = build_combined_text([title] if title else [], [content] if content else [])
        if not full_text:
            continue

        h = _sha1(full_text)

        if h in sha1_seen:
            continue
        sha1_seen.add(h)

        if _load_cache(cache_dir, h) is not None:
            cached_count += 1
            continue

        sha1_map[h] = {
            "full_text": full_text,
            "primary_ticker": str(row.get("primary_ticker") or ""),
            "date": str(row.get("date", "")),
        }

    return sha1_map, cached_count


def _batch_state_path(cache_dir: str, chunk_i: int) -> str:
    return os.path.join(cache_dir, f"_batch_state_chunk{chunk_i}.json")


def _save_batch_state(cache_dir: str, chunk_i: int, job_name: str, sha1_list: List[str]):
    """Save batch job state for resume after interrupt."""
    path = _batch_state_path(cache_dir, chunk_i)
    with open(path, "w") as f:
        json.dump({"job_name": job_name, "sha1s": sha1_list, "chunk_i": chunk_i}, f)


def _load_batch_state(cache_dir: str, chunk_i: int) -> Optional[Dict]:
    path = _batch_state_path(cache_dir, chunk_i)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _remove_batch_state(cache_dir: str, chunk_i: int):
    path = _batch_state_path(cache_dir, chunk_i)
    if os.path.exists(path):
        os.remove(path)


def _poll_batch_job(client, job_name: str, poll_interval: int = None, max_wait: int = None):
    """Poll batch job until terminal state. Returns batch_job object."""
    if poll_interval is None:
        poll_interval = getattr(GlobalConfig, 'KG_BATCH_POLL_INTERVAL', 30)
    if max_wait is None:
        max_wait = getattr(GlobalConfig, 'KG_BATCH_MAX_WAIT', 86400)

    terminal = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}
    elapsed = 0
    last_print = 0

    while elapsed < max_wait:
        time.sleep(poll_interval)
        elapsed += poll_interval
        batch_job = client.batches.get(name=job_name)

        # Print every 2 minutes
        if elapsed - last_print >= 120:
            print(f"    [{elapsed//60:>3}min] {batch_job.state.name}")
            last_print = elapsed

        if batch_job.state.name in terminal:
            print(f"    [{elapsed//60:>3}min] {batch_job.state.name} (final)")
            return batch_job

    print(f"    TIMEOUT after {max_wait//3600}h")
    return client.batches.get(name=job_name)


def _parse_batch_results(batch_job, min_relevance: float, min_confidence: float) -> Dict[str, List]:
    """
    Parse inlined_responses from completed batch job.
    Returns: {sha1: [triples]}
    Key is stored in resp.metadata["sha1"].
    """
    sha1_triples = defaultdict(list)
    dest = getattr(batch_job, "dest", None)
    if dest is None:
        return sha1_triples

    inlined = getattr(dest, "inlined_responses", None) or []
    parsed, failed = 0, 0

    for resp in inlined:
        try:
            meta = getattr(resp, "metadata", None) or {}
            key = meta.get("sha1", "")
            if not key:
                failed += 1
                continue
            response = resp.response
            text = response.candidates[0].content.parts[0].text
            raw = json.loads(text)
            triples = _filter_and_clamp(raw, min_relevance, min_confidence)
            sha1_triples[key].extend(triples)
            parsed += 1
        except Exception:
            failed += 1

    print(f"    Parsed {parsed} responses, {failed} failed")
    return sha1_triples


def run_stage_a_batch(
    news_df: pd.DataFrame,
    cache_dir: str,
    min_relevance:  float = None,
    min_confidence: float = None,
    ticker_filter:  Optional[str] = None,
    date_prefix:    Optional[str] = None,
    batch_chunk_size: int = None,
):
    """
    Batch-optimized extraction: gom articles → submit Gemini Batch API → 50% saving.

    Phase 1: Scan corpus, collect uncached articles (deduplicate by SHA1)
    Phase 2: Submit as batch job(s), save results to per-article cache

    Interrupt-safe:
      - Batch job state saved to disk → resume polling on restart
      - Each article cached individually after batch completes
      - Cache check before each chunk → skip already-cached articles

    After completion: run `python embed_news.py` to read from cache.
    """
    os.makedirs(cache_dir, exist_ok=True)
    _mr = min_relevance  if min_relevance  is not None else GlobalConfig.KG_MIN_RELEVANCE
    _mc = min_confidence if min_confidence is not None else GlobalConfig.KG_MIN_CONFIDENCE
    _chunk_size = batch_chunk_size or getattr(GlobalConfig, 'KG_BATCH_CHUNK_SIZE', 5000)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set.")

    # ── Normalize & filter ────────────────────────────────────────────────
    df = normalize_news_df(news_df)
    if ticker_filter:
        df = df[df["equity"] == ticker_filter.upper()]
    if date_prefix:
        df = df[df["date"].astype(str).str.startswith(date_prefix)]
    if len(df) == 0:
        print("No data after filter.")
        return

    print("=" * 70)
    print("  Stage A — Batch Mode (Gemini Batch API, 50% cost saving)")
    print("=" * 70)
    print(f"  Thresholds: min_relevance={_mr}  min_confidence={_mc}")
    print(f"  Batch chunk size: {_chunk_size}")
    print(f"  Cache: {cache_dir}")

    # ── Phase 1: Collect uncached ─────────────────────────────────────────
    print(f"\n  Phase 1: Scanning articles...")
    sha1_map, cached_count = _collect_all_uncached(df, cache_dir)
    total_unique = cached_count + len(sha1_map)

    print(f"    Total unique articles: {total_unique:,}")
    print(f"    Already cached:        {cached_count:,}")
    print(f"    To extract:            {len(sha1_map):,}")

    if not sha1_map:
        print("\n  All articles already cached. Nothing to extract.")
        print("  Next: python embed_news.py")
        return

    # ── Estimated cost ────────────────────────────────────────────────────
    avg_tokens_per_article = 4000  # prompt + response estimate
    total_tokens = len(sha1_map) * avg_tokens_per_article
    est_cost_interactive = total_tokens / 1_000_000 * 0.10  # ~$0.10/M tokens
    est_cost_batch = est_cost_interactive * 0.5
    print(f"\n    Estimated cost (batch, 50% off): ~${est_cost_batch:.2f}")
    print(f"    Estimated cost (interactive):    ~${est_cost_interactive:.2f}")
    print(f"    Saving: ~${est_cost_interactive - est_cost_batch:.2f}")

    # ── Phase 2: Submit batch jobs ────────────────────────────────────────
    client = genai.Client(api_key=api_key)
    model = f"models/{MODEL_ID}"

    items = list(sha1_map.items())  # [(sha1, meta), ...]
    n_chunks = (len(items) + _chunk_size - 1) // _chunk_size

    print(f"\n  Phase 2: Submitting {n_chunks} batch job(s)...")

    total_saved = 0
    for chunk_i in range(n_chunks):
        start = chunk_i * _chunk_size
        end   = min(start + _chunk_size, len(items))
        chunk = items[start:end]

        # ── Check for resume state (interrupted polling) ──────────────────
        saved_state = _load_batch_state(cache_dir, chunk_i)
        if saved_state:
            print(f"\n  Chunk {chunk_i+1}/{n_chunks}: RESUMING job {saved_state['job_name']}")
            batch_job = _poll_batch_job(client, saved_state["job_name"])

            if batch_job.state.name == "JOB_STATE_SUCCEEDED":
                sha1_results = _parse_batch_results(batch_job, _mr, _mc)
                chunk_saved = 0
                for h, meta in chunk:
                    if _load_cache(cache_dir, h) is not None:
                        continue
                    triples = sha1_results.get(h, [])
                    deduped = apply_quality_filters(triples)
                    deduped = tag_triples_source(deduped, h)
                    _save_cache(cache_dir, h, deduped, meta=meta)
                    chunk_saved += 1
                total_saved += chunk_saved
                print(f"    Saved {chunk_saved} articles to cache")
                _remove_batch_state(cache_dir, chunk_i)
                continue
            else:
                print(f"    Resumed job FAILED: {batch_job.state.name}")
                _remove_batch_state(cache_dir, chunk_i)
                # Fall through to re-submit

        # ── Filter out already-cached from this chunk ─────────────────────
        still_needed = [(h, meta) for h, meta in chunk if _load_cache(cache_dir, h) is None]
        if not still_needed:
            print(f"\n  Chunk {chunk_i+1}/{n_chunks}: all {len(chunk)} already cached (skipped)")
            continue

        print(f"\n  Chunk {chunk_i+1}/{n_chunks}: {len(still_needed)} articles")

        # ── Build inline requests (typed InlinedRequest objects) ─────────
        from google.genai import types as genai_types
        inline_requests = []
        for h, meta in still_needed:
            pieces = get_article_pieces(meta["full_text"])
            for piece_i, piece in enumerate(pieces):
                prompt = build_user_prompt(piece, meta["primary_ticker"], meta["date"])
                key = h if len(pieces) == 1 else f"{h}_p{piece_i}"
                inline_requests.append(genai_types.InlinedRequest(
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=FINDKG_LITE_SYSTEM_PROMPT,
                        response_mime_type="application/json",
                        response_schema=_GEN_CONFIG_DICT["response_schema"],
                        temperature=_GEN_CONFIG_DICT["temperature"],
                        max_output_tokens=_GEN_CONFIG_DICT["max_output_tokens"],
                    ),
                    metadata={"sha1": key},
                ))

        print(f"    Submitting {len(inline_requests)} requests...")

        # ── Submit batch job ──────────────────────────────────────────────
        try:
            batch_job = client.batches.create(
                model=model,
                src=inline_requests,
                config=genai_types.CreateBatchJobConfig(
                    displayName=f"findkg-v5-chunk-{chunk_i+1}-of-{n_chunks}",
                ),
            )
        except Exception as e:
            print(f"    Submit FAILED: {e}")
            print(f"    Re-run to retry this chunk.")
            continue

        print(f"    Job: {batch_job.name}")
        print(f"    State: {batch_job.state.name}")

        # ── Save state for resume ─────────────────────────────────────────
        _save_batch_state(cache_dir, chunk_i, batch_job.name, [h for h, _ in still_needed])

        # ── Poll until done ───────────────────────────────────────────────
        batch_job = _poll_batch_job(client, batch_job.name)

        if batch_job.state.name != "JOB_STATE_SUCCEEDED":
            print(f"    Batch job FAILED: {batch_job.state.name}")
            print(f"    Re-run to retry this chunk (state saved).")
            continue

        # ── Parse results and save to cache ───────────────────────────────
        sha1_results = _parse_batch_results(batch_job, _mr, _mc)

        chunk_saved = 0
        for h, meta in still_needed:
            # Collect triples (handle multi-piece keys)
            triples = list(sha1_results.get(h, []))
            # Also check piece keys if chunking was on
            for piece_i in range(10):  # max 10 pieces per article
                piece_key = f"{h}_p{piece_i}"
                if piece_key in sha1_results:
                    triples.extend(sha1_results[piece_key])

            deduped = apply_quality_filters(triples)
            deduped = tag_triples_source(deduped, h)
            _save_cache(cache_dir, h, deduped, meta=meta)
            chunk_saved += 1

        total_saved += chunk_saved
        print(f"    Saved {chunk_saved} articles to cache")

        # ── Remove state file ─────────────────────────────────────────────
        _remove_batch_state(cache_dir, chunk_i)

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  Phase 2 complete.")
    print(f"  Articles extracted & cached: {total_saved}")
    print(f"  Total cached (including previous): {cached_count + total_saved}")
    print(f"\n  Next: python embed_news.py")
    print(f"{'='*70}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Stage A: KG Extraction (V5.2)")
    p.add_argument("--batch", action="store_true",
                   help="Use Gemini Batch API (50%% cost saving, slower)")
    p.add_argument("--max-concurrent", type=int,   default=None,
                   help="Max concurrent API calls (interactive mode only)")
    p.add_argument("--batch-chunk-size", type=int, default=None,
                   help=f"Articles per batch job (default: {GlobalConfig.KG_BATCH_CHUNK_SIZE})")
    p.add_argument("--min-relevance",  type=float, default=None)
    p.add_argument("--min-confidence", type=float, default=None)
    p.add_argument("--ticker", default=None, help="Filter to 1 ticker")
    p.add_argument("--date",   default=None, help="Date prefix filter, e.g. '2022-07'")
    p.add_argument("--news",   default=None, help="Path to news parquet")
    p.add_argument("--enable-chunking", action="store_true",
                   help="Enable legacy chunking (default: full-article mode)")
    args = p.parse_args()

    if args.enable_chunking:
        GlobalConfig.KG_ENABLE_CHUNKING = True

    news_path = args.news or os.path.join(GlobalConfig.INTERIM_PATH, "concatenated_news_filtered.parquet")
    if not os.path.exists(news_path):
        print(f"Not found: {news_path}")
        sys.exit(1)

    df = pd.read_parquet(news_path)
    print(f"Loaded {len(df):,} rows from {news_path}")

    cache_dir = GlobalConfig.kg_cache_dir()

    if args.batch:
        # ── Batch mode (50% cost saving) ──────────────────────────────────
        run_stage_a_batch(
            news_df=df,
            cache_dir=cache_dir,
            min_relevance=args.min_relevance,
            min_confidence=args.min_confidence,
            ticker_filter=args.ticker,
            date_prefix=args.date,
            batch_chunk_size=args.batch_chunk_size,
        )
    else:
        # ── Interactive mode (fast, full price) ───────────────────────────
        run_stage_a(
            news_df=df,
            cache_dir=cache_dir,
            max_concurrent=args.max_concurrent,
            min_relevance=args.min_relevance,
            min_confidence=args.min_confidence,
            ticker_filter=args.ticker,
            date_prefix=args.date,
        )


if __name__ == "__main__":
    main()