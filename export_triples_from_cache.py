import os
import json
import re
import hashlib
from typing import List, Tuple, Any, Dict, Optional

import pandas as pd


# =========================
# CONFIG — SỬA Ở ĐÂY
# =========================
PROJECT_ROOT = r"D:\ProjectNCKH\deep_finance"

NEWS_PARQUET = os.path.join(PROJECT_ROOT, "data", "interim", "concatenated_news_filtered.parquet")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "interim", "kg_article_cache")  # cache per-article triples

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "interim", "kg_exports")
os.makedirs(OUT_DIR, exist_ok=True)

TOP_TRIPLES_PER_ARTICLE = 5  # bạn đang dùng 5
SAVE_FULL_TRIPLES = True     # lưu cả full triples trong cache (nếu cache lưu >5)


# =========================
# UTILS (phải giống logic trong news_processor)
# =========================
Triple = Tuple[str, str, str]

def normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def sha1_text(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

def cache_path_for_text(text: str) -> str:
    return os.path.join(CACHE_DIR, f"{sha1_text(text)}.json")

def load_cached_triples(text: str) -> Optional[List[Triple]]:
    """
    Cache format: {"text_sha1": "...", "triples": [[s,p,o], ...]}
    """
    p = cache_path_for_text(text)
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        triples = obj.get("triples", [])
        out: List[Triple] = []
        for t in triples:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                s, p_, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
                if s and p_ and o:
                    out.append((s, p_, o))
        return out
    except Exception:
        return None

def dedup_preserve_order(triples: List[Triple]) -> List[Triple]:
    seen = set()
    out = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def triples_to_jsonable(triples: List[Triple]) -> List[List[str]]:
    return [[a, b, c] for (a, b, c) in triples]


# =========================
# MAIN
# =========================
def main():
    print("🚀 EXPORT triples from cache")
    print("NEWS_PARQUET:", NEWS_PARQUET)
    print("CACHE_DIR:", CACHE_DIR)
    print("OUT_DIR:", OUT_DIR)

    if not os.path.exists(NEWS_PARQUET):
        raise FileNotFoundError(f"❌ Cannot find news parquet: {NEWS_PARQUET}")
    if not os.path.exists(CACHE_DIR):
        raise FileNotFoundError(f"❌ Cannot find cache dir: {CACHE_DIR} (Bạn đã build cache per-article chưa?)")

    df = pd.read_parquet(NEWS_PARQUET)

    # schema normalize
    if "equity" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "equity"})
    if "title" not in df.columns and "headline" in df.columns:
        df = df.rename(columns={"headline": "title"})

    if "date" not in df.columns:
        raise ValueError(f"❌ Missing 'date' col in news. cols={list(df.columns)}")
    if "equity" not in df.columns:
        raise ValueError(f"❌ Missing 'equity' col in news. cols={list(df.columns)}")

    # pick text col
    text_col = "content" if "content" in df.columns else "title"
    if text_col not in df.columns:
        raise ValueError(f"❌ Need either 'content' or 'title'. cols={list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"]).dt.date
    df[text_col] = df[text_col].fillna("").astype(str)

    # --- ARTICLE-LEVEL EXPORT ---
    article_rows = []
    cache_hit = 0

    for i, row in df.iterrows():
        ticker = str(row["equity"])
        d = str(row["date"])
        text = normalize_space(row[text_col])

        cached = load_cached_triples(text) if text else None
        hit = cached is not None and len(cached) > 0
        if hit:
            cache_hit += 1
            topk = cached[:TOP_TRIPLES_PER_ARTICLE]
            full = cached if SAVE_FULL_TRIPLES else None
        else:
            topk = []
            full = [] if SAVE_FULL_TRIPLES else None

        article_rows.append({
            "date": d,
            "equity": ticker,
            "text_col": text_col,
            "cache_hit": hit,
            "num_triples_in_cache": (len(cached) if cached is not None else 0),
            "triples_topk": triples_to_jsonable(topk),
            "triples_full": (triples_to_jsonable(full) if SAVE_FULL_TRIPLES else None),
            # optional metadata to help inspection
            "title": row.get("title", None),
            "url": row.get("url", None),
            "source": row.get("source", None),
        })

    article_df = pd.DataFrame(article_rows)
    print(f"✅ Articles: {len(article_df)} | Cache-hit: {cache_hit} ({cache_hit/len(article_df):.2%})")

    # --- DAY-LEVEL EXPORT (group by date + ticker) ---
    day_rows = []
    grouped = article_df.groupby(["date", "equity"], sort=True)

    for (d, ticker), g in grouped:
        # list of list-of-triples per article
        per_article = g["triples_topk"].tolist()
        flat: List[Triple] = []
        for art_triples in per_article:
            # art_triples is jsonable list [[s,p,o],...]
            for t in art_triples:
                if isinstance(t, list) and len(t) == 3:
                    flat.append((t[0], t[1], t[2]))
        flat = dedup_preserve_order(flat)

        day_rows.append({
            "date": d,
            "equity": ticker,
            "num_articles": int(len(g)),
            "num_articles_cache_hit": int(g["cache_hit"].sum()),
            "num_triples_after_flatten_dedup": int(len(flat)),
            "triples_day_flat": triples_to_jsonable(flat),
            "triples_day_per_article": per_article,  # giữ cấu trúc để debug “news nào ra triple nào”
        })

    day_df = pd.DataFrame(day_rows)
    print(f"✅ Day-level rows: {len(day_df)}")

    # --- SAVE: parquet + pkl ---
    article_parquet = os.path.join(OUT_DIR, "kg_triples_article_level.parquet")
    day_parquet = os.path.join(OUT_DIR, "kg_triples_day_level.parquet")

    article_pkl = os.path.join(OUT_DIR, "kg_triples_article_level.pkl")
    day_pkl = os.path.join(OUT_DIR, "kg_triples_day_level.pkl")

    # parquet may require pyarrow/fastparquet; fallback to pkl only if not available
    parquet_ok = True
    try:
        article_df.to_parquet(article_parquet, index=False)
        day_df.to_parquet(day_parquet, index=False)
    except Exception as e:
        parquet_ok = False
        print("⚠️ Parquet export failed (missing pyarrow/fastparquet?).")
        print("   Error:", e)

    article_df.to_pickle(article_pkl)
    day_df.to_pickle(day_pkl)

    print("\n====================")
    print("✅ EXPORT DONE")
    if parquet_ok:
        print("Parquet:")
        print(" -", article_parquet)
        print(" -", day_parquet)
    print("Pickle:")
    print(" -", article_pkl)
    print(" -", day_pkl)
    print("====================\n")


if __name__ == "__main__":
    main()