# data_pipeline/processors/news_processor.py

import os
import json
import re
import ast
import time
from typing import List, Tuple, Dict, Any, Optional
import hashlib
from pathlib import Path

import dspy
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans

from data_pipeline.kg.prompts import PRICE_IMPACT_PROMPT

# ==============================
# News Processor (baseline API)
# ==============================

class NewsProcessor:
    """
    Baseline contract used by main_test.py:
      - align_to_trading_days(news_input, trading_days) -> DataFrame
    Required: date, equity
    At least one of: content or title
    """

    def __init__(self):
        pass

    def align_to_trading_days(self, news_input, trading_days):
        # 1) Load df
        if isinstance(news_input, pd.DataFrame):
            df = news_input.copy()
        elif isinstance(news_input, str):
            if news_input.endswith(".parquet"):
                df = pd.read_parquet(news_input)
            else:
                df = pd.read_csv(news_input)
        else:
            raise TypeError(f"news_input must be DataFrame or str path, got: {type(news_input)}")

        # 2) normalize columns
        if "equity" not in df.columns and "ticker" in df.columns:
            df = df.rename(columns={"ticker": "equity"})

        if "title" not in df.columns and "headline" in df.columns:
            df = df.rename(columns={"headline": "title"})

        if "content" not in df.columns:
            if "body" in df.columns:
                df = df.rename(columns={"body": "content"})
            elif "text" in df.columns:
                df = df.rename(columns={"text": "content"})

        # 3) enforce date type
        if "date" not in df.columns:
            raise ValueError(f"News data missing 'date' column. Has {list(df.columns)}")
        df["date"] = pd.to_datetime(df["date"]).dt.date

        # 4) align to trading days
        if trading_days is not None:
            td = set(pd.to_datetime(trading_days).date)
            df = df[df["date"].isin(td)]

        # 5) required base columns
        needed_base = {"date", "equity"}
        missing = needed_base - set(df.columns)
        if missing:
            raise ValueError(f"News data missing columns {missing}. Has {list(df.columns)}")

        if ("content" not in df.columns) and ("title" not in df.columns):
            raise ValueError("News data must contain either 'content' or 'title' column.")

        cols = ["date", "equity"]
        if "content" in df.columns:
            cols.append("content")
        if "title" in df.columns:
            cols.append("title")
        return df[cols].copy()


class NewsEmbedder:
    """
    Placeholder to keep old imports working.
    """
    def __init__(self):
        pass

    def process_and_save(self, aligned_news: pd.DataFrame) -> str:
        raise RuntimeError("NewsEmbedder disabled. Use KGGenNewsEmbedder.")


# ==============================
# KGGen Types & Helpers
# ==============================

Triple = Tuple[str, str, str]


def _safe_parse_py_list(x):
    """
    DSPy/LLM đôi khi trả về string dạng list.
    Parse an toàn sang Python list.
    """
    if isinstance(x, list):
        return x
    if x is None:
        return []
    s = str(x).strip()

    m = re.search(r"\[[\s\S]*\]", s)
    if m:
        s = m.group(0)

    try:
        return ast.literal_eval(s)
    except Exception:
        return []


# ==============================
# Price-Impact DSPy Extractor (Single Call)
# ==============================

class PriceImpactSig(dspy.Signature):
    """
    Single-call signature:
      Input: full news text
      Output: list of (subject, predicate, object) tuples,
              ordered from most to least impactful on stock price.
    """
    text = dspy.InputField(desc=PRICE_IMPACT_PROMPT)
    triples = dspy.OutputField(
        desc="Python list of (subject, predicate, object) tuples, ordered by price impact."
    )


class KGGenDSPyExtractor:
    """
    One LLM call per article:
      - Uses PRICE_IMPACT_PROMPT to extract up to K triples (default K=5).
    """

    def __init__(self, model: str = None, api_key: str = None, temperature: float = 0.0):
        if model is None:
            model = os.getenv("KG_LLM_MODEL", "gemini/gemini-2.0-flash")
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY env var. Set it before running KG extraction.")

        lm = dspy.LM(model=model, api_key=api_key, temperature=temperature)
        dspy.settings.configure(lm=lm)

        self.prog = dspy.ChainOfThought(PriceImpactSig)

    def extract_triples(self, text: str, limit_k: int = 5) -> List[Triple]:
        if not text or not str(text).strip():
            return []

        out = self.prog(text=text).triples
        triples_raw = _safe_parse_py_list(out)

        triples: List[Triple] = []
        for t in triples_raw:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                s = str(t[0]).strip()
                p = str(t[1]).strip()
                o = str(t[2]).strip()
                if s and p and o:
                    triples.append((s, p, o))

        # remove duplicates, preserve order
        seen = set()
        deduped: List[Triple] = []
        for t in triples:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        return deduped[:limit_k]

# ==============================
# KGGen OFFLINE Embedder (NEW LOGIC)
# ==============================

class KGGenNewsEmbedder:
    """
    NEW LOGIC (theo yêu cầu):
    - Per-article: mỗi bài news -> top_triples_per_article triples -> LƯU FILE
    - Per-day: gom tất cả bài trong ngày => N * top_triples_per_article triples (KHÔNG top-k/day)
    - Không gọi LLM nếu per-article file đã tồn tại
    - Graph rebuild (raw/stable/tensors/index) tách riêng, NO LLM
    """

    def __init__(
        self,
        interim_root: str = None,
        window_days: int = 20,
        kmeans_k: int = 128,
        top_triples_per_article: int = 5,
        debug_print_samples: bool = False,
        # resolution
        use_voyage_resolution: bool = True,
        voyage_model: str = "voyage-3-large",
        enable_cache: bool = True,
        cache_dirname: str = "kg_article_cache",
        **kwargs,
    ):
        if interim_root is None:
            interim_root = os.path.join("data", "interim")

        self.window_days = window_days
        self.kmeans_k = kmeans_k
        self.top_triples_per_article = top_triples_per_article
        self.debug_print_samples = debug_print_samples

        self.use_voyage_resolution = use_voyage_resolution
        self.voyage_model = voyage_model

        # SBERT node features (giữ nguyên)
        self.sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # LLM extractor (chỉ dùng khi thiếu)
        self.kg_extractor = KGGenDSPyExtractor()

        # output dirs
        self.base_dir = os.path.join(interim_root, "kg")
        self.dir_triples = os.path.join(self.base_dir, "extracted_triples")            # per-day (merged)
        self.dir_articles = os.path.join(self.base_dir, "extracted_articles")          # ✅ NEW per-article
        self.dir_raw = os.path.join(self.base_dir, "window_graph_raw")
        self.dir_stable = os.path.join(self.base_dir, "window_graph_stable")
        self.dir_tensors = os.path.join(self.base_dir, "tensors")
        for d in [self.dir_triples, self.dir_articles, self.dir_raw, self.dir_stable, self.dir_tensors]:
            os.makedirs(d, exist_ok=True)

        self.emb_dir = os.path.join(interim_root, "kg_embeddings")
        os.makedirs(self.emb_dir, exist_ok=True)

        # legacy cache (optional)
        self.enable_cache = enable_cache
        self.cache_dir = os.path.join(interim_root, cache_dirname)
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------
    # ID helpers (per article)
    # ------------------------------
    def _article_id(self, row: pd.Series) -> str:
        """
        Tạo ID ổn định cho 1 news row:
        ưu tiên url; nếu không có thì hash(content)
        """
        url = str(row.get("url", "") or "").strip()
        if url:
            base = url
        else:
            base = str(row.get("content", "") or "").strip()
        if not base:
            base = str(row.get("title", "") or "").strip()

        h = hashlib.sha1(base.encode("utf-8")).hexdigest()
        return h

    def _article_path(self, ticker: str, date_str: str, article_id: str) -> str:
        out_dir = os.path.join(self.dir_articles, ticker, date_str)
        os.makedirs(out_dir, exist_ok=True)
        return os.path.join(out_dir, f"{article_id}.json")

    # ------------------------------
    # Legacy cache by text (optional)
    # ------------------------------
    def _hash_text(self, s: str) -> str:
        return hashlib.sha1((s or "").encode("utf-8")).hexdigest()

    def _cache_path(self, h: str) -> str:
        return os.path.join(self.cache_dir, f"{h}.json")

    def _load_legacy_cache(self, text: str):
        if not self.enable_cache:
            return None
        h = self._hash_text(text)
        p = self._cache_path(h)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                triples = obj.get("triples", [])
                triples = [tuple(t) for t in triples if isinstance(t, (list, tuple)) and len(t) == 3]
                return triples
            except Exception:
                return None
        return None

    def _save_legacy_cache(self, text: str, triples: List[Triple]):
        if not self.enable_cache:
            return
        h = self._hash_text(text)
        p = self._cache_path(h)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"triples": [list(t) for t in triples]}, f, ensure_ascii=False)

    # ------------------------------
    # Per-article triples: load or extract
    # ------------------------------
    def get_or_extract_article_triples(self, ticker: str, date_str: str, row: pd.Series, text_col: str) -> List[Triple]:
        article_id = self._article_id(row)
        apath = self._article_path(ticker, date_str, article_id)

        # 1) Nếu đã có file per-article => load luôn, NO LLM
        if os.path.exists(apath):
            try:
                with open(apath, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                triples = obj.get("triples", [])
                triples = [tuple(t) for t in triples if isinstance(t, (list, tuple)) and len(t) == 3]
                return triples
            except Exception:
                pass  # nếu hỏng file thì fallthrough để trích lại

        text = str(row.get(text_col, "") or "").strip()
        if not text:
            return []

        # 2) thử legacy cache (hash theo text)
        cached = self._load_legacy_cache(text)
        if cached is not None and len(cached) > 0:
            triples = cached[: self.top_triples_per_article]
        else:
            # 3) gọi LLM (chỉ khi thiếu)
            triples = self.kg_extractor.extract_triples(text, limit_k=self.top_triples_per_article)
            self._save_legacy_cache(text, triples)

        # 4) save per-article file
        with open(apath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ticker": ticker,
                    "date": date_str,
                    "article_id": article_id,
                    "url": str(row.get("url", "") or ""),
                    "title": str(row.get("title", "") or ""),
                    "triples": [list(t) for t in triples],
                },
                f,
                ensure_ascii=False,
            )
        return triples

    # ------------------------------
    # Voyage embeddings for resolution
    # ------------------------------
    def _voyage_embed(self, texts: List[str]):
        import numpy as np
        import requests
        import time

        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing VOYAGE_API_KEY for Voyage resolution.")

        url = "https://api.voyageai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": self.voyage_model, "input": texts}

        for attempt in range(6):
            r = requests.post(url, headers=headers, json=payload, timeout=60)
            if r.status_code == 200:
                data = r.json()
                vecs = [x["embedding"] for x in data["data"]]
                return np.array(vecs, dtype=np.float32)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Voyage embedding error: {r.status_code} {r.text}")

        raise RuntimeError("Voyage embedding failed after retries.")

    # ------------------------------
    # Resolve entities (Voyage + KMeans)
    # ------------------------------
    def resolve_triples(self, triples: List[Triple]) -> List[Triple]:
        ents = list({s for s, _, _ in triples} | {o for _, _, o in triples})
        if len(ents) <= 1:
            return triples

        if self.use_voyage_resolution:
            emb = self._voyage_embed(ents)
        else:
            emb = self.sbert.encode(ents, normalize_embeddings=True)

        k = min(self.kmeans_k, len(ents))
        labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(emb)

        canon: Dict[int, str] = {}
        for e, cid in zip(ents, labels):
            canon.setdefault(int(cid), e)
        mapping = {e: canon[int(cid)] for e, cid in zip(ents, labels)}

        out: List[Triple] = []
        for s, p, o in triples:
            out.append((mapping.get(s, s), p, mapping.get(o, o)))

        # dedup preserve order
        seen = set()
        deduped = []
        for t in out:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped

    # ------------------------------
    # Aggregate rolling window
    # ------------------------------
    def aggregate_window(self, per_day_triples: List[List[Triple]]) -> List[Triple]:
        agg: List[Triple] = []
        for t in per_day_triples:
            agg.extend(t)

        seen = set()
        deduped = []
        for tri in agg:
            if tri not in seen:
                seen.add(tri)
                deduped.append(tri)
        return deduped

    # ------------------------------
    # Tensorize graph
    # ------------------------------
    def tensorize(self, ticker: str, date_str: str, triples: List[Triple]) -> str:
        nodes = sorted({s for s, _, _ in triples} | {o for _, _, o in triples})
        node2id = {n: i for i, n in enumerate(nodes)}

        edges = [(node2id[s], node2id[o]) for s, _, o in triples if s in node2id and o in node2id]
        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges else torch.zeros((2, 0), dtype=torch.long)
        )

        node_x = torch.tensor(self.sbert.encode(nodes, normalize_embeddings=True), dtype=torch.float32)
        ticker_idx = node2id.get(ticker, 0)

        obj: Dict[str, Any] = {
            "node_x": node_x,
            "edge_index": edge_index,
            "ticker_idx": ticker_idx,
            "nodes": nodes,
        }

        out_dir = os.path.join(self.dir_tensors, ticker)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{date_str}.pt")
        torch.save(obj, out_path)
        return out_path

    # ------------------------------
    # MAIN: build extracted_triples (per-day) using per-article (reuse or extract)
    # ------------------------------
    def process_and_save(self, news_df: pd.DataFrame) -> str:
        df = news_df.copy()

        if "equity" not in df.columns and "ticker" in df.columns:
            df = df.rename(columns={"ticker": "equity"})
        if "title" not in df.columns and "headline" in df.columns:
            df = df.rename(columns={"headline": "title"})
        if "content" not in df.columns:
            if "body" in df.columns:
                df = df.rename(columns={"body": "content"})
            elif "text" in df.columns:
                df = df.rename(columns={"text": "content"})

        if "date" not in df.columns or "equity" not in df.columns:
            raise ValueError("aligned_news must contain date & equity")

        df["date"] = pd.to_datetime(df["date"]).dt.date

        text_col = "content" if "content" in df.columns else "title"
        df[text_col] = df[text_col].fillna("").astype(str)

        grouped = df.groupby(["equity", "date"])

        # 1) Build per-day merged triples from per-article files
        for (ticker, day), g in grouped:
            date_str = str(day)

            day_triples: List[Triple] = []

            for _, row in g.iterrows():
                triples_article = self.get_or_extract_article_triples(
                    ticker=ticker,
                    date_str=date_str,
                    row=row,
                    text_col=text_col
                )
                # giữ đúng top-5 mỗi bài (đã enforce ở extractor)
                day_triples.extend(triples_article)

            # ✅ IMPORTANT: yêu cầu bạn muốn giữ hết N*5, không top-k/day
            # Nếu bạn vẫn muốn dedup để giảm noise, có thể bật đoạn dưới:
            # seen = set(); deduped=[]
            # for t in day_triples:
            #     if t not in seen:
            #         seen.add(t); deduped.append(t)
            # day_triples = deduped

            out_dir = os.path.join(self.dir_triples, ticker)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {"date": date_str, "ticker": ticker, "triples": [list(t) for t in day_triples]},
                    f,
                    ensure_ascii=False,
                )

        # 2) Rebuild graph-only (NO LLM)
        return self.rebuild_graph_only()

    # ------------------------------
    # Graph-only rebuild from per-day extracted_triples
    # ------------------------------
    def rebuild_graph_only(self) -> str:
        results_json: Dict[str, List[Dict[str, Any]]] = {}

        for ticker in sorted(os.listdir(self.dir_triples)):
            tdir = os.path.join(self.dir_triples, ticker)
            if not os.path.isdir(tdir):
                continue

            files = sorted([f for f in os.listdir(tdir) if f.endswith(".json")])
            if not files:
                continue

            window_triples: List[List[Triple]] = []

            for fn in files:
                date_str = fn.replace(".json", "")
                fp = os.path.join(tdir, fn)
                with open(fp, "r", encoding="utf-8") as f:
                    obj = json.load(f)

                triples = obj.get("triples", [])
                triples = [tuple(t) for t in triples if isinstance(t, (list, tuple)) and len(t) == 3]

                window_triples.append(triples)
                if len(window_triples) > self.window_days:
                    window_triples.pop(0)

                raw_graph = self.aggregate_window(window_triples)

                out_dir = os.path.join(self.dir_raw, ticker)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
                    json.dump({"date": date_str, "ticker": ticker, "triples": [list(t) for t in raw_graph]}, f, ensure_ascii=False)

                stable = self.resolve_triples(raw_graph)

                out_dir = os.path.join(self.dir_stable, ticker)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
                    json.dump({"date": date_str, "ticker": ticker, "triples": [list(t) for t in stable]}, f, ensure_ascii=False)

                kg_path = self.tensorize(ticker, date_str, stable)

                results_json.setdefault(date_str, []).append(
                    {"date": date_str, "equity": ticker, "kg_tensor_path": kg_path}
                )

        out_path = os.path.join(self.emb_dir, "embedded_kg.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False)

        print("✅ Graph-only rebuild done. Index:", out_path)
        return out_path

