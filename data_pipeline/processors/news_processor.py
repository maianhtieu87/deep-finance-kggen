# data_pipeline/processors/news_processor.py

import os
import re
import json
import ast
import time
import random
import hashlib
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import torch
import torch.nn as nn

import requests
from sklearn.cluster import KMeans

import dspy

from configs.config import GlobalConfig
from data_pipeline.kg.prompts import PRICE_IMPACT_PROMPT

# ✅ GNN Encoder Import (corrected path)
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from encoders.kg_graph_encoder import KGGraphEncoder, build_edge_index_from_triples


# ============================================================
# 0) Baseline NewsProcessor (align trading days)
# ============================================================

class NewsProcessor:
    def __init__(self):
        pass

    def align_to_trading_days(self, news_input, trading_days):
        if isinstance(news_input, pd.DataFrame):
            df = news_input.copy()
        elif isinstance(news_input, str):
            if news_input.endswith(".parquet"):
                df = pd.read_parquet(news_input)
            else:
                df = pd.read_csv(news_input)
        else:
            raise TypeError(f"news_input must be DataFrame or str path, got: {type(news_input)}")

        if "equity" not in df.columns and "ticker" in df.columns:
            df = df.rename(columns={"ticker": "equity"})
        if "title" not in df.columns and "headline" in df.columns:
            df = df.rename(columns={"headline": "title"})
        if "content" not in df.columns:
            if "body" in df.columns:
                df = df.rename(columns={"body": "content"})
            elif "text" in df.columns:
                df = df.rename(columns={"text": "content"})

        if "date" not in df.columns:
            raise ValueError(f"News data missing 'date' column. Has {list(df.columns)}")
        df["date"] = pd.to_datetime(df["date"]).dt.date

        if trading_days is not None:
            td = set(pd.to_datetime(trading_days).date)
            df = df[df["date"].isin(td)]

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
    def __init__(self):
        pass

    def process_and_save(self, aligned_news: pd.DataFrame) -> str:
        raise RuntimeError("NewsEmbedder disabled. Use KGGenNewsEmbedder.")


# ============================================================
# 1) Utils
# ============================================================

Triple = Tuple[str, str, str]


def _safe_parse_py_list(x):
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


def _normalize_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


# ============================================================
# 2) LLM extractor (PriceImpact) — only used when cache missing
# ============================================================

class PriceImpactSig(dspy.Signature):
    text = dspy.InputField(desc=PRICE_IMPACT_PROMPT)
    triples = dspy.OutputField(desc="Python list of (subject, predicate, object) tuples, ordered by price impact.")


class KGGenDSPyExtractor:
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

    def extract_topk(self, text: str, top_k: int = 5) -> List[Triple]:
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

        # dedup preserve order
        seen = set()
        deduped: List[Triple] = []
        for t in triples:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        return deduped[:top_k]


# ============================================================
# 3) Voyage embedder (rate limit + retry + cache)
# ============================================================

class VoyageEmbedder:
    """
    Voyage-only embedder:
      - cache per text: {cache_dir}/{sha1}.json
      - rate limit by RPM (simple sliding window)
      - retry w/ exponential backoff
      - batch by MAX_TEXTS_PER_REQ
    """
    def __init__(self, cache_dir: str):
        self.api_key = os.getenv("VOYAGE_API_KEY", GlobalConfig.VOYAGE_API_KEY)
        if not self.api_key or self.api_key == "---":
            raise RuntimeError("Missing VOYAGE_API_KEY. Please set env: $env:VOYAGE_API_KEY='...'")

        self.model = getattr(GlobalConfig, "EMBED_MODEL", "voyage-3-large")
        self.max_texts = getattr(GlobalConfig, "MAX_TEXTS_PER_REQ", 40)
        self.max_retries = getattr(GlobalConfig, "MAX_RETRIES", 6)
        self.backoff_base = getattr(GlobalConfig, "BACKOFF_BASE", 30)

        payment_added = bool(getattr(GlobalConfig, "PAYMENT_ADDED", True))
        rl = GlobalConfig.VOYAGE_RATE_LIMITS[True if payment_added else False]
        self.rpm = int(rl["RPM"])
        self.base_sleep = float(rl["SLEEP"])

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        # request timestamps for RPM limiting
        self._req_times: List[float] = []

    def _cache_path(self, text: str) -> str:
        return os.path.join(self.cache_dir, f"{_sha1(text)}.json")

    def _load_cached(self, text: str) -> Optional[List[float]]:
        p = self._cache_path(text)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                emb = obj.get("embedding", None)
                if isinstance(emb, list) and len(emb) > 0:
                    return emb
            except Exception:
                return None
        return None

    def _save_cache(self, text: str, emb: List[float]) -> None:
        p = self._cache_path(text)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"text_sha1": _sha1(text), "embedding": emb}, f, ensure_ascii=False)

    def _rpm_sleep_if_needed(self):
        now = time.time()
        # keep only last 60s
        self._req_times = [t for t in self._req_times if now - t < 60.0]
        if len(self._req_times) >= self.rpm:
            # sleep until we are under rpm
            oldest = min(self._req_times)
            wait = max(0.0, 60.0 - (now - oldest)) + 0.5
            print(f"⏳ Voyage RPM limit reached ({self.rpm}/min). Sleep {wait:.1f}s ...")
            time.sleep(wait)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Returns embeddings aligned with input texts.
        Uses cache to avoid repeat calls.
        """
        texts = [(_normalize_space(t)[:6000] if t else "") for t in texts]  # safety clip
        out: List[Optional[List[float]]] = [None] * len(texts)

        # 1) fill from cache
        missing_idx = []
        missing_texts = []
        for i, t in enumerate(texts):
            if not t.strip():
                out[i] = []
                continue
            cached = self._load_cached(t)
            if cached is not None:
                out[i] = cached
            else:
                missing_idx.append(i)
                missing_texts.append(t)

        if not missing_texts:
            return [o if o is not None else [] for o in out]

        # 2) call voyage for missing (batched)
        url = "https://api.voyageai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        pos = 0
        for batch in chunks(missing_texts, self.max_texts):
            batch_indices = missing_idx[pos:pos + len(batch)]
            pos += len(batch)

            payload = {"model": self.model, "input": batch}

            # retry loop
            for attempt in range(self.max_retries):
                try:
                    self._rpm_sleep_if_needed()
                    if self.base_sleep > 0:
                        time.sleep(self.base_sleep)

                    # connect timeout 15s, read timeout 120s
                    r = requests.post(url, headers=headers, json=payload, timeout=(15, 120))
                    self._req_times.append(time.time())

                    if r.status_code == 429:
                        # rate limited -> backoff
                        wait = self.backoff_base * (2 ** attempt) + random.uniform(0, 3)
                        print(f"⚠️ Voyage 429. Backoff {wait:.1f}s (attempt {attempt+1}/{self.max_retries})")
                        time.sleep(wait)
                        continue

                    r.raise_for_status()
                    data = r.json()
                    embs = data.get("data", [])
                    if len(embs) != len(batch):
                        raise RuntimeError(f"Voyage returned {len(embs)} embeddings for {len(batch)} texts")

                    for bi, one in enumerate(embs):
                        emb = one.get("embedding", None)
                        if not isinstance(emb, list):
                            raise RuntimeError("Voyage embedding format invalid")
                        idx = batch_indices[bi]
                        out[idx] = emb
                        self._save_cache(texts[idx], emb)
                    break  # success

                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise  # re-raise final error
                    wait = self.backoff_base * (2 ** attempt) + random.uniform(0, 3)
                    print(f"⚠️ Voyage error ({type(e).__name__}): {e}")
                    print(f"   -> retry backoff {wait:.1f}s (attempt {attempt+1}/{self.max_retries})")
                    time.sleep(wait)

        return [o if o is not None else [] for o in out]


# ============================================================
# 4) KGGenNewsEmbedder (Voyage-only graph build with GNN)
# ============================================================

class KGGenNewsEmbedder:
    """
    ✅ Updated KGGenNewsEmbedder with Real GNN Encoder

    Key changes:
    - Uses KGGraphEncoder (GraphSAGE) instead of Linear
    - Properly builds edge_index from triples
    - Enables message passing between nodes
    - Output dim = 128 (matching config)
    """

    def __init__(
        self,
        interim_root: str = None,
        window_days: int = 20,
        kmeans_k: int = 128,
        top_triples_per_article: int = 5,
        debug_print_samples: bool = False,
        enable_cache: bool = True,
        cache_dirname: str = "kg_article_cache",
        allow_llm_when_missing: bool = False,
        use_voyage_resolution: bool = True,
        use_voyage_node_features: bool = True,
        voyage_cache_dirname: str = "kg_voyage_emb_cache",
        max_triples_cap_per_day: Optional[int] = None,
        # ✅ GNN encoder params (corrected dimensions)
        use_graph_encoder_embedding: bool = True,
        graph_out_dim: int = 128,  # ✅ FIXED: 128 not 1024
        graph_hidden_dim: int = 128,
        graph_num_layers: int = 2,
        graph_dropout: float = 0.1,
        graph_use_gat: bool = False,  # Set True for attention
    ):
        if interim_root is None:
            interim_root = os.path.join("data", "interim")

        self.interim_root = interim_root
        self.window_days = window_days
        self.kmeans_k = kmeans_k
        self.top_triples_per_article = top_triples_per_article
        self.debug_print_samples = debug_print_samples

        self.enable_cache = enable_cache
        self.allow_llm_when_missing = allow_llm_when_missing

        self.use_voyage_resolution = use_voyage_resolution
        self.use_voyage_node_features = use_voyage_node_features
        self.max_triples_cap_per_day = max_triples_cap_per_day

        self.use_graph_encoder_embedding = use_graph_encoder_embedding
        self.graph_out_dim = graph_out_dim

        # LLM extractor (ONLY if allowed and cache missing)
        self.kg_extractor = None

        # output dirs
        self.base_dir = os.path.join(interim_root, "kg")
        self.dir_triples = os.path.join(self.base_dir, "extracted_triples")
        self.dir_raw = os.path.join(self.base_dir, "window_graph_raw")
        self.dir_stable = os.path.join(self.base_dir, "window_graph_stable")
        self.dir_tensors = os.path.join(self.base_dir, "tensors")
        for d in [self.dir_triples, self.dir_raw, self.dir_stable, self.dir_tensors]:
            os.makedirs(d, exist_ok=True)

        self.emb_dir = os.path.join(interim_root, "kg_embeddings")
        os.makedirs(self.emb_dir, exist_ok=True)

        # cache for article triples
        self.cache_dir = os.path.join(interim_root, cache_dirname)
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # voyage embedder cache
        self.voyage_cache_dir = os.path.join(interim_root, voyage_cache_dirname)
        self.voyage = VoyageEmbedder(cache_dir=self.voyage_cache_dir)

        # ✅ Real GNN Encoder (GraphSAGE with message passing)
        if self.use_graph_encoder_embedding:
            self.graph_encoder = KGGraphEncoder(
                node_dim=1024,              # Voyage embeddings
                hidden_dim=graph_hidden_dim,  # 128
                output_dim=graph_out_dim,     # 128
                num_sage_layers=graph_num_layers,
                use_gat=graph_use_gat,
                dropout=graph_dropout,
            )
            self.graph_encoder.eval()
            print("✅ Using Real GNN Encoder (GraphSAGE with message passing)")
            print(f"   Config: hidden_dim={graph_hidden_dim}, output_dim={graph_out_dim}, layers={graph_num_layers}")
        else:
            self.graph_encoder = None
            print("⚠️  Graph encoder disabled (use_graph_encoder_embedding=False)")

    # ---------- article triple cache ----------
    def _article_cache_path(self, text: str) -> str:
        return os.path.join(self.cache_dir, f"{_sha1(text)}.json")

    def _load_article_cache(self, text: str) -> Optional[List[Triple]]:
        if not self.enable_cache:
            return None
        p = self._article_cache_path(text)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                triples = obj.get("triples", [])
                out = []
                for t in triples:
                    if isinstance(t, (list, tuple)) and len(t) == 3:
                        out.append((str(t[0]), str(t[1]), str(t[2])))
                return out
            except Exception:
                return None
        return None

    def _save_article_cache(self, text: str, triples: List[Triple]) -> None:
        if not self.enable_cache:
            return
        p = self._article_cache_path(text)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"text_sha1": _sha1(text), "triples": [list(t) for t in triples]}, f, ensure_ascii=False)

    def _get_top_triples_for_article(self, text: str) -> List[Triple]:
        text = _normalize_space(text)
        if not text:
            return []

        cached = self._load_article_cache(text)
        if cached is not None:
            return cached[: self.top_triples_per_article]

        if not self.allow_llm_when_missing:
            # graph-only mode: do not call LLM
            return []

        # ✅ lazy init extractor ONLY when we truly need LLM
        if self.kg_extractor is None:
            self.kg_extractor = KGGenDSPyExtractor()

        triples = self.kg_extractor.extract_topk(text, top_k=self.top_triples_per_article)
        self._save_article_cache(text, triples)
        return triples

    # ---------- build per day ----------
    def _collect_day_triples(self, day_df: pd.DataFrame, text_col: str) -> List[Triple]:
        all_triples: List[Triple] = []
        for _, r in day_df.iterrows():
            text = str(r.get(text_col, "") or "")
            triples = self._get_top_triples_for_article(text)
            all_triples.extend(triples)

        # dedup preserve order
        seen = set()
        deduped: List[Triple] = []
        for t in all_triples:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        if self.max_triples_cap_per_day is not None and len(deduped) > self.max_triples_cap_per_day:
            deduped = deduped[: self.max_triples_cap_per_day]
        return deduped

    # ---------- aggregate rolling window ----------
    @staticmethod
    def aggregate_window(per_day_triples: List[List[Triple]]) -> List[Triple]:
        agg: List[Triple] = []
        for t in per_day_triples:
            agg.extend(t)
        seen = set()
        deduped = []
        for x in agg:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        return deduped

    # ---------- resolve via voyage + kmeans ----------
    def resolve_triples(self, triples: List[Triple]) -> List[Triple]:
        if not triples:
            return []
        if not self.use_voyage_resolution:
            return triples

        ents = list({s for s, _, _ in triples} | {o for _, _, o in triples})
        if len(ents) <= 1:
            return triples

        emb_list = self.voyage.embed_texts(ents)
        emb = torch.tensor(emb_list, dtype=torch.float32).numpy()

        k = min(self.kmeans_k, len(ents))
        labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(emb)

        canon: Dict[int, str] = {}
        for e, cid in zip(ents, labels):
            canon.setdefault(int(cid), e)
        mapping = {e: canon[int(cid)] for e, cid in zip(ents, labels)}

        out: List[Triple] = []
        for s, p, o in triples:
            out.append((mapping.get(s, s), p, mapping.get(o, o)))

        seen = set()
        deduped = []
        for t in out:
            if t not in seen:
                seen.add(t)
                deduped.append(t)
        return deduped

    # ---------- tensorize graph + ✅ GNN embedding ----------
    def tensorize_and_embed(self, ticker: str, date_str: str, triples: List[Triple]) -> Tuple[str, List[float]]:
        """
        ✅ Updated to use real GNN encoder with edge_index
        """
        nodes = sorted({s for s, _, _ in triples} | {o for _, _, o in triples})
        
        if not nodes:
            # Empty graph - return zero embedding
            graph_emb = [0.0] * self.graph_out_dim
            obj = {
                "node_x": torch.zeros((0, 1024)),
                "edge_index": torch.zeros((2, 0), dtype=torch.long),
                "ticker_idx": 0,
                "nodes": [],
                "graph_emb": torch.tensor(graph_emb, dtype=torch.float32),
            }
            out_dir = os.path.join(self.dir_tensors, ticker)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{date_str}.pt")
            torch.save(obj, out_path)
            return out_path, graph_emb
        
        node2id = {n: i for i, n in enumerate(nodes)}

        # ✅ Build edge_index properly
        edge_index = build_edge_index_from_triples(triples, node2id)

        if self.debug_print_samples:
            print(f"   Graph {ticker} {date_str}: {len(nodes)} nodes, {edge_index.shape[1]} edges")

        # Handle empty graph (no edges)
        if edge_index.shape[1] == 0:
            if self.debug_print_samples:
                print(f"   ⚠️  Empty graph (no edges) for {ticker} on {date_str}")
            # Return zero embedding
            graph_emb = [0.0] * self.graph_out_dim
        else:
            # Node features
            if self.use_voyage_node_features:
                node_emb = self.voyage.embed_texts(nodes)
                node_x = torch.tensor(node_emb, dtype=torch.float32)
            else:
                node_x = torch.zeros((len(nodes), 1024), dtype=torch.float32)

            # ✅ GNN encoding with edge_index
            if self.use_graph_encoder_embedding and self.graph_encoder is not None:
                with torch.no_grad():
                    g = self.graph_encoder(
                        x=node_x,
                        edge_index=edge_index,
                        batch=None,  # Single graph
                    )  # Output: (1, output_dim)
                    g = g.squeeze(0)  # (output_dim,)
                graph_emb = g.detach().cpu().tolist()
            else:
                graph_emb = [0.0] * self.graph_out_dim

        ticker_idx = node2id.get(ticker, 0) if nodes else 0

        # Save tensors
        if self.use_voyage_node_features:
            node_emb = self.voyage.embed_texts(nodes) if nodes else []
            node_x = torch.tensor(node_emb, dtype=torch.float32) if node_emb else torch.zeros((0, 1024))
        else:
            node_x = torch.zeros((len(nodes), 1024), dtype=torch.float32)

        obj: Dict[str, Any] = {
            "node_x": node_x,
            "edge_index": edge_index,
            "ticker_idx": ticker_idx,
            "nodes": nodes,
            "graph_emb": torch.tensor(graph_emb, dtype=torch.float32),
        }

        out_dir = os.path.join(self.dir_tensors, ticker)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{date_str}.pt")
        torch.save(obj, out_path)
        
        return out_path, graph_emb

    # ============================================================
    # A) Full build: may call LLM when missing cache
    # ============================================================
    def process_and_save(self, news_df: pd.DataFrame) -> str:
        df = news_df.copy()

        if "equity" not in df.columns and "ticker" in df.columns:
            df = df.rename(columns={"ticker": "equity"})
        if "content" not in df.columns and "text" in df.columns:
            df = df.rename(columns={"text": "content"})
        df["date"] = pd.to_datetime(df["date"]).dt.date

        text_col = "content" if "content" in df.columns else "title"
        df[text_col] = df[text_col].fillna("").astype(str)

        results_json: Dict[str, List[Dict[str, Any]]] = {}

        for ticker in sorted(df["equity"].unique()):
            df_t = df[df["equity"] == ticker].copy()
            dates_sorted = sorted(df_t["date"].unique())

            window_triples: List[List[Triple]] = []

            for d in dates_sorted:
                date_str = str(d)
                day_df = df_t[df_t["date"] == d]

                day_triples = self._collect_day_triples(day_df, text_col=text_col)

                # save extracted triples per-day
                out_dir = os.path.join(self.dir_triples, ticker)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
                    json.dump({"date": date_str, "ticker": ticker, "triples": day_triples}, f, ensure_ascii=False)

                window_triples.append(day_triples)
                if len(window_triples) > self.window_days:
                    window_triples.pop(0)

                raw_graph = self.aggregate_window(window_triples)

                out_dir = os.path.join(self.dir_raw, ticker)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
                    json.dump({"date": date_str, "ticker": ticker, "triples": raw_graph}, f, ensure_ascii=False)

                stable = self.resolve_triples(raw_graph)

                out_dir = os.path.join(self.dir_stable, ticker)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
                    json.dump({"date": date_str, "ticker": ticker, "triples": stable}, f, ensure_ascii=False)

                kg_path, graph_emb = self.tensorize_and_embed(ticker, date_str, stable)

                rec = {
                    "date": date_str,
                    "equity": ticker,
                    "kg_tensor_path": kg_path,
                    "embedding": graph_emb,  # ✅ critical for builder -> news_embedding
                }
                results_json.setdefault(date_str, []).append(rec)

        out_path = os.path.join(self.emb_dir, "embedded_kg.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False)

        print("✅ KG outputs saved at:", self.base_dir)
        print("✅ Builder index saved:", out_path)
        print("✅ Article triple cache dir:", self.cache_dir)
        print("✅ Voyage emb cache dir:", self.voyage_cache_dir)
        return out_path

    # ============================================================
    # B) Graph-only rebuild: NO LLM. Uses cached article triples.
    # ============================================================
    def rebuild_graph_only(self) -> str:
        """
        Rebuild window_graph_raw + window_graph_stable + tensors + embedded_kg.json
        WITHOUT calling LLM. It will skip articles that are not in cache.

        ⚠️ Still needs Voyage embeddings if:
            use_voyage_resolution=True or use_voyage_node_features=True
        """
        news_path = os.path.join(self.interim_root, "concatenated_news_filtered.parquet")
        if not os.path.exists(news_path):
            raise FileNotFoundError(f"Cannot find news parquet at: {news_path}")

        df = pd.read_parquet(news_path)
        if "equity" not in df.columns and "ticker" in df.columns:
            df = df.rename(columns={"ticker": "equity"})
        if "content" not in df.columns and "text" in df.columns:
            df = df.rename(columns={"text": "content"})
        df["date"] = pd.to_datetime(df["date"]).dt.date

        text_col = "content" if "content" in df.columns else "title"
        df[text_col] = df[text_col].fillna("").astype(str)

        print("🚀 REBUILD GRAPH-ONLY (NO LLM) — Using Real GNN Encoder")

        results_json: Dict[str, List[Dict[str, Any]]] = {}

        missing_articles = 0
        total_articles = 0

        for ticker in sorted(df["equity"].unique()):
            df_t = df[df["equity"] == ticker].copy()
            dates_sorted = sorted(df_t["date"].unique())

            window_triples: List[List[Triple]] = []

            for d in dates_sorted:
                date_str = str(d)
                day_df = df_t[df_t["date"] == d]

                day_triples: List[Triple] = []
                for _, r in day_df.iterrows():
                    total_articles += 1
                    text = _normalize_space(str(r.get(text_col, "") or ""))
                    if not text:
                        continue
                    cached = self._load_article_cache(text)
                    if cached is None:
                        missing_articles += 1
                        continue
                    day_triples.extend(cached[: self.top_triples_per_article])

                # dedup preserve order
                seen = set()
                deduped = []
                for t in day_triples:
                    if t not in seen:
                        seen.add(t)
                        deduped.append(t)
                if self.max_triples_cap_per_day is not None and len(deduped) > self.max_triples_cap_per_day:
                    deduped = deduped[: self.max_triples_cap_per_day]
                day_triples = deduped

                # save extracted triples per-day
                out_dir = os.path.join(self.dir_triples, ticker)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
                    json.dump({"date": date_str, "ticker": ticker, "triples": day_triples}, f, ensure_ascii=False)

                window_triples.append(day_triples)
                if len(window_triples) > self.window_days:
                    window_triples.pop(0)

                raw_graph = self.aggregate_window(window_triples)

                out_dir = os.path.join(self.dir_raw, ticker)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
                    json.dump({"date": date_str, "ticker": ticker, "triples": raw_graph}, f, ensure_ascii=False)

                stable = self.resolve_triples(raw_graph)

                out_dir = os.path.join(self.dir_stable, ticker)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
                    json.dump({"date": date_str, "ticker": ticker, "triples": stable}, f, ensure_ascii=False)

                kg_path, graph_emb = self.tensorize_and_embed(ticker, date_str, stable)

                rec = {
                    "date": date_str,
                    "equity": ticker,
                    "kg_tensor_path": kg_path,
                    "embedding": graph_emb,  # ✅ critical for builder
                }
                results_json.setdefault(date_str, []).append(rec)

        out_path = os.path.join(self.emb_dir, "embedded_kg.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False)

        hit = total_articles - missing_articles
        print(f"✅ Graph-only rebuild done. Cache-hit articles: {hit}/{total_articles} (miss={missing_articles})")
        print("✅ Builder index saved:", out_path)
        print("✅ Voyage emb cache dir:", self.voyage_cache_dir)
        return out_path