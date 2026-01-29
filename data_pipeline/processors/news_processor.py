# data_pipeline/processors/news_processor.py

import os
import json
import re
import ast
from typing import List, Tuple, Dict, Any

import dspy
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans

from data_pipeline.kg.prompts import ENTITY_PROMPT, RELATION_PROMPT


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
        """
        Accepts:
          - path string (.parquet/.csv)
          - pandas DataFrame already loaded

        Returns DataFrame with:
          - date (python date)
          - equity
          - content (if available)
          - title (if available)
        """
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


def _normalize_entity(e: str) -> str:
    return re.sub(r"\s+", " ", (e or "").strip())


# ==============================
# KGGen DSPy Extractor (Entity -> Triple)
# ==============================

class EntitySig(dspy.Signature):
    text = dspy.InputField(desc=ENTITY_PROMPT)
    entities = dspy.OutputField(desc="Python list of strings")


class TripleSig(dspy.Signature):
    text = dspy.InputField(desc=RELATION_PROMPT)
    entities = dspy.InputField(desc="Python list of strings extracted from the same text")
    triples = dspy.OutputField(desc="Python list of (subject, predicate, object) tuples")


class KGGenDSPyExtractor:
    """
    KGGen 2-stage extraction:
      1) entities from text
      2) triples constrained to extracted entities
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

        self.ent_prog = dspy.ChainOfThought(EntitySig)
        self.tri_prog = dspy.ChainOfThought(TripleSig)

    def extract(self, text: str):
        # 1) entities
        ent_raw = self.ent_prog(text=text).entities
        entities = [_normalize_entity(e) for e in _safe_parse_py_list(ent_raw)]
        entities = [e for e in entities if e]
        entities = list(dict.fromkeys(entities))

        # 2) triples constrained to entities
        tri_raw = self.tri_prog(text=text, entities=entities).triples
        triples_raw = _safe_parse_py_list(tri_raw)

        ent_set = set(entities)
        triples: List[Triple] = []
        for t in triples_raw:
            if isinstance(t, (list, tuple)) and len(t) == 3:
                s, p, o = str(t[0]).strip(), str(t[1]).strip(), str(t[2]).strip()
                if s in ent_set and o in ent_set and p:
                    triples.append((s, p, o))

        triples = list(dict.fromkeys(triples))
        return entities, triples


# ==============================
# KGGen OFFLINE Embedder (Phase B′)
# ==============================

class KGGenNewsEmbedder:
    """
    KGGen offline builder (Phase B′):
      Input: aligned_news DataFrame với các cột:
        - date, equity
        - content (ưu tiên) hoặc title (fallback)

      Output dirs:
        data/interim/kg/
          - extracted_triples/{ticker}/{date}.json
          - window_graph_raw/{ticker}/{date}.json
          - window_graph_stable/{ticker}/{date}.json
          - tensors/{ticker}/{date}.pt
        data/interim/kg_embeddings/
          - embedded_kg.json (index cho builder)
    """

    def __init__(
        self,
        interim_root: str = None,
        window_days: int = 20,
        kmeans_k: int = 128,
        chunk_max_chars: int = 1200,
        chunk_min_chars: int = 200,
        max_triples_per_day: int = 250,
        debug_print_samples: bool = False,
        enable_cache: bool = True,
        cache_dirname: str = "kg_cache_chunks",
    ):
        if interim_root is None:
            interim_root = os.path.join("data", "interim")

        self.window_days = window_days
        self.kmeans_k = kmeans_k
        self.chunk_max_chars = chunk_max_chars
        self.chunk_min_chars = chunk_min_chars
        self.max_triples_per_day = max_triples_per_day
        self.debug_print_samples = debug_print_samples

        # SBERT cho node feature + similarity
        self.sbert = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # KGGen LLM extractor
        self.kg_extractor = KGGenDSPyExtractor()

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

        # cache
        self.enable_cache = enable_cache
        self.cache_dir = os.path.join(interim_root, cache_dirname)
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------
    # Chunking
    # ------------------------------
    @staticmethod
    def _normalize_space(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip()

    def split_into_chunks(self, text: str) -> List[str]:
        """
        Chunk long content:
          - split by paragraph
          - sentence split if paragraph too long
          - final hard split if still too long
        """
        text = (text or "").strip()
        if not text:
            return []

        paras = [p.strip() for p in re.split(r"\n{2,}|\r\n{2,}", text) if p.strip()]
        chunks: List[str] = []

        sent_split_re = re.compile(r"(?<=[\.\?\!;])\s+")
        for p in paras:
            p = self._normalize_space(p)
            if len(p) <= self.chunk_max_chars:
                chunks.append(p)
                continue

            sents = [s.strip() for s in sent_split_re.split(p) if s.strip()]
            buf = ""
            for s in sents:
                if not buf:
                    buf = s
                elif len(buf) + 1 + len(s) <= self.chunk_max_chars:
                    buf = buf + " " + s
                else:
                    if len(buf) >= self.chunk_min_chars:
                        chunks.append(buf)
                    else:
                        chunks.append(buf)
                    buf = s

            if buf:
                chunks.append(buf)

        final_chunks: List[str] = []
        for c in chunks:
            c = c.strip()
            if len(c) <= self.chunk_max_chars:
                final_chunks.append(c)
            else:
                for i in range(0, len(c), self.chunk_max_chars):
                    final_chunks.append(c[i:i + self.chunk_max_chars])

        return [c for c in final_chunks if c.strip()]

    # ------------------------------
    # Cache helpers
    # ------------------------------
    def _chunk_hash(self, chunk: str) -> str:
        import hashlib
        return hashlib.sha1(chunk.encode("utf-8")).hexdigest()

    def _cache_path(self, h: str) -> str:
        return os.path.join(self.cache_dir, f"{h}.json")

    def _load_cache(self, chunk: str):
        if not self.enable_cache:
            return None
        h = self._chunk_hash(chunk)
        p = self._cache_path(h)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                return obj.get("entities", []), [tuple(t) for t in obj.get("triples", [])]
            except Exception:
                return None
        return None

    def _save_cache(self, chunk: str, entities: List[str], triples: List[Triple]):
        if not self.enable_cache:
            return
        h = self._chunk_hash(chunk)
        p = self._cache_path(h)
        obj = {"entities": entities, "triples": [list(t) for t in triples]}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)

    # ------------------------------
    # Extract (REAL KGGen via DSPy+Gemini)
    # ------------------------------
    def extract_entities_triples(self, text: str) -> Tuple[List[str], List[Triple]]:
        """
        KGGen extraction cho 1 chunk (có cache).
        """
        cached = self._load_cache(text)
        if cached is not None:
            return cached

        entities, triples = self.kg_extractor.extract(text)
        self._save_cache(text, entities, triples)
        return entities, triples

    def filter_top3_triples(self, triples: List[Triple], full_text: str) -> List[Triple]:
        """
        Giữ tối đa 3 triples có similarity cao nhất với toàn bộ news.
        """
        if len(triples) <= 3:
            return triples

        full_emb = self.sbert.encode(full_text, convert_to_tensor=True)
        scored = []

        for s, p, o in triples:
            triple_text = f"{s} {p} {o}"
            triple_emb = self.sbert.encode(triple_text, convert_to_tensor=True)
            sim = util.cos_sim(triple_emb, full_emb).item()
            scored.append((sim, (s, p, o)))

        scored.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in scored[:3]]

    def extract_entities_triples_chunked(self, text: str) -> Tuple[List[str], List[Triple]]:
        """
        Extract per chunk and merge; filter top-3 triples by semantic similarity.
        """
        chunks = self.split_into_chunks(text)
        all_entities: List[str] = []
        all_triples: List[Triple] = []

        for ch in chunks:
            ents, triples = self.extract_entities_triples(ch)
            all_entities.extend([str(e) for e in ents if str(e).strip()])
            for t in triples:
                if isinstance(t, (list, tuple)) and len(t) == 3:
                    all_triples.append((str(t[0]), str(t[1]), str(t[2])))

        all_entities = list(dict.fromkeys(all_entities))
        all_triples = list(dict.fromkeys(all_triples))

        if self.max_triples_per_day is not None and len(all_triples) > self.max_triples_per_day:
            all_triples = all_triples[: self.max_triples_per_day]

        # ✅ Lọc top-3 triples quan trọng nhất bằng SBERT
        filtered_triples = self.filter_top3_triples(all_triples, text)
        return all_entities, filtered_triples

    # ------------------------------
    # Aggregate rolling window
    # ------------------------------
    def aggregate_window(self, per_day_triples: List[List[Triple]]) -> List[Triple]:
        agg: List[Triple] = []
        for t in per_day_triples:
            agg.extend(t)
        return list(dict.fromkeys(agg))

    # ------------------------------
    # Resolve entities (SBERT + kmeans)
    # ------------------------------
    def resolve_triples(self, triples: List[Triple]) -> List[Triple]:
        ents = list({s for s, _, _ in triples} | {o for _, _, o in triples})
        if len(ents) <= 1:
            return triples

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
        return list(dict.fromkeys(out))

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

        node_x = torch.tensor(
            self.sbert.encode(nodes, normalize_embeddings=True), dtype=torch.float32
        )

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
    # Main runner
    # ------------------------------
    def process_and_save(self, news_df: pd.DataFrame) -> str:
        df = news_df.copy()

        # normalize column names
        if "equity" not in df.columns and "ticker" in df.columns:
            df = df.rename(columns={"ticker": "equity"})
        if "title" not in df.columns and "headline" in df.columns:
            df = df.rename(columns={"headline": "title"})
        if "content" not in df.columns:
            if "body" in df.columns:
                df = df.rename(columns={"body": "content"})
            elif "text" in df.columns:
                df = df.rename(columns={"text": "content"})

        base_req = {"date", "equity"}
        missing = base_req - set(df.columns)
        if missing:
            raise ValueError(f"aligned_news missing columns: {missing}. Has: {list(df.columns)}")

        if ("content" not in df.columns) and ("title" not in df.columns):
            raise ValueError("aligned_news must contain either 'content' or 'title' for KG input.")

        df["date"] = pd.to_datetime(df["date"]).dt.date

        text_col = "content" if "content" in df.columns else "title"
        df[text_col] = df[text_col].fillna("").astype(str)

        grouped = (
            df.groupby(["equity", "date"])[text_col]
              .apply(lambda x: "\n\n".join([t for t in x.astype(str) if t.strip()]))
              .reset_index()
              .rename(columns={text_col: "merged_text"})
        )

        ticker_days: Dict[str, Dict[Any, str]] = {}
        for _, r in grouped.iterrows():
            ticker_days.setdefault(r["equity"], {})[r["date"]] = r["merged_text"]

        results_json: Dict[str, List[Dict[str, Any]]] = {}

        for ticker, day_map in ticker_days.items():
            dates_sorted = sorted(day_map.keys())
            window_triples: List[List[Triple]] = []

            for d in dates_sorted:
                date_str = str(d)
                text = day_map[d]

                if self.debug_print_samples:
                    print(f"[KG INPUT] {ticker} {date_str} chars={len(text)} preview={text[:200]}")

                entities, triples = self.extract_entities_triples_chunked(text)

                # save extracted triples (per-day)
                out_dir = os.path.join(self.dir_triples, ticker)
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{date_str}.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {"date": date_str, "ticker": ticker, "entities": entities, "triples": triples},
                        f,
                        ensure_ascii=False,
                    )

                window_triples.append(triples)
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

                kg_path = self.tensorize(ticker, date_str, stable)

                rec = {"date": date_str, "equity": ticker, "kg_tensor_path": kg_path}
                results_json.setdefault(date_str, []).append(rec)

        out_path = os.path.join(self.emb_dir, "embedded_kg.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False)

        print("✅ KGGen offline outputs saved at:", self.base_dir)
        print("✅ Builder index saved:", out_path)
        if self.enable_cache:
            print("✅ KG chunk cache dir:", self.cache_dir)
        return out_path
