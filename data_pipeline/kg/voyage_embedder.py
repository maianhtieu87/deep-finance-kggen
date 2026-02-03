# data_pipeline/kg/voyage_embedder.py
import os
import time
import json
import hashlib
from typing import List, Optional
import requests

from configs.config import GlobalConfig

class VoyageEmbedder:
    """
    Minimal Voyage embedding client with simple disk cache (optional).
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        output_dimension: Optional[int] = None,
        max_retries: int = None,
        backoff_base: float = None,
        cache_dir: Optional[str] = None,
        enable_cache: bool = True,
        timeout: int = 60,
    ):
        self.api_key = api_key or GlobalConfig.VOYAGE_API_KEY
        if not self.api_key or self.api_key.strip() in ["---", "YOUR_API_KEY"]:
            raise RuntimeError("Missing VOYAGE_API_KEY. Please set env VOYAGE_API_KEY or configs/config.py")

        self.model = model or GlobalConfig.EMBED_MODEL  # e.g. "voyage-3-large"
        self.output_dimension = output_dimension  # works best for newer models, optional
        self.max_retries = max_retries if max_retries is not None else GlobalConfig.MAX_RETRIES
        self.backoff_base = backoff_base if backoff_base is not None else GlobalConfig.BACKOFF_BASE
        self.timeout = timeout

        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        if self.enable_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        self.url = "https://api.voyageai.com/v1/embeddings"

        # Rate-limit hints from your config
        rate = GlobalConfig.VOYAGE_RATE_LIMITS.get(GlobalConfig.PAYMENT_ADDED, {"SLEEP": 1.0})
        self.sleep_sec = float(rate.get("SLEEP", 1.0))

    def _hash(self, s: str) -> str:
        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def _cache_path(self, text: str) -> str:
        return os.path.join(self.cache_dir, f"{self._hash(text)}.json")

    def _try_load_cache(self, text: str):
        if not (self.enable_cache and self.cache_dir):
            return None
        p = self._cache_path(text)
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)["embedding"]
            except Exception:
                return None
        return None

    def _save_cache(self, text: str, emb: List[float]):
        if not (self.enable_cache and self.cache_dir):
            return
        p = self._cache_path(text)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"embedding": emb}, f)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        # Try per-item cache first
        cached = []
        to_query = []
        to_query_idx = []
        for i, t in enumerate(texts):
            t = (t or "").strip()
            if not t:
                cached.append(None)
                continue
            hit = self._try_load_cache(t)
            if hit is not None:
                cached.append(hit)
            else:
                cached.append(None)
                to_query.append(t)
                to_query_idx.append(i)

        if not to_query:
            return [c if c is not None else [] for c in cached]

        payload = {
            "model": self.model,
            "input": to_query
        }
        if self.output_dimension is not None:
            payload["output_dimension"] = int(self.output_dimension)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    embs = [item["embedding"] for item in data["data"]]

                    # Fill back + save cache
                    for j, emb in enumerate(embs):
                        idx = to_query_idx[j]
                        cached[idx] = emb
                        self._save_cache(to_query[j], emb)

                    time.sleep(self.sleep_sec)
                    return [c if c is not None else [] for c in cached]

                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
            except Exception as e:
                last_err = str(e)

            # backoff
            sleep_t = self.backoff_base * attempt
            time.sleep(sleep_t)

        raise RuntimeError(f"Voyage embed failed after retries. Last error: {last_err}")
