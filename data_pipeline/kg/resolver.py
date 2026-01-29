from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

class KGResolver:
    def __init__(self, model_name="all-MiniLM-L6-v2", k=128):
        self.encoder = SentenceTransformer(model_name)
        self.k = k

    def resolve(self, triples):
        entities = list({e for t in triples for e in (t[0], t[2])})
        if len(entities) < 2:
            return triples

        emb = self.encoder.encode(entities, normalize_embeddings=True)
        k = min(self.k, len(entities))
        labels = KMeans(n_clusters=k, n_init="auto").fit_predict(emb)

        canon = {}
        for e, l in zip(entities, labels):
            canon.setdefault(l, e)

        mapping = {e: canon[l] for e, l in zip(entities, labels)}
        return [(mapping[s], p, mapping[o]) for s, p, o in triples]
