"""Pluggable text -> vector encoders.

Default: CharNGramVectorizer (numpy only, deterministic, no downloads) -- good
typo-robust baseline and the sparse side of the hybrid.
Optional: STVectorizer wraps a sentence-transformers model (USER-bge-m3, e5...)
for the neural iterations.

All encoders return L2-normalized float32 [n, D], so cosine = dot product.
"""
import hashlib
import numpy as np


def _l2norm(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return (m / n).astype(np.float32)


class CharNGramVectorizer:
    """Hashed character n-gram TF vectorizer (typo-robust, dependency-free)."""

    def __init__(self, dim=4096, ngram_range=(3, 4)):
        self.dim = dim
        self.ngram_range = ngram_range

    def fit(self, corpus):
        return self  # hashing needs no vocabulary

    def _ngrams(self, text):
        t = "^" + text.lower().replace(" ", "_") + "$"
        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(t) - n + 1):
                yield t[i:i + n]

    def transform(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for r, text in enumerate(texts):
            for g in self._ngrams(text):
                h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16) % self.dim
                out[r, h] += 1.0
        return _l2norm(out)

    def fit_transform(self, corpus):
        return self.fit(corpus).transform(corpus)


class STVectorizer:
    """sentence-transformers wrapper for the neural iterations."""

    def __init__(self, model_name="deepvk/USER-bge-m3", query_prefix="",
                 doc_prefix=""):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    def fit(self, corpus):
        return self

    def transform(self, texts, is_query=False):
        prefix = self.query_prefix if is_query else self.doc_prefix
        texts = [prefix + t for t in texts] if prefix else texts
        v = self.model.encode(texts, normalize_embeddings=True,
                              convert_to_numpy=True, batch_size=128,
                              show_progress_bar=False)
        return v.astype(np.float32)

    def fit_transform(self, corpus):
        return self.transform(corpus)


def make_vectorizer(kind="charngram", **kw):
    if kind == "charngram":
        return CharNGramVectorizer(**kw)
    if kind == "st":
        return STVectorizer(**kw)
    raise ValueError(kind)
