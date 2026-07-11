"""Demo service - FastAPI + mini UI.

Dirty address in -> canonical FIAS candidate out (hybrid + precise rerank).
Builds a small synthetic index at startup (swap in a saved index / real GAR for
production).

    pip install fastapi uvicorn
    uvicorn service.app:app --reload      ->  open http://127.0.0.1:8000
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import generate_synthetic
from src.vectorizer import make_vectorizer
from src.index import FlatIndex
from src.lexical import BM25
from src.hybrid import rrf_fuse, precise_rerank


class Engine:
    """Hybrid (dense char-n-gram + BM25) + precise rerank."""

    def __init__(self, n=20000):
        self.canon = generate_synthetic(n)
        self.texts = [c["text"] for c in self.canon]
        self.vec = make_vectorizer("charngram")
        self.dense = FlatIndex(self.vec.fit_transform(self.texts))
        self.bm25 = BM25().fit(self.texts)

    def search(self, query, k=5, shortlist=50):
        qv = self.vec.transform([query])[0]
        d_ids = self.dense.search(qv, k=shortlist)[0].tolist()
        b_ids = self.bm25.search(query, k=shortlist)
        fused = rrf_fuse([d_ids, b_ids], top=shortlist)
        ranked = precise_rerank(query, fused, self.texts, top=k)
        return [{"fias_id": self.canon[d]["id"],
                 "address": self.texts[d],
                 "region": self.canon[d]["region"]} for d in ranked]


def build_engine(n=20000):
    return Engine(n)


# --- FastAPI app (import-safe: only built when fastapi is installed) ---
try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Address Normalization")
    _engine = None

    def engine():
        global _engine
        if _engine is None:
            _engine = build_engine(int(os.environ.get("BASE_N", "20000")))
        return _engine

    @app.get("/search")
    def search(q: str, k: int = 5):
        return JSONResponse({"query": q, "results": engine().search(q, k)})

    @app.get("/", response_class=HTMLResponse)
    def home():
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "index.html"), encoding="utf-8") as f:
            return f.read()
except ImportError:
    app = None  # fastapi not installed; Engine still usable for tests
