"""Demo service - FastAPI + Yandex map UI.

Dirty address in -> canonical candidates with coordinates out. Uses the
fine-tuned encoder + prebuilt HNSW index from build_index.py (index/).

    python build_index.py --limit 60000        # once (needs the fine-tuned model)
    set YANDEX_MAPS_API_KEY=<your key>          # Yandex Maps JS API v2.1 key
    uvicorn service.app:app                     # http://127.0.0.1:8000

If index/ is missing, falls back to a small synthetic char-n-gram engine
(search works, but without coordinates/map).
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


class NeuralEngine:
    """Fine-tuned encoder + prebuilt FAISS HNSW index (with coordinates)."""

    has_coords = True

    def __init__(self, index_dir):
        import numpy as np
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer

        cfg = json.load(open(os.path.join(index_dir, "config.json"), encoding="utf-8"))
        self.qp = cfg.get("query_prefix", "")
        self.meta = [json.loads(l) for l in
                     open(os.path.join(index_dir, "meta.jsonl"), encoding="utf-8")]
        self.index = faiss.read_index(os.path.join(index_dir, "hnsw.faiss"))
        self.index.hnsw.efSearch = 64
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(cfg["model"], device=dev)
        self.model.max_seq_length = 64
        self._np = np
        print(f"NeuralEngine: {len(self.meta)} addresses, model={cfg['model']}, dev={dev}")

    def search(self, query, k=8):
        v = self.model.encode([self.qp + query], normalize_embeddings=True,
                              convert_to_numpy=True).astype(self._np.float32)
        D, I = self.index.search(self._np.ascontiguousarray(v), k)
        out = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            m = self.meta[idx]
            out.append({"address": m["text"], "region": m.get("region"),
                        "city": m.get("city"), "lat": m.get("lat"),
                        "lon": m.get("lon"), "osm_id": m.get("osm_id"),
                        "score": round(float(score), 4)})
        return out


class FallbackEngine:
    """Synthetic char-n-gram engine when no prebuilt index exists (no coords)."""

    has_coords = False

    def __init__(self, n=20000):
        from src.data import generate_synthetic
        from src.vectorizer import make_vectorizer
        from src.index import FlatIndex
        from src.lexical import BM25
        self.canon = generate_synthetic(n)
        self.texts = [c["text"] for c in self.canon]
        self.vec = make_vectorizer("charngram")
        self.dense = FlatIndex(self.vec.fit_transform(self.texts))
        self.bm25 = BM25().fit(self.texts)

    def search(self, query, k=8):
        from src.hybrid import rrf_fuse, precise_rerank
        qv = self.vec.transform([query])[0]
        d = self.dense.search(qv, k=50)[0].tolist()
        b = self.bm25.search(query, k=50)
        fused = rrf_fuse([d, b], top=50)
        ranked = precise_rerank(query, fused, self.texts, top=k)
        return [{"address": self.texts[i], "region": self.canon[i]["region"],
                 "city": self.canon[i].get("city"), "lat": None, "lon": None,
                 "score": None} for i in ranked]


def build_engine():
    index_dir = os.path.join(ROOT, "index")
    if os.path.exists(os.path.join(index_dir, "hnsw.faiss")):
        return NeuralEngine(index_dir)
    print("index/ not found -> FallbackEngine (no coordinates). Run build_index.py.")
    return FallbackEngine()


# --- FastAPI ---------------------------------------------------------------
try:
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="Address Normalization")
    _engine = None

    def engine():
        global _engine
        if _engine is None:
            _engine = build_engine()
        return _engine

    @app.get("/search")
    def search(q: str, k: int = 8):
        return JSONResponse({"query": q, "has_coords": engine().has_coords,
                             "results": engine().search(q, k)})

    @app.get("/", response_class=HTMLResponse)
    def home():
        with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
            html = f.read()
        key = os.environ.get("YANDEX_MAPS_API_KEY", "")
        return html.replace("__YANDEX_KEY__", key)
except ImportError:
    app = None
