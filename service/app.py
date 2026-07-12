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


def _load_dotenv():
    """Read KEY=VALUE lines from a gitignored .env at project root into env.
    Lets you drop YANDEX_MAPS_API_KEY there instead of exporting it each run."""
    path = os.path.join(ROOT, ".env")
    if not os.path.exists(path):
        return
    try:
        for line in open(path, encoding="utf-8-sig"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        pass


_load_dotenv()


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


class NeuralEnginePQ:
    """National base: fine-tuned encoder + FAISS IVF-PQ + SQLite metadata.

    Vectors are PQ-compressed (lossy), so we over-fetch `rerank` candidates,
    pull their text from SQLite, re-encode them and score exactly against the
    query vector (cheap, recovers most of the PQ recall loss, L9)."""

    has_coords = True

    def __init__(self, index_dir):
        import sqlite3
        import threading
        import numpy as np
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer

        cfg = json.load(open(os.path.join(index_dir, "config.json"), encoding="utf-8"))
        self.qp = cfg.get("query_prefix", "")
        self.dp = "passage: " if "e5" in cfg["model"].lower() else ""
        self.rerank = int(cfg.get("rerank", 100))
        self.index = faiss.read_index(os.path.join(index_dir, "ivfpq.faiss"))
        self.index.nprobe = int(cfg.get("nprobe", 32))
        self.db = sqlite3.connect(os.path.join(index_dir, "meta.sqlite"),
                                  check_same_thread=False)
        self.lock = threading.Lock()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(cfg["model"], device=dev)
        self.model.max_seq_length = 64
        self._np = np
        print(f"NeuralEnginePQ: {cfg['n']:,} addresses, IVF-PQ nprobe="
              f"{self.index.nprobe} rerank={self.rerank}, dev={dev}")

    def _rows(self, ids):
        q = "SELECT rowid,text,region,city,lat,lon FROM addr WHERE rowid IN (%s)" % \
            ",".join("?" * len(ids))
        with self.lock:
            cur = self.db.execute(q, ids)
            by_id = {r[0]: r for r in cur.fetchall()}
        return by_id

    def search(self, query, k=8):
        np = self._np
        qv = self.model.encode([self.qp + query], normalize_embeddings=True,
                               convert_to_numpy=True).astype(np.float32)
        n = max(self.rerank, k)
        _, I = self.index.search(np.ascontiguousarray(qv), n)
        ids = [int(i) for i in I[0] if i >= 0]
        if not ids:
            return []
        rows = self._rows(ids)
        cand = [rows[i] for i in ids if i in rows]
        # exact re-rank: re-encode candidate texts, score vs query vector
        cvec = self.model.encode([self.dp + (r[1] or "") for r in cand],
                                 normalize_embeddings=True,
                                 convert_to_numpy=True).astype(np.float32)
        scores = (cvec @ qv[0])
        order = np.argsort(-scores)[:k]
        out = []
        for j in order:
            _, text, region, city, lat, lon = cand[int(j)]
            out.append({"address": text, "region": region, "city": city,
                        "lat": lat, "lon": lon,
                        "score": round(float(scores[int(j)]), 4)})
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
    # Prefer the national IVF-PQ base (index_ru/), then the HNSW base (index/),
    # then the synthetic fallback.
    pq_dir = os.path.join(ROOT, "index_ru")
    if os.path.exists(os.path.join(pq_dir, "ivfpq.faiss")):
        return NeuralEnginePQ(pq_dir)
    index_dir = os.path.join(ROOT, "index")
    if os.path.exists(os.path.join(index_dir, "hnsw.faiss")):
        return NeuralEngine(index_dir)
    print("no prebuilt index -> FallbackEngine (no coordinates). Run build_index.py.")
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
