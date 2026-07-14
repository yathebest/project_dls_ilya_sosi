"""Demo service - FastAPI + Yandex map UI.

Dirty address in -> canonical candidates with coordinates out. Uses the
fine-tuned encoder + prebuilt HNSW index from build_index.py (index/).

    python build_index.py --limit 60000        # once (needs the fine-tuned model)
    set YANDEX_MAPS_API_KEY=<your key>          # Yandex Maps JS API v2.1 key
    uvicorn service.app:app                     # http://127.0.0.1:8000

If index/ is missing, falls back to a small synthetic char-n-gram engine
(search works, but without coordinates/map).
"""
import difflib
import json
import os
import re
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


_PUNCT = re.compile(r"[^0-9A-Za-zА-Яа-яЁё/-]+")   # keep house separators / and -
# type words stay lowercase to match the canon ("г. Казань, ул. Баумана"); only
# NAME words get capitalized. (Capitalizing "г/ул/д" measurably hurt R@1 — caught
# by eval_index.py; see results_fias_index_ablation.json.)
try:
    from src.matching import _TYPE_WORDS as _NAME_TYPES
except Exception:
    _NAME_TYPES = set()
_TYPE_WORDS = _NAME_TYPES | {
    "к", "корп", "корпус", "стр", "строение", "влд", "двлд", "соор", "лит",
    "обл", "респ", "край", "ао", "аобл", "р-н", "округ", "дом",
}


def _norm_query(q):
    """Normalize a query so trivial variations don't change the result.

    multilingual-e5 is case- and punctuation-SENSITIVE and the index holds
    proper-cased GAR names ("ул. Баумана"). We always fold punctuation/whitespace.
    Casing is only rewritten when the user gave NO casing signal — i.e. the query
    is all-lowercase or ALL-CAPS — in which case NAME words are capitalized to the
    canon's proper case (type words kept lowercase). Mixed-case input is trusted
    as-is. This fixes 'баумана' vs 'Баумана' vs 'БАУМАНА' without penalising the
    already-proper-cased queries (measured in eval_index.py: uniform-case rewrite
    keeps R@1 ~0.955 on proper input while lifting lowercase input)."""
    q = _PUNCT.sub(" ", q)
    words = q.split()
    if q and (q == q.lower() or q == q.upper()):     # no intentional casing
        out = []
        for w in words:
            lw = w.lower()
            out.append(lw if lw in _TYPE_WORDS else lw[:1].upper() + lw[1:])
        words = out
    return " ".join(words)


# --- settlement gazetteer: real place names, to reject fake towns ------------
# The lexical guard (distinctive word must appear in the result) is defeated by
# joke towns that sound real ("Мухосранск"≈real village "Муханки", ratio 0.71).
# The fix: check the claimed settlement against the ACTUAL set of GAR place names.
_PLACES = set()
_PLACES_BY_PFX = {}
try:
    for _ln in open(os.path.join(ROOT, "data", "place_names.txt"), encoding="utf-8"):
        _n = _ln.strip()
        if _n:
            _PLACES.add(_n)
            _PLACES_BY_PFX.setdefault(_n[:3], []).append(_n)
    print(f"gazetteer: {len(_PLACES):,} place names")
except Exception:
    pass

_SETTLEMENT_MARKERS = {"город", "гор", "г", "пгт", "рп", "с", "село", "д",
                       "деревня", "дер", "пос", "поселок", "п", "х", "хутор",
                       "аул", "станица", "ст-ца"}


def _valid_place(name):
    """True if `name` is a real settlement (exact match or a close typo)."""
    if not _PLACES or name in _PLACES:
        return True                                   # no gazetteer -> don't block
    for cand in _PLACES_BY_PFX.get(name[:3], ()):     # typo tolerance (edit ~1)
        if abs(len(cand) - len(name)) <= 2 and \
                difflib.SequenceMatcher(None, name, cand).ratio() >= 0.85:
            return True
    return False


def _confidence(query, results):
    """Confidence of the top match: 'high' / 'medium' / 'low'.

    Score alone can't catch a made-up place ("Задрищенск дом 5") — the exact
    number + a village-like name embed near a real address (~0.8). So we add a
    lexical guard: for a Cyrillic query, the most distinctive word (longest,
    non-type) must fuzzily appear in the top result. 'задрищенск'/'колатушкина'
    don't → low, even at a high score; a typo ('бауманна'≈'баумана') still
    passes. Skipped for Latin (transliteration) queries, where scripts differ."""
    if not results:
        return "low"
    best = results[0].get("score") or 0.0
    # fake-settlement guard: an explicit "город/с/д <name>" whose <name> is not a
    # real place ("Мухосранск", "Задрищенск") -> low, whatever the score.
    toks = re.findall(r"[а-яa-z-]+|\d+", query.lower().replace("ё", "е"))
    for i in range(len(toks) - 1):
        if toks[i] in _SETTLEMENT_MARKERS:
            nxt = toks[i + 1]
            if re.fullmatch(r"[а-я-]{3,}", nxt) and nxt not in _TYPE_WORDS \
                    and not _valid_place(nxt):
                return "low"
    found = True
    if not re.search(r"[a-z]", query.lower()):        # Cyrillic query only
        content = [w for w in re.findall(r"[а-яё]+", query.lower().replace("ё", "е"))
                   if len(w) >= 4 and w not in _TYPE_WORDS]
        if content:
            key = max(content, key=len)
            rtoks = re.findall(r"[а-яё]+", (results[0].get("address") or "").lower().replace("ё", "е"))
            best_ratio = max((difflib.SequenceMatcher(None, key, t).ratio()
                              for t in rtoks), default=0.0)
            found = best_ratio >= 0.70
    if best < 0.5 or not found:
        return "low"
    return "medium" if best < 0.65 else "high"


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
        v = self.model.encode([self.qp + _norm_query(query)], normalize_embeddings=True,
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
                        "fias_guid": m.get("fias_guid"), "postal": m.get("postal"),
                        "region_code": m.get("region_code"),
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
        q = ("SELECT rowid,text,region,city,lat,lon,fias_guid,postal,region_code "
             "FROM addr WHERE rowid IN (%s)") % ",".join("?" * len(ids))
        with self.lock:
            cur = self.db.execute(q, ids)
            by_id = {r[0]: r for r in cur.fetchall()}
        return by_id

    def search(self, query, k=8, norm=True, num_boost=True):
        np = self._np
        q = _norm_query(query) if norm else query
        qv = self.model.encode([self.qp + q], normalize_embeddings=True,
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
        # exact-number boost: dense retrieval blurs numbers (корпус 2013 vs дом 3,
        # or дом 15 vs дом 5), so nudge candidates containing the query's exact
        # number tokens above near-equal neighbours.
        qnums = set(re.findall(r"\d+", query)) if num_boost else set()
        if qnums:
            for i, r in enumerate(cand):
                m = qnums & set(re.findall(r"\d+", r[1] or ""))
                if m:
                    scores[i] += 0.05 * len(m) / len(qnums)
        order = np.argsort(-scores)[:k]
        out = []
        for j in order:
            _, text, region, city, lat, lon, fias_guid, postal, region_code = cand[int(j)]
            out.append({"address": text, "region": region, "city": city,
                        "lat": lat, "lon": lon, "fias_guid": fias_guid,
                        "postal": postal, "region_code": region_code,
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
        results = engine().search(q, k)
        return JSONResponse({"query": q, "has_coords": engine().has_coords,
                             "confidence": _confidence(q, results),
                             "results": results})

    @app.get("/", response_class=HTMLResponse)
    def home():
        with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
            html = f.read()
        key = os.environ.get("YANDEX_MAPS_API_KEY", "")
        return HTMLResponse(html.replace("__YANDEX_KEY__", key),
                            headers={"Cache-Control": "no-store"})
except ImportError:
    app = None
