"""Iteration 2 - hybrid retrieval + precise rerank.

Compares four systems on the same dirty queries:
    dense        - char-n-gram cosine (Iter 0)
    bm25         - word-level BM25 (exact tokens)
    hybrid       - dense + bm25 fused by RRF
    hybrid+rerank- hybrid shortlist reordered by a precise joint scorer

    python run_hybrid.py
"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import numpy as np
from src.data import generate_synthetic
from src.noise import make_eval_set
from src.vectorizer import make_vectorizer
from src.index import FlatIndex
from src.lexical import BM25
from src.hybrid import rrf_fuse, precise_rerank
from src.metrics import evaluate

N, PER_CAT, SHORTLIST = 5000, 200, 50


def main():
    canon = generate_synthetic(N)
    texts = [c["text"] for c in canon]

    vec = make_vectorizer("charngram")
    doc_vecs = vec.fit_transform(texts)
    dense = FlatIndex(doc_vecs)
    bm25 = BM25().fit(texts)

    eval_set = make_eval_set(canon, PER_CAT)
    q_texts = [q for q, _, _ in eval_set]
    q_vecs = vec.transform(q_texts)

    systems = {"dense": [], "bm25": [], "hybrid": [], "hybrid+rerank": []}
    lat = {k: [] for k in systems}
    for i, (qtext, gold, cat) in enumerate(eval_set):
        # dense
        t = time.perf_counter()
        d_ids = dense.search(q_vecs[i], k=SHORTLIST)[0].tolist()
        lat["dense"].append((time.perf_counter() - t) * 1000)
        systems["dense"].append((d_ids[:10], gold, cat))
        # bm25
        t = time.perf_counter()
        b_ids = bm25.search(qtext, k=SHORTLIST)
        lat["bm25"].append((time.perf_counter() - t) * 1000)
        systems["bm25"].append((b_ids[:10], gold, cat))
        # hybrid (RRF of the two shortlists)
        t = time.perf_counter()
        h_ids = rrf_fuse([d_ids, b_ids], top=SHORTLIST)
        lat["hybrid"].append(lat["dense"][-1] + lat["bm25"][-1] +
                             (time.perf_counter() - t) * 1000)
        systems["hybrid"].append((h_ids[:10], gold, cat))
        # hybrid + precise rerank of the fused shortlist
        t = time.perf_counter()
        r_ids = precise_rerank(qtext, h_ids, texts, top=10)
        lat["hybrid+rerank"].append(lat["hybrid"][-1] + (time.perf_counter() - t) * 1000)
        systems["hybrid+rerank"].append((r_ids, gold, cat))

    print(f"n={N} queries={len(eval_set)} shortlist={SHORTLIST}\n")
    header = f"{'system':16s}{'R@1':>8s}{'R@5':>8s}{'R@10':>8s}{'MRR@10':>9s}{'p50 ms':>9s}"
    print(header); print("-" * len(header))
    out = {}
    for name, res in systems.items():
        m = evaluate(res)
        p50 = float(np.percentile(lat[name], 50))
        out[name] = dict(metrics=m, p50_ms=round(p50, 3))
        print(f"{name:16s}{m['recall_at_1']:8.3f}{m['recall_at_5']:8.3f}"
              f"{m['recall_at_10']:8.3f}{m['mrr_at_10']:9.3f}{p50:9.3f}")

    print("\n-- recall@1 by noise category --")
    cats = sorted(next(iter(out.values()))["metrics"]["per_category_recall_at_1"])
    print(f"{'category':18s}" + "".join(f"{s:>16s}" for s in systems))
    for c in cats:
        print(f"{c:18s}" + "".join(
            f"{out[s]['metrics']['per_category_recall_at_1'][c]:16.3f}" for s in systems))

    with open("results_hybrid.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nsaved -> results_hybrid.json")


if __name__ == "__main__":
    main()
