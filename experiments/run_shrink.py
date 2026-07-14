"""Iteration 1 (shrink) - quantization and dimensionality reduction (L9).

Shows the recall / memory trade-off of shrinking the vectors:
    fp32     - baseline float32
    int8     - 8-bit scalar quantization (x4 smaller)
    binary   - presence bits + Hamming search (x32 smaller)
    pca-d    - PCA down to d dims

    python run_shrink.py
"""
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import argparse
import numpy as np
from src.data import get_canonicals
from src.noise import make_eval_set
from src.vectorizer import make_vectorizer
from src.index import FlatIndex
from src.metrics import evaluate

N, PER_CAT = 5000, 200


def eval_flat(doc_vecs, q_vecs, eval_set):
    index = FlatIndex(doc_vecs)
    res, lat = [], []
    for i, (_, gold, cat) in enumerate(eval_set):
        t = time.perf_counter()
        ranked = index.search(q_vecs[i], k=10)[0].tolist()
        lat.append((time.perf_counter() - t) * 1000)
        res.append((ranked, gold, cat))
    return evaluate(res), float(np.percentile(lat, 50))


def int8_quant(v):
    scale = np.abs(v).max() / 127.0
    q = np.round(v / scale).astype(np.int8)
    return q, scale


def binary_search_eval(doc_vecs, q_vecs, eval_set):
    """Presence bits (v>0) + Hamming; memory = N*D/8 bytes."""
    docb = np.packbits(doc_vecs > 0, axis=1)          # [N, D/8] uint8
    qb = np.packbits(q_vecs > 0, axis=1)
    res, lat = [], []
    for i, (_, gold, cat) in enumerate(eval_set):
        t = time.perf_counter()
        xor = np.bitwise_xor(docb, qb[i])
        ham = np.unpackbits(xor, axis=1).sum(axis=1)   # hamming distance
        ranked = np.argsort(ham, kind="stable")[:10].tolist()
        lat.append((time.perf_counter() - t) * 1000)
        res.append((ranked, gold, cat))
    mem = docb.nbytes / 1e6
    return evaluate(res), float(np.percentile(lat, 50)), mem


def pca_project(doc_vecs, q_vecs, d):
    mu = doc_vecs.mean(axis=0, keepdims=True)
    Xc = doc_vecs - mu
    # top-d principal directions
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    W = Vt[:d].T                                       # [D, d]
    def proj(m):
        p = (m - mu) @ W
        n = np.linalg.norm(p, axis=1, keepdims=True); n[n == 0] = 1
        return (p / n).astype(np.float32)
    return proj(doc_vecs), proj(q_vecs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="real jsonl base (else synthetic)")
    ap.add_argument("--n", type=int, default=N)
    args = ap.parse_args()

    canon = get_canonicals(args.n, args.dataset)
    texts = [c["text"] for c in canon]
    vec = make_vectorizer("charngram")
    doc_vecs = vec.fit_transform(texts)
    eval_set = make_eval_set(canon, PER_CAT)
    q_vecs = vec.transform([q for q, _, _ in eval_set])
    D = doc_vecs.shape[1]

    rows = []
    header = f"{'variant':12s}{'R@1':>8s}{'R@10':>8s}{'MRR@10':>9s}{'mem MB':>9s}{'p50 ms':>9s}"
    print(f"n={N} D={D} queries={len(eval_set)}\n{header}\n" + "-" * len(header))

    # fp32
    m, p50 = eval_flat(doc_vecs, q_vecs, eval_set)
    mem = doc_vecs.nbytes / 1e6
    rows.append(("fp32", m, mem, p50))

    # int8 (dequantize -> search; memory counts int8 storage)
    dq, sc = int8_quant(doc_vecs); qq, _ = int8_quant(q_vecs)
    m, p50 = eval_flat((dq.astype(np.float32) * sc), (qq.astype(np.float32) * sc), eval_set)
    rows.append(("int8", m, doc_vecs.nbytes / 4 / 1e6, p50))

    # binary
    m, p50, mem_b = binary_search_eval(doc_vecs, q_vecs, eval_set)
    rows.append(("binary", m, mem_b, p50))

    # PCA
    for d in [512, 256, 128, 64]:
        dv, qv = pca_project(doc_vecs, q_vecs, d)
        m, p50 = eval_flat(dv, qv, eval_set)
        rows.append((f"pca-{d}", m, dv.nbytes / 1e6, p50))

    out = []
    for name, m, mem, p50 in rows:
        print(f"{name:12s}{m['recall_at_1']:8.3f}{m['recall_at_10']:8.3f}"
              f"{m['mrr_at_10']:9.3f}{mem:9.1f}{p50:9.3f}")
        out.append(dict(variant=name, recall_at_1=m["recall_at_1"],
                        recall_at_10=m["recall_at_10"], mrr_at_10=m["mrr_at_10"],
                        memory_mb=round(mem, 2), p50_ms=round(p50, 3)))

    with open("results/shrink.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nsaved -> results/shrink.json")


if __name__ == "__main__":
    main()
