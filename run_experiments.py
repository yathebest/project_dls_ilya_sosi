"""Iteration 1 - efficiency comparison across indexes (Pareto: recall vs latency
vs memory). Builds the base once, then compares Flat / HNSW / IVF-PQ.

    python run_experiments.py
    python run_experiments.py --embedder st --model deepvk/USER-bge-m3
"""
import argparse
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "src")

import numpy as np
from data import get_canonicals
from noise import make_eval_set
from vectorizer import make_vectorizer
from index import FlatIndex, make_faiss
from metrics import evaluate


def run_index(index, q_vecs, eval_set):
    results, lat = [], []
    for i, (_, gold, cat) in enumerate(eval_set):
        t = time.perf_counter()
        ranked = index.search(q_vecs[i], k=10)[0].tolist()
        lat.append((time.perf_counter() - t) * 1000)
        results.append((ranked, gold, cat))
    m = evaluate(results)
    return m, np.array(lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--embedder", choices=["charngram", "st"], default="charngram")
    ap.add_argument("--model", default="deepvk/USER-bge-m3")
    ap.add_argument("--dataset", default=None, help="real jsonl base (else synthetic)")
    args = ap.parse_args()

    canon = get_canonicals(args.n, args.dataset)
    texts = [c["text"] for c in canon]
    if args.embedder == "st":
        vec = make_vectorizer("st", model_name=args.model,
                              query_prefix="query: ", doc_prefix="passage: ")
        doc_vecs = vec.transform(texts)
    else:
        vec = make_vectorizer("charngram")
        doc_vecs = vec.fit_transform(texts)

    eval_set = make_eval_set(canon, args.per_category)
    q_texts = [q for q, _, _ in eval_set]
    q_vecs = (vec.transform(q_texts, is_query=True) if args.embedder == "st"
              else vec.transform(q_texts))

    indexes = {
        "flat":  lambda: FlatIndex(doc_vecs),
        "hnsw":  lambda: make_faiss("hnsw", doc_vecs),
        "ivfpq": lambda: make_faiss("ivfpq", doc_vecs),
    }

    print(f"n={len(canon)} queries={len(eval_set)} embedder={args.embedder}\n")
    header = f"{'index':10s}{'R@1':>8s}{'R@10':>8s}{'MRR@10':>9s}{'mem MB':>9s}{'p50 ms':>9s}{'p99 ms':>9s}"
    print(header)
    print("-" * len(header))
    rows = []
    for name, build in indexes.items():
        index = build()
        m, lat = run_index(index, q_vecs, eval_set)
        row = dict(index=index.kind, recall_at_1=m["recall_at_1"],
                   recall_at_10=m["recall_at_10"], mrr_at_10=m["mrr_at_10"],
                   memory_mb=round(index.memory_mb(), 1),
                   p50_ms=round(float(np.percentile(lat, 50)), 3),
                   p99_ms=round(float(np.percentile(lat, 99)), 3))
        rows.append(row)
        print(f"{name:10s}{m['recall_at_1']:8.3f}{m['recall_at_10']:8.3f}"
              f"{m['mrr_at_10']:9.3f}{row['memory_mb']:9.1f}"
              f"{row['p50_ms']:9.3f}{row['p99_ms']:9.3f}")

    with open("results_experiments.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\nsaved -> results_experiments.json")
    print("Note: IVF-PQ trades recall for ~15x smaller memory; add exact re-rank "
          "of the shortlist to recover recall (L9).")


if __name__ == "__main__":
    main()
