"""Iteration 0 - baseline retrieval, end-to-end.

Build canonical base -> encode -> index -> generate dirty queries -> search ->
evaluate (quality + robustness by noise category + latency + memory).

Runs offline with the char-n-gram encoder by default:
    python run_baseline.py
Neural encoder + FAISS index (downloads a model on first run):
    python run_baseline.py --embedder st --model deepvk/USER-bge-m3 --index hnsw
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
from metrics import evaluate, acc_at_radius


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000, help="canonical base size")
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--embedder", choices=["charngram", "st"], default="charngram")
    ap.add_argument("--model", default="deepvk/USER-bge-m3")
    ap.add_argument("--index", choices=["flat", "faiss-flat", "hnsw", "ivfpq"],
                    default="flat")
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--dataset", default=None, help="real jsonl base (else synthetic)")
    args = ap.parse_args()

    print(f"== Iteration 0 baseline | embedder={args.embedder} index={args.index} ==")

    # 1. canonical base
    canon = get_canonicals(args.n, args.dataset, seed=args.seed)
    texts = [c["text"] for c in canon]
    print(f"canonical base: {len(canon)} addresses")

    # 2. encode
    if args.embedder == "st":
        vec = make_vectorizer("st", model_name=args.model,
                              query_prefix="query: ", doc_prefix="passage: ")
    else:
        vec = make_vectorizer("charngram")
    t = time.perf_counter()
    doc_vecs = vec.fit_transform(texts) if args.embedder == "charngram" \
        else vec.transform(texts)
    build_s = time.perf_counter() - t
    print(f"encoded docs: {doc_vecs.shape} in {build_s:.1f}s")

    # 3. index
    if args.index == "flat":
        index = FlatIndex(doc_vecs)
    else:
        kind = args.index.replace("faiss-", "")
        index = make_faiss(kind, doc_vecs)
    print(f"index: {index.kind}  memory={index.memory_mb():.1f} MB")

    # 4. dirty queries (query -> known gold canonical id)
    eval_set = make_eval_set(canon, args.per_category, seed=args.seed)
    q_texts = [q for q, _, _ in eval_set]
    if args.embedder == "st":
        q_vecs = vec.transform(q_texts, is_query=True)
    else:
        q_vecs = vec.transform(q_texts)
    print(f"eval queries: {len(eval_set)}")

    # 5. search + latency
    results = []
    lat = []
    for i, (_, gold, cat) in enumerate(eval_set):
        t = time.perf_counter()
        ranked = index.search(q_vecs[i], k=10)[0].tolist()
        lat.append((time.perf_counter() - t) * 1000)
        results.append((ranked, gold, cat))

    # 6. report
    m = evaluate(results)
    lat = np.array(lat)
    print("\n-- quality --")
    for k in ["recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10", "ndcg_at_10"]:
        print(f"  {k:14s} {m[k]:.4f}")
    print("-- robustness: recall@1 by noise category --")
    for c, v in m["per_category_recall_at_1"].items():
        print(f"  {c:18s} {v:.4f}")
    print("-- latency per query (ms) --")
    print(f"  p50={np.percentile(lat,50):.2f}  p95={np.percentile(lat,95):.2f}  "
          f"p99={np.percentile(lat,99):.2f}")
    if canon and canon[0].get("lat") is not None:      # geo metric (AddrLLM style)
        for rad in (300, 500):
            a = acc_at_radius(results, canon, rad)
            if a is not None:
                print(f"  Acc@{rad}m={a:.4f}")

    out = dict(config=vars(args), n=len(canon), n_queries=len(eval_set),
               index=index.kind, index_mb=round(index.memory_mb(), 2),
               build_s=round(build_s, 2),
               latency_ms={"p50": float(np.percentile(lat, 50)),
                           "p95": float(np.percentile(lat, 95)),
                           "p99": float(np.percentile(lat, 99))},
               metrics=m)
    with open("results/baseline.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nsaved -> results/baseline.json")


if __name__ == "__main__":
    main()
