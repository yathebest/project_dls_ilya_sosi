"""Precompute embeddings + FAISS HNSW index for the demo service.

    python build_index.py --model models/addr-e5-ft --dataset data/canon.jsonl \
        --limit 60000 --out index          # subset for a fast demo
    python build_index.py --limit 0                                # full 623k

Saves index/{vectors.npy, meta.jsonl, hnsw.faiss, config.json}. The service
(service/app.py) loads these and encodes only the query at request time.
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import numpy as np
from src.data import load_canon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/addr-e5-ft")
    ap.add_argument("--dataset", default="data/canon.jsonl")
    ap.add_argument("--limit", type=int, default=60000, help="0 = all")
    ap.add_argument("--out", default="index")
    args = ap.parse_args()

    import torch
    import faiss
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    canon = load_canon(args.dataset)
    if args.limit:
        canon = canon[:args.limit]
    print(f"model={args.model} device={dev} | encoding {len(canon)} addresses")

    e5 = "e5" in args.model.lower()
    dp = "passage: " if e5 else ""
    model = SentenceTransformer(args.model, device=dev)
    model.max_seq_length = 64
    vecs = model.encode([dp + c["text"] for c in canon], normalize_embeddings=True,
                        batch_size=512, convert_to_numpy=True,
                        show_progress_bar=True).astype(np.float32)

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "vectors.npy"), vecs)
    with open(os.path.join(args.out, "meta.jsonl"), "w", encoding="utf-8") as f:
        for c in canon:
            f.write(json.dumps({"id": c["id"], "text": c["text"],
                                "region": c.get("region"), "city": c.get("city"),
                                "lat": c.get("lat"), "lon": c.get("lon"),
                                "osm_id": c.get("osm_id")}, ensure_ascii=False) + "\n")

    index = faiss.IndexHNSWFlat(vecs.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200
    index.add(vecs)
    faiss.write_index(index, os.path.join(args.out, "hnsw.faiss"))

    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "query_prefix": "query: " if e5 else "",
                   "n": len(canon), "dim": int(vecs.shape[1])}, f)
    print(f"saved index -> {args.out}/  (n={len(canon)}, dim={vecs.shape[1]})")


if __name__ == "__main__":
    main()
