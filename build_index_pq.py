"""Scalable index for the *national* base (millions of addresses).

Why a separate builder: build_index.py keeps every 384-d vector in an HNSW-Flat
index (N*384*4 bytes) and every metadata row as a Python dict in RAM. That is
fine for ~600k but explodes for tens of millions (30+ GB RAM). Here instead:

  * encoder runs in a **streaming** two-pass loop (chunk in -> vectors -> add ->
    drop), so peak RAM ~= one chunk, not the whole base;
  * vectors are stored **compressed** in a FAISS **IVF-PQ** index (~48 bytes/vec
    vs 1536), so 20M vectors ~= <1 GB on disk instead of ~30 GB;
  * metadata goes to **SQLite** (meta.sqlite); the service fetches only the ~k
    rows it returns per query, so serving RAM stays tiny.

    python build_index_pq.py --dataset data/canon_ru.jsonl --out index_ru
    python build_index_pq.py --dataset data/canon_ru.jsonl --limit 2000000   # test

Search recall of PQ is lower than Flat -> the service can re-rank the top-N with
the (exact) query vector (L9). config.json records nprobe/rerank hints.
"""
import argparse
import json
import math
import os
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import numpy as np


def count_lines(path):
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def iter_rows(path, limit=0):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                yield json.loads(line)


def reservoir_sample_texts(path, k, limit, seed=20260605):
    """Deterministic reservoir sample of texts to train the quantizer."""
    rng = np.random.RandomState(seed)
    sample = []
    for i, r in enumerate(iter_rows(path, limit)):
        t = r.get("text") or ""
        if len(sample) < k:
            sample.append(t)
        else:
            j = rng.randint(0, i + 1)
            if j < k:
                sample[j] = t
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/addr-e5-ft")
    ap.add_argument("--dataset", default="data/canon_ru.jsonl")
    ap.add_argument("--out", default="index_ru")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--m", type=int, default=48, help="PQ subquantizers (divides dim)")
    ap.add_argument("--nbits", type=int, default=8)
    ap.add_argument("--train-size", type=int, default=300000)
    ap.add_argument("--chunk", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    e5 = "e5" in args.model.lower()
    dp = "passage: " if e5 else ""

    total = count_lines(args.dataset)
    if args.limit:
        total = min(total, args.limit)
    nlist = min(65536, max(1024, int(4 * math.sqrt(total))))
    print(f"model={args.model} dev={dev} | N={total:,} nlist={nlist} "
          f"m={args.m} nbits={args.nbits}")

    model = SentenceTransformer(args.model, device=dev)
    model.max_seq_length = 64

    def encode(texts):
        return model.encode([dp + t for t in texts], normalize_embeddings=True,
                            batch_size=args.batch, convert_to_numpy=True
                            ).astype(np.float32)

    dim = model.get_sentence_embedding_dimension()
    assert dim % args.m == 0, f"m={args.m} must divide dim={dim}"

    # --- train IVF-PQ on a sample ------------------------------------------
    print(f"sampling {args.train_size:,} texts to train the quantizer...")
    sample = reservoir_sample_texts(args.dataset, args.train_size, args.limit)
    tvec = encode(sample)
    quant = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quant, dim, nlist, args.m, args.nbits,
                             faiss.METRIC_INNER_PRODUCT)
    print("training index...")
    index.train(tvec)
    del tvec, sample

    # --- SQLite meta --------------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    db_path = os.path.join(args.out, "meta.sqlite")
    if os.path.exists(db_path):
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE addr(rowid INTEGER PRIMARY KEY, text TEXT, "
               "region TEXT, city TEXT, lat REAL, lon REAL)")

    # --- streaming encode + add --------------------------------------------
    print("encoding + adding in chunks...")
    buf, pos = [], 0
    def flush():
        nonlocal pos
        if not buf:
            return
        vecs = encode([r.get("text") or "" for r in buf])
        index.add(vecs)
        db.executemany(
            "INSERT INTO addr(rowid,text,region,city,lat,lon) VALUES(?,?,?,?,?,?)",
            [(pos + i, r.get("text"), r.get("region"), r.get("city"),
              r.get("lat"), r.get("lon")) for i, r in enumerate(buf)])
        pos += len(buf)
        buf.clear()
        if pos % (args.chunk * 5) == 0 or pos >= total:
            db.commit()
            print(f"  ... {pos:,}/{total:,}")

    for r in iter_rows(args.dataset, args.limit):
        buf.append(r)
        if len(buf) >= args.chunk:
            flush()
    flush()
    db.commit()
    db.close()

    faiss.write_index(index, os.path.join(args.out, "ivfpq.faiss"))
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "query_prefix": "query: " if e5 else "",
                   "n": pos, "dim": dim, "nlist": nlist, "nprobe": 32,
                   "m": args.m, "nbits": args.nbits, "kind": "ivfpq",
                   "rerank": 200}, f)
    print(f"saved -> {args.out}/  (n={pos:,}, ivfpq.faiss + meta.sqlite)")


if __name__ == "__main__":
    main()
