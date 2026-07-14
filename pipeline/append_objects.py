"""Append extra records (e.g. street/city OBJECTS) to a prebuilt IVF-PQ index
WITHOUT re-encoding what's already there.

FAISS IVF-PQ supports adding vectors to a trained index, and SQLite rows just get
new rowids that continue from the current count — so a street-only query can match
the STREET object instead of the houses on it, for a few minutes of encoding.

    python append_objects.py --dataset data/canon_fias_objects.jsonl --index index_ru

IMPORTANT: stop the service first (it holds the index in memory / SQLite open),
then restart it afterwards to load the enlarged index.
"""
import argparse
import json
import os
import sqlite3
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")
import numpy as np


def iter_rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--index", default="index_ru")
    ap.add_argument("--chunk", type=int, default=100000)
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    cfg = json.load(open(os.path.join(args.index, "config.json"), encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dp = "passage: " if "e5" in cfg["model"].lower() else ""
    model = SentenceTransformer(cfg["model"], device=dev)
    model.max_seq_length = 64

    index = faiss.read_index(os.path.join(args.index, "ivfpq.faiss"))
    base = index.ntotal
    db = sqlite3.connect(os.path.join(args.index, "meta.sqlite"))
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    print(f"model={cfg['model']} dev={dev} | current index ntotal={base:,}")

    buf, pos = [], base

    def flush():
        nonlocal pos
        if not buf:
            return
        vecs = model.encode([dp + (r.get("text") or "") for r in buf],
                            normalize_embeddings=True, batch_size=args.batch,
                            convert_to_numpy=True).astype(np.float32)
        index.add(vecs)
        db.executemany(
            "INSERT INTO addr(rowid,text,region,city,lat,lon,fias_guid,"
            "postal,region_code) VALUES(?,?,?,?,?,?,?,?,?)",
            [(pos + i, r.get("text"), r.get("region"), r.get("city"),
              r.get("lat"), r.get("lon"), r.get("fias_guid"),
              r.get("postal"), r.get("region_code")) for i, r in enumerate(buf)])
        pos += len(buf)
        buf.clear()
        db.commit()
        print(f"  ... appended up to {pos:,}", flush=True)

    for r in iter_rows(args.dataset):
        buf.append(r)
        if len(buf) >= args.chunk:
            flush()
    flush()
    db.close()

    faiss.write_index(index, os.path.join(args.index, "ivfpq.faiss"))
    cfg["n"] = pos
    with open(os.path.join(args.index, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    print(f"done: appended {pos - base:,} objects -> total {pos:,}")


if __name__ == "__main__":
    main()
