"""Ablation / evaluation of the LIVE FIAS index (index_ru).

Measures retrieval quality on the deployed index with dirty queries generated
from held-out houses (AddrLLM noise taxonomy), and quantifies the contribution
of each engineering trick in the serving pipeline:

    dense (baseline)   — raw query, no normalization, no number boost
    + case/punct norm  — _norm_query (case + punctuation folding)
    + number boost      — exact-number re-rank bonus
    full                — both

Gold is matched by the house's official OBJECTGUID (returned by /search), so a
hit means we retrieved the exact GAR object. Also reports per-category robustness
(Recall@1) and serving latency.

    python eval_index.py --per-category 100
"""
import argparse
import json
import os
import random
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

from src.noise import CATEGORIES, make_dirty
from src.metrics import evaluate


def sample_houses(path, k, seed):
    """Reservoir-sample k houses that have city+street+guid (one streaming pass)."""
    rng = random.Random(seed)
    out, n = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not (r.get("street") and r.get("city") and r.get("fias_guid")):
                continue
            n += 1
            if len(out) < k:
                out.append(r)
            else:
                j = rng.randrange(n)
                if j < k:
                    out[j] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/canon_fias_demo.jsonl")
    ap.add_argument("--index", default="index_ru")
    ap.add_argument("--per-category", type=int, default=100)
    ap.add_argument("--pool", type=int, default=4000, help="house sample to draw queries from")
    ap.add_argument("--seed", type=int, default=20260605)
    ap.add_argument("--lower", action="store_true",
                    help="lowercase dirty queries (simulate real user typing; "
                         "make_dirty otherwise preserves the FIAS capitalization)")
    args = ap.parse_args()

    from service.app import NeuralEnginePQ
    eng = NeuralEnginePQ(args.index)

    print(f"sampling {args.pool} houses from {args.dataset} ...")
    houses = sample_houses(args.dataset, args.pool, args.seed)
    rng = random.Random(args.seed)
    eval_set = []                                    # (dirty, gold_guid, category)
    for cat in CATEGORIES:
        for c in rng.sample(houses, min(args.per_category, len(houses))):
            dq = make_dirty(c, cat, rng)
            eval_set.append((dq.lower() if args.lower else dq, c["fias_guid"], cat))
    rng.shuffle(eval_set)
    print(f"eval queries: {len(eval_set)}  ({args.per_category}/category)\n")

    configs = [
        ("dense (baseline)", dict(norm=False, num_boost=False)),
        ("+ case/punct norm", dict(norm=True,  num_boost=False)),
        ("+ number boost",    dict(norm=False, num_boost=True)),
        ("full (norm+boost)", dict(norm=True,  num_boost=True)),
    ]

    hdr = f"{'config':20}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'MRR@10':>8}{'p50 ms':>8}{'p95 ms':>8}"
    print(hdr + "\n" + "-" * len(hdr))
    rows = []
    for name, cfg in configs:
        results, lat = [], []
        for dirty, gold, cat in eval_set:
            t = time.perf_counter()
            res = eng.search(dirty, k=10, **cfg)
            lat.append((time.perf_counter() - t) * 1000)
            results.append(([x["fias_guid"] for x in res], gold, cat))
        m = evaluate(results)
        lat.sort()
        p50 = lat[len(lat) // 2]
        p95 = lat[int(len(lat) * 0.95)]
        print(f"{name:20}{m['recall_at_1']:7.3f}{m['recall_at_5']:7.3f}"
              f"{m['recall_at_10']:7.3f}{m['mrr_at_10']:8.3f}{p50:8.1f}{p95:8.1f}")
        rows.append((name, m, p50, p95))

    # per-category robustness for the full pipeline
    full = rows[-1][1]["per_category_recall_at_1"]
    print("\nper-category Recall@1 (full pipeline):")
    for c in CATEGORIES:
        print(f"  {c:18}{full.get(c, 0):.3f}")

    out = {name: {"metrics": m, "p50_ms": round(p50, 1), "p95_ms": round(p95, 1)}
           for name, m, p50, p95 in rows}
    os.makedirs("results", exist_ok=True)
    with open("results/fias_index_ablation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\nsaved -> results/fias_index_ablation.json")


if __name__ == "__main__":
    main()
