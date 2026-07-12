"""Iteration 3 - fine-tune a domain encoder (headline "beyond baseline").

Contrastive fine-tuning (InfoNCE via in-batch negatives, L6/L7): pull a dirty
address variant to its canonical form, push away others. Pairs are generated
from the AddrLLM noise taxonomy on REAL OSM addresses.

To show generalization (not "teaching to the test"), training pairs are built
from addresses OUTSIDE the held-out eval subset (rows >= --holdout).

    python train_encoder.py --base intfloat/multilingual-e5-small \
        --dataset data/canon.jsonl --pairs 30000 --epochs 1 --batch 64 \
        --out models/addr-e5-ft
    # then compare off-the-shelf vs fine-tuned on the 8k eval subset:
    python run_baseline.py --embedder st --model intfloat/multilingual-e5-small \
        --dataset data/canon_8k.jsonl --index hnsw            # baseline
    python run_baseline.py --embedder st --model models/addr-e5-ft \
        --dataset data/canon_8k.jsonl --index hnsw            # fine-tuned

Geocoding-aware multi-task head (AddrLLM Fig.4) and distillation to a tiny model
are wired as documented extensions at the bottom (need coords / a teacher pass).
"""
import argparse
import random
import sys

sys.path.insert(0, ".")
from src.data import load_canon, generate_synthetic
from src.noise import CATEGORIES, make_dirty


def build_pairs(canon, n, q_prefix, d_prefix, seed=20260605):
    """(dirty, canonical) positive pairs for contrastive fine-tuning."""
    rng = random.Random(seed)
    sample = canon if len(canon) <= n else rng.sample(canon, n)
    pairs = []
    for c in sample:
        cat = rng.choice(CATEGORIES)
        pairs.append((q_prefix + make_dirty(c, cat, rng), d_prefix + c["text"]))
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="intfloat/multilingual-e5-small")
    ap.add_argument("--dataset", default=None, help="real jsonl base (else synthetic)")
    ap.add_argument("--pairs", type=int, default=30000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--holdout", type=int, default=8000,
                    help="first N rows are the eval subset -> excluded from training")
    ap.add_argument("--out", default="models/addr-e5-ft")
    args = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer, InputExample, losses
    from torch.utils.data import DataLoader

    torch.manual_seed(20260605)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    canon = load_canon(args.dataset) if args.dataset else generate_synthetic(args.pairs + args.holdout)
    train_pool = canon[args.holdout:] or canon        # unseen addresses
    print(f"base={args.base} device={dev} | canon={len(canon)} train_pool={len(train_pool)}")

    # e5 models expect "query:" / "passage:" prefixes; keep them consistent with eval
    e5 = "e5" in args.base.lower()
    qp, dp = ("query: ", "passage: ") if e5 else ("", "")
    pairs = build_pairs(train_pool, args.pairs, qp, dp)
    print(f"training pairs: {len(pairs)}  (e.g. {pairs[0][0]!r} -> {pairs[0][1]!r})")

    model = SentenceTransformer(args.base, device=dev)
    model.max_seq_length = 64                          # addresses are short -> fast
    loader = DataLoader([InputExample(texts=[q, d]) for q, d in pairs],
                        shuffle=True, batch_size=args.batch)
    loss = losses.MultipleNegativesRankingLoss(model)  # InfoNCE, in-batch negatives
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs,
              warmup_steps=int(0.1 * len(loader)), show_progress_bar=True)
    model.save(args.out)
    print(f"saved fine-tuned model -> {args.out}")


# --- Extension (b): geocoding-aware head (needs coords, in our OSM base) ---------
# Multi-task: loss = contrastive + λ·MSE(FC(embedding), (lat,lon)). Attach a small
# nn.Linear(dim, 2) head; train with a custom loop mixing MultipleNegativesRanking
# and coordinate MSE. Afterwards plot embedding-distance vs geo-distance (R^2) and
# t-SNE by city, as in AddrLLM Fig.3-4.
# --- Extension (c): distillation -> sergeyzh/rubert-tiny-retriever (L7) -----------
# Teacher = model above; student = tiny; losses.MSELoss on teacher/student
# embeddings for CPU-servable inference.

if __name__ == "__main__":
    main()
