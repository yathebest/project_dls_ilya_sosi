"""Iteration 3 - fine-tune a domain encoder (headline "beyond baseline").

Two objectives, AddrLLM-inspired:
  (a) CONTRASTIVE  - pull dirty variants to their canonical address, push away
      others (InfoNCE via in-batch negatives). Needs only text pairs.
  (b) GEOCODING (optional, AddrLLM Fig.4) - auxiliary head predicting lat/lon so
      embedding distance ~ geographic distance. Needs <address, lat/lon> pairs.
  Then DISTILL the fine-tuned model into a tiny fast one (rubert-tiny) for CPU.

Run (needs torch + sentence-transformers + a GPU is strongly recommended):
    pip install sentence-transformers torch
    python train_encoder.py --base deepvk/USER-bge-m3 --epochs 1

This trains part (a) on synthetic pairs generated from the same noise taxonomy.
Part (b) and distillation are wired as documented extensions below.
"""
import argparse
import sys

sys.path.insert(0, ".")
from src.data import generate_synthetic, canonical_string
from src.noise import CATEGORIES, make_dirty
import random


def build_pairs(n=20000, seed=20260605):
    """(dirty, canonical) positive pairs for contrastive fine-tuning."""
    rng = random.Random(seed)
    canon = generate_synthetic(n)
    pairs = []
    for c in canon:
        cat = rng.choice(CATEGORIES)
        pairs.append((make_dirty(c, cat, rng), c["text"]))
    return pairs, canon


def train_contrastive(base, epochs, out_dir):
    from sentence_transformers import (SentenceTransformer, InputExample,
                                       losses)
    from torch.utils.data import DataLoader

    pairs, _ = build_pairs()
    examples = [InputExample(texts=[q, d]) for q, d in pairs]
    model = SentenceTransformer(base)
    loader = DataLoader(examples, shuffle=True, batch_size=64)
    # MultipleNegativesRankingLoss = InfoNCE with in-batch negatives (L6/L7)
    loss = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=epochs,
              warmup_steps=100, show_progress_bar=True)
    model.save(out_dir)
    print(f"saved fine-tuned model -> {out_dir}")
    return out_dir


# ---------------------------------------------------------------------------
# Extension (b): geocoding-aware multi-task head  (needs coordinates)
# ---------------------------------------------------------------------------
# Attach an FC head on top of the pooled embedding that regresses (lat, lon):
#     h = encoder(address)            # [B, 768]
#     coord = FC(h)                   # [B, 2]
#     loss = contrastive(h) + λ * MSE(coord, true_latlon)
# Data: join GAR addresses with OpenAddresses / OSM / Nominatim to get
# <address, lat, lon>. After training, plot embedding-distance vs geo-distance
# and report R^2 / t-SNE station separation, as in AddrLLM Fig.3-4.
#
# Extension (c): distillation to a tiny fast student (L7)
# Teacher = fine-tuned model above; student = sergeyzh/rubert-tiny-retriever.
# Use sentence_transformers.losses.MSELoss on teacher vs student embeddings,
# or margin-MSE on (query, pos, neg) score gaps. Result: CPU-servable encoder.
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="deepvk/USER-bge-m3")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out", default="models/addr-encoder")
    args = ap.parse_args()
    train_contrastive(args.base, args.epochs, args.out)
    print("Next: evaluate with  python run_baseline.py --embedder st --model", args.out)


if __name__ == "__main__":
    main()
