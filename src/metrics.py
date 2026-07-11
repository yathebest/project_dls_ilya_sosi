"""Hand-implemented ranking metrics (no pytrec_eval).

Relevance is single-target: each query has exactly one correct canonical id.
"""
import math
from collections import defaultdict


def recall_at_k(ranked, gold, k):
    return 1.0 if gold in ranked[:k] else 0.0


def reciprocal_rank(ranked, gold, k=10):
    for i, d in enumerate(ranked[:k], start=1):
        if d == gold:
            return 1.0 / i
    return 0.0


def ndcg_at_10(ranked, gold):
    # binary gain, ideal DCG = 1 (single relevant doc at rank 1)
    for i, d in enumerate(ranked[:10], start=1):
        if d == gold:
            return 1.0 / math.log2(i + 1)
    return 0.0


def evaluate(results):
    """results: list of (ranked_ids, gold_id, category).
    Returns overall metrics + per-category recall@1."""
    n = len(results)
    agg = dict(
        recall_at_1=sum(recall_at_k(r, g, 1) for r, g, _ in results) / n,
        recall_at_5=sum(recall_at_k(r, g, 5) for r, g, _ in results) / n,
        recall_at_10=sum(recall_at_k(r, g, 10) for r, g, _ in results) / n,
        mrr_at_10=sum(reciprocal_rank(r, g) for r, g, _ in results) / n,
        ndcg_at_10=sum(ndcg_at_10(r, g) for r, g, _ in results) / n,
    )
    by_cat = defaultdict(list)
    for r, g, c in results:
        by_cat[c].append(recall_at_k(r, g, 1))
    agg["per_category_recall_at_1"] = {
        c: sum(v) / len(v) for c, v in sorted(by_cat.items())}
    return agg
