"""Hand-implemented ranking metrics (no pytrec_eval).

Relevance is single-target: each query has exactly one correct canonical id.
"""
import math
from collections import defaultdict


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def acc_at_radius(results, canon, radius_m=500):
    """AddrLLM-style geo metric: top-1 prediction within radius_m of the gold
    address. results: list of (ranked_ids, gold_id, category). Needs lat/lon."""
    hit, tot = 0, 0
    for ranked, gold, _ in results:
        if not ranked:
            continue
        pred, g = canon[ranked[0]], canon[gold]
        if g.get("lat") is None or pred.get("lat") is None:
            continue
        tot += 1
        if haversine_m(pred["lat"], pred["lon"], g["lat"], g["lon"]) <= radius_m:
            hit += 1
    return hit / tot if tot else None


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
