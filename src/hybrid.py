"""RRF fusion + precise rerank (Iteration 2).

rrf_fuse    - merge several ranked lists by rank (RRF, k=60), scale-free (L3/L8).
precise_rerank - re-order a shortlist by an exact joint score (offline stand-in
                 for a cross-encoder); CrossEncoder path is optional (L7).
"""
import difflib
from src.lexical import tokenize


def rrf_fuse(ranked_lists, k=60, top=10):
    score = {}
    for lst in ranked_lists:
        for rank, doc in enumerate(lst, start=1):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda d: (-score[d], d))[:top]


def _precise_score(query, doc_text):
    """Exact joint similarity: token Jaccard + char sequence ratio."""
    qt, dt = set(tokenize(query)), set(tokenize(doc_text))
    jac = len(qt & dt) / max(1, len(qt | dt))
    seq = difflib.SequenceMatcher(None, query.lower(), doc_text.lower()).ratio()
    return 0.5 * jac + 0.5 * seq


def precise_rerank(query, candidate_ids, doc_texts, top=10):
    scored = [(d, _precise_score(query, doc_texts[d])) for d in candidate_ids]
    scored.sort(key=lambda x: -x[1])
    return [d for d, _ in scored[:top]]


class CrossEncoderReranker:
    """Neural reranker for the production run (needs a model download)."""

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidate_ids, doc_texts, top=10):
        pairs = [(query, doc_texts[d]) for d in candidate_ids]
        scores = self.model.predict(pairs)
        order = sorted(range(len(candidate_ids)), key=lambda i: -scores[i])
        return [candidate_ids[i] for i in order[:top]]
