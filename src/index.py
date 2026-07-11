"""Vector indexes.

FlatIndex   - numpy brute-force cosine (exact baseline, no deps).
make_faiss  - FAISS Flat / HNSW / IVF-PQ for the efficiency iteration (L9).

All vectors are assumed L2-normalized, so inner product == cosine.
"""
import numpy as np


class FlatIndex:
    """Exact cosine search via a single dense matrix (Iteration 0 baseline)."""

    kind = "flat-numpy"

    def __init__(self, vectors):
        self.vectors = vectors.astype(np.float32)   # [N, D], normalized

    def search(self, q, k=10):
        # q: [D] or [n, D]
        if q.ndim == 1:
            q = q[None, :]
        sims = q @ self.vectors.T                    # cosine
        k = min(k, self.vectors.shape[0])
        idx = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        # sort the top-k per row
        rows = np.arange(q.shape[0])[:, None]
        order = np.argsort(-sims[rows, idx], axis=1)
        return idx[rows, order]                      # [n, k] doc indices

    def memory_mb(self):
        return self.vectors.nbytes / 1e6


def make_faiss(kind, vectors, nlist=None, m=8, hnsw_m=32):
    """Build a FAISS index. kind in {flat, hnsw, ivfpq}. Returns a wrapper with
    the same .search(q, k) -> indices interface."""
    import faiss
    v = np.ascontiguousarray(vectors.astype(np.float32))
    d = v.shape[1]

    if kind == "flat":
        index = faiss.IndexFlatIP(d)
        index.add(v)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        index.add(v)
    elif kind == "ivfpq":
        nlist = nlist or max(1, int(np.sqrt(len(v))))
        quant = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFPQ(quant, d, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(v)
        index.add(v)
        index.nprobe = min(16, nlist)
    else:
        raise ValueError(kind)

    class _W:
        def __init__(self, index, kind):
            self.index = index
            self.kind = "faiss-" + kind
        def search(self, q, k=10):
            if q.ndim == 1:
                q = q[None, :]
            _, idx = self.index.search(np.ascontiguousarray(q, dtype=np.float32), k)
            return idx
        def memory_mb(self):
            import faiss, tempfile, os
            f = tempfile.NamedTemporaryFile(delete=False)
            f.close()
            faiss.write_index(self.index, f.name)
            mb = os.path.getsize(f.name) / 1e6
            os.unlink(f.name)
            return mb

    return _W(index, kind)
