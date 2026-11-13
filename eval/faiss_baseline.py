from __future__ import annotations

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "FAISS baseline requires `faiss-cpu`. Install: pip install faiss-cpu"
    ) from e


class FaissHNSW:
    """FAISS IndexHNSWFlat wrapper with the same interface as our HNSW.

    Uses L2 metric on normalized vectors, which is equivalent to cosine
    similarity for ranking (||a-b||^2 = 2 - 2·a·b for unit vectors).
    """

    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 128,
    ) -> None:
        self.dim = dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self.ids: list[str] = []

    def insert_many(self, vectors: np.ndarray, doc_ids: list[str]) -> None:
        """FAISS HNSW does not support incremental single-vector insert
        efficiently — batch add is the recommended path."""
        if vectors.shape[0] != len(doc_ids):
            raise ValueError("vectors and doc_ids must have the same length")
        self.index.add(vectors.astype(np.float32))
        self.ids.extend(doc_ids)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef_search: int | None = None,
    ) -> list[tuple[str, float]]:
        if ef_search is not None:
            self.index.hnsw.efSearch = ef_search
        if self.index.ntotal == 0:
            return []

        q = query.astype(np.float32).reshape(1, -1)
        distances, indices = self.index.search(q, k)

        out: list[tuple[str, float]] = []
        for d2, idx in zip(distances[0], indices[0]):
            if idx < 0:  # FAISS pad for missing results
                continue
            # for unit vectors: cos_sim = 1 - d2/2
            score = 1.0 - float(d2) / 2.0
            out.append((self.ids[idx], score))
        return out

    def __len__(self) -> int:
        return self.index.ntotal