from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from index.bm25 import BM25Index
    from index.hnsw import HNSW


# --------------------------------------------------------------------- RRF
def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[tuple[str, float]]],
    k: int = 60,
    top_k: int = 10,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked lists via Reciprocal Rank Fusion (Cormack et al. 2009).

    Args:
        result_lists: list of ranked lists, each [(doc_id, score), ...]
            sorted in descending score order. Scores are ignored — only
            rank positions matter.
        k: RRF constant. Larger k -> slower decay, all ranks contribute
            more evenly.
        top_k: number of results to return.
        weights: optional per-list weights. Default: uniform (all 1.0).

    Returns:
        [(doc_id, rrf_score), ...] sorted by rrf_score descending.
    """
    if k < 1:
        raise ValueError("RRF k must be >= 1")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if weights is not None:
        if len(weights) != len(result_lists):
            raise ValueError(
                f"weights ({len(weights)}) and result_lists "
                f"({len(result_lists)}) must have the same length"
            )
        if any(w < 0 for w in weights):
            raise ValueError("weights must be >= 0")
    else:
        weights = [1.0] * len(result_lists)

    scores: dict[str, float] = {}
    for w, results in zip(weights, result_lists):
        if w == 0:
            continue
        seen: set[str] = set()
        for rank, (doc_id, _) in enumerate(results, start=1):
            if doc_id in seen:
                # duplicate within the same list — only the first hit counts
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + w / (k + rank)

    # stable sort — ties broken by dict insertion order
    return sorted(scores.items(), key=lambda x: -x[1])[:top_k]


# ---------------------------------------------------------- HybridRetriever
@dataclass
class HybridRetriever:
    """Combines dense (HNSW) and sparse (BM25) retrieval via RRF.

    The encoder is injected as an `encode_query` callable, so
    `index/hybrid.py` does not depend on `embeddings/`. Tests can pass a
    trivial encoder.
    """

    dense: "HNSW"
    sparse: "BM25Index"
    encode_query: Callable[[str], np.ndarray]
    rrf_k: int = 60

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        candidate_k: int | None = None,
        ef_search: int | None = None,
    ) -> list[tuple[str, float]]:
        """Run retrieval.

        Args:
            query: query text.
            top_k: number of results.
            mode: "dense" | "sparse" | "hybrid".
            candidate_k: how many candidates to pull from each retriever
                before fusion. Default: max(top_k * 5, 50). Ignored when
                mode != "hybrid".
            ef_search: override efSearch for the dense branch (for
                benchmarking).
        """
        if mode == "dense":
            return self._dense(query, top_k, ef_search)
        if mode == "sparse":
            return self.sparse.search(query, k=top_k)
        if mode != "hybrid":
            raise ValueError(
                f"unknown mode: {mode!r} (expected dense/sparse/hybrid)"
            )

        ck = candidate_k or max(top_k * 5, 50)
        dense_res = self._dense(query, ck, ef_search)
        sparse_res = self.sparse.search(query, k=ck)
        return reciprocal_rank_fusion(
            [dense_res, sparse_res],
            k=self.rrf_k,
            top_k=top_k,
        )

    def _dense(
        self,
        query: str,
        k: int,
        ef_search: int | None,
    ) -> list[tuple[str, float]]:
        qv = self.encode_query(query)
        return self.dense.search(qv, k=k, ef_search=ef_search)