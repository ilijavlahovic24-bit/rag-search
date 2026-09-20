from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder


@dataclass
class CrossEncoderReranker:
    """Second-pass reranker over a candidate list from hybrid retrieval.

    The cross-encoder scores (query, document) pairs jointly — much more
    accurate than dot-product similarity, but O(pool_size) forward passes,
    so it only makes sense over a small candidate pool (top-50 to top-100).

    `model` can be injected for testing; otherwise a real CrossEncoder is
    loaded from `model_name`.
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 32
    max_length: int = 512
    model: Any = None

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = CrossEncoder(self.model_name, max_length=self.max_length)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, float]],
        documents: dict[str, str],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """Rescore `candidates` with the cross-encoder.

        Args:
            query: query text.
            candidates: [(doc_id, prev_score), ...] from the first stage.
                The previous scores are discarded.
            documents: doc_id -> text mapping for lookup.
            top_k: truncate to this many results. None keeps all.

        Returns:
            [(doc_id, rerank_score), ...] sorted descending by rerank score.
            Candidates whose doc_id is missing from `documents` are dropped.
        """
        if not candidates:
            return []

        pairs: list[tuple[str, str]] = []
        kept_ids: list[str] = []
        for doc_id, _ in candidates:
            text = documents.get(doc_id)
            if text is None:
                continue
            pairs.append((query, text))
            kept_ids.append(doc_id)

        if not pairs:
            return []

        raw_scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        ranked = sorted(
            zip(kept_ids, (float(s) for s in raw_scores)),
            key=lambda x: -x[1],
        )
        if top_k is not None:
            ranked = ranked[:top_k]
        return ranked