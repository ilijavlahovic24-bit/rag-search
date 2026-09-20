from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from embeding_model.encoder import Encoder
from index.bm25 import BM25Index
from index.hnsw import HNSW
from index.hybrid import HybridRetriever
from rerank.cross_encoder import CrossEncoderReranker


@dataclass
class SearchHit:
    doc_id: str
    score: float
    text: str | None = None


@dataclass
class MiniRagService:
    """Holds all retrieval state for the API.

    Single instance per process. `index()` replaces all state atomically
    (under a lock). `search()` is lock-free for reads — HNSW and BM25 are
    read-only after indexing.

    Encoder is created lazily on first `index()` call so the service can
    start (and answer /health) without loading the model.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    query_prompt_name: str | None = None
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enable_reranker: bool = False
    rerank_pool: int = 50

    hnsw_M: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 128

    rrf_k: int = 60

    # ---- runtime state (populated by index()) ----
    _encoder: Encoder | None = field(default=None, init=False, repr=False)
    _hnsw: HNSW | None = field(default=None, init=False, repr=False)
    _bm25: BM25Index | None = field(default=None, init=False, repr=False)
    _reranker: CrossEncoderReranker | None = field(default=None, init=False, repr=False)
    _documents: dict[str, str] = field(default_factory=dict, init=False, repr=False)
    _retriever: HybridRetriever | None = field(default=None, init=False, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False, repr=False)

    # ------------------------------------------------------------------ meta
    @property
    def is_indexed(self) -> bool:
        return self._retriever is not None

    @property
    def n_documents(self) -> int:
        return len(self._documents)

    def _ensure_encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder(
                model_name=self.model_name,
                query_prompt_name=self.query_prompt_name,
                model_kwargs=self.model_kwargs,
            )
        return self._encoder

    def _ensure_reranker(self) -> CrossEncoderReranker:
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                model_name=self.reranker_model
            )
        return self._reranker

    # ---------------------------------------------------------------- index
    def index(self, documents: list[tuple[str, str]]) -> int:
        """Build HNSW + BM25 from (doc_id, text) pairs.

        Replaces any previous index. Blocking — pure-Python HNSW insert is
        slow for large corpora. Caller should run in a threadpool or accept
        the latency.
        """
        if not documents:
            raise ValueError("documents must not be empty")

        doc_ids = [d for d, _ in documents]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("duplicate doc_ids in request")

        texts = [t for _, t in documents]

        encoder = self._ensure_encoder()
        embeddings = encoder.encode_documents(texts, show_progress=False)

        hnsw = HNSW(
            dim=embeddings.shape[1],
            M=self.hnsw_M,
            ef_construction=self.hnsw_ef_construction,
            ef_search=self.hnsw_ef_search,
        )
        for i, doc_id in enumerate(doc_ids):
            hnsw.insert(embeddings[i], doc_id)

        bm25 = BM25Index()
        bm25.index(doc_ids, texts)

        def encode_query(q: str) -> np.ndarray:
            return encoder.encode_queries([q], show_progress=False)[0]

        retriever = HybridRetriever(
            dense=hnsw,
            sparse=bm25,
            encode_query=encode_query,
            rrf_k=self.rrf_k,
        )

        with self._lock:
            self._hnsw = hnsw
            self._bm25 = bm25
            self._retriever = retriever
            self._documents = dict(documents)

        return len(documents)

    # --------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "hybrid",
        rrf_k: int | None = None,
        ef_search: int | None = None,
        rerank: bool = False,
        rerank_pool: int | None = None,
    ) -> list[SearchHit]:
        with self._lock:
            retriever = self._retriever
            documents = self._documents

        if retriever is None:
            raise RuntimeError("index is empty — call /index first")

        if rrf_k is not None:
            retriever = HybridRetriever(
                dense=retriever.dense,
                sparse=retriever.sparse,
                encode_query=retriever.encode_query,
                rrf_k=rrf_k,
            )

        if rerank:
            pool = rerank_pool or self.rerank_pool
            candidates = retriever.search(query, top_k=pool, mode=mode)
            reranker = self._ensure_reranker()
            ranked = reranker.rerank(query, candidates, documents, top_k=top_k)
        else:
            ranked = retriever.search(
                query, top_k=top_k, mode=mode, ef_search=ef_search
            )

        return [
            SearchHit(doc_id=doc_id, score=score, text=documents.get(doc_id))
            for doc_id, score in ranked
        ]

    # --------------------------------------------------------------- stats
    def stats(self) -> dict[str, Any]:
        return {
            "indexed": self.is_indexed,
            "n_documents": self.n_documents,
            "model_name": self.model_name,
            "query_prompt_name": self.query_prompt_name,
            "reranker_enabled": self.enable_reranker,
            "hnsw": self._hnsw.stats() if self._hnsw else None,
            "rrf_k": self.rrf_k,
        }


# --------------------------------------------------------------------- DI
_service: MiniRagService | None = None
_service_lock = threading.Lock()


def get_service() -> MiniRagService:
    """Process-wide singleton. Override in tests by calling set_service()."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = MiniRagService()
    return _service


def set_service(service: MiniRagService) -> None:
    """Test hook / startup hook to inject a configured service."""
    global _service
    with _service_lock:
        _service = service