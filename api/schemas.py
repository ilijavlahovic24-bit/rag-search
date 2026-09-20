from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ------------------------------------------------------------------ /index
class Document(BaseModel):
    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class IndexRequest(BaseModel):
    documents: list[Document] = Field(..., min_length=1)


class IndexResponse(BaseModel):
    indexed: int
    n_documents: int


# ----------------------------------------------------------------- /search
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(10, ge=1, le=100)
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    rrf_k: int | None = Field(None, ge=1)
    ef_search: int | None = Field(None, ge=1)
    rerank: bool = False
    rerank_pool: int | None = Field(None, ge=1, le=500)


class SearchHit(BaseModel):
    doc_id: str
    score: float
    text: str | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    rerank: bool
    results: list[SearchHit]


# ------------------------------------------------------------------ /stats
class StatsResponse(BaseModel):
    indexed: bool
    n_documents: int
    model_name: str
    query_prompt_name: str | None
    reranker_enabled: bool
    hnsw: dict | None
    rrf_k: int