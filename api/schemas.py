from typing import Literal

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    text: str


class IndexRequest(BaseModel):
    documents: list[Document]


class IndexResponse(BaseModel):
    indexed: int
    mode: str = "dense"


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(10, ge=1, le=100)
    mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    rrf_k: int = Field(60, ge=1)


class SearchResult(BaseModel):
    doc_id: str
    score: float
    text: str | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult]