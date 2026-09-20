from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from schemas import (
    IndexRequest,
    IndexResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)
from service import MiniRagService, get_service

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/stats", response_model=StatsResponse)
def stats(service: MiniRagService = Depends(get_service)) -> StatsResponse:
    return StatsResponse(**service.stats())


@router.post("/index", response_model=IndexResponse)
def index_documents(
    req: IndexRequest,
    service: MiniRagService = Depends(get_service),
) -> IndexResponse:
    docs = [(d.id, d.text) for d in req.documents]
    try:
        n = service.index(docs)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e
    except Exception as e:  # noqa: BLE001 — surface encoder/model errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"indexing failed: {e}",
        ) from e
    return IndexResponse(indexed=n, n_documents=service.n_documents)


@router.post("/search", response_model=SearchResponse)
def search(
    req: SearchRequest,
    service: MiniRagService = Depends(get_service),
) -> SearchResponse:
    try:
        hits = service.search(
            query=req.query,
            top_k=req.top_k,
            mode=req.mode,
            rrf_k=req.rrf_k,
            ef_search=req.ef_search,
            rerank=req.rerank,
            rerank_pool=req.rerank_pool,
        )
    except RuntimeError as e:
        # "index is empty"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(e)
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e

    return SearchResponse(
        query=req.query,
        mode=req.mode,
        rerank=req.rerank,
        results=[SearchHit(**h.__dict__) for h in hits],
    )