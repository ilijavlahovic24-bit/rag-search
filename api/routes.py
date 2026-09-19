from fastapi import APIRouter, HTTPException

from schemas import IndexRequest, IndexResponse, SearchRequest, SearchResponse

router = APIRouter()
@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/index", response_model=IndexResponse)
def index_documents(req: IndexRequest) -> IndexResponse:
    # TODO embedding + HNSW/BM25 indexing
    raise HTTPException(status_code=501, detail="Indexing not implemented yet")


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    # TODO dense/sparse/hybrid search
    raise HTTPException(status_code=501, detail="Search not implemented yet")