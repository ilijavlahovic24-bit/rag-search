from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from routes import router
from service import MiniRagService, set_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = MiniRagService(
        model_name=os.getenv(
            "MINIRAG_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        query_prompt_name=os.getenv("MINIRAG_QUERY_PROMPT") or None,
        reranker_model=os.getenv(
            "MINIRAG_RERANKER",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        enable_reranker=os.getenv("MINIRAG_ENABLE_RERANK", "0") == "1",
        rrf_k=int(os.getenv("MINIRAG_RRF_K", "60")),
    )
    set_service(service)
    yield


app = FastAPI(title="Mini-RAG", version="0.1.0", lifespan=lifespan)
app.include_router(router)