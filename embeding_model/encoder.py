from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class Encoder:
    """Sentence-transformers wrapper with separate query/document paths
    and on-disk embedding cache.

    Asymmetric models (E5, BGE, Harrier, etc.) require a prompt on the query
    side and no prompt on the document side. Pass `query_prompt_name` to
    enable this; leave it None for symmetric models like all-MiniLM-L6-v2.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        query_prompt_name: str | None = None,
        cache_dir: str | Path = "data/cache/embeddings",
        batch_size: int = 64,
        normalize: bool = True,
        model_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.query_prompt_name = query_prompt_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = SentenceTransformer(
            model_name, model_kwargs=model_kwargs or {}
        )

    # ---------------------------------------------------------------- cache
    def _cache_path(self, key: str) -> Path:
        safe_model = self.model_name.replace("/", "_").replace("\\", "_")
        return self.cache_dir / f"{safe_model}__{key}.npy"

    # --------------------------------------------------------------- encode
    def _encode(
        self,
        texts: list[str],
        prompt_name: str | None,
        cache_key: str | None,
        show_progress: bool,
    ) -> np.ndarray:
        if cache_key:
            path = self._cache_path(cache_key)
            if path.exists():
                return np.load(path)

        kwargs: dict[str, Any] = {
            "batch_size": self.batch_size,
            "show_progress_bar": show_progress,
            "convert_to_numpy": True,
            "normalize_embeddings": self.normalize,
        }
        if prompt_name is not None:
            kwargs["prompt_name"] = prompt_name

        emb = self.model.encode(texts, **kwargs)
        if cache_key:
            np.save(self._cache_path(cache_key), emb)
        return emb

    def encode_documents(
        self,
        texts: list[str],
        cache_key: str | None = None,
        show_progress: bool = True,
    ) -> np.ndarray:
        return self._encode(texts, None, cache_key, show_progress)

    def encode_queries(
        self,
        texts: list[str],
        cache_key: str | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        return self._encode(texts, self.query_prompt_name, cache_key, show_progress)