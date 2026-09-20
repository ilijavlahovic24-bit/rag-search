from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def default_tokenize(text: str) -> list[str]:
    """Lowercase + extract alphanumeric tokens

    Simple, without stemming and stopword filtering — for BEIR that is
    standard and sufficient. If you want a stemmer, insert it here and consistently
    used for both corpus and query.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Sparse index over document corpus.

    Interface deliberately the same as HNSW.search: returns [(doc_id, score), ...]
    sorted in descending order of score.
    """

    def __init__(
            self,
            k1: float = 1.5,
            b: float = 0.75,
            tokenizer=default_tokenize,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.tokenizer = tokenizer

        self.doc_ids: list[str] = []
        self.tokenized: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

        # ------------------------------------------------------------- build

    def index(self, doc_ids: list[str], texts: list[str]) -> None:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have same length")
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("doc_ids contains duplicate doc_ids")

        self.doc_ids = list(doc_ids)
        self.tokenized = [self.tokenizer(t) for t in texts]
        self._bm25 = BM25Okapi(self.tokenized)
        self._bm25.k1 = self.k1
        self._bm25.b = self.b

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        tokens = self.tokenizer(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # partial sort
        order = scores.argsort()[::-1][:k]
        return [(self.doc_ids[i], float(scores[i])) for i in order]

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in set(self.doc_ids)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"k1": self.k1,"b": self.b,"doc_ids": self.doc_ids, "tokenized": self.tokenized,
                    # _bm25 is not pickled: it is reconstructed from tokenized
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        path = Path(path)
        with path.open("rb") as f:
            data = pickle.load(f)
        idx = cls(k1=data["k1"], b=data["b"])
        idx.doc_ids = data["doc_ids"]
        idx.tokenized = data["tokenized"]
        idx._bm25 = BM25Okapi(idx.tokenized)
        idx._bm25.k1 = idx.k1
        idx._bm25.b = idx.b
        return idx