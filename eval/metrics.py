from __future__ import annotations
from collections.abc import Mapping, Sequence
import math

def recall_at_k(retrieved: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    relevant_ids = {doc_id for doc_id, rel in relevant.items() if rel > 0}
    if not relevant_ids:
        return 0.0
    return len(top & relevant_ids) / len(relevant_ids)

def dcg_at_k(retrieved: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    score = 0.0
    for i, doc_id in enumerate(retrieved[:k], start=1):
        rel = relevant.get(doc_id, 0)
        score += (2**rel - 1) / math.log2(i + 1)
    return score

def ndcg_at_k(retrieved: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    dcg = dcg_at_k(retrieved, relevant, k)
    ideal = sorted((rel for rel in relevant.values() if rel > 0), reverse=True)[:k]
    idcg = sum((2**rel - 1) / math.log2(i + 1) for i, rel in enumerate(ideal, start=1))
    return dcg / idcg if idcg > 0 else 0.0