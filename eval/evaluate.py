from __future__ import annotations

import argparse
from collections.abc import Callable

import numpy as np

from data.beir_loader import load_beir
from embeding_model.encoder import Encoder
from eval.metrics import ndcg_at_k, recall_at_k
from index.bm25 import BM25Index
from index.hnsw import HNSW
from index.hybrid import HybridRetriever
from rerank.cross_encoder import CrossEncoderReranker

RetrieveFn = Callable[[str, int], list[tuple[str, float]]]


def evaluate(
    retrieve: RetrieveFn,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k_values: tuple[int, ...] = (10,),
) -> dict[str, float]:
    """Return {metric: avg_value, ..., "n_queries": int}.

    Queries without any relevant document are skipped (BEIR convention).
    """
    max_k = max(k_values)
    totals: dict[str, float] = {}
    for k in k_values:
        totals[f"recall@{k}"] = 0.0
        totals[f"ndcg@{k}"] = 0.0

    n = 0
    for qid, rel in qrels.items():
        if not any(v > 0 for v in rel.values()):
            continue
        qtext = queries.get(qid)
        if qtext is None:
            continue
        results = retrieve(qtext, max_k)
        retrieved = [doc_id for doc_id, _ in results]
        for k in k_values:
            totals[f"recall@{k}"] += recall_at_k(retrieved, rel, k)
            totals[f"ndcg@{k}"] += ndcg_at_k(retrieved, rel, k)
        n += 1

    if n == 0:
        raise RuntimeError("No queries evaluated — check qrels/queries")

    out = {name: v / n for name, v in totals.items()}
    out["n_queries"] = float(n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="embedding model (e.g. microsoft/harrier-oss-v1-27b)",
    )
    ap.add_argument(
        "--query-prompt",
        default=None,
        help="prompt_name for query encoding (e.g. web_search_query)",
    )
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--efs", type=int, default=128)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--reranker-model",
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
    )
    ap.add_argument(
        "--rerank-pool",
        type=int,
        default=50,
        help="hybrid candidates fed into the reranker",
    )
    ap.add_argument(
        "--no-rerank",
        action="store_true",
        help="skip the hybrid+rerank row",
    )
    args = ap.parse_args()

    ds = load_beir(args.dataset, split=args.split)
    print(
        f"corpus={len(ds.corpus)}  queries={len(ds.queries)}  "
        f"qrels={len(ds.qrels)}"
    )

    encoder = Encoder(
        model_name=args.model,
        query_prompt_name=args.query_prompt,
    )
    doc_ids = list(ds.corpus.keys())
    doc_texts = [ds.corpus[i] for i in doc_ids]

    corpus_emb = encoder.encode_documents(
        doc_texts, cache_key=f"{args.dataset}_corpus"
    )
    print(f"encoded corpus: {corpus_emb.shape}")

    hnsw = HNSW(
        dim=corpus_emb.shape[1],
        M=args.M,
        ef_construction=args.efc,
        ef_search=args.efs,
    )
    for i, doc_id in enumerate(doc_ids):
        hnsw.insert(corpus_emb[i], doc_id)
    print(f"HNSW: {hnsw.stats()}")

    bm25 = BM25Index()
    bm25.index(doc_ids, doc_texts)
    print(f"BM25: {len(bm25)} documents")

    def encode_query(q: str) -> np.ndarray:
        return encoder.encode_queries([q], show_progress=False)[0]

    retriever = HybridRetriever(
        dense=hnsw, sparse=bm25, encode_query=encode_query, rrf_k=args.rrf_k
    )

    modes: list[str] = ["dense", "sparse", "hybrid"]
    reranker: CrossEncoderReranker | None = None
    if not args.no_rerank:
        reranker = CrossEncoderReranker(model_name=args.reranker_model)
        modes.append("hybrid+rerank")

    rows: list[dict] = []
    for mode in modes:

        def retrieve(
            q: str,
            k: int,
            _mode: str = mode,
        ) -> list[tuple[str, float]]:
            if _mode == "hybrid+rerank":
                assert reranker is not None
                pool = retriever.search(
                    q, top_k=args.rerank_pool, mode="hybrid"
                )
                return reranker.rerank(q, pool, ds.corpus, top_k=k)
            return retriever.search(q, top_k=k, mode=_mode)

        metrics = evaluate(
            retrieve, ds.queries, ds.qrels, k_values=(args.top_k,)
        )
        metrics["mode"] = mode
        rows.append(metrics)
        print(f"\n[{mode}]")
        for name, val in metrics.items():
            if name == "mode":
                continue
            if name == "n_queries":
                print(f"  n_queries = {int(val)}")
            else:
                print(f"  {name}: {val:.4f}")

    print(
        "\n\n| mode | "
        + " | ".join(f"nDCG@{args.top_k}", f"Recall@{args.top_k}")
        + " |"
    )
    print("|---|---|---|")
    for r in rows:
        print(
            f"| {r['mode']} | "
            f"{r[f'ndcg@{args.top_k}']:.4f} | "
            f"{r[f'recall@{args.top_k}']:.4f} |"
        )


if __name__ == "__main__":
    main()