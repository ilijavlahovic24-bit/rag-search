"""End-to-end benchmark: all numbers for the CV / report."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# --- project-root bootstrap: works no matter where you launch from ---
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ---------------------------------------------------------------------

import numpy as np
from tqdm import tqdm

from data.beir_loader import load_beir
# ISPRAVLJENO: Tvoj folder se zove embeding_model
from embeding_model.encoder import Encoder
from eval.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from index.bm25 import BM25Index
from index.hnsw import HNSW
from index.hybrid import HybridRetriever
from rerank.cross_encoder import CrossEncoderReranker

# Graceful FAISS import
try:
    from eval.faiss_baseline import FaissHNSW
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# ----------------------------------------------------------------- constants
MODELS = {
    "minilm": (
        "sentence-transformers/all-MiniLM-L6-v2",
        None,  # no query prompt
    ),
    "harrier": (
        "microsoft/harrier-oss-v1-27b",
        "web_search_query",
    ),
}


class FakeEncoder:
    def __init__(self, dim: int = 64) -> None:
        self.dim = dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(0)
        out = rng.normal(size=(len(texts), self.dim)).astype(np.float32)
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        return out

    def encode_documents(self, texts, cache_key=None, show_progress=True):
        return self._encode(texts)

    def encode_queries(self, texts, cache_key=None, show_progress=False):
        return self._encode(texts)


# ------------------------------------------------------------------ helpers
def _percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    idx = max(0, min(len(xs) - 1, int(p * len(xs)) - 1))
    return xs[idx]


def evaluate_retrieval(
    retrieve,
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
    k: int,
    max_queries: int | None = None,
    desc: str = "Eval",
) -> dict[str, float]:
    """Compute nDCG@k, Recall@k, MRR@k averages over queries with qrels."""
    totals = {"nDCG": 0.0, "Recall": 0.0, "MRR": 0.0}
    n = 0

    valid_queries = []
    for qid, rel in qrels.items():
        if not any(v > 0 for v in rel.values()):
            continue
        qtext = queries.get(qid)
        if qtext is not None:
            valid_queries.append((qid, rel, qtext))

    if max_queries is not None:
        valid_queries = valid_queries[:max_queries]

    for qid, rel, qtext in tqdm(valid_queries, desc=desc, leave=False):
        results = retrieve(qtext, k)
        retrieved = [doc_id for doc_id, _ in results]
        totals["nDCG"] += ndcg_at_k(retrieved, rel, k)
        totals["Recall"] += recall_at_k(retrieved, rel, k)
        totals["MRR"] += mrr_at_k(retrieved, rel, k)
        n += 1

    if n == 0:
        raise RuntimeError("No queries evaluated")
    return {f"{name}@{k}": v / n for name, v in totals.items()} | {"n_queries": n}


def measure_latency(
    retrieve,
    queries: list[str],
    k: int,
    warmup: int = 3,
    desc: str = "Latency"
) -> dict:
    """Time a retrieval callable over a query list. Warmup passes excluded."""
    for q in queries[:warmup]:
        retrieve(q, k)

    latencies: list[float] = []
    for q in tqdm(queries, desc=desc, leave=False):
        t0 = time.perf_counter()
        retrieve(q, k)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return {
        "p50_ms": statistics.median(latencies),
        "p95_ms": _percentile(latencies, 0.95),
        "mean_ms": statistics.mean(latencies),
        "n": len(latencies),
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = "| " + " | ".join(headers) + " |\n"
    out += "|" + "|".join("---" for _ in headers) + "|\n"
    for row in rows:
        out += "| " + " | ".join(row) + " |\n"
    return out


# -------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--split", default="test")
    ap.add_argument("--encoder", choices=["minilm", "harrier", "fake"], default="minilm")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--efs", type=int, default=128)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--rerank-pool", type=int, default=50)
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--no-faiss", action="store_true")
    ap.add_argument("--sweep-ef", action="store_true",
                    help="also run ef_search sensitivity (slower)")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="evaluate on first N queries only (for quick runs)")
    ap.add_argument("--out", default="results/full_report.json")
    args = ap.parse_args()

    if not args.no_faiss and not HAS_FAISS:
        print("⚠️ FAISS nije instaliran. Preskačem FAISS baseline. (pip install faiss-cpu)")
        args.no_faiss = True

    # ------------------------------------------------ load + encode
    ds = load_beir(args.dataset, split=args.split)
    print(f"# dataset: {args.dataset}  corpus={len(ds.corpus)}  "
          f"queries={len(ds.queries)}  qrels={len(ds.qrels)}\n")

    if args.encoder == "fake":
        encoder = FakeEncoder()
    else:
        model_name, prompt = MODELS[args.encoder]
        encoder = Encoder(model_name=model_name, query_prompt_name=prompt)

    doc_ids = list(ds.corpus.keys())
    texts = [ds.corpus[i] for i in doc_ids]

    t0 = time.perf_counter()
    print(f"# Encoding {len(texts)} documents...")
    corpus_emb = encoder.encode_documents(texts, cache_key=f"{args.dataset}_corpus")
    t_encode = time.perf_counter() - t0
    print(f"# encoded corpus in {t_encode:.2f}s  shape={corpus_emb.shape}  "
          f"dim={corpus_emb.shape[1]}\n")

    def encode_query(q: str) -> np.ndarray:
        return encoder.encode_queries([q], show_progress=False)[0]

    # ------------------------------------------------ build: ours
    print("# Building our HNSW...")
    t0 = time.perf_counter()
    my_hnsw = HNSW(dim=corpus_emb.shape[1], M=args.M,
                   ef_construction=args.efc, ef_search=args.efs)
    for i, did in enumerate(tqdm(doc_ids, desc="Our HNSW Insert")):
        my_hnsw.insert(corpus_emb[i], did)
    t_index_ours = time.perf_counter() - t0
    print(f"# ours indexed in {t_index_ours:.2f}s\n")

    # ------------------------------------------------ build: FAISS
    faiss_hnsw = None
    t_index_faiss = 0.0
    if not args.no_faiss:
        print("# Building FAISS HNSW...")
        t0 = time.perf_counter()
        faiss_hnsw = FaissHNSW(dim=corpus_emb.shape[1], M=args.M,
                               ef_construction=args.efc, ef_search=args.efs)
        faiss_hnsw.insert_many(corpus_emb, doc_ids)
        t_index_faiss = time.perf_counter() - t0
        print(f"# FAISS indexed in {t_index_faiss:.2f}s\n")

    # ------------------------------------------------ build: BM25 + hybrid
    print("# Building BM25...")
    t0 = time.perf_counter()
    bm25 = BM25Index()
    bm25.index(doc_ids, texts)
    t_index_bm25 = time.perf_counter() - t0
    print(f"# BM25 indexed in {t_index_bm25:.2f}s\n")

    hybrid = HybridRetriever(
        dense=my_hnsw, sparse=bm25,
        encode_query=encode_query, rrf_k=args.rrf_k,
    )

    reranker = None
    if not args.no_rerank:
        print("# Loading Reranker...")
        reranker = CrossEncoderReranker(model_name=args.reranker)
        print("# Reranker ready.\n")

    # ------------------------------------------------ evaluation
    query_list = [q for q in ds.queries.values()]
    results_quality: dict[str, dict[str, float]] = {}

    def eval_mode(name: str, retrieve):
        m = evaluate_retrieval(
            retrieve, ds.queries, ds.qrels,
            k=args.top_k, max_queries=args.max_queries, desc=f"Eval {name}"
        )
        results_quality[name] = m

    print("# --- Evaluating Retrieval Quality ---")
    eval_mode("dense (ours)", lambda q, k: my_hnsw.search(encode_query(q), k=k))
    if faiss_hnsw is not None:
        eval_mode("dense (faiss)", lambda q, k: faiss_hnsw.search(encode_query(q), k=k))
    eval_mode("sparse (BM25)", lambda q, k: bm25.search(q, k=k))
    eval_mode("hybrid (RRF)", lambda q, k: hybrid.search(q, top_k=k, mode="hybrid"))

    if reranker is not None:
        def _rerank(q, k):
            cand = hybrid.search(q, top_k=args.rerank_pool, mode="hybrid")
            return reranker.rerank(q, cand, ds.corpus, top_k=k)
        eval_mode("hybrid + rerank", _rerank)

    # ------------------------------------------------ latency
    print("\n# --- Measuring Query Latency ---")
    latencies: dict[str, dict] = {}
    lat_queries = query_list[:args.max_queries] if args.max_queries else query_list

    latencies["dense (ours)"] = measure_latency(
        lambda q, k: my_hnsw.search(encode_query(q), k=k), lat_queries, args.top_k, desc="Lat dense (ours)")
    if faiss_hnsw is not None:
        latencies["dense (faiss)"] = measure_latency(
            lambda q, k: faiss_hnsw.search(encode_query(q), k=k), lat_queries, args.top_k, desc="Lat dense (faiss)")
    latencies["sparse (BM25)"] = measure_latency(
        lambda q, k: bm25.search(q, k=k), lat_queries, args.top_k, desc="Lat sparse (BM25)")
    latencies["hybrid (RRF)"] = measure_latency(
        lambda q, k: hybrid.search(q, top_k=k, mode="hybrid"), lat_queries, args.top_k, desc="Lat hybrid (RRF)")
    if reranker is not None:
        latencies["hybrid + rerank"] = measure_latency(
            _rerank, lat_queries, args.top_k, desc="Lat hybrid + rerank")

    # ------------------------------------------------ ef_search sweep
    ef_sweep: list[dict] = []
    if args.sweep_ef:
        print("\n# --- ef_search sensitivity sweep ---")
        for ef in (10, 20, 50, 100, 200, 400):
            def retrieve_sweep(q, k, _ef=ef):
                return my_hnsw.search(encode_query(q), k=k, ef_search=_ef)

            m = evaluate_retrieval(
                retrieve_sweep, ds.queries, ds.qrels, k=args.top_k,
                max_queries=50, desc=f"Eval efS={ef}"
            )
            ef_sweep.append({
                "ef_search": ef,
                "recall@k": m[f"Recall@{args.top_k}"],
            })

    # ------------------------------------------------ print tables
    print("\n\n")

    # Table 1: retrieval quality
    headers = ["Mode", f"nDCG@{args.top_k}", f"Recall@{args.top_k}", f"MRR@{args.top_k}"]
    rows = [[name,
             f"{m[f'nDCG@{args.top_k}']:.4f}",
             f"{m[f'Recall@{args.top_k}']:.4f}",
             f"{m[f'MRR@{args.top_k}']:.4f}"]
            for name, m in results_quality.items()]
    print("## Retrieval quality\n")
    print(markdown_table(headers, rows))

    # Table 2: latency
    headers = ["Mode", "p50 (ms)", "p95 (ms)", "mean (ms)"]
    rows = [[name,
             f"{lat['p50_ms']:.2f}",
             f"{lat['p95_ms']:.2f}",
             f"{lat['mean_ms']:.2f}"]
            for name, lat in latencies.items()]
    print("## Query latency\n")
    print(markdown_table(headers, rows))

    # Table 3: HNSW vs FAISS
    if faiss_hnsw is not None:
        ours_q = results_quality["dense (ours)"][f"nDCG@{args.top_k}"]
        faiss_q = results_quality["dense (faiss)"][f"nDCG@{args.top_k}"]
        ours_rec = results_quality["dense (ours)"][f"Recall@{args.top_k}"]
        faiss_rec = results_quality["dense (faiss)"][f"Recall@{args.top_k}"]

        # ISPRAVLJENO: t_index_ours / t_index_faiss (da bi dobio npr. 94.2x)
        speedup_idx = t_index_ours / max(t_index_faiss, 1e-9)
        speedup_q = latencies["dense (ours)"]["p50_ms"] / max(
            latencies["dense (faiss)"]["p50_ms"], 1e-9)

        rows = [
            ["nDCG@10", f"{ours_q:.4f}", f"{faiss_q:.4f}",
             f"{(ours_q - faiss_q) * 100:+.2f} pp"],
            ["Recall@10", f"{ours_rec:.4f}", f"{faiss_rec:.4f}",
             f"{(ours_rec - faiss_rec) * 100:+.2f} pp"],
            ["Indexing time (s)", f"{t_index_ours:.2f}", f"{t_index_faiss:.2f}",
             f"{speedup_idx:.1f}x faster (FAISS)"],
            ["Query p50 (ms)", f"{latencies['dense (ours)']['p50_ms']:.2f}",
             f"{latencies['dense (faiss)']['p50_ms']:.2f}",
             f"{speedup_q:.1f}x faster (FAISS)"],
        ]
        headers = ["Metric", "Our HNSW", "FAISS HNSW", "Delta"]
        print(f"## Our HNSW vs FAISS HNSW (M={args.M}, efC={args.efc}, efS={args.efs})\n")
        print(markdown_table(headers, rows))

    # Table 4: ef_search sweep
    if ef_sweep:
        # ISPRAVLJENO: Sada traži "recall@k" i ispisuje "Recall@k"
        headers = ["ef_search", "Recall@k"]
        rows = [[str(r["ef_search"]), f"{r['recall@k']:.4f}"] for r in ef_sweep]
        print("## ef_search sensitivity (dense branch only)\n")
        print(markdown_table(headers, rows))

    # ------------------------------------------------ persistence
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": vars(args),
        "quality": results_quality,
        "latency": latencies,
        "indexing_times_s": {
            "ours": t_index_ours,
            "faiss": t_index_faiss,
            "bm25": t_index_bm25,
            "encode": t_encode,
        },
        "ef_sweep": ef_sweep,
        "corpus_stats": {
            "n_docs": len(ds.corpus),
            "n_queries": len(ds.queries),
            "embedding_dim": int(corpus_emb.shape[1]),
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\n# saved: {args.out}")


if __name__ == "__main__":
    main()