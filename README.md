# Mini-RAG

A search engine with a custom HNSW vector index implemented from scratch (not a FAISS black box) and hybrid retrieval (dense + sparse), wrapped in a FastAPI service layer.

## Motivation

The goal is to demonstrate understanding of retrieval mechanisms from the inside out, not just using off-the-shelf libraries (implement the mechanism, don't rely on a black box). 

## Architecture

```
mini-rag/
├── index/          # HNSW (core), BM25, hybrid (RRF) fusion
├── embeddings/     # sentence-transformers encoder
├── rerank/         # optional cross-encoder second pass
├── data/           # BEIR loader, embedding cache
├── eval/           # recall@k, nDCG@k, FAISS baseline comparison
├── api/            # FastAPI app (/index, /search)
├── benchmarks/     # latency/throughput scripts
└── doc/            # write-up, references
```

## Key decisions

- **HNSW implemented from scratch** in Python (layered graph structure, insert + greedy search), with M/efConstruction/efSearch as tunable parameters
- **BM25 via the `rank-bm25` library** (project focus is HNSW, not the sparse component)
- **Hybrid fusion via Reciprocal Rank Fusion (RRF)** - combines ranks rather than raw scores, avoiding score-range normalization issues
- **Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (small, runs on CPU)
- **Dataset**: a BEIR subset (SciFact or FiQA)
- **FAISS used only as a comparison baseline**, not a core system dependency
- **Service layer**: FastAPI (`/index`, `/search`)

## Evaluation

- Recall@k of the custom HNSW implementation vs. brute-force exact search (correctness) and vs. FAISS IndexHNSWFlat (accuracy/speed)
- nDCG@10, recall@10 for hybrid vs. dense-only vs. sparse-only retrieval
- Optional: nDCG improvement after cross-encoder re-ranking
- Latency benchmark (p50/p95) as a function of index size

##  Benchmarks (BEIR SciFact)

*Dataset: SciFact (5,183 docs, 300 test queries). Embeddings: `all-MiniLM-L6-v2` (dim=384). HNSW Params: M=16, efConstruction=200, efSearch=128.*

### 1. Retrieval Quality
Combining dense and sparse retrieval (Hybrid RRF) outperforms either method individually. Cross-encoder re-ranking pushes precision even higher.

| Mode | nDCG@10 | Recall@10 | MRR@10 |
|---|---|---|---|
| Dense (Our HNSW) | 0.6451 | 0.7833 | 0.6047 |
| Sparse (BM25) | 0.6523 | 0.7757 | 0.6184 |
| **Hybrid (RRF)** | **0.6821** | **0.8209** | **0.6426** |
| **Hybrid + Re-rank** | **0.6933** | **0.8222** | **0.6610** |

> *Insight: Hybrid fusion improves nDCG@10 by +3.7 percentage points over dense-only retrieval.*

### 2. HNSW vs. FAISS (Correctness & Latency)

Our pure-Python HNSW implementation achieves **identical retrieval quality** to FAISS (C++) at the same hyperparameters, proving algorithmic correctness.

| Metric | Our HNSW (Python) | FAISS HNSW (C++) | Delta |
|---|---|---|---|
| nDCG@10 | 0.6451 | 0.6451 | +0.00 pp (Parity) |
| Recall@10 | 0.7833 | 0.7833 | +0.00 pp (Parity) |
| Indexing time | 16.39 s | 0.23 s | **70.5x faster** (FAISS) |
| Query p50 (ms) | 14.22 ms | 11.11 ms | **1.3x faster** (FAISS) |

> *Insight: While C++ FAISS dominates in indexing speed, my Python HNSW is competitive in **query latency** (only 1.3x slower), proving that graph traversal overhead is minimal in Python.*

### 3. Query Latency Trade-offs

| Mode | p50 (ms) | p95 (ms) | Mean (ms) |
|---|---|---|---|
| Dense (HNSW) | 14.22 | 16.64 | 14.31 |
| Sparse (BM25) | 14.36 | 26.28 | 15.23 |
| Hybrid (RRF) | 32.46 | 48.24 | 33.94 |
| Hybrid + Re-rank | 2840.04 | 3194.67 | 2857.70 |

> *Insight: Cross-encoder re-ranking provides a +1.1 pp nDCG boost but costs ~2.8s per query on CPU, highlighting the classic latency-accuracy trade-off.*

### 4. ef_search Sensitivity (Dense Branch)

| ef_search | Recall@10 |
|---|---|
| 10 | 0.8160 |
| 20 | 0.8520 |
| 50 | 0.8720 |
| 100 | 0.8720 |
| 200 | 0.8720 |
| 400 | 0.8720 |

> *Insight: Recall plateaus at `ef_search >= 50`, confirming the algorithm's sensitivity to the search frontier width and providing a clear tuning target for production.*

## Resources
1. (["https://huggingface.co/blog/ngxson/make-your-own-rag"]) Create A simple RAG Hugging Face- starter tutorial
2. Malkov & Yashunin 2016/2018 (HNSW paper)
3. Johnson et al. 2017 (FAISS)
4. Karpukhin et al. 2020 (DPR)
5. Nogueira & Cho 2019 (BERT re-ranking)


