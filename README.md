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


## Resources

Malkov & Yashunin 2016/2018 (HNSW paper), Johnson et al. 2017 (FAISS), Karpukhin et al. 2020 (DPR), Nogueira & Cho 2019 (BERT re-ranking)

